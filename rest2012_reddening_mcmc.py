#!/usr/bin/env python3
"""
Joint MCMC: stellar Teff + line-of-sight E(B-V) from EC1 light-echo spectra.

Combines two independent constraints on the same Pickles (1998) LC-I templates:
  1. Continuum color ratios in clean bandpasses (flux-calibrated; reddening-sensitive)
  2. Absorption-line shapes in Mg b and Ca II / Paschen windows (from rest2012_joint_mcmc)

Free parameters
  Teff            effective temperature (interpolated Pickles grid)
  ebv             line-of-sight color excess E(B-V)
  ln_sigma_mgb    log noise scale, Mg b window
  ln_sigma_caii   log noise scale, Ca II / Paschen window

Teff carries a Gaussian prior anchored to the line-shape MCMC (5429 ± 180 K).
Fixed: v = -210 km/s (Rest+ Ca II); R_V (Smith+2018 Carina default 4.8).

Outputs → figures/rest2012_spectral_type/reddening/
"""

from __future__ import annotations

import json
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rest2012_joint_mcmc as jm
import rest2012_spectral_type as r12

ROOT = Path(__file__).resolve().parent
FIGDIR = r12.FIGDIR / "reddening"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Continuum bands (Å): avoid strong nebular / ISM lines (Rest+ Table S4 style).
COLOR_BANDS = {
    "blue": (4500.0, 4700.0),
    "green": (5500.0, 5700.0),
    "red": (6400.0, 6600.0),
    "nir": (8420.0, 8580.0),
}
COLOR_RATIOS = [
    ("blue", "red"),
    ("blue", "nir"),
    ("green", "red"),
    ("blue", "green"),
]

RV_DEFAULT = 4.8  # Smith et al. 2018, Carina / Eta Car sightline
EBV_LO, EBV_HI = 0.0, 1.5
# Gaussian prior on Teff from line-shape MCMC (rest2012_joint_mcmc); breaks color degeneracy.
TEFF_PRIOR_CENTER = 5429.0
TEFF_PRIOR_SIGMA = 180.0

N_WALKERS = 48
N_STEPS = 4000
N_BURN = 1000
SEED = 7


def ccm89_alambda_av(wave_angstrom, rv: float = RV_DEFAULT) -> np.ndarray:
    """A(λ)/A(V) from Cardelli, Clayton & Mathis (1989), 0.3 < λ⁻¹ < 3.3 μm⁻¹."""
    x = 1.0 / (np.asarray(wave_angstrom, dtype=float) * 1e-4)
    a = np.empty_like(x)
    b = np.empty_like(x)
    ir = x <= 1.1
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61
    opt = ~ir
    y = x[opt] - 1.82
    a[opt] = (
        1.0
        + 0.17699 * y
        - 0.50447 * y**2
        - 0.02427 * y**3
        + 0.72085 * y**4
        + 0.01979 * y**5
        - 0.77530 * y**6
        + 0.32999 * y**7
    )
    b[opt] = (
        1.41338 * y
        + 2.28305 * y**2
        + 1.07233 * y**3
        - 5.38434 * y**4
        - 0.62251 * y**5
        + 5.30260 * y**6
        - 2.09002 * y**7
    )
    return a + b / rv


def extinction_factor(wave_angstrom, ebv: float, rv: float = RV_DEFAULT) -> np.ndarray:
    """Multiplicative flux factor 10^(-0.4 A_λ) with A_λ = (A_λ/A_V) R_V E(B-V)."""
    al = ccm89_alambda_av(wave_angstrom, rv=rv)
    return 10.0 ** (-0.4 * al * rv * ebv)


def band_median_flux(wave, flux, lo: float, hi: float) -> float:
    w = np.asarray(wave, dtype=float)
    f = np.asarray(flux, dtype=float)
    m = (w >= lo) & (w <= hi) & np.isfinite(f) & (f > 0)
    if m.sum() < 8:
        return np.nan
    return float(np.nanmedian(f[m]))


def template_band_fluxes(teff, teffs, wave_grid, flux_grid, rv=RV_DEFAULT, ebv=0.0):
    """Median reddened template flux in each COLOR_BAND."""
    tmpl = jm.interp_template(teff, teffs, flux_grid)
    ext = extinction_factor(wave_grid, ebv, rv=rv)
    f_red = tmpl * ext
    out = {}
    for name, (lo, hi) in COLOR_BANDS.items():
        m = (wave_grid >= lo) & (wave_grid <= hi)
        if m.sum() < 5:
            out[name] = np.nan
        else:
            out[name] = float(np.nanmedian(f_red[m]))
    return out


def prepare_flux_template_grid(ids, wmin, wmax, dw=1.0):
    """Raw Pickles flux templates on a common rest-frame grid (not continuum-normalized)."""
    wave_grid = np.arange(wmin, wmax + 0.5 * dw, dw)
    rows = []
    for n in ids:
        sptype, teff = r12.PICKLES_I[n]
        wt, ft = r12.load_pickles(n)
        f_grid = r12.resample(wt, ft, wave_grid)
        rows.append((teff, sptype, n, f_grid))
    rows.sort(key=lambda x: x[0])
    teffs = np.array([r[0] for r in rows])
    sptypes = [r[1] for r in rows]
    ids_sorted = [r[2] for r in rows]
    flux = np.vstack([r[3] for r in rows])
    return teffs, sptypes, ids_sorted, wave_grid, flux


class ColorRatioDataset:
    """Observed continuum color ratios per epoch."""

    def __init__(self, rv: float = RV_DEFAULT):
        self.rv = rv
        self.epochs = []
        # Build template grid on full wavelength span covering all bands
        wmin = min(lo for lo, _ in COLOR_BANDS.values()) - 50
        wmax = max(hi for _, hi in COLOR_BANDS.values()) + 50
        self.teffs, self.sptypes, self.ids, self.wave_t, self.flux_t = (
            prepare_flux_template_grid(jm.FIT_TEMPLATE_IDS, wmin, wmax, dw=1.0)
        )

        for epoch, kind, pa, pb in r12.discover_epochs():
            df = r12.load_epoch(kind, pa, pb)
            w = df["wavelength_A"].values
            f = df["flux"].values
            band_f = {
                name: band_median_flux(w, f, lo, hi)
                for name, (lo, hi) in COLOR_BANDS.items()
            }
            ratios = {}
            sigmas = {}
            for a, b in COLOR_RATIOS:
                key = f"{a}/{b}"
                if not np.isfinite(band_f[a]) or not np.isfinite(band_f[b]):
                    continue
                ratios[key] = band_f[a] / band_f[b]
                # Fractional uncertainty from pixel scatter within bands
                fa = _band_flux_samples(w, f, *COLOR_BANDS[a])
                fb = _band_flux_samples(w, f, *COLOR_BANDS[b])
                if len(fa) > 5 and len(fb) > 5:
                    rs = fa[:, None] / fb[None, :]
                    sig = float(np.nanstd(rs))
                    sig = max(sig, 0.015 * ratios[key])
                else:
                    sig = 0.04 * ratios[key]
                sigmas[key] = sig
            if len(ratios) == len(COLOR_RATIOS):
                self.epochs.append(
                    {
                        "epoch": epoch,
                        "band_flux": band_f,
                        "ratios": ratios,
                        "sigmas": sigmas,
                    }
                )
                print(f"Color ratios {epoch}: " + ", ".join(f"{k}={v:.3f}" for k, v in ratios.items()))

    def model_ratio(self, teff: float, ebv: float, num: str, den: str) -> float:
        bf = template_band_fluxes(
            teff, self.teffs, self.wave_t, self.flux_t, rv=self.rv, ebv=ebv
        )
        return bf[num] / bf[den]

    def lnlike_colors(self, teff: float, ebv: float) -> float:
        ll = 0.0
        for ep in self.epochs:
            for a, b in COLOR_RATIOS:
                key = f"{a}/{b}"
                obs = ep["ratios"][key]
                sig = ep["sigmas"][key]
                pred = self.model_ratio(teff, ebv, a, b)
                if not np.isfinite(pred) or pred <= 0:
                    return -np.inf
                ll += jm.ln_gaussian(np.array([obs - pred]), np.log(sig))
        return ll


def _band_flux_samples(wave, flux, lo, hi, n_boot: int = 40):
    w = np.asarray(wave, dtype=float)
    f = np.asarray(flux, dtype=float)
    m = (w >= lo) & (w <= hi) & np.isfinite(f) & (f > 0)
    if m.sum() < 8:
        return np.array([])
    idx = np.where(m)[0]
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        out.append(np.nanmedian(f[pick]))
    return np.asarray(out)


def ln_prior(theta):
    teff, ebv, ln_s_m, ln_s_ca = theta
    if not (jm.TEFF_LO < teff < jm.TEFF_HI):
        return -np.inf
    if not (EBV_LO < ebv < EBV_HI):
        return -np.inf
    if not (jm.LN_SIG_LO < ln_s_m < jm.LN_SIG_HI):
        return -np.inf
    if not (jm.LN_SIG_LO < ln_s_ca < jm.LN_SIG_HI):
        return -np.inf
    # Soft prior from independent line-shape MCMC
    lp_teff = -0.5 * ((teff - TEFF_PRIOR_CENTER) / TEFF_PRIOR_SIGMA) ** 2
    return lp_teff


def ln_posterior(theta, color_data: ColorRatioDataset, line_data: jm.JointDataset):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    teff, ebv, ln_s_m, ln_s_ca = theta
    ll = color_data.lnlike_colors(teff, ebv)
    ll += jm.ln_likelihood([teff, ln_s_m, ln_s_ca], line_data)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def profile_ebv(color_data: ColorRatioDataset, teff: float, rv: float, n_grid: int = 200):
    """Profile likelihood for E(B-V) at fixed Teff (colors only)."""
    grid = np.linspace(EBV_LO + 0.01, EBV_HI - 0.01, n_grid)
    lnL = np.array([color_data.lnlike_colors(teff, ebv) for ebv in grid])
    lnL -= np.nanmax(lnL)
    return grid, lnL


def summarize_chain(flat):
    names = ["Teff", "ebv", "ln_sigma_mgb", "ln_sigma_caii"]
    out = {}
    for i, name in enumerate(names):
        x = flat[:, i]
        q = np.percentile(x, [2.5, 16, 50, 84, 97.5])
        out[name] = {
            "median": float(q[2]),
            "p16": float(q[1]),
            "p84": float(q[3]),
            "p2.5": float(q[0]),
            "p97.5": float(q[4]),
            "minus_1sigma": float(q[2] - q[1]),
            "plus_1sigma": float(q[3] - q[2]),
        }
    return out


def plot_results(color_data, line_data, sampler, summary, rv: float):
    flat = sampler.get_chain(discard=N_BURN, flat=True, thin=5)
    teff = flat[:, 0]
    ebv = flat[:, 1]
    teff_med = summary["Teff"]["median"]
    ebv_med = summary["ebv"]["median"]

    # --- Corner plot ---
    try:
        import corner

        labels = [
            r"$T_{\rm eff}$ (K)",
            r"$E(B-V)$",
            r"$\ln\sigma_{\rm Mg}$",
            r"$\ln\sigma_{\rm Ca}$",
        ]
        fig = corner.corner(
            flat,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3f",
        )
        fig.savefig(FIGDIR / "joint_teff_ebv_corner.png", dpi=140)
        plt.close(fig)
    except Exception as exc:
        print(f"corner plot skipped: {exc}")

    # --- Teff vs E(B-V) ---
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    h = ax.hist2d(ebv, teff, bins=50, cmap="Blues", cmin=1)
    plt.colorbar(h[3], ax=ax, label="samples")
    ax.axvline(ebv_med, color="C3", ls="--", lw=1.2)
    ax.axhline(teff_med, color="C0", ls="--", lw=1.2)
    ax.set_xlabel(r"$E(B-V)$")
    ax.set_ylabel(r"$T_{\rm eff}$ (K)")
    tp, tm = summary["Teff"]["plus_1sigma"], summary["Teff"]["minus_1sigma"]
    ep, em = summary["ebv"]["plus_1sigma"], summary["ebv"]["minus_1sigma"]
    ax.set_title(
        "Joint posterior: colors + line shapes\n"
        rf"$T_{{\rm eff}}={teff_med:.0f}^{{+{tp:.0f}}}_{{-{tm:.0f}}}$ K,  "
        rf"$E(B-V)={ebv_med:.3f}^{{+{ep:.3f}}}_{{-{em:.3f}}}$  ($R_V={rv}$)"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "teff_ebv_posterior.png", dpi=160)
    plt.close(fig)

    # --- Profile likelihood at MCMC Teff median ---
    grid, dlnL = profile_ebv(color_data, teff_med, rv)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(grid, dlnL, "C0", lw=2)
    ax.axhline(-0.5, color="0.5", ls=":", label=r"68% ($\Delta\ln L=-0.5$)")
    ax.axhline(-2.0, color="0.7", ls=":", label=r"95% ($\Delta\ln L=-2$)")
    ax.axvline(ebv_med, color="C3", ls="--", label=f"MCMC median = {ebv_med:.3f}")
    ax.axvline(1.0, color="k", ls=":", alpha=0.6, label="Smith+2018 LE value ≈ 1.0")
    good = dlnL >= -0.5
    if good.any():
        ax.axvspan(grid[good][0], grid[good][-1], color="C0", alpha=0.15)
    ax.set_xlabel(r"$E(B-V)$")
    ax.set_ylabel(r"$\Delta \ln \mathcal{L}$ (colors only)")
    ax.set_title(f"Profile likelihood at $T_{{\\rm eff}}={teff_med:.0f}$ K, $R_V={rv}$")
    ax.legend(fontsize=8)
    ax.set_ylim(-5, 0.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "ebv_profile_likelihood.png", dpi=160)
    plt.close(fig)

    # --- Diagnostic: observed vs model color ratios ---
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ax_obs, ax_res = axs[0, 0], axs[0, 1]
    ax_prof, ax_sed = axs[1, 0], axs[1, 1]

    preds = []
    obses = []
    sigs = []
    labels = []
    for ep in color_data.epochs:
        for a, b in COLOR_RATIOS:
            key = f"{a}/{b}"
            obs = ep["ratios"][key]
            pred = color_data.model_ratio(teff_med, ebv_med, a, b)
            sig = ep["sigmas"][key]
            preds.append(pred)
            obses.append(obs)
            sigs.append(sig)
            labels.append(f"{ep['epoch'][:3]} {key}")

    ax_obs.plot(preds, obses, "o", ms=8)
    lims = [min(preds + obses) * 0.9, max(preds + obses) * 1.05]
    ax_obs.plot(lims, lims, "k--", lw=1)
    ax_obs.set_xlabel("Model color ratio (reddened template)")
    ax_obs.set_ylabel("Observed color ratio")
    ax_obs.set_title("(a) Observed vs. model color ratios")
    ax_obs.text(
        0.05,
        0.95,
        f"$R_V={rv}$, $E(B-V)={ebv_med:.3f}$\n"
        f"$T_{{\\rm eff}}={teff_med:.0f}$ K",
        transform=ax_obs.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
    )

    resid = [(o - p) / s for o, p, s in zip(obses, preds, sigs)]
    xpos = np.arange(len(labels))
    ax_res.bar(xpos, resid, color="C0", alpha=0.85)
    ax_res.axhline(0, color="k", lw=0.8)
    ax_res.set_ylabel(r"$(\mathrm{obs}-\mathrm{pred})/\sigma$")
    ax_res.set_title("(b) Normalized residuals")
    ax_res.set_xticks(xpos)
    ax_res.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)

    ax_prof.plot(grid, dlnL, "C0", lw=2)
    ax_prof.axvline(ebv_med, color="C3", ls="--")
    ax_prof.set_xlabel(r"$E(B-V)$")
    ax_prof.set_ylabel(r"$\Delta \ln \mathcal{L}$")
    ax_prof.set_title(f"(c) Profile likelihood ($R_V={rv}$)")

    # Continuum shape panel
    wave_plot = np.arange(4200, 9000, 20)
    tmpl = jm.interp_template(teff_med, color_data.teffs, color_data.flux_t)
    ext = extinction_factor(wave_plot, ebv_med, rv=rv)
    f_int = np.interp(wave_plot, color_data.wave_t, tmpl, left=np.nan, right=np.nan)
    f_red = f_int * ext
    f_red_norm = f_red / np.interp(6500, wave_plot, f_red)
    ax_sed.plot(wave_plot, f_int / np.interp(6500, wave_plot, f_int), "k:", lw=1.2, label="intrinsic template")
    ax_sed.plot(wave_plot, f_red_norm, "k-", lw=1.5, label=f"reddened, $E(B-V)={ebv_med:.2f}$")
    for name, (lo, hi) in COLOR_BANDS.items():
        ax_sed.axvspan(lo, hi, color="0.9", zorder=0)
    for ep in color_data.epochs:
        fr = ep["band_flux"]["red"]
        ys = [ep["band_flux"][n] / fr for n in COLOR_BANDS]
        xs = [0.5 * (COLOR_BANDS[n][0] + COLOR_BANDS[n][1]) for n in COLOR_BANDS]
        ax_sed.plot(xs, ys, "o-", ms=5, lw=1, label=ep["epoch"][:7])
    ax_sed.set_xscale("log")
    ax_sed.set_xlabel("Wavelength (Å)")
    ax_sed.set_ylabel(r"Band flux / $F_{\rm red}$")
    ax_sed.set_title("(d) Continuum shape: observed vs. reddened template")
    ax_sed.legend(fontsize=7, loc="upper right")
    ax_sed.set_xlim(4200, 9000)

    sp, _ = r12.sptype_from_teff(teff_med)
    fig.suptitle(
        f"EC1 joint reddening + spectral-type MCMC  "
        f"({sp}, $T_{{\\rm eff}}={teff_med:.0f}$ K, $E(B-V)={ebv_med:.3f}$, $R_V={rv}$)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "ebv_fit_diagnostic.png", dpi=160)
    fig.savefig(FIGDIR / "ebv_fit_diagnostic.pdf")
    plt.close(fig)

    # --- MCMC traces ---
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    chain = sampler.get_chain(discard=N_BURN, thin=5)
    names = ["Teff", "E(B-V)", "ln σ_Mg", "ln σ_Ca"]
    for i, ax in enumerate(axs):
        ax.plot(chain[:, :, i], "k", alpha=0.15, lw=0.5)
        ax.set_ylabel(names[i])
    axs[-1].set_xlabel("step")
    fig.suptitle("MCMC traces (post burn-in, thinned)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "mcmc_traces.png", dpi=140)
    plt.close(fig)


def run_mcmc(color_data, line_data):
    ndim = 4
    rng = np.random.default_rng(SEED)
    pos = np.zeros((N_WALKERS, ndim))
    pos[:, 0] = TEFF_PRIOR_CENTER + 80 * rng.normal(size=N_WALKERS)
    pos[:, 1] = rng.uniform(0.7, 1.2, size=N_WALKERS)
    pos[:, 2] = np.log(0.08) + 0.2 * rng.normal(size=N_WALKERS)
    pos[:, 3] = np.log(0.10) + 0.2 * rng.normal(size=N_WALKERS)
    pos[:, 0] = np.clip(pos[:, 0], jm.TEFF_LO + 50, jm.TEFF_HI - 50)
    pos[:, 2] = np.clip(pos[:, 2], jm.LN_SIG_LO + 0.01, jm.LN_SIG_HI - 0.01)
    pos[:, 3] = np.clip(pos[:, 3], jm.LN_SIG_LO + 0.01, jm.LN_SIG_HI - 0.01)

    sampler = emcee.EnsembleSampler(
        N_WALKERS, ndim, ln_posterior, args=(color_data, line_data)
    )
    print(f"Running joint MCMC: {N_WALKERS} walkers × {N_STEPS} steps …")
    sampler.run_mcmc(pos, N_STEPS, progress=True)
    return sampler


def main(rv: float = RV_DEFAULT):
    print(f"Preparing color-ratio dataset (R_V={rv}) …")
    color_data = ColorRatioDataset(rv=rv)
    if not color_data.epochs:
        raise SystemExit("No epochs with full color-ratio coverage")

    print("\nPreparing line-shape dataset …")
    line_data = jm.JointDataset()

    # Profile likelihood at joint-MCMC Teff for comparison
    with open(r12.FIGDIR / "joint_mcmc" / "joint_mcmc_summary.json") as f:
        jm_summary = json.load(f)
    teff_ref = jm_summary["teff_K"]["best"]
    grid, dlnL = profile_ebv(color_data, teff_ref, rv)
    i68 = dlnL >= -0.5
    ebv_prof = {
        "teff_fixed_K": teff_ref,
        "rv": rv,
        "best": float(grid[np.argmax(dlnL)]),
        "profile_68": [float(grid[i68][0]), float(grid[i68][-1])] if i68.any() else None,
    }
    print(f"Color-only profile at Teff={teff_ref:.0f} K: E(B-V) best = {ebv_prof['best']:.3f}")

    sampler = run_mcmc(color_data, line_data)
    print(f"Mean acceptance: {np.mean(sampler.acceptance_fraction):.3f}")

    flat = sampler.get_chain(discard=N_BURN, flat=True, thin=5)
    summary = summarize_chain(flat)
    sp, teff_near = r12.sptype_from_teff(summary["Teff"]["median"])

    measurements = []
    for ep in color_data.epochs:
        for key, val in ep["ratios"].items():
            measurements.append(
                {
                    "epoch": ep["epoch"],
                    "key": key,
                    "ratio": val,
                    "sigma": ep["sigmas"][key],
                }
            )

    result = {
        "method": (
            "Joint MCMC: continuum color ratios + Mg b / Ca II line shapes; "
            f"Pickles I templates, CCM89, R_V={rv}, v fixed {jm.V_FIXED:.0f} km/s"
        ),
        "rv": rv,
        "parameters": summary,
        "teff_K": {
            "best": summary["Teff"]["median"],
            "minus_1sigma": summary["Teff"]["minus_1sigma"],
            "plus_1sigma": summary["Teff"]["plus_1sigma"],
            "string": (
                f"{summary['Teff']['median']:.0f}"
                f"+{summary['Teff']['plus_1sigma']:.0f}"
                f"-{summary['Teff']['minus_1sigma']:.0f}"
            ),
        },
        "ebv": {
            "best": summary["ebv"]["median"],
            "minus_1sigma": summary["ebv"]["minus_1sigma"],
            "plus_1sigma": summary["ebv"]["plus_1sigma"],
            "string": (
                f"{summary['ebv']['median']:.3f}"
                f"+{summary['ebv']['plus_1sigma']:.3f}"
                f"-{summary['ebv']['minus_1sigma']:.3f}"
            ),
        },
        "nearest_pickles": {"sptype": sp, "teff_K": teff_near},
        "teff_prior_from_lines": {
            "center_K": TEFF_PRIOR_CENTER,
            "sigma_K": TEFF_PRIOR_SIGMA,
        },
        "color_bands_A": COLOR_BANDS,
        "measurements": measurements,
        "color_only_profile_at_joint_teff": ebv_prof,
        "mcmc": {
            "n_walkers": N_WALKERS,
            "n_steps": N_STEPS,
            "n_burn": N_BURN,
            "mean_acceptance": float(np.mean(sampler.acceptance_fraction)),
        },
    }
    with open(FIGDIR / "reddening_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    pd.DataFrame(
        flat,
        columns=["Teff", "ebv", "ln_sigma_mgb", "ln_sigma_caii"],
    ).to_csv(FIGDIR / "mcmc_chain_thinned.csv", index=False)

    plot_results(color_data, line_data, sampler, summary, rv)

    print("\n=== Joint Teff + E(B-V) MCMC ===")
    print(
        f"Teff = {summary['Teff']['median']:.0f} "
        f"+{summary['Teff']['plus_1sigma']:.0f}/-{summary['Teff']['minus_1sigma']:.0f} K  "
        f"→ {sp}"
    )
    print(
        f"E(B-V) = {summary['ebv']['median']:.3f} "
        f"+{summary['ebv']['plus_1sigma']:.3f}/-{summary['ebv']['minus_1sigma']:.3f}  "
        f"(R_V={rv})"
    )
    print(f"Color-only profile at Teff={teff_ref:.0f} K: E(B-V) ≈ {ebv_prof['best']:.3f}")
    print(f"Figures → {FIGDIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rv",
        type=float,
        default=RV_DEFAULT,
        help=f"Total-to-selective extinction R_V (default {RV_DEFAULT})",
    )
    args = parser.parse_args()
    main(rv=args.rv)
