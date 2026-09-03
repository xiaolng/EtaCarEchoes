#!/usr/bin/env python3
"""
Joint MCMC template matching for Rest+ EC1 light-echo spectra.

Fits Pickles (1998) luminosity-class I templates simultaneously in:
  • Mg b / 5270 window (5060–5500 Å)
  • Ca II IR triplet + H Paschen window (8200–8720 Å)

Free parameters
  Teff          continuous effective temperature (interpolated Pickles grid)
  ln_sigma_mgb  log noise scale in the Mg b window
  ln_sigma_caii log noise scale in the Ca II / Paschen window

Velocity is fixed to the Rest+ Ca II value (v = -210 km/s). At IMACS/WFCCD
resolution, free-v is strongly degenerate with line-depth / continuum mismatch
and drives unphysical edge solutions; Ca II line cores independently imply
v ≈ -220 km/s, consistent with Rest+.

All Mar/Apr 2011 EC1 epochs share Teff; noise scales are global per window.

Outputs → figures/rest2012_spectral_type/joint_mcmc/
"""

from __future__ import annotations

import json
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rest2012_spectral_type as r12

ROOT = Path(__file__).resolve().parent
FIGDIR = r12.FIGDIR / "joint_mcmc"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Rest+ diagnostic windows
MGB_LO, MGB_HI = r12.WAVE_MGB_LO, r12.WAVE_MGB_HI
CA_LO, CA_HI = r12.WAVE_CAII_LO, r12.WAVE_CAII_HI

# Priors / fixed velocity
TEFF_LO, TEFF_HI = 4000.0, 7000.0
V_FIXED = -210.0  # Rest+ Ca II cross-correlation mean
LN_SIG_LO, LN_SIG_HI = np.log(0.02), np.log(0.80)

# MCMC
N_WALKERS = 48
N_STEPS = 4000
N_BURN = 1000
SEED = 42

# Use F0I–K4I for interpolation (A/M edges poorly constrained by these features)
FIT_TEMPLATE_IDS = [121, 122, 123, 124, 125, 126, 127, 128, 129, 130]


def prepare_obs_window(wave, flux, wmin, wmax, exclude_for_cont=None):
    """Continuum-normalize an observed-frame window (resolution-matched)."""
    w = np.asarray(wave, dtype=float)
    f = np.asarray(flux, dtype=float)
    sel = (w >= wmin) & (w <= wmax) & np.isfinite(f) & (f > 0)
    ww, ff = w[sel], f[sel]
    ff = r12.match_resolution(ww, ff, fwhm_A=7.0)
    fn = r12.continuum_normalize_gauss(
        ww, ff, fwhm_cont=200.0, exclude_windows=exclude_for_cont
    )
    good = np.isfinite(fn)
    return ww[good], fn[good]


def prepare_template_grid(ids, wmin, wmax, exclude_for_cont=None, dw=1.0):
    """
    Continuum-normalized Pickles templates on a common rest-frame grid.
    Returns teff ascending, sptype list, flux array (n_teff, n_wave), wave grid.
    """
    wave_grid = np.arange(wmin, wmax + 0.5 * dw, dw)
    rows = []
    for n in ids:
        sptype, teff = r12.PICKLES_I[n]
        wt, ft = r12.load_pickles(n)
        ww, fn = prepare_obs_window(wt, ft, wmin, wmax, exclude_for_cont=exclude_for_cont)
        # templates are rest-frame; resample onto grid
        f_grid = r12.resample(ww, fn, wave_grid)
        rows.append((teff, sptype, n, f_grid))
    rows.sort(key=lambda x: x[0])
    teffs = np.array([r[0] for r in rows])
    sptypes = [r[1] for r in rows]
    ids_sorted = [r[2] for r in rows]
    flux = np.vstack([r[3] for r in rows])
    return teffs, sptypes, ids_sorted, wave_grid, flux


def interp_template(teff, teffs, flux_grid):
    """Linear interpolation of continuum-normalized template in Teff."""
    if teff <= teffs[0]:
        return flux_grid[0].copy()
    if teff >= teffs[-1]:
        return flux_grid[-1].copy()
    i = int(np.searchsorted(teffs, teff) - 1)
    i = max(0, min(i, len(teffs) - 2))
    t0, t1 = teffs[i], teffs[i + 1]
    w = (teff - t0) / (t1 - t0)
    return (1.0 - w) * flux_grid[i] + w * flux_grid[i + 1]


def model_on_obs(wave_obs, v_kms, teff, teffs, wave_tmpl, flux_grid):
    """
    Predict continuum-normalized template at observed wavelengths for velocity v.
    model(λ_obs) = template_rest(λ_obs / (1 + v/c))
    """
    tmpl = interp_template(teff, teffs, flux_grid)
    wave_rest = wave_obs / (1.0 + v_kms / r12.C_KMS)
    return r12.resample(wave_tmpl, tmpl, wave_rest)


def ln_gaussian(resid, ln_sigma):
    sigma = np.exp(ln_sigma)
    return -0.5 * np.sum((resid / sigma) ** 2 + np.log(2.0 * np.pi * sigma**2))


def shape_vectors(obs, model):
    """
    Rest+/xcsao-like shape match: demean and unit-normalize both vectors.

    Absolute line depth then cannot dominate (LE Ca II is deeper than Pickles
    G/K templates); the fit tracks absorption *pattern*, as in Rest+ xcsao.
    """
    m = np.isfinite(obs) & np.isfinite(model)
    if m.sum() < 30:
        return None, None, None
    o = obs[m] - np.mean(obs[m])
    t = model[m] - np.mean(model[m])
    so, st = np.std(o), np.std(t)
    if so < 1e-8 or st < 1e-8:
        return None, None, None
    return o / so, t / st, m


class JointDataset:
    def __init__(self):
        self.epochs = []
        self.teffs_m, self.spt_m, self.ids_m, self.w_m, self.f_m = prepare_template_grid(
            FIT_TEMPLATE_IDS, MGB_LO, MGB_HI, exclude_for_cont=None
        )
        self.teffs_c, self.spt_c, self.ids_c, self.w_c, self.f_c = prepare_template_grid(
            FIT_TEMPLATE_IDS, CA_LO, CA_HI, exclude_for_cont=r12.CAII_CONT_MASK
        )
        # Same Teff grid for both (same ids)
        assert np.allclose(self.teffs_m, self.teffs_c)
        self.teffs = self.teffs_m
        self.sptypes = self.spt_m

        for epoch, kind, pa, pb in r12.discover_epochs():
            df = r12.load_epoch(kind, pa, pb)
            w, f = df["wavelength_A"].values, df["flux"].values
            w_mg, f_mg = prepare_obs_window(w, f, MGB_LO, MGB_HI)
            w_ca, f_ca = prepare_obs_window(
                w, f, CA_LO, CA_HI, exclude_for_cont=r12.CAII_CONT_MASK
            )
            if len(w_mg) < 50 or len(w_ca) < 50:
                print(f"Skipping {epoch}: insufficient coverage")
                continue
            self.epochs.append(
                {"epoch": epoch, "w_mg": w_mg, "f_mg": f_mg, "w_ca": w_ca, "f_ca": f_ca}
            )
            print(
                f"Loaded {epoch}: Mg b N={len(w_mg)}, Ca II N={len(w_ca)}"
            )

    @property
    def n_epochs(self):
        return len(self.epochs)


def ln_prior(theta):
    teff, ln_s_m, ln_s_c = theta
    if not (TEFF_LO < teff < TEFF_HI):
        return -np.inf
    if not (LN_SIG_LO < ln_s_m < LN_SIG_HI):
        return -np.inf
    if not (LN_SIG_LO < ln_s_c < LN_SIG_HI):
        return -np.inf
    return 0.0


def ln_likelihood(theta, data: JointDataset, v_kms=V_FIXED):
    teff, ln_s_m, ln_s_c = theta
    ll = 0.0
    # Equalize the two diagnostic windows (Ca II has more pixels).
    n_m_tot = sum(len(ep["f_mg"]) for ep in data.epochs)
    n_c_tot = sum(len(ep["f_ca"]) for ep in data.epochs)
    w_m = 0.5 * (n_m_tot + n_c_tot) / max(n_m_tot, 1)
    w_c = 0.5 * (n_m_tot + n_c_tot) / max(n_c_tot, 1)
    for ep in data.epochs:
        m_mg = model_on_obs(ep["w_mg"], v_kms, teff, data.teffs, data.w_m, data.f_m)
        m_ca = model_on_obs(ep["w_ca"], v_kms, teff, data.teffs, data.w_c, data.f_c)
        o_m, t_m, _ = shape_vectors(ep["f_mg"], m_mg)
        o_c, t_c, _ = shape_vectors(ep["f_ca"], m_ca)
        if o_m is None or o_c is None:
            return -np.inf
        ll += w_m * ln_gaussian(o_m - t_m, ln_s_m)
        ll += w_c * ln_gaussian(o_c - t_c, ln_s_c)
    return ll


def ln_posterior(theta, data: JointDataset):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = ln_likelihood(theta, data)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def discrete_template_likelihoods(data: JointDataset, v_kms=V_FIXED):
    """
    For each discrete Pickles template, profile noise scales and evaluate
    joint Mg b + Ca II/Paschen likelihood at fixed Rest+ velocity.
    """
    rows = []
    for teff, sptype, pid in zip(data.teffs, data.sptypes, data.ids_m):
        r2_m, r2_c, nm, nc = 0.0, 0.0, 0, 0
        ok = True
        for ep in data.epochs:
            m_mg = model_on_obs(ep["w_mg"], v_kms, teff, data.teffs, data.w_m, data.f_m)
            m_ca = model_on_obs(ep["w_ca"], v_kms, teff, data.teffs, data.w_c, data.f_c)
            o_m, t_m, _ = shape_vectors(ep["f_mg"], m_mg)
            o_c, t_c, _ = shape_vectors(ep["f_ca"], m_ca)
            if o_m is None or o_c is None:
                ok = False
                break
            r2_m += np.sum((o_m - t_m) ** 2)
            r2_c += np.sum((o_c - t_c) ** 2)
            nm += len(o_m)
            nc += len(o_c)
        if not ok or nm == 0 or nc == 0:
            best, s_m, s_c = -np.inf, np.nan, np.nan
        else:
            s_m = float(np.clip(max(np.sqrt(r2_m / nm), 1e-4), np.exp(LN_SIG_LO), np.exp(LN_SIG_HI)))
            s_c = float(np.clip(max(np.sqrt(r2_c / nc), 1e-4), np.exp(LN_SIG_LO), np.exp(LN_SIG_HI)))
            best = ln_likelihood([teff, np.log(s_m), np.log(s_c)], data, v_kms=v_kms)
        rows.append(
            {
                "pickles_id": pid,
                "sptype": sptype,
                "teff": teff,
                "lnL_max": best,
                "v_fixed": v_kms,
                "sigma_mgb": s_m,
                "sigma_caii": s_c,
            }
        )
    df = pd.DataFrame(rows)
    m = df["lnL_max"].max()
    w = np.exp(df["lnL_max"] - m)
    w[~np.isfinite(w)] = 0
    df["posterior_weight"] = w / w.sum() if w.sum() > 0 else w
    return df.sort_values("teff")


def run_mcmc(data: JointDataset):
    ndim = 3
    rng = np.random.default_rng(SEED)
    # Spread walkers across the full Teff prior to capture multimodal structure
    pos = np.zeros((N_WALKERS, ndim))
    pos[:, 0] = rng.uniform(TEFF_LO + 50, TEFF_HI - 50, size=N_WALKERS)
    pos[:, 1] = np.log(0.08) + 0.25 * rng.normal(size=N_WALKERS)
    pos[:, 2] = np.log(0.10) + 0.25 * rng.normal(size=N_WALKERS)
    pos[:, 1] = np.clip(pos[:, 1], LN_SIG_LO + 0.01, LN_SIG_HI - 0.01)
    pos[:, 2] = np.clip(pos[:, 2], LN_SIG_LO + 0.01, LN_SIG_HI - 0.01)

    sampler = emcee.EnsembleSampler(N_WALKERS, ndim, ln_posterior, args=(data,))
    print(f"Running MCMC: {N_WALKERS} walkers × {N_STEPS} steps (v fixed = {V_FIXED:.0f} km/s) …")
    sampler.run_mcmc(pos, N_STEPS, progress=True)
    return sampler


def summarize_chain(flat):
    names = ["Teff", "ln_sigma_mgb", "ln_sigma_caii"]
    out = {}
    for i, name in enumerate(names):
        x = flat[:, i]
        q = np.percentile(x, [16, 50, 84])
        out[name] = {
            "median": float(q[1]),
            "p16": float(q[0]),
            "p84": float(q[2]),
            "minus_1sigma": float(q[1] - q[0]),
            "plus_1sigma": float(q[2] - q[1]),
        }
    for key, col in [("sigma_mgb", 1), ("sigma_caii", 2)]:
        x = np.exp(flat[:, col])
        q = np.percentile(x, [16, 50, 84])
        out[key] = {
            "median": float(q[1]),
            "p16": float(q[0]),
            "p84": float(q[2]),
        }
    out["v_fixed_kms"] = {
        "median": V_FIXED,
        "p16": V_FIXED,
        "p84": V_FIXED,
        "minus_1sigma": 0.0,
        "plus_1sigma": 0.0,
    }
    return out


def plot_results(data, sampler, disc_df, summary):
    flat = sampler.get_chain(discard=N_BURN, flat=True, thin=5)
    teff = flat[:, 0]
    med = summary["Teff"]["median"]
    lo = summary["Teff"]["p16"]
    hi = summary["Teff"]["p84"]
    v_b = V_FIXED

    # --- Teff posterior ---
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(teff, bins=50, density=True, color="C0", alpha=0.75, histtype="stepfilled")
    ax.axvline(med, color="k", lw=1.5, label=f"median = {med:.0f} K")
    ax.axvspan(lo, hi, color="C0", alpha=0.2, label=f"16–84%: [{lo:.0f}, {hi:.0f}] K")
    ax.axvspan(4850, 5550, color="C2", alpha=0.12, label="Rest+ EC1B 95% CI")
    ax2 = ax.twinx()
    ax2.plot(
        disc_df["teff"],
        disc_df["posterior_weight"],
        "o-",
        color="C3",
        ms=6,
        label="discrete P(template)",
    )
    ax2.set_ylabel("Discrete template weight", color="C3")
    ax.set_xlabel(r"$T_{\rm eff}$ (K)")
    ax.set_ylabel("Posterior density")
    ax.set_title(
        f"Joint Mg b + Ca II/Paschen MCMC  →  "
        f"$T_{{\\rm eff}}={med:.0f}^{{+{summary['Teff']['plus_1sigma']:.0f}}}_{{-{summary['Teff']['minus_1sigma']:.0f}}}$ K"
        f"  ($v={V_FIXED:.0f}$ km/s fixed)"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "teff_posterior.png", dpi=160)
    plt.close(fig)

    # --- summary panel ---
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].hist(teff, bins=40, color="C0", alpha=0.85)
    axs[0].axvline(med, color="k")
    axs[0].axvspan(lo, hi, color="C0", alpha=0.2)
    axs[0].set_xlabel(r"$T_{\rm eff}$ (K)")
    axs[0].set_ylabel("N")
    axs[0].set_title("Teff posterior")
    sp, tt = r12.sptype_from_teff(med)
    i_map = disc_df["posterior_weight"].idxmax()
    axs[1].axis("off")
    axs[1].text(
        0.05,
        0.55,
        f"Joint Mg b + Ca II/Paschen MCMC\n"
        f"v fixed = {V_FIXED:.0f} km/s (Rest+)\n\n"
        f"Teff = {med:.0f} "
        f"+{summary['Teff']['plus_1sigma']:.0f}/-{summary['Teff']['minus_1sigma']:.0f} K\n"
        f"1σ range: [{lo:.0f}, {hi:.0f}] K\n"
        f"Nearest Pickles: {sp} ({tt:.0f} K)\n\n"
        f"Best discrete template:\n"
        f"  {disc_df.loc[i_map, 'sptype']} "
        f"({disc_df.loc[i_map, 'teff']:.0f} K)\n"
        f"Rest+2012: G2–G5, ~5000 K",
        transform=axs[1].transAxes,
        fontsize=11,
        va="center",
        family="monospace",
    )
    fig.suptitle("Joint MCMC posteriors (Mg b + Ca II/Paschen)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "mcmc_summary.png", dpi=160)
    plt.close(fig)

    try:
        import corner

        labels = [r"$T_{\rm eff}$", r"$\ln\sigma_{\rm Mg}$", r"$\ln\sigma_{\rm Ca}$"]
        fig = corner.corner(
            flat,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".2f",
        )
        fig.savefig(FIGDIR / "corner.png", dpi=140)
        plt.close(fig)
    except Exception as exc:
        print(f"corner plot skipped: {exc}")

    # --- Best-fit spectral match ---
    teff_b = med
    sp_b, _ = r12.sptype_from_teff(teff_b)
    sp_disc = disc_df.loc[i_map, "sptype"]
    teff_disc = float(disc_df.loc[i_map, "teff"])

    fig, axs = plt.subplots(data.n_epochs, 2, figsize=(11, 2.8 * data.n_epochs), sharex="col")
    if data.n_epochs == 1:
        axs = np.array([axs])
    for i, ep in enumerate(data.epochs):
        m_mg = model_on_obs(ep["w_mg"], v_b, teff_b, data.teffs, data.w_m, data.f_m)
        m_ca = model_on_obs(ep["w_ca"], v_b, teff_b, data.teffs, data.w_c, data.f_c)
        axs[i, 0].plot(ep["w_mg"], ep["f_mg"], "k", lw=0.9, label=ep["epoch"])
        axs[i, 0].plot(ep["w_mg"], m_mg, "C3", lw=1.2, label=f"model {sp_b} ({teff_b:.0f} K)")
        for w0 in (5167, 5173, 5184, 5270):
            axs[i, 0].axvline(w0, color="0.6", ls=":", lw=0.7)
        axs[i, 0].set_ylabel("Fλ/cont.")
        axs[i, 0].set_ylim(0.55, 1.35)
        axs[i, 0].legend(fontsize=7, loc="lower right")
        if i == 0:
            axs[i, 0].set_title("Mg b / 5270 (joint MCMC)")

        w_rest = ep["w_ca"] / (1.0 + v_b / r12.C_KMS)
        axs[i, 1].plot(w_rest, ep["f_ca"], "k", lw=0.9, label=ep["epoch"])
        axs[i, 1].plot(w_rest, m_ca, "C3", lw=1.2, label=f"v={v_b:.0f} km/s")
        for w0 in (8498, 8542, 8662):
            axs[i, 1].axvline(w0, color="C0", ls=":", lw=0.8)
        for w0 in (8502.5, 8598.4):
            axs[i, 1].axvline(w0, color="C2", ls=":", lw=0.7)
        axs[i, 1].set_ylim(0.35, 1.4)
        axs[i, 1].legend(fontsize=7, loc="lower right")
        if i == 0:
            axs[i, 1].set_title("Ca II + Paschen (rest frame)")
    axs[-1, 0].set_xlabel("Observed wavelength (Å)")
    axs[-1, 1].set_xlabel("Rest wavelength (Å)")
    fig.suptitle(
        f"Joint best fit: {sp_b} / discrete MAP {sp_disc} ({teff_disc:.0f} K);  "
        f"$T_{{\\rm eff}}={teff_b:.0f}^{{+{summary['Teff']['plus_1sigma']:.0f}}}"
        f"_{{-{summary['Teff']['minus_1sigma']:.0f}}}$ K",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "joint_best_match.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(disc_df["sptype"], disc_df["posterior_weight"], color="C3", alpha=0.85)
    ax.set_ylabel("Posterior weight (flat template prior)")
    ax.set_xlabel("Pickles I template")
    ax.set_title("Discrete template posterior from joint Mg b + Ca II/Paschen likelihood")
    fig.tight_layout()
    fig.savefig(FIGDIR / "discrete_template_weights.png", dpi=150)
    plt.close(fig)


def main():
    print("Preparing joint Mg b + Ca II/Paschen dataset …")
    data = JointDataset()
    if data.n_epochs == 0:
        raise SystemExit("No epochs loaded")

    print("\nDiscrete template scan …")
    disc_df = discrete_template_likelihoods(data)
    disc_df.to_csv(FIGDIR / "discrete_template_likelihoods.csv", index=False)
    i_map = disc_df["posterior_weight"].idxmax()
    print(
        f"MAP discrete template: {disc_df.loc[i_map, 'sptype']} "
        f"({disc_df.loc[i_map, 'teff']:.0f} K), "
        f"weight={disc_df.loc[i_map, 'posterior_weight']:.3f} "
        f"(v fixed = {V_FIXED:.0f} km/s)"
    )
    print(disc_df[["sptype", "teff", "lnL_max", "posterior_weight"]].to_string(index=False))

    sampler = run_mcmc(data)
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"Autocorr time: {tau}")
    except Exception as exc:
        print(f"Autocorr estimate failed: {exc}")
    print(f"Mean acceptance: {np.mean(sampler.acceptance_fraction):.3f}")

    flat = sampler.get_chain(discard=N_BURN, flat=True, thin=5)
    summary = summarize_chain(flat)
    sp_near, teff_near = r12.sptype_from_teff(summary["Teff"]["median"])

    result = {
        "method": (
            "Joint MCMC: Mg b (5060-5500A) + Ca II/Paschen (8200-8720A) vs Pickles I; "
            f"v fixed to Rest+ {V_FIXED:.0f} km/s"
        ),
        "reference": "Rest et al. 2012, Nature, 482, 375",
        "epochs": [e["epoch"] for e in data.epochs],
        "parameters": summary,
        "teff_K": {
            "best": summary["Teff"]["median"],
            "minus_1sigma": summary["Teff"]["minus_1sigma"],
            "plus_1sigma": summary["Teff"]["plus_1sigma"],
            "p16": summary["Teff"]["p16"],
            "p84": summary["Teff"]["p84"],
            "string": (
                f"{summary['Teff']['median']:.0f}"
                f"+{summary['Teff']['plus_1sigma']:.0f}"
                f"-{summary['Teff']['minus_1sigma']:.0f}"
            ),
        },
        "nearest_pickles_to_median": {"sptype": sp_near, "teff_K": teff_near},
        "best_discrete_template": {
            "sptype": disc_df.loc[i_map, "sptype"],
            "teff_K": float(disc_df.loc[i_map, "teff"]),
            "pickles_id": int(disc_df.loc[i_map, "pickles_id"]),
            "posterior_weight": float(disc_df.loc[i_map, "posterior_weight"]),
            "v_kms": V_FIXED,
        },
        "rest2012": {"sptype": "G2-G5", "teff_K": 5000},
        "mcmc": {
            "n_walkers": N_WALKERS,
            "n_steps": N_STEPS,
            "n_burn": N_BURN,
            "v_fixed_kms": V_FIXED,
            "mean_acceptance": float(np.mean(sampler.acceptance_fraction)),
        },
    }
    with open(FIGDIR / "joint_mcmc_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    pd.DataFrame(
        flat, columns=["Teff", "ln_sigma_mgb", "ln_sigma_caii"]
    ).to_csv(FIGDIR / "mcmc_chain_thinned.csv", index=False)

    plot_results(data, sampler, disc_df, summary)

    print("\n=== Joint MCMC Teff result ===")
    print(
        f"Teff = {summary['Teff']['median']:.0f} "
        f"+{summary['Teff']['plus_1sigma']:.0f}/-{summary['Teff']['minus_1sigma']:.0f} K "
        f"(16–84%: {summary['Teff']['p16']:.0f}–{summary['Teff']['p84']:.0f} K)"
    )
    print(f"Nearest Pickles type: {sp_near} ({teff_near:.0f} K)")
    print(
        f"Best discrete template: {result['best_discrete_template']['sptype']} "
        f"({result['best_discrete_template']['teff_K']:.0f} K)"
    )
    print(f"v fixed = {V_FIXED:.0f} km/s (Rest+ Ca II)")
    print(f"Rest+2012: G2–G5, ~5000 K")
    print(f"Figures → {FIGDIR}")


if __name__ == "__main__":
    main()
