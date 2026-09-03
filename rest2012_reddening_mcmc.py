#!/usr/bin/env python3
"""
Separate Mg and Ca MCMC fits: Teff + E(B-V) from EC1 light-echo spectra.

Each feature fit combines local continuum color ratios with absorption-line shape.
Per-epoch parameters (spectrum-by-spectrum):
  v_kms         outflow velocity for line matching
  ln_cal_blue   log flux calibration offset for the blue continuum band

Global: Teff, E(B-V), ln sigma_line.
Blue-band ratio uncertainties are inflated (BLUE_SIGMA_INFLATE).

Outputs → figures/rest2012_spectral_type/reddening/{mg,ca}/
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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

MG_COLOR_BANDS = {
    "blue": (4500.0, 4700.0),
    "green": (5500.0, 5700.0),
    "red": (6400.0, 6600.0),
}
MG_COLOR_RATIOS = [("blue", "red"), ("green", "red"), ("blue", "green")]

CA_COLOR_BANDS = {
    "blue": (4500.0, 4700.0),
    "green": (5500.0, 5700.0),
    "red": (6400.0, 6600.0),
    "nir": (8420.0, 8580.0),
}
CA_COLOR_RATIOS = [("nir", "red"), ("blue", "nir"), ("green", "red")]

RV_DEFAULT = 4.8
EBV_LO, EBV_HI = 0.0, 1.5
# Blueshifted edge must sit beyond the joint posterior (Mar/WFCCD pile up near −370…−400).
V_LO, V_HI = -450.0, -80.0
V_PRIOR_CENTER = -210.0
V_PRIOR_SIGMA = 45.0
LN_CAL_PRIOR_SIGMA = 0.30
BLUE_SIGMA_INFLATE = 3.0
NIR_SIGMA_INFLATE = 2.0
# Ca / joint priors: inflate Mg posterior widths so the sampler can still move,
# but stay near the Mg solution (not the cool/low-E(B-V) island).
MG_PRIOR_WIDTH_SCALE = 4.0
TEFF_PRIOR_FLOOR_K = 40.0
TEFF_BOX_LO = 5350.0  # cool edge of Teff box (must contain the posterior)
EBV_PRIOR_FLOOR = 0.04
V_PRIOR_FLOOR_KMS = 25.0

N_WALKERS = 56
N_STEPS = 4500
N_BURN = 1200
SEED = 7


def epoch_short(ep: str) -> str:
    return ep.replace("2011_", "").replace("_IMACS", "I").replace("_WFCCD", "W")


def ccm89_alambda_av(wave_angstrom, rv: float = RV_DEFAULT) -> np.ndarray:
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
    al = ccm89_alambda_av(wave_angstrom, rv=rv)
    return 10.0 ** (-0.4 * al * rv * ebv)


def band_median_flux(wave, flux, lo: float, hi: float) -> float:
    w = np.asarray(wave, dtype=float)
    f = np.asarray(flux, dtype=float)
    m = (w >= lo) & (w <= hi) & np.isfinite(f) & (f > 0)
    if m.sum() < 8:
        return np.nan
    return float(np.nanmedian(f[m]))


def _band_flux_samples(wave, flux, lo, hi, n_boot: int = 40):
    w = np.asarray(wave, dtype=float)
    f = np.asarray(flux, dtype=float)
    m = (w >= lo) & (w <= hi) & np.isfinite(f) & (f > 0)
    if m.sum() < 8:
        return np.array([])
    idx = np.where(m)[0]
    rng = np.random.default_rng(0)
    return np.array([np.nanmedian(f[rng.choice(idx, size=len(idx), replace=True)]) for _ in range(n_boot)])


def prepare_flux_template_grid(ids, wmin, wmax, dw=1.0):
    wave_grid = np.arange(wmin, wmax + 0.5 * dw, dw)
    rows = []
    for n in ids:
        sptype, teff = r12.PICKLES_I[n]
        wt, ft = r12.load_pickles(n)
        rows.append((teff, sptype, n, r12.resample(wt, ft, wave_grid)))
    rows.sort(key=lambda x: x[0])
    teffs = np.array([r[0] for r in rows])
    sptypes = [r[1] for r in rows]
    ids_sorted = [r[2] for r in rows]
    flux = np.vstack([r[3] for r in rows])
    return teffs, sptypes, ids_sorted, wave_grid, flux


def template_band_fluxes(teff, teffs, wave_grid, flux_grid, bands, rv=RV_DEFAULT, ebv=0.0):
    tmpl = jm.interp_template(teff, teffs, flux_grid)
    ext = extinction_factor(wave_grid, ebv, rv=rv)
    f_red = tmpl * ext
    out = {}
    for name, (lo, hi) in bands.items():
        m = (wave_grid >= lo) & (wave_grid <= hi)
        out[name] = float(np.nanmedian(f_red[m])) if m.sum() >= 5 else np.nan
    return out


def ratio_sigma_inflate(key: str, feature: str) -> float:
    if "blue" in key:
        return BLUE_SIGMA_INFLATE
    if feature == "ca" and "nir" in key:
        return NIR_SIGMA_INFLATE
    return 1.0


def apply_ln_cal_to_ratio(ratio: float, key: str, ln_cal_blue: float, ln_cal_nir: float = 0.0) -> float:
    """Correct observed ratio for per-epoch log flux offsets."""
    num, den = key.split("/")
    ln_corr = 0.0
    if num == "blue":
        ln_corr += ln_cal_blue
    elif num == "nir":
        ln_corr += ln_cal_nir
    if den == "blue":
        ln_corr -= ln_cal_blue
    elif den == "nir":
        ln_corr -= ln_cal_nir
    return ratio * np.exp(ln_corr)


@dataclass
class MgConsistentPrior:
    """Gaussian + box prior on Teff, E(B-V), and per-epoch v from the Mg fit."""

    teff: float
    teff_sigma: float
    ebv: float
    ebv_sigma: float
    v_by_epoch: dict[str, tuple[float, float]]  # epoch -> (median, sigma)

    def teff_box(self) -> tuple[float, float]:
        orig = self.teff_sigma / MG_PRIOR_WIDTH_SCALE
        half = max(5.0 * orig, 80.0)
        # Extend the cool edge at least to TEFF_BOX_LO so the posterior is not truncated.
        lo = min(self.teff - half, TEFF_BOX_LO)
        hi = self.teff + half
        return lo, hi

    def ebv_box(self) -> tuple[float, float]:
        orig = self.ebv_sigma / MG_PRIOR_WIDTH_SCALE
        half = max(5.0 * orig, 0.08)
        return max(EBV_LO, self.ebv - half), min(EBV_HI, self.ebv + half)

    def v_box(self, epoch: str) -> tuple[float, float]:
        med, sig = self.v_by_epoch[epoch]
        orig = sig / MG_PRIOR_WIDTH_SCALE
        half = max(4.0 * orig, 40.0)
        # Extend the blueshifted edge at least to V_LO so velocities are not truncated.
        lo = min(med - half, V_LO)
        hi = min(V_HI, med + half)
        return lo, hi


def load_mg_consistent_prior(path: Path | None = None) -> MgConsistentPrior:
    path = path or (FIGDIR / "mg" / "summary.json")
    if not path.exists():
        raise SystemExit(f"Need Mg summary at {path} to constrain the Ca prior")
    with open(path) as f:
        mg = json.load(f)
    p = mg["parameters"]

    def _sig(block: dict, floor: float) -> float:
        s = 0.5 * (block["minus_1sigma"] + block["plus_1sigma"])
        return max(MG_PRIOR_WIDTH_SCALE * s, floor)

    v_by_epoch = {}
    for ep, vs in mg["velocity_kms_per_epoch"].items():
        key = f"v_{epoch_short(ep)}"
        block = p[key]
        v_by_epoch[ep] = (float(vs["median"]), _sig(block, V_PRIOR_FLOOR_KMS))
    prior = MgConsistentPrior(
        teff=float(p["Teff"]["median"]),
        teff_sigma=_sig(p["Teff"], TEFF_PRIOR_FLOOR_K),
        ebv=float(p["ebv"]["median"]),
        ebv_sigma=_sig(p["ebv"], EBV_PRIOR_FLOOR),
        v_by_epoch=v_by_epoch,
    )
    tlo, thi = prior.teff_box()
    elo, ehi = prior.ebv_box()
    print(
        f"Ca prior from Mg: Teff={prior.teff:.0f}±{prior.teff_sigma:.0f} K "
        f"(box {tlo:.0f}–{thi:.0f}); "
        f"E(B-V)={prior.ebv:.3f}±{prior.ebv_sigma:.3f} (box {elo:.3f}–{ehi:.3f})"
    )
    for ep, (med, sig) in prior.v_by_epoch.items():
        vlo, vhi = prior.v_box(ep)
        print(f"  v({ep}) = {med:.0f}±{sig:.0f} km/s (box {vlo:.0f}–{vhi:.0f})")
    return prior


@dataclass
class FeatureFitConfig:
    feature: str
    label: str
    line_window: str
    bands: dict
    ratios: list
    line_kind: str  # "mg" or "ca"
    free_ln_cal_nir: bool = False
    mg_prior: MgConsistentPrior | None = None


class ColorRatioDataset:
    def __init__(self, cfg: FeatureFitConfig, rv: float = RV_DEFAULT):
        self.cfg = cfg
        self.bands = cfg.bands
        self.ratios = cfg.ratios
        self.rv = rv
        self.epochs = []
        wmin = min(lo for lo, _ in cfg.bands.values()) - 50
        wmax = max(hi for _, hi in cfg.bands.values()) + 50
        self.teffs, self.sptypes, self.ids, self.wave_t, self.flux_t = (
            prepare_flux_template_grid(jm.FIT_TEMPLATE_IDS, wmin, wmax, dw=1.0)
        )

        for epoch, kind, pa, pb in r12.discover_epochs():
            df = r12.load_epoch(kind, pa, pb)
            w, f = df["wavelength_A"].values, df["flux"].values
            band_f = {name: band_median_flux(w, f, lo, hi) for name, (lo, hi) in cfg.bands.items()}
            ratios_d, sigmas = {}, {}
            for a, b in cfg.ratios:
                key = f"{a}/{b}"
                if not np.isfinite(band_f[a]) or not np.isfinite(band_f[b]):
                    continue
                ratios_d[key] = band_f[a] / band_f[b]
                fa = _band_flux_samples(w, f, *cfg.bands[a])
                fb = _band_flux_samples(w, f, *cfg.bands[b])
                if len(fa) > 5 and len(fb) > 5:
                    sig = float(np.nanstd(fa[:, None] / fb[None, :]))
                    sig = max(sig, 0.015 * ratios_d[key])
                else:
                    sig = 0.04 * ratios_d[key]
                sig *= ratio_sigma_inflate(key, cfg.feature)
                sigmas[key] = sig
            if len(ratios_d) == len(cfg.ratios):
                self.epochs.append(
                    {"epoch": epoch, "band_flux": band_f, "ratios": ratios_d, "sigmas": sigmas}
                )
                print(
                    f"[{cfg.feature}] {epoch}: "
                    + ", ".join(f"{k}={v:.3f}" for k, v in ratios_d.items())
                )

    @property
    def n_epochs(self):
        return len(self.epochs)

    def epoch_names(self):
        return [ep["epoch"] for ep in self.epochs]

    def model_ratio(self, teff: float, ebv: float, num: str, den: str) -> float:
        bf = template_band_fluxes(
            teff, self.teffs, self.wave_t, self.flux_t, self.bands, rv=self.rv, ebv=ebv
        )
        return bf[num] / bf[den]

    def lnlike_colors(
        self,
        teff: float,
        ebv: float,
        ln_cal_blue: np.ndarray,
        ln_cal_nir: np.ndarray | None = None,
    ) -> float:
        if ln_cal_nir is None:
            ln_cal_nir = np.zeros_like(ln_cal_blue)
        ll = 0.0
        for i, ep in enumerate(self.epochs):
            for a, b in self.ratios:
                key = f"{a}/{b}"
                pred = self.model_ratio(teff, ebv, a, b)
                if not np.isfinite(pred) or pred <= 0:
                    return -np.inf
                obs = apply_ln_cal_to_ratio(
                    ep["ratios"][key], key, ln_cal_blue[i], ln_cal_nir[i]
                )
                ll += jm.ln_gaussian(
                    np.array([obs - pred]), np.log(ep["sigmas"][key])
                )
        return ll


def n_params(n_epochs: int, free_ln_cal_nir: bool) -> int:
    # Teff, ebv, ln_sigma, v[n], ln_cal_blue[n], [ln_cal_nir[n]]
    return 3 + n_epochs + n_epochs + (n_epochs if free_ln_cal_nir else 0)


def unpack_theta(theta, n_epochs: int, free_ln_cal_nir: bool):
    teff, ebv, ln_s = theta[0], theta[1], theta[2]
    i = 3
    v = theta[i : i + n_epochs]
    i += n_epochs
    ln_cal_blue = theta[i : i + n_epochs]
    i += n_epochs
    ln_cal_nir = theta[i : i + n_epochs] if free_ln_cal_nir else np.zeros(n_epochs)
    return teff, ebv, ln_s, v, ln_cal_blue, ln_cal_nir


def param_names(epoch_names: list[str], line_param: str, free_ln_cal_nir: bool) -> list[str]:
    names = ["Teff", "ebv", line_param]
    for ep in epoch_names:
        names.append(f"v_{epoch_short(ep)}")
    for ep in epoch_names:
        names.append(f"ln_cal_blue_{epoch_short(ep)}")
    if free_ln_cal_nir:
        for ep in epoch_names:
            names.append(f"ln_cal_nir_{epoch_short(ep)}")
    return names


def corner_labels(epoch_names: list[str], line_latex: str) -> list[str]:
    labels = [r"$T_{\rm eff}$ (K)", r"$E(B-V)$", line_latex]
    for ep in epoch_names:
        short = ep.split("_")[0][:3] + ep.split("_")[-1][:3]
        labels.append(rf"$v$ ({short}) km/s")
    for ep in epoch_names:
        short = ep.split("_")[0][:3] + ep.split("_")[-1][:3]
        labels.append(rf"$\ln cal$ ({short})")
    return labels


def ln_prior_extended(
    theta,
    n_epochs: int,
    free_ln_cal_nir: bool,
    epoch_names: list[str] | None = None,
    mg_prior: MgConsistentPrior | None = None,
):
    teff, ebv, ln_s, v, ln_cal_blue, ln_cal_nir = unpack_theta(
        theta, n_epochs, free_ln_cal_nir
    )
    if not (jm.LN_SIG_LO < ln_s < jm.LN_SIG_HI):
        return -np.inf
    lp = 0.0
    if mg_prior is not None:
        tlo, thi = mg_prior.teff_box()
        elo, ehi = mg_prior.ebv_box()
        if not (tlo < teff < thi):
            return -np.inf
        if not (elo < ebv < ehi):
            return -np.inf
        lp += -0.5 * ((teff - mg_prior.teff) / mg_prior.teff_sigma) ** 2
        lp += -0.5 * ((ebv - mg_prior.ebv) / mg_prior.ebv_sigma) ** 2
        names = epoch_names or []
        for i, vi in enumerate(v):
            ep = names[i] if i < len(names) else None
            if ep is None or ep not in mg_prior.v_by_epoch:
                if not (V_LO < vi < V_HI):
                    return -np.inf
                lp += -0.5 * ((vi - V_PRIOR_CENTER) / V_PRIOR_SIGMA) ** 2
                continue
            vlo, vhi = mg_prior.v_box(ep)
            if not (vlo < vi < vhi):
                return -np.inf
            med, sig = mg_prior.v_by_epoch[ep]
            lp += -0.5 * ((vi - med) / sig) ** 2
    else:
        if not (jm.TEFF_LO < teff < jm.TEFF_HI):
            return -np.inf
        if not (EBV_LO < ebv < EBV_HI):
            return -np.inf
        for vi in v:
            if not (V_LO < vi < V_HI):
                return -np.inf
            lp += -0.5 * ((vi - V_PRIOR_CENTER) / V_PRIOR_SIGMA) ** 2
    for lc in ln_cal_blue:
        lp += -0.5 * (lc / LN_CAL_PRIOR_SIGMA) ** 2
    if free_ln_cal_nir:
        for lc in ln_cal_nir:
            lp += -0.5 * (lc / LN_CAL_PRIOR_SIGMA) ** 2
    return lp


def make_ln_posterior(cfg: FeatureFitConfig, color_data: ColorRatioDataset, line_data: jm.JointDataset):
    n_ep = color_data.n_epochs
    line_fn = jm.ln_likelihood_mgb_v if cfg.line_kind == "mg" else jm.ln_likelihood_caii_v

    def ln_posterior(theta):
        lp = ln_prior_extended(
            theta,
            n_ep,
            cfg.free_ln_cal_nir,
            epoch_names=color_data.epoch_names(),
            mg_prior=cfg.mg_prior,
        )
        if not np.isfinite(lp):
            return -np.inf
        teff, ebv, ln_s, v, ln_cal_blue, ln_cal_nir = unpack_theta(
            theta, n_ep, cfg.free_ln_cal_nir
        )
        ll = color_data.lnlike_colors(teff, ebv, ln_cal_blue, ln_cal_nir)
        ll += line_fn(teff, ln_s, v, line_data)
        return lp + ll if np.isfinite(ll) else -np.inf

    return ln_posterior


def initial_positions(
    n_epochs: int,
    free_ln_cal_nir: bool,
    seed: int,
    epoch_names: list[str] | None = None,
    mg_prior: MgConsistentPrior | None = None,
):
    ndim = n_params(n_epochs, free_ln_cal_nir)
    rng = np.random.default_rng(seed)
    pos = np.zeros((N_WALKERS, ndim))
    if mg_prior is not None:
        pos[:, 0] = mg_prior.teff + 0.3 * mg_prior.teff_sigma * rng.normal(size=N_WALKERS)
        pos[:, 1] = mg_prior.ebv + 0.3 * mg_prior.ebv_sigma * rng.normal(size=N_WALKERS)
        tlo, thi = mg_prior.teff_box()
        elo, ehi = mg_prior.ebv_box()
        pos[:, 0] = np.clip(pos[:, 0], tlo + 1, thi - 1)
        pos[:, 1] = np.clip(pos[:, 1], elo + 0.002, ehi - 0.002)
        for j, ep in enumerate(epoch_names or []):
            med, sig = mg_prior.v_by_epoch[ep]
            vlo, vhi = mg_prior.v_box(ep)
            pos[:, 3 + j] = med + 0.3 * sig * rng.normal(size=N_WALKERS)
            pos[:, 3 + j] = np.clip(pos[:, 3 + j], vlo + 1, vhi - 1)
    else:
        pos[:, 0] = rng.uniform(5200, 5550, size=N_WALKERS)
        pos[:, 1] = rng.uniform(0.85, 1.15, size=N_WALKERS)
        pos[:, 0] = np.clip(pos[:, 0], jm.TEFF_LO + 50, jm.TEFF_HI - 50)
        pos[:, 3 : 3 + n_epochs] = V_PRIOR_CENTER + 20 * rng.normal(size=(N_WALKERS, n_epochs))
        pos[:, 3 : 3 + n_epochs] = np.clip(pos[:, 3 : 3 + n_epochs], V_LO + 1, V_HI - 1)
    pos[:, 2] = np.log(0.08) + 0.15 * rng.normal(size=N_WALKERS)
    pos[:, 2] = np.clip(pos[:, 2], jm.LN_SIG_LO + 0.01, jm.LN_SIG_HI - 0.01)
    i = 3 + n_epochs
    pos[:, i : i + n_epochs] = 0.05 * rng.normal(size=(N_WALKERS, n_epochs))
    if free_ln_cal_nir:
        i += n_epochs
        pos[:, i : i + n_epochs] = 0.05 * rng.normal(size=(N_WALKERS, n_epochs))
    return pos


def summarize_chain(flat, names: list[str]):
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


def run_feature_mcmc(cfg: FeatureFitConfig, color_data: ColorRatioDataset, line_data: jm.JointDataset):
    n_ep = color_data.n_epochs
    ndim = n_params(n_ep, cfg.free_ln_cal_nir)
    ln_post = make_ln_posterior(cfg, color_data, line_data)
    pos = initial_positions(
        n_ep,
        cfg.free_ln_cal_nir,
        SEED + (0 if cfg.feature == "mg" else 100),
        epoch_names=color_data.epoch_names(),
        mg_prior=cfg.mg_prior,
    )
    sampler = emcee.EnsembleSampler(N_WALKERS, ndim, ln_post)
    print(f"\n[{cfg.feature.upper()}] MCMC: {N_WALKERS} walkers × {N_STEPS} steps, {ndim} params …")
    sampler.run_mcmc(pos, N_STEPS, progress=True)
    print(f"[{cfg.feature.upper()}] Mean acceptance: {np.mean(sampler.acceptance_fraction):.3f}")
    return sampler


def plot_feature_results(
    cfg: FeatureFitConfig,
    color_data: ColorRatioDataset,
    line_data: jm.JointDataset,
    sampler,
    summary: dict,
    names: list[str],
    rv: float,
    outdir: Path,
):
    outdir.mkdir(parents=True, exist_ok=True)
    flat = sampler.get_chain(discard=N_BURN, flat=True, thin=8)
    teff_med = summary["Teff"]["median"]
    ebv_med = summary["ebv"]["median"]
    sp, _ = r12.sptype_from_teff(teff_med)
    line_key = "ln_sigma_mgb" if cfg.line_kind == "mg" else "ln_sigma_caii"
    line_latex = r"$\ln\sigma_{\rm Mg}$" if cfg.line_kind == "mg" else r"$\ln\sigma_{\rm Ca}$"
    epoch_names = color_data.epoch_names()

    # Full corner (all parameters)
    try:
        import corner

        labels = corner_labels(epoch_names, line_latex)
        if cfg.free_ln_cal_nir:
            for ep in epoch_names:
                short = ep.split("_")[0][:3] + ep.split("_")[-1][:3]
                labels.append(rf"$\ln cal_{{nir}}$ ({short})")
        fig = corner.corner(
            flat,
            labels=labels[: flat.shape[1]],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".2f",
        )
        fig.suptitle(
            f"{cfg.label}\n{sp}, $T_{{\\rm eff}}={teff_med:.0f}$ K, $E(B-V)={ebv_med:.3f}$",
            y=1.02,
            fontsize=10,
        )
        fig.savefig(outdir / "corner_full.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

        # Focused corner: Teff, E(B-V), per-epoch v, ln_sigma
        idx = [0, 1, 2] + list(range(3, 3 + color_data.n_epochs))
        sub_labels = [
            r"$T_{\rm eff}$ (K)",
            r"$E(B-V)$",
            line_latex,
        ] + [rf"$v$ ({ep.split('_')[0][:3]}{ep.split('_')[-1][:3]})" for ep in epoch_names]
        fig = corner.corner(
            flat[:, idx],
            labels=sub_labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".2f",
        )
        fig.suptitle(f"{cfg.label}: $T_{{\\rm eff}}$, $E(B-V)$, per-epoch $v$", fontsize=11)
        fig.savefig(outdir / "corner_teff_ebv.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"corner plot skipped: {exc}")

    # Teff vs E(B-V)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist2d(flat[:, 1], flat[:, 0], bins=45, cmap="Blues", cmin=1)
    ax.axvline(ebv_med, color="C3", ls="--")
    ax.axhline(teff_med, color="C0", ls="--")
    ax.set_xlabel(r"$E(B-V)$")
    ax.set_ylabel(r"$T_{\rm eff}$ (K)")
    ax.set_title(cfg.label)
    fig.tight_layout()
    fig.savefig(outdir / "teff_ebv_posterior.png", dpi=160)
    plt.close(fig)

    # Per-epoch velocity posteriors
    fig, ax = plt.subplots(figsize=(7, 4))
    vcols = range(3, 3 + color_data.n_epochs)
    for j, (i, ep) in enumerate(zip(vcols, epoch_names)):
        ax.hist(flat[:, i], bins=35, histtype="step", lw=1.5, label=ep.replace("_", " "))
    ax.axvline(V_PRIOR_CENTER, color="k", ls=":", alpha=0.6, label="Rest+ prior")
    ax.set_xlabel(r"$v$ (km s$^{-1}$)")
    ax.set_ylabel("density")
    ax.set_title("Per-epoch velocity posteriors")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "velocity_posteriors.png", dpi=150)
    plt.close(fig)

    # Diagnostic (use median calibration + velocity)
    teff, ebv, ln_s, v, ln_cal_blue, ln_cal_nir = unpack_theta(
        np.median(flat, axis=0), color_data.n_epochs, cfg.free_ln_cal_nir
    )
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    preds, obses, sigs, labels = [], [], [], []
    for i, ep in enumerate(color_data.epochs):
        for a, b in color_data.ratios:
            key = f"{a}/{b}"
            preds.append(color_data.model_ratio(teff, ebv, a, b))
            obses.append(
                apply_ln_cal_to_ratio(ep["ratios"][key], key, ln_cal_blue[i], ln_cal_nir[i])
            )
            sigs.append(ep["sigmas"][key])
            labels.append(f"{ep['epoch'][:3]} {key}")

    axs[0, 0].plot(preds, obses, "o", ms=8)
    lims = [min(preds + obses) * 0.85, max(preds + obses) * 1.08]
    axs[0, 0].plot(lims, lims, "k--")
    axs[0, 0].set_xlabel("Model ratio")
    axs[0, 0].set_ylabel("Observed (cal-corrected)")
    axs[0, 0].set_title("(a) Continuum color ratios")

    resid = [(o - p) / s for o, p, s in zip(obses, preds, sigs)]
    axs[0, 1].bar(range(len(labels)), resid, color="C0", alpha=0.85)
    axs[0, 1].axhline(0, color="k", lw=0.8)
    axs[0, 1].set_xticks(range(len(labels)))
    axs[0, 1].set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    axs[0, 1].set_title("(b) Continuum residuals")

    if cfg.line_kind == "mg":
        w_lo, w_hi = jm.MGB_LO, jm.MGB_HI
        w_t, f_t = line_data.w_m, line_data.f_m
        title_line = "Mg b"
    else:
        w_lo, w_hi = jm.CA_LO, jm.CA_HI
        w_t, f_t = line_data.w_c, line_data.f_c
        title_line = "Ca II + Paschen"

    ax = axs[1, 0]
    for i, ep in enumerate(line_data.epochs):
        if cfg.line_kind == "mg":
            w_obs, f_obs = ep["w_mg"], ep["f_mg"]
        else:
            w_obs, f_obs = ep["w_ca"], ep["f_ca"]
        m_line = jm.model_on_obs(w_obs, v[i], teff, line_data.teffs, w_t, f_t)
        ax.plot(w_obs, f_obs, lw=0.8, alpha=0.85, label=f"{ep['epoch'][:12]}  v={v[i]:.0f}")
        ax.plot(w_obs, m_line, lw=1.0, ls="--", alpha=0.85)
    ax.set_xlim(w_lo, w_hi)
    ax.set_ylabel("Fλ / cont.")
    ax.set_xlabel("Observed λ (Å)")
    ax.set_title(f"(c) {title_line} (all epochs, median fit)")
    ax.legend(fontsize=6)

    ax = axs[1, 1]
    xpos = np.arange(color_data.n_epochs)
    ax.bar(xpos - 0.15, ln_cal_blue, width=0.3, label=r"$\ln cal_{blue}$")
    if cfg.free_ln_cal_nir:
        ax.bar(xpos + 0.15, ln_cal_nir, width=0.3, label=r"$\ln cal_{nir}$")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([e[:12] for e in epoch_names], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("ln flux offset")
    ax.set_title("(d) Per-epoch calibration offsets")
    ax.legend(fontsize=8)

    fig.suptitle(f"{cfg.label}: {sp}, $R_V={rv}$", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "fit_diagnostic.png", dpi=160)
    plt.close(fig)

    # Traces for global + velocity params
    fig, axs = plt.subplots(3 + color_data.n_epochs, 1, figsize=(10, 2 + 1.2 * color_data.n_epochs), sharex=True)
    chain = sampler.get_chain(discard=N_BURN, thin=8)
    trace_names = ["Teff", "E(B-V)", line_key] + [f"v:{ep[:12]}" for ep in epoch_names]
    for i, (ax, nm) in enumerate(zip(axs, trace_names)):
        ax.plot(chain[:, :, i], "k", alpha=0.12, lw=0.5)
        ax.set_ylabel(nm, fontsize=8)
    axs[-1].set_xlabel("step")
    fig.suptitle(f"{cfg.label} traces")
    fig.tight_layout()
    fig.savefig(outdir / "mcmc_traces.png", dpi=140)
    plt.close(fig)


def run_one_feature(cfg: FeatureFitConfig, line_data: jm.JointDataset, rv: float):
    outdir = FIGDIR / cfg.feature
    color_data = ColorRatioDataset(cfg, rv=rv)
    if color_data.n_epochs == 0:
        raise SystemExit(f"No color coverage for {cfg.feature}")

    line_param = "ln_sigma_mgb" if cfg.line_kind == "mg" else "ln_sigma_caii"
    names = param_names(color_data.epoch_names(), line_param, cfg.free_ln_cal_nir)
    sampler = run_feature_mcmc(cfg, color_data, line_data)
    flat = sampler.get_chain(discard=N_BURN, flat=True, thin=8)
    summary = summarize_chain(flat, names)
    sp, teff_near = r12.sptype_from_teff(summary["Teff"]["median"])

    v_summary = {
        ep: {
            "median": summary[f"v_{epoch_short(ep)}"]["median"],
            "p16": summary[f"v_{epoch_short(ep)}"]["p16"],
            "p84": summary[f"v_{epoch_short(ep)}"]["p84"],
        }
        for ep in color_data.epoch_names()
    }

    result = {
        "feature": cfg.feature,
        "label": cfg.label,
        "line_window_A": cfg.line_window,
        "method": (
            f"{cfg.label}; per-epoch v and ln_cal; blue σ × {BLUE_SIGMA_INFLATE}; "
            f"Pickles I, CCM89, R_V={rv}"
        ),
        "rv": rv,
        "parameters": summary,
        "teff_K": {
            "best": summary["Teff"]["median"],
            "string": (
                f"{summary['Teff']['median']:.0f}"
                f"+{summary['Teff']['plus_1sigma']:.0f}"
                f"-{summary['Teff']['minus_1sigma']:.0f}"
            ),
        },
        "ebv": {
            "best": summary["ebv"]["median"],
            "string": (
                f"{summary['ebv']['median']:.3f}"
                f"+{summary['ebv']['plus_1sigma']:.3f}"
                f"-{summary['ebv']['minus_1sigma']:.3f}"
            ),
        },
        "velocity_kms_per_epoch": v_summary,
        "nearest_pickles": {"sptype": sp, "teff_K": teff_near},
        "blue_sigma_inflate": BLUE_SIGMA_INFLATE,
        "mg_consistent_prior": (
            None
            if cfg.mg_prior is None
            else {
                "teff_K": cfg.mg_prior.teff,
                "teff_sigma_K": cfg.mg_prior.teff_sigma,
                "ebv": cfg.mg_prior.ebv,
                "ebv_sigma": cfg.mg_prior.ebv_sigma,
                "v_kms": {
                    ep: {"median": m, "sigma": s}
                    for ep, (m, s) in cfg.mg_prior.v_by_epoch.items()
                },
            }
        ),
        "mcmc": {
            "n_walkers": N_WALKERS,
            "n_steps": N_STEPS,
            "n_burn": N_BURN,
            "n_params": len(names),
            "mean_acceptance": float(np.mean(sampler.acceptance_fraction)),
        },
    }

    with open(outdir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(flat, columns=names).to_csv(outdir / "mcmc_chain_thinned.csv", index=False)
    plot_feature_results(cfg, color_data, line_data, sampler, summary, names, rv, outdir)

    print(f"\n=== {cfg.label} ===")
    print(f"Teff = {result['teff_K']['string']} K  → {sp}")
    print(f"E(B-V) = {result['ebv']['string']}  (R_V={rv})")
    for ep, vs in v_summary.items():
        print(f"  v({ep}) = {vs['median']:.0f} [{vs['p16']:.0f}, {vs['p84']:.0f}] km/s")
    print(f"Figures → {outdir}")
    return result


JOINT_PARAM_NAMES = [
    "Teff",
    "ebv",
    "v_MarIMACS",
    "v_AprIMACS",
    "v_AprWFC",
    "ln_sigma_mgb",
    "ln_sigma_caii",
]
JOINT_CORNER_LABELS = [
    r"$T_{\rm eff}$ (K)",
    r"$E(B-V)$",
    r"$v_{\rm Mar, IMACS}$ (km s$^{-1}$)",
    r"$v_{\rm Apr, IMACS}$ (km s$^{-1}$)",
    r"$v_{\rm Apr, WFCCD}$ (km s$^{-1}$)",
    r"$\ln\sigma_{\rm Mg}$",
    r"$\ln\sigma_{\rm Ca}$",
]
# Shorter names for diagonal titles (units stay on axis labels).
JOINT_CORNER_TITLES = [
    r"$T_{\rm eff}$",
    r"$E(B-V)$",
    r"$v_{\rm Mar, IMACS}$",
    r"$v_{\rm Apr, IMACS}$",
    r"$v_{\rm Apr, WFCCD}$",
    r"$\ln\sigma_{\rm Mg}$",
    r"$\ln\sigma_{\rm Ca}$",
]
# Per-epoch blue calibration offsets are free nuisance params (not shown on main corner).
JOINT_N_NUISANCE = 3  # ln_cal_blue × 3 epochs
JOINT_NDIM = 7 + JOINT_N_NUISANCE


def unpack_joint(theta):
    """theta = [Teff, ebv, v0, v1, v2, ln_s_mg, ln_s_ca, ln_cal_blue×3]."""
    teff, ebv = theta[0], theta[1]
    v = np.asarray(theta[2:5], dtype=float)
    ln_s_m, ln_s_c = theta[5], theta[6]
    ln_cal_blue = np.asarray(theta[7:10], dtype=float)
    return teff, ebv, ln_s_m, ln_s_c, v, ln_cal_blue


def ln_prior_joint(theta, epoch_names: list[str], mg_prior: MgConsistentPrior):
    teff, ebv, ln_s_m, ln_s_c, v, ln_cal_blue = unpack_joint(theta)
    if not (jm.LN_SIG_LO < ln_s_m < jm.LN_SIG_HI):
        return -np.inf
    if not (jm.LN_SIG_LO < ln_s_c < jm.LN_SIG_HI):
        return -np.inf
    tlo, thi = mg_prior.teff_box()
    elo, ehi = mg_prior.ebv_box()
    if not (tlo < teff < thi) or not (elo < ebv < ehi):
        return -np.inf
    lp = -0.5 * ((teff - mg_prior.teff) / mg_prior.teff_sigma) ** 2
    lp += -0.5 * ((ebv - mg_prior.ebv) / mg_prior.ebv_sigma) ** 2
    for i, vi in enumerate(v):
        ep = epoch_names[i]
        vlo, vhi = mg_prior.v_box(ep)
        if not (vlo < vi < vhi):
            return -np.inf
        med, sig = mg_prior.v_by_epoch[ep]
        lp += -0.5 * ((vi - med) / sig) ** 2
    for lc in ln_cal_blue:
        lp += -0.5 * (lc / LN_CAL_PRIOR_SIGMA) ** 2
    return lp


def ln_posterior_joint(theta, color_mg, color_ca, line_data, epoch_names, mg_prior):
    lp = ln_prior_joint(theta, epoch_names, mg_prior)
    if not np.isfinite(lp):
        return -np.inf
    teff, ebv, ln_s_m, ln_s_c, v, ln_cal_blue = unpack_joint(theta)
    # Optical continuum once (Mg bands). Ca II contributes via line shapes only
    # — NIR color ratios are poorly calibrated and pin E(B-V) to the prior edge.
    ll = color_mg.lnlike_colors(teff, ebv, ln_cal_blue)
    ll += jm.ln_likelihood_mgb_v(teff, ln_s_m, v, line_data)
    ll += jm.ln_likelihood_caii_v(teff, ln_s_c, v, line_data)
    return lp + ll if np.isfinite(ll) else -np.inf


def initial_positions_joint(epoch_names: list[str], mg_prior: MgConsistentPrior, seed: int):
    rng = np.random.default_rng(seed)
    pos = np.zeros((N_WALKERS, JOINT_NDIM))
    tlo, thi = mg_prior.teff_box()
    elo, ehi = mg_prior.ebv_box()
    pos[:, 0] = np.clip(
        mg_prior.teff + 0.3 * mg_prior.teff_sigma * rng.normal(size=N_WALKERS),
        tlo + 1,
        thi - 1,
    )
    pos[:, 1] = np.clip(
        mg_prior.ebv + 0.3 * mg_prior.ebv_sigma * rng.normal(size=N_WALKERS),
        elo + 0.002,
        ehi - 0.002,
    )
    for j, ep in enumerate(epoch_names):
        med, sig = mg_prior.v_by_epoch[ep]
        vlo, vhi = mg_prior.v_box(ep)
        pos[:, 2 + j] = np.clip(med + 0.3 * sig * rng.normal(size=N_WALKERS), vlo + 1, vhi - 1)
    pos[:, 5] = np.clip(np.log(0.08) + 0.15 * rng.normal(size=N_WALKERS), jm.LN_SIG_LO + 0.01, jm.LN_SIG_HI - 0.01)
    pos[:, 6] = np.clip(np.log(0.10) + 0.15 * rng.normal(size=N_WALKERS), jm.LN_SIG_LO + 0.01, jm.LN_SIG_HI - 0.01)
    pos[:, 7:10] = 0.05 * rng.normal(size=(N_WALKERS, 3))
    return pos


def plot_joint_corner(flat, summary, rv: float, outdir: Path):
    teff_med = summary["Teff"]["median"]
    ebv_med = summary["ebv"]["median"]
    try:
        import corner

        # Main corner: Teff, E(B-V), velocities, then ln σ (physics params only)
        fig = corner.corner(
            flat[:, :7],
            labels=JOINT_CORNER_LABELS,
            titles=JOINT_CORNER_TITLES,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".2f",
            # Left-align titles so long velocity labels don't overlap the panel to the left.
            title_kwargs={"loc": "left", "fontsize": 10},
            label_kwargs={"fontsize": 15},
        )
        # Two-line, left-flush titles on velocity diagonals (name / quantile).
        axes = np.asarray(fig.axes).reshape(7, 7)
        for i in (2, 3, 4):
            ax = axes[i, i]
            title = ""
            for loc in ("left", "center", "right"):
                t = ax.get_title(loc=loc)
                if t:
                    title = t
                    ax.set_title("", loc=loc)
            if " = " in title:
                name, val = title.split(" = ", 1)
                title = f"{name}\n= {val}"
            if title:
                ax.set_title(title, loc="left", fontsize=10)
        fig.savefig(outdir / "joint_teff_ebv_corner.png", dpi=140, bbox_inches="tight")
        fig.savefig(FIGDIR / "joint_teff_ebv_corner.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"joint corner plot skipped: {exc}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist2d(flat[:, 1], flat[:, 0], bins=45, cmap="Blues", cmin=1)
    ax.axvline(ebv_med, color="C3", ls="--")
    ax.axhline(teff_med, color="C0", ls="--")
    ax.set_xlabel(r"$E(B-V)$")
    ax.set_ylabel(r"$T_{\rm eff}$ (K)")
    ax.set_title("Joint Mg + Ca")
    fig.tight_layout()
    fig.savefig(outdir / "teff_ebv_posterior.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, lab in enumerate(
        [r"$v_{\rm Mar, IMACS}$", r"$v_{\rm Apr, IMACS}$", r"$v_{\rm Apr, WFCCD}$"]
    ):
        ax.hist(flat[:, 2 + i], bins=35, histtype="step", lw=1.5, label=lab)
    ax.set_xlabel(r"$v$ (km s$^{-1}$)")
    ax.set_ylabel("counts")
    ax.set_title("Joint per-epoch velocity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "velocity_posteriors.png", dpi=150)
    plt.close(fig)


def run_joint_fit(line_data: jm.JointDataset, rv: float, mg_prior: MgConsistentPrior):
    outdir = FIGDIR / "joint"
    outdir.mkdir(parents=True, exist_ok=True)

    mg_cfg = FeatureFitConfig(
        "mg",
        "Mg (joint)",
        f"{jm.MGB_LO:.0f}-{jm.MGB_HI:.0f}",
        MG_COLOR_BANDS,
        MG_COLOR_RATIOS,
        "mg",
    )
    color_mg = ColorRatioDataset(mg_cfg, rv=rv)
    epoch_names = color_mg.epoch_names()
    if [e["epoch"] for e in line_data.epochs] != epoch_names:
        raise SystemExit("Line-shape epoch order differs from color datasets")
    if len(epoch_names) != 3:
        raise SystemExit(f"Expected 3 epochs, got {epoch_names}")

    pos = initial_positions_joint(epoch_names, mg_prior, SEED + 200)
    sampler = emcee.EnsembleSampler(
        N_WALKERS,
        JOINT_NDIM,
        ln_posterior_joint,
        args=(color_mg, None, line_data, epoch_names, mg_prior),
    )
    print(f"\n[JOINT] MCMC: {N_WALKERS} walkers × {N_STEPS} steps, {JOINT_NDIM} params …")
    sampler.run_mcmc(pos, N_STEPS, progress=True)
    print(f"[JOINT] Mean acceptance: {np.mean(sampler.acceptance_fraction):.3f}")

    flat = sampler.get_chain(discard=N_BURN, flat=True, thin=8)
    all_names = JOINT_PARAM_NAMES + [f"ln_cal_blue_{epoch_short(ep)}" for ep in epoch_names]
    summary = summarize_chain(flat, all_names)
    sp, teff_near = r12.sptype_from_teff(summary["Teff"]["median"])
    v_keys = ["v_MarIMACS", "v_AprIMACS", "v_AprWFC"]
    v_summary = {
        ep: {
            "median": summary[k]["median"],
            "p16": summary[k]["p16"],
            "p84": summary[k]["p84"],
        }
        for ep, k in zip(epoch_names, v_keys)
    }
    result = {
        "feature": "joint",
        "label": "Joint Mg + Ca (shared Teff, E(B-V), v; independent ln σ)",
        "method": (
            "Joint likelihood: optical continuum colors + Mg b and Ca II line shapes; "
            "shared Teff, E(B-V), per-epoch v; independent ln σ_Mg, ln σ_Ca; "
            f"per-epoch ln_cal_blue nuisance; Mg-consistent Gaussian+box prior; "
            f"blue σ × {BLUE_SIGMA_INFLATE}; NIR continuum omitted"
        ),
        "rv": rv,
        "parameters": summary,
        "teff_K": {
            "best": summary["Teff"]["median"],
            "string": (
                f"{summary['Teff']['median']:.0f}"
                f"+{summary['Teff']['plus_1sigma']:.0f}"
                f"-{summary['Teff']['minus_1sigma']:.0f}"
            ),
        },
        "ebv": {
            "best": summary["ebv"]["median"],
            "string": (
                f"{summary['ebv']['median']:.3f}"
                f"+{summary['ebv']['plus_1sigma']:.3f}"
                f"-{summary['ebv']['minus_1sigma']:.3f}"
            ),
        },
        "velocity_kms_per_epoch": v_summary,
        "nearest_pickles": {"sptype": sp, "teff_K": teff_near},
        "mg_consistent_prior": {
            "teff_K": mg_prior.teff,
            "teff_sigma_K": mg_prior.teff_sigma,
            "ebv": mg_prior.ebv,
            "ebv_sigma": mg_prior.ebv_sigma,
            "v_kms": {
                ep: {"median": m, "sigma": s} for ep, (m, s) in mg_prior.v_by_epoch.items()
            },
        },
        "mcmc": {
            "n_walkers": N_WALKERS,
            "n_steps": N_STEPS,
            "n_burn": N_BURN,
            "n_params": JOINT_NDIM,
            "mean_acceptance": float(np.mean(sampler.acceptance_fraction)),
        },
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(flat, columns=all_names).to_csv(
        outdir / "mcmc_chain_thinned.csv", index=False
    )
    plot_joint_corner(flat, summary, rv, outdir)

    print("\n=== Joint Mg + Ca ===")
    print(f"Teff = {result['teff_K']['string']} K  → {sp}")
    print(f"E(B-V) = {result['ebv']['string']}  (R_V={rv})")
    for ep, vs in v_summary.items():
        print(f"  v({ep}) = {vs['median']:.0f} [{vs['p16']:.0f}, {vs['p84']:.0f}] km/s")
    print(f"Figures → {outdir}")
    return result


def main(rv: float = RV_DEFAULT, ca_only: bool = False, joint_only: bool = False):
    print(f"Line-shape + reddening MCMC (R_V={rv}) …")
    line_data = jm.JointDataset()

    mg_cfg = FeatureFitConfig(
        "mg",
        "Mg b (optical continuum + line shape)",
        f"{jm.MGB_LO:.0f}-{jm.MGB_HI:.0f}",
        MG_COLOR_BANDS,
        MG_COLOR_RATIOS,
        "mg",
        free_ln_cal_nir=False,
    )
    skip_separate = joint_only or ca_only
    if skip_separate and (FIGDIR / "mg" / "summary.json").exists():
        with open(FIGDIR / "mg" / "summary.json") as f:
            mg_result = json.load(f)
        print("Using existing mg/summary.json")
    else:
        mg_result = run_one_feature(mg_cfg, line_data, rv)

    mg_prior = load_mg_consistent_prior()
    if joint_only:
        ca_result = None
        if (FIGDIR / "ca" / "summary.json").exists():
            with open(FIGDIR / "ca" / "summary.json") as f:
                ca_result = json.load(f)
    else:
        ca_cfg = FeatureFitConfig(
            "ca",
            "Ca II / Paschen (NIR continuum + line shape; Mg-consistent prior)",
            f"{jm.CA_LO:.0f}-{jm.CA_HI:.0f}",
            CA_COLOR_BANDS,
            CA_COLOR_RATIOS,
            "ca",
            free_ln_cal_nir=True,
            mg_prior=mg_prior,
        )
        ca_result = run_one_feature(ca_cfg, line_data, rv)

    joint_result = run_joint_fit(line_data, rv, mg_prior)

    combined = {
        "method": (
            "Mg unconstrained; Ca with Mg-consistent priors; "
            "joint Mg+Ca with shared Teff, E(B-V), v"
        ),
        "rv": rv,
        "mg": mg_result,
        "ca": ca_result,
        "joint": joint_result,
    }
    with open(FIGDIR / "reddening_summary.json", "w") as f:
        json.dump(combined, f, indent=2)

    print("\n=== Comparison ===")
    print(f"Mg:    Teff={mg_result['teff_K']['best']:.0f} K, E(B-V)={mg_result['ebv']['best']:.3f}")
    if ca_result is not None:
        print(f"Ca:    Teff={ca_result['teff_K']['best']:.0f} K, E(B-V)={ca_result['ebv']['best']:.3f}")
    print(f"Joint: Teff={joint_result['teff_K']['best']:.0f} K, E(B-V)={joint_result['ebv']['best']:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rv", type=float, default=RV_DEFAULT)
    parser.add_argument(
        "--ca-only",
        action="store_true",
        help="Reuse existing Mg summary and re-run only the Ca MCMC",
    )
    parser.add_argument(
        "--joint-only",
        action="store_true",
        help="Reuse existing Mg summary and run only the joint Mg+Ca MCMC",
    )
    args = parser.parse_args()
    main(rv=args.rv, ca_only=args.ca_only, joint_only=args.joint_only)
