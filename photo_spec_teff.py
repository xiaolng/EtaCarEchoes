#!/usr/bin/env python3
"""
Same-temperature test for EC1 photometry and spectroscopy.

One forward model is applied to both datasets:

    Pickles I spectrum, interpolated in Teff
    × CCM89 extinction at fixed R_V
    convolved with CTIO/DECam griz

Photometry: DECam light-curve colors at the 2011 spectral epoch.
Spectroscopy: the same colors, measured by integrating each EC1 spectrum
through those filter curves. No line fitting, no separate color calibration.

The temperature is the posterior in Teff after marginalizing E(B-V).
If the two measurements are the same, the posteriors overlap.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import color_to_Teff_mc as ct
import rest2012_joint_mcmc as jm
import rest2012_reddening_mcmc as rm
import rest2012_spectral_type as r12

ROOT = Path(__file__).resolve().parent
FILTER_DIR = ROOT / "data" / "filters" / "decam"
FIGDIR = r12.FIGDIR
RV = 4.8
# Spectral-epoch photometry. April 6 2011 is the IMACS 300 night.
PHOTO_MJD = 55657.0
# Colors used for the temperature. g is left out: Hβ and [O III] move it by
# tenths of a magnitude, which is not part of this continuum test.
COLORS = ("r-i", "i-z")
# Floor under the color uncertainty. Photometric errors are ~0.01 mag;
# 0.05 mag is the band-to-band calibration floor, larger than the 0.02 mag
# Ca II shift in i-z and smaller than the 0.1 mag Hα shift in r.
COLOR_SIGMA = {"r-i": 0.05, "i-z": 0.04}
LINE_TEFF = 5411.0  # joint Mg+Ca line fit


def load_filters() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for band in "griz":
        w, t = np.loadtxt(FILTER_DIR / f"decam_{band}.dat").T
        out[band] = (w, t)
    return out


def band_signal(wave: np.ndarray, flux: np.ndarray, filt, min_cov: float = 0.85):
    ww, tp = filt
    fi = np.interp(ww, wave, flux, left=np.nan, right=np.nan)
    m = np.isfinite(fi) & (tp > 0)
    cov = np.trapezoid(tp[m], ww[m]) / np.trapezoid(tp, ww)
    if cov < min_cov:
        return np.nan, cov
    wt = tp[m] * ww[m]
    sig = np.trapezoid(fi[m] * wt, ww[m]) / np.trapezoid(wt, ww[m])
    return sig, cov


def colors_from_spectrum(wave, flux, filters) -> dict[str, float]:
    sig = {b: band_signal(wave, flux, filters[b])[0] for b in "riz"}
    out = {}
    for key in COLORS:
        a, b = key.split("-")
        if not np.isfinite(sig[a]) or not np.isfinite(sig[b]) or sig[a] <= 0 or sig[b] <= 0:
            out[key] = np.nan
        else:
            out[key] = -2.5 * np.log10(sig[a] / sig[b])
    return out


def photometric_colors() -> dict[str, float]:
    frames = ct.load_models()
    inds = ct.add_ab_mags(*frames)
    t, i, _, g, _, r, _, z = ct.interpolate_mags(*frames, inds)
    j = int(np.argmin(np.abs(t - PHOTO_MJD)))
    return {"g-r": g[j] - r[j], "r-i": r[j] - i[j], "i-z": i[j] - z[j]}


def template_library(filters):
    ids = [n for n, (_, te) in r12.PICKLES_I.items() if 3900.0 <= te <= 10000.0]
    teffs, sptypes, _, wave, flux = rm.prepare_flux_template_grid(ids, 3500.0, 10500.0, dw=2.0)
    # Colors are scale-free. Normalize before interpolating so a bright template
    # does not dominate the blend.
    norm = np.nanmedian(flux[:, (wave > 5400) & (wave < 5600)], axis=1)
    flux = flux / norm[:, None]
    return teffs, sptypes, wave, flux


def model_colors(teff, ebv, teffs, wave, flux, filters) -> dict[str, float]:
    tmpl = jm.interp_template(teff, teffs, flux)
    reddened = tmpl * rm.extinction_factor(wave, ebv, rv=RV)
    return colors_from_spectrum(wave, reddened, filters)


def loglike(obs: dict[str, float], pred: dict[str, float]) -> float:
    chi = 0.0
    for key in COLORS:
        if not np.isfinite(obs[key]) or not np.isfinite(pred[key]):
            return -np.inf
        chi += ((obs[key] - pred[key]) / COLOR_SIGMA[key]) ** 2
    return -0.5 * chi


def teff_posterior(obs, teffs, wave, flux, filters, ebv_grid, teff_grid):
    """p(Teff) with E(B-V) marginalized on a flat grid."""
    logw = np.full(len(teff_grid), -np.inf)
    best = np.full(len(teff_grid), np.inf)
    for i, teff in enumerate(teff_grid):
        terms = []
        for ebv in ebv_grid:
            ll = loglike(obs, model_colors(teff, ebv, teffs, wave, flux, filters))
            terms.append(ll)
            if np.isfinite(ll):
                best[i] = min(best[i], -2.0 * ll)
        terms = np.asarray(terms, dtype=float)
        if np.any(np.isfinite(terms)):
            m = np.nanmax(terms)
            logw[i] = m + np.log(np.sum(np.exp(terms - m)))
    finite = np.isfinite(logw)
    logw = np.where(finite, logw - np.nanmax(logw[finite]), -np.inf)
    prob = np.exp(logw)
    prob /= np.trapezoid(prob, teff_grid)
    cdf = np.cumsum(prob)
    cdf /= cdf[-1]

    def quantile(q):
        return float(np.interp(q, cdf, teff_grid))

    return {
        "teff": teff_grid,
        "prob": prob,
        "chi2_min": best,
        "p16": quantile(0.16),
        "p50": quantile(0.50),
        "p84": quantile(0.84),
        "peak": float(teff_grid[int(np.argmax(prob))]),
        "chi2_at_peak": float(best[int(np.argmax(prob))]),
    }


def main():
    filters = load_filters()
    teffs, _, wave, flux = template_library(filters)
    photo = photometric_colors()
    datasets = {"DECam photometry": photo}
    for epoch, kind, pa, pb in r12.discover_epochs():
        df = r12.load_epoch(kind, pa, pb)
        cols = colors_from_spectrum(df.wavelength_A.values, df.flux.values, filters)
        if not all(np.isfinite(cols[k]) for k in COLORS):
            print(f"Skipping {epoch}: a color is uncovered by the spectrum ({cols})")
            continue
        datasets[epoch] = cols

    teff_grid = np.linspace(4000.0, 9000.0, 81)
    ebv_grid = np.linspace(0.0, 1.4, 29)
    results = {
        name: teff_posterior(obs, teffs, wave, flux, filters, ebv_grid, teff_grid)
        for name, obs in datasets.items()
    }

    print(f"R_V = {RV} fixed. Colors: {', '.join(COLORS)}. E(B-V) marginalized 0–1.4.")
    print(f"{'dataset':22s} {'r-i':7s} {'i-z':7s} {'peak':7s} {'p50':7s} {'16–84':14s} {'chi2':6s}")
    for name, obs in datasets.items():
        res = results[name]
        print(
            f"{name:22s} {obs['r-i']:7.3f} {obs['i-z']:7.3f} "
            f"{res['peak']:7.0f} {res['p50']:7.0f} "
            f"{res['p16']:.0f}–{res['p84']:.0f}{'':4s} {res['chi2_at_peak']:6.2f}"
        )
    print(f"Line temperature (joint Mg+Ca) = {LINE_TEFF:.0f} K")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    styles = {
        "DECam photometry": ("k", 2.2),
        "Mar2011_IMACS200": ("C0", 1.6),
        "Apr2011_IMACS300": ("C1", 1.6),
        "Apr2011_WFCCD": ("C3", 1.4),
    }
    for name, res in results.items():
        color, lw = styles.get(name, ("0.4", 1.2))
        ax.plot(res["teff"], res["prob"], color=color, lw=lw, label=name.replace("_", " "))
    ax.axvline(LINE_TEFF, color="0.35", ls="--", lw=1.0, label=f"line fit {LINE_TEFF:.0f} K")
    ax.set_xlabel(r"$T_{\rm eff}$ (K)")
    ax.set_ylabel(r"$p(T_{\rm eff})$")
    ax.set_xlim(teff_grid[0], teff_grid[-1])
    ax.set_title(
        r"Same model on DECam $r-i$, $i-z$"
        "\n"
        r"Pickles I, CCM89, $R_V=4.8$, $E(B-V)$ marginalized"
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = FIGDIR / "photo_spec_teff.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
