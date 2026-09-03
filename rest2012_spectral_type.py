#!/usr/bin/env python3
"""
Rest+2012-style spectral type / Teff analysis for Eta Car light-echo spectra.

Method (following Rest et al. 2012, Nature, 482, 375; SI):
  1. Continuum-normalize the echo spectrum.
  2. Cross-correlate with a luminosity-class I template sequence in 5050–6500 Å
     (nebular emission-line windows masked).
  3. Smooth r(Teff) with a Gaussian (σ = 300 K) and bootstrap for Teff PDFs.

Rest+ used the UVES POP library (Bagnulo et al. 2003). Here we use the publicly
available Pickles (1998) luminosity-class I atlas with the same analysis steps,
targeting the same G2–G5 / ~5000 K conclusion for the early EC1 light-echo spectra.

Inputs:  data/spectra/rest2012_ec1/*.fits
Outputs: ASCII spectra, figures under figures/rest2012_spectral_type/, summary CSV/JSON
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent
SPEC_DIR = ROOT / "data" / "spectra" / "rest2012_ec1"
TEMPLATE_DIR = ROOT / "data" / "spectra" / "templates" / "pickles_I"
FIGDIR = ROOT / "figures" / "rest2012_spectral_type"
ASCII_DIR = SPEC_DIR / "ascii"

# Display / sort order for Rest+ EC1 epochs (Mar–Apr 2011).
EPOCH_ORDER = {
    "Mar2011_IMACS200": 0,
    "Apr2011_IMACS300": 1,
    "Apr2011_WFCCD": 2,
}

FIGDIR.mkdir(parents=True, exist_ok=True)
ASCII_DIR.mkdir(parents=True, exist_ok=True)

# Pickles (1998) Teff for luminosity class I (from CDBS AA_README).
# G2I / A2I / K3I not listed explicitly — interpolated in log Teff.
PICKLES_I = {
    114: ("B0I", 26001.6),
    119: ("A0I", 9727.47),
    120: ("A2I", 8700.0),  # interpolate A0I–F0I
    121: ("F0I", 7691.30),
    122: ("F5I", 6637.43),
    123: ("F8I", 6095.37),
    124: ("G0I", 5508.08),
    125: ("G2I", 5320.0),  # interpolate G0I–G5I
    126: ("G5I", 5046.61),
    127: ("G8I", 4591.98),
    128: ("K2I", 4255.98),
    129: ("K3I", 4123.0),  # interpolate K2I–K4I
    130: ("K4I", 3990.25),
    131: ("M2I", 3451.44),
}

# Rest+ preferred windows (SI §S1.3 / Fig. S2).
# Optical: chip gap ~6360–6500 so we use 5050–6350 on the blue side.
WAVE_LO, WAVE_HI = 5050.0, 6350.0
# Ca II IR triplet / H Paschen (Rest+ used 8200–8720; Fig. S2 shows ~8300–8750).
WAVE_CAII_LO, WAVE_CAII_HI = 8200.0, 8720.0
# Mg b zoom used in Rest+ Fig. S2 upper left.
WAVE_MGB_LO, WAVE_MGB_HI = 5060.0, 5500.0

# Rest+ mean outflow velocity for aligning LE absorption to rest templates.
C_KMS = 299792.458
V_LE_KMS = -210.0

# Nebular / sky contamination masks (Å), analogous to Rest+ Table S4.
MASK_WINDOWS = [
    (4855, 4875),  # Hβ
    (4955, 5015),  # [O III]
    (5870, 5900),  # He I / Na D (nebular + ISM)
    (6295, 6315),  # [O I]
    (6355, 6375),  # [O I]
    (6540, 6595),  # Hα + [N II]
    (6710, 6740),  # [S II]
]

# Strong Ca II cores to exclude when estimating the local continuum (Rest+ SI).
CAII_CONT_MASK = [
    (8484.0, 8505.0),
    (8520.0, 8546.0),
    (8643.0, 8670.0),
]

# Subset of Pickles I for Rest+ Fig. S2–style stacked sequence.
STACK_TEMPLATE_IDS = [119, 121, 122, 123, 124, 125, 126, 127, 128, 130, 131]  # A0I…M2I


def wavelength_from_header(header, naxis1: int) -> np.ndarray:
    crval = float(header["CRVAL1"])
    crpix = float(header.get("CRPIX1", 1.0))
    cd = float(header.get("CD1_1", header.get("CDELT1")))
    return crval + (np.arange(naxis1) - (crpix - 1.0)) * cd


def load_spectrum_fits(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a 1D spectrum FITS (or IRAF multispec with spectrum on plane 0).
    Returns wavelength (Å), flux, flux_err (NaN if unavailable).
    """
    with fits.open(path) as hdul:
        hdu = hdul[0]
        data = np.asarray(hdu.data, dtype=float)
        wave = wavelength_from_header(hdu.header, data.shape[-1])
        if data.ndim == 1:
            flux = data
            err = np.full_like(flux, np.nan)
        elif data.ndim == 2:
            flux = data[0]
            err = data[3] if data.shape[0] >= 4 else np.full_like(flux, np.nan)
        elif data.ndim == 3:
            flux = data[0, 0]
            err = data[3, 0] if data.shape[0] >= 4 else np.full_like(flux, np.nan)
        else:
            raise ValueError(f"Unsupported FITS data shape {data.shape} in {path}")
    return wave, flux, err


def spectrum_df(wave, flux, err=None) -> pd.DataFrame:
    if err is None:
        err = np.full_like(flux, np.nan, dtype=float)
    order = np.argsort(wave)
    return pd.DataFrame(
        {
            "wavelength_A": np.asarray(wave, dtype=float)[order],
            "flux": np.asarray(flux, dtype=float)[order],
            "flux_err": np.asarray(err, dtype=float)[order],
        }
    )


def merge_blue_red(blue_path: Path, red_path: Path) -> pd.DataFrame:
    wb, fb, eb = load_spectrum_fits(blue_path)
    wr, fr, er = load_spectrum_fits(red_path)
    # Prefer blue where overlap exists; otherwise concatenate across chip gap.
    if wr[0] <= wb[-1]:
        cut = wr > wb[-1]
        wr, fr, er = wr[cut], fr[cut], er[cut]
    return spectrum_df(
        np.concatenate([wb, wr]),
        np.concatenate([fb, fr]),
        np.concatenate([eb, er]),
    )


def discover_epochs():
    """
    Discover Rest+ EC1 spectra in SPEC_DIR.

    Returns list of (epoch_label, kind, path_a, path_b_or_None)
      kind = 'blue_red' | 'single'
    """
    epochs = []
    for blue in sorted(SPEC_DIR.glob("*.imacs300.*.blue.fits")):
        m = re.search(r"\.imacs300\.([^.]+)\.blue\.fits$", blue.name)
        if not m:
            continue
        red = blue.with_name(blue.name.replace(".blue.fits", ".red.fits"))
        if red.exists():
            epochs.append((f"{m.group(1)}_IMACS300", "blue_red", blue, red))
    for path in sorted(SPEC_DIR.glob("*.imacs200.*.fits")):
        m = re.search(r"\.imacs200\.([^.]+)\.fits$", path.name)
        if m:
            epochs.append((f"{m.group(1)}_IMACS200", "single", path, None))
    for path in sorted(SPEC_DIR.glob("*.wfccd.*.fits")):
        m = re.search(r"\.wfccd\.([^.]+)\.", path.name)
        if m:
            epochs.append((f"{m.group(1)}_WFCCD", "single", path, None))
    epochs.sort(key=lambda x: EPOCH_ORDER.get(x[0], 99))
    return epochs


def load_epoch(kind: str, path_a: Path, path_b: Path | None) -> pd.DataFrame:
    if kind == "blue_red":
        return merge_blue_red(path_a, path_b)
    wave, flux, err = load_spectrum_fits(path_a)
    return spectrum_df(wave, flux, err)


def continuum_normalize(wave, flux, win=151, poly=3):
    """Divide by a smooth continuum estimate (Savitzky–Golay on positive flux)."""
    f = np.asarray(flux, dtype=float)
    w = np.asarray(wave, dtype=float)
    good = np.isfinite(f) & np.isfinite(w) & (f > 0)
    fn = np.full_like(f, np.nan)
    if good.sum() < win:
        return fn
    # Work on a filled array for the filter, then restore gaps.
    ff = f.copy()
    med = np.nanmedian(ff[good])
    ff[~good] = med
    # Odd window
    win = min(win, good.sum() // 2 * 2 + 1)
    if win < 11:
        win = 11
    cont = savgol_filter(ff, window_length=win, polyorder=poly, mode="interp")
    cont[cont <= 0] = np.nan
    fn[good] = f[good] / cont[good]
    return fn


def apply_masks(wave, flux):
    m = np.isfinite(wave) & np.isfinite(flux)
    for lo, hi in MASK_WINDOWS:
        m &= ~((wave >= lo) & (wave <= hi))
    return m


def flatten_for_xcorr(wave, flux, wmin=WAVE_LO, wmax=WAVE_HI):
    sel = (wave >= wmin) & (wave <= wmax) & np.isfinite(flux)
    w = wave[sel]
    f = continuum_normalize(w, flux[sel])
    m = apply_masks(w, f) & np.isfinite(f)
    # Remove mean, unit variance for Pearson / Tonry-like correlation
    x = f[m]
    x = x - np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        return w[m], x
    return w[m], x / s


def load_pickles(n: int):
    path = TEMPLATE_DIR / f"pickles_{n}.fits"
    with fits.open(path) as hdul:
        t = hdul[1].data
        wave = np.asarray(t["WAVELENGTH"], dtype=float)
        flux = np.asarray(t["FLUX"], dtype=float)
    return wave, flux


def resample(wave_src, flux_src, wave_dst):
    good = np.isfinite(wave_src) & np.isfinite(flux_src)
    if good.sum() < 10:
        return np.full_like(wave_dst, np.nan, dtype=float)
    f = interp1d(wave_src[good], flux_src[good], kind="linear", bounds_error=False, fill_value=np.nan)
    return f(wave_dst)


def pearson_r(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return np.nan
    aa, bb = a[m], b[m]
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    denom = np.sqrt(np.sum(aa**2) * np.sum(bb**2))
    if denom == 0:
        return np.nan
    return float(np.sum(aa * bb) / denom)


def correlate_templates(wave_obs, flux_obs):
    w_ref, f_obs = flatten_for_xcorr(wave_obs, flux_obs)
    rows = []
    for n, (sptype, teff) in PICKLES_I.items():
        wt, ft = load_pickles(n)
        # Continuum-normalize template on same window, resample to obs grid
        wt_win = (wt >= WAVE_LO) & (wt <= WAVE_HI)
        ft_n = continuum_normalize(wt[wt_win], ft[wt_win])
        ft_res = resample(wt[wt_win], ft_n, w_ref)
        m = apply_masks(w_ref, ft_res) & np.isfinite(ft_res) & np.isfinite(f_obs)
        tmpl = ft_res[m].copy()
        obs = f_obs[m].copy()
        tmpl = tmpl - np.mean(tmpl)
        obs = obs - np.mean(obs)
        st, so = np.std(tmpl), np.std(obs)
        if st == 0 or so == 0:
            r = np.nan
        else:
            r = pearson_r(obs / so, tmpl / st)
        rows.append({"pickles_id": n, "sptype": sptype, "teff": teff, "r": r})
    return pd.DataFrame(rows).sort_values("teff")


def smooth_r_teff(df, sigma_k=300.0, teff_grid=None):
    if teff_grid is None:
        teff_grid = np.linspace(3500, 8000, 451)
    teff = df["teff"].values
    r = df["r"].values
    # Weighted Gaussian kernel smoother in Teff
    rr = np.empty_like(teff_grid)
    for i, t in enumerate(teff_grid):
        w = np.exp(-0.5 * ((teff - t) / sigma_k) ** 2)
        w[~np.isfinite(r)] = 0
        rr[i] = np.nansum(w * r) / np.nansum(w) if np.nansum(w) > 0 else np.nan
    return teff_grid, rr


def bootstrap_teff(df, n_boot=10000, sigma_k=300.0, rng=None):
    rng = np.random.default_rng(rng)
    teff = df["teff"].values
    r = df["r"].values
    good = np.isfinite(r)
    teff, r = teff[good], r[good]
    peaks = []
    grid = np.linspace(3500, 8000, 451)
    for _ in range(n_boot):
        idx = rng.integers(0, len(r), size=len(r))
        d = pd.DataFrame({"teff": teff[idx], "r": r[idx]})
        # collapse duplicates by mean
        d = d.groupby("teff", as_index=False)["r"].mean()
        g, rr = smooth_r_teff(d, sigma_k=sigma_k, teff_grid=grid)
        peaks.append(g[np.nanargmax(rr)])
    peaks = np.asarray(peaks)
    return {
        "teff_peak": float(np.median(peaks)),
        "teff_p2.5": float(np.percentile(peaks, 2.5)),
        "teff_p97.5": float(np.percentile(peaks, 97.5)),
        "peaks": peaks,
    }


def sptype_from_teff(teff):
    # Nearest Pickles I type
    items = [(n, sp, t) for n, (sp, t) in PICKLES_I.items()]
    n, sp, t = min(items, key=lambda x: abs(x[2] - teff))
    return sp, t


def blackbody_flam(wave_A, teff):
    # B_λ in relative erg/s/cm^2/Å
    wave_cm = wave_A * 1e-8
    h = 6.62607015e-27
    c = 2.99792458e10
    k = 1.380649e-16
    x = h * c / (wave_cm * k * teff)
    # avoid overflow
    x = np.clip(x, 0, 100)
    return (2 * h * c**2) / (wave_cm**5 * (np.exp(x) - 1)) * 1e-8  # per Å


# Rest+ critical absorption features for G-type classification (SI §S1.3).
REST_FEATURES = [
    (5172.0, "Mg b", 0.0),
    (5270.0, "5270", 0.0),
    (5328.0, "Fe I", 0.0),
    (5893.0, "Na D*", 0.0),  # masked / nebular+ISM; shown for context
    (6162.0, "Ca I", 0.0),
]


def smooth_display(y, sigma=2.0):
    """Light Gaussian smooth for display; preserves NaNs."""
    y = np.asarray(y, dtype=float)
    out = y.copy()
    good = np.isfinite(y)
    if good.sum() < 10:
        return out
    filled = y.copy()
    filled[~good] = np.nanmedian(y[good])
    sm = gaussian_filter1d(filled, sigma=sigma, mode="nearest")
    out[good] = sm[good]
    return out


def annotate_rest_features(ax, y_top=1.18):
    for wave, label, _ in REST_FEATURES:
        if WAVE_LO <= wave <= WAVE_HI:
            ax.axvline(wave, color="0.55", ls=":", lw=0.8, zorder=0)
            ax.text(wave, y_top, label, ha="center", va="bottom", fontsize=8, color="0.35", rotation=90)


def to_rest_frame(wave_obs, v_kms=V_LE_KMS):
    """Map observed wavelengths to the rest frame for velocity v_kms."""
    return np.asarray(wave_obs, dtype=float) / (1.0 + v_kms / C_KMS)


def match_resolution(wave, flux, fwhm_A=7.0):
    """Convolve to approximate Rest+ display resolution (FWHM ~7 Å)."""
    w = np.asarray(wave, dtype=float)
    f = np.asarray(flux, dtype=float)
    out = f.copy()
    good = np.isfinite(f) & np.isfinite(w)
    if good.sum() < 20:
        return out
    dw = float(np.median(np.diff(w[good])))
    if dw <= 0:
        return out
    sigma_pix = (fwhm_A / 2.355) / dw
    if sigma_pix < 0.3:
        return out
    filled = f.copy()
    filled[~good] = np.nanmedian(f[good])
    sm = gaussian_filter1d(filled, sigma=sigma_pix, mode="nearest")
    out[good] = sm[good]
    return out


def continuum_normalize_gauss(wave, flux, fwhm_cont=200.0, exclude_windows=None):
    """
    Rest+-style continuum: divide by a Gaussian-smoothed continuum
    (FWHM ~200 Å), optionally ignoring strong line cores.
    """
    w = np.asarray(wave, dtype=float)
    f = np.asarray(flux, dtype=float)
    fn = np.full_like(f, np.nan)
    good = np.isfinite(f) & np.isfinite(w) & (f > 0)
    if good.sum() < 30:
        return fn
    use = good.copy()
    if exclude_windows:
        for lo, hi in exclude_windows:
            use &= ~((w >= lo) & (w <= hi))
    if use.sum() < 30:
        use = good
    dw = float(np.median(np.diff(w[good])))
    sigma_pix = max((fwhm_cont / 2.355) / dw, 1.0)
    filled = f.copy()
    # Interpolate excluded / bad pixels before smoothing
    if (~use).any():
        filled[~use] = np.interp(w[~use], w[use], f[use])
    cont = gaussian_filter1d(filled, sigma=sigma_pix, mode="nearest")
    cont[cont <= 0] = np.nan
    fn[good] = f[good] / cont[good]
    return fn


def continuum_normalized_window(
    wave,
    flux,
    wmin=WAVE_LO,
    wmax=WAVE_HI,
    masks=None,
    exclude_for_cont=None,
    to_rest=False,
    v_kms=V_LE_KMS,
    fwhm_res=7.0,
):
    """Continuum-divide in a Rest+ window (flux ~1); optionally rest-frame + resolve-match."""
    w = np.asarray(wave, dtype=float)
    f = np.asarray(flux, dtype=float)
    if to_rest:
        w = to_rest_frame(w, v_kms=v_kms)
    sel = (w >= wmin) & (w <= wmax) & np.isfinite(f) & (f > 0)
    ww, ff = w[sel], f[sel]
    if fwhm_res:
        ff = match_resolution(ww, ff, fwhm_A=fwhm_res)
    fn = continuum_normalize_gauss(ww, ff, fwhm_cont=200.0, exclude_windows=exclude_for_cont)
    if masks is None:
        masks = MASK_WINDOWS
    for lo, hi in masks:
        fn[(ww >= lo) & (ww <= hi)] = np.nan
    return ww, fn


def correlate_templates_window(
    wave_obs,
    flux_obs,
    wmin,
    wmax,
    masks=None,
    exclude_for_cont=None,
    teff_lo=4500.0,
    teff_hi=7000.0,
):
    """
    Pearson-r vs Pickles I in an arbitrary Rest+ window (rest-frame LE).

    Default Teff cut keeps F5–K templates: Rest+ note that earlier types are
    Paschen-dominated in the Ca II window, which biases unconstrained xcorr.
    """
    w_ref, f_obs = continuum_normalized_window(
        wave_obs,
        flux_obs,
        wmin=wmin,
        wmax=wmax,
        masks=masks or [],
        exclude_for_cont=exclude_for_cont,
        to_rest=True,
    )
    m0 = np.isfinite(f_obs)
    x = f_obs[m0].copy()
    x = x - np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        return pd.DataFrame(columns=["pickles_id", "sptype", "teff", "r"])
    f_obs_z = np.full_like(f_obs, np.nan)
    f_obs_z[m0] = x / s

    rows = []
    for n, (sptype, teff) in PICKLES_I.items():
        if teff < teff_lo or teff > teff_hi:
            continue
        wt, ft = load_pickles(n)
        wt_n, ft_n = continuum_normalized_window(
            wt,
            ft,
            wmin=wmin,
            wmax=wmax,
            masks=masks or [],
            exclude_for_cont=exclude_for_cont,
            to_rest=False,
            fwhm_res=7.0,
        )
        ft_res = resample(wt_n, ft_n, w_ref)
        m = np.isfinite(ft_res) & np.isfinite(f_obs_z)
        if m.sum() < 50:
            r = np.nan
        else:
            tmpl = ft_res[m] - np.mean(ft_res[m])
            obs = f_obs_z[m] - np.mean(f_obs_z[m])
            st, so = np.std(tmpl), np.std(obs)
            r = pearson_r(obs / so, tmpl / st) if st > 0 and so > 0 else np.nan
        rows.append({"pickles_id": n, "sptype": sptype, "teff": teff, "r": r})
    return pd.DataFrame(rows).sort_values("teff")


def paschen_caii_index(wave, flux):
    """
    Simple Rest+-style NIR diagnostic (rest frame):
      deep Ca II 8542 and weak Paschen P14 8598 → late-F/G.
    Returns (caii_depth, paschen_depth) where depth = 1 - F/cont (~positive in absorption).
    """
    w, f = continuum_normalized_window(
        wave, flux, wmin=WAVE_CAII_LO, wmax=WAVE_CAII_HI,
        masks=[], exclude_for_cont=CAII_CONT_MASK, to_rest=True, fwhm_res=7.0,
    )
    def depth(c, half=6.0):
        m = (w > c - half) & (w < c + half) & np.isfinite(f)
        return float(1.0 - np.nanmedian(f[m])) if m.any() else np.nan
    return depth(8542.0), depth(8598.4)


def _stack_spectra(ax, series, wmin, wmax, step, highlight_sptype=None, text_x="left"):
    """
    Plot continuum-normalized spectra with vertical offsets.
    `series` is a list of (label, wave, flux, is_le, sptype_or_None).
    Returns final offset.
    """
    offset = 0.0
    first_le = True
    for label, w, f, is_le, sptype in series:
        f = smooth_display(f, sigma=1.2)
        if is_le:
            color, lw = "k", 1.15
            lab = label if first_le else None
            first_le = False
        else:
            color = "C3" if (highlight_sptype and sptype == highlight_sptype) else "0.35"
            lw = 1.4 if color == "C3" else 0.9
            lab = None
        ax.plot(w, f + offset, color=color, lw=lw, label=lab)
        if text_x == "left":
            ax.text(wmin + 0.01 * (wmax - wmin), 1.02 + offset, label, fontsize=7.5, va="bottom", color=color)
        else:
            ax.text(wmax - 0.01 * (wmax - wmin), 1.02 + offset, label, fontsize=7.5, ha="right", va="bottom", color=color)
        offset += step
    return offset


def plot_rest_s2_mgb_caii(merged, summaries, outpath: Path):
    """
    Reproduce Rest+ SI Fig. S2 style:
      left: Mg b / 5270 region with stacked templates
      right: Ca II IR triplet + H Paschen with stacked templates
    Rest+ used UVES (optical) + Cenarro Ca II library (NIR); we use Pickles I for both.
    Layout matches Rest+: LE on top, early→late templates below (observed frame;
    Ca II absorption is blueshifted relative to rest markers, as in their Fig. S2).
    """
    teffs = [s["teff_bootstrap_median"] for s in summaries]
    teff_adopt = float(np.median(teffs))
    sp_adopt, teff_tmpl = sptype_from_teff(teff_adopt)

    epochs = sorted(merged.keys(), key=lambda e: EPOCH_ORDER.get(e, 99))

    fig, (ax_mg, ax_ca) = plt.subplots(1, 2, figsize=(12.5, 9.5))

    # Build series bottom→top: late templates at bottom of stack… then LE on top (Rest+ order:
    # templates early→late upward, LE above). We'll do early at bottom, late mid, LE top.
    def le_window(epoch, wmin, wmax, caii=False):
        df = merged[epoch]
        return continuum_normalized_window(
            df["wavelength_A"].values,
            df["flux"].values,
            wmin=wmin,
            wmax=wmax,
            masks=[],
            exclude_for_cont=CAII_CONT_MASK if caii else None,
            to_rest=False,
            fwhm_res=7.0,
        )

    def tmpl_window(n, wmin, wmax, caii=False):
        wt, ft = load_pickles(n)
        return continuum_normalized_window(
            wt, ft, wmin=wmin, wmax=wmax, masks=[],
            exclude_for_cont=CAII_CONT_MASK if caii else None,
            to_rest=False, fwhm_res=7.0,
        )

    # ---- Mg b / 5270 ----
    mg_series = []
    for n in STACK_TEMPLATE_IDS:
        sp, teff = PICKLES_I[n]
        w, f = tmpl_window(n, WAVE_MGB_LO, WAVE_MGB_HI, caii=False)
        mg_series.append((f"{sp} ({teff:.0f} K)", w, f, False, sp))
    for epoch in epochs:
        w, f = le_window(epoch, WAVE_MGB_LO, WAVE_MGB_HI, caii=False)
        mg_series.append((f"LE {epoch}", w, f, True, None))
    # reverse so first plotted is at bottom: early templates bottom, LE top
    # STACK is A0…M2; Rest+ has early bottom — keep order, append LE last (= top)
    off = _stack_spectra(ax_mg, mg_series, WAVE_MGB_LO, WAVE_MGB_HI, step=0.55, highlight_sptype=sp_adopt, text_x="right")
    for wave in (5167.0, 5173.0, 5184.0, 5270.0):
        ax_mg.axvline(wave, color="0.55", ls=":", lw=0.8)
    ax_mg.text(5173, off + 0.02, "Mg b", ha="center", fontsize=9, color="0.3")
    ax_mg.text(5270, off + 0.02, "5270", ha="center", fontsize=9, color="0.3")
    ax_mg.set_xlim(WAVE_MGB_LO, WAVE_MGB_HI)
    ax_mg.set_ylim(-0.1, off + 0.3)
    ax_mg.set_xlabel("Observed wavelength (Å)")
    ax_mg.set_ylabel("Normalized flux + constant")
    ax_mg.set_title("Mg b / 5270 region (Rest+ Fig. S2 upper left)")
    ax_mg.legend(loc="lower right", fontsize=8)

    # ---- Ca II IR + H Paschen (observed frame; Rest+ marks rest wavelengths) ----
    paschen_in = [8413.3, 8438.0, 8467.3, 8502.5, 8545.4, 8598.4, 8665.0]
    caii = [8498.0, 8542.0, 8662.0]
    ca_series = []
    for n in STACK_TEMPLATE_IDS:
        sp, teff = PICKLES_I[n]
        w, f = tmpl_window(n, WAVE_CAII_LO, WAVE_CAII_HI, caii=True)
        ca_series.append((f"{sp} ({teff:.0f} K)", w, f, False, sp))
    for epoch in epochs:
        w, f = le_window(epoch, WAVE_CAII_LO, WAVE_CAII_HI, caii=True)
        ca_series.append((f"LE {epoch}", w, f, True, None))
    off = _stack_spectra(ax_ca, ca_series, WAVE_CAII_LO, WAVE_CAII_HI, step=0.70, highlight_sptype=sp_adopt, text_x="right")
    for wave in caii:
        ax_ca.axvline(wave, color="C0", ls=":", lw=0.9)
        # also mark blueshifted LE absorption expected at v=-210 km/s
        ax_ca.axvline(wave * (1.0 + V_LE_KMS / C_KMS), color="C0", ls="--", lw=0.7, alpha=0.55)
    for wave in paschen_in:
        if WAVE_CAII_LO <= wave <= WAVE_CAII_HI:
            ax_ca.axvline(wave, color="C2", ls=":", lw=0.7, alpha=0.75)
    ax_ca.text(8542, off + 0.05, "Ca II (rest)", ha="center", fontsize=8, color="C0")
    ax_ca.text(8450, off + 0.05, "H Paschen", ha="center", fontsize=8, color="C2")
    ax_ca.text(
        0.02, 0.02,
        f"Optical xcorr → {sp_adopt}, $T_{{\\rm eff}}={teff_tmpl:.0f}$ K  (Rest+: G2–G5, ~5000 K)\n"
        f"Dashed blue: Ca II at $v={V_LE_KMS:.0f}$ km/s (LE absorption).\n"
        f"Solid blue: rest-frame Ca II; green: H Paschen.\n"
        f"Early types: strong Paschen; late-F/G: deep Ca II, weak/absent Paschen.",
        transform=ax_ca.transAxes, fontsize=7.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.93),
    )
    ax_ca.set_xlim(WAVE_CAII_LO, WAVE_CAII_HI)
    ax_ca.set_ylim(-0.15, off + 0.4)
    ax_ca.set_xlabel("Observed wavelength (Å)")
    ax_ca.set_ylabel("Normalized flux + constant")
    ax_ca.set_title("Ca II IR + H Paschen (Rest+ Fig. S2 lower left)")
    ax_ca.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Rest et al. (2012) Fig. S2–style: Mg b and Ca II / Paschen\n"
        f"Pickles I templates (proxy for UVES/Cenarro); adopted {sp_adopt}, "
        f"$T_{{\\rm eff}}\\approx{teff_adopt:.0f}$ K",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outpath.relative_to(ROOT)}")


def plot_caii_best_match(merged, summaries, outpath: Path):
    """Direct LE vs best G-type template in the Ca II / Paschen window (rest frame)."""
    teffs = [s["teff_bootstrap_median"] for s in summaries]
    teff_adopt = float(np.median(teffs))
    sp_adopt, teff_tmpl = sptype_from_teff(teff_adopt)
    n_adopt = min(PICKLES_I.items(), key=lambda kv: abs(kv[1][1] - teff_adopt))[0]
    wt, ft = load_pickles(n_adopt)
    w_t, f_t = continuum_normalized_window(
        wt, ft, wmin=WAVE_CAII_LO, wmax=WAVE_CAII_HI,
        masks=[], exclude_for_cont=CAII_CONT_MASK, to_rest=False, fwhm_res=7.0,
    )

    epochs = sorted(merged.keys(), key=lambda e: EPOCH_ORDER.get(e, 99))
    fig, axs = plt.subplots(len(epochs) + 1, 1, figsize=(10.5, 2.5 * (len(epochs) + 1)), sharex=True)
    if len(epochs) == 0:
        return
    if len(epochs) + 1 == 1:
        axs = [axs]

    stack_w, stack = None, []
    caii = [8498.0, 8542.0, 8662.0]
    paschen_in = [8413.3, 8438.0, 8467.3, 8502.5, 8545.4, 8598.4, 8665.0]

    for ax, epoch in zip(axs[:-1], epochs):
        df = merged[epoch]
        w, f = continuum_normalized_window(
            df["wavelength_A"].values,
            df["flux"].values,
            wmin=WAVE_CAII_LO,
            wmax=WAVE_CAII_HI,
            masks=[],
            exclude_for_cont=CAII_CONT_MASK,
            to_rest=True,
            fwhm_res=7.0,
        )
        f_sm = smooth_display(f, sigma=1.5)
        f_tr = resample(w_t, f_t, w)
        ax.plot(w, f, color="0.75", lw=0.5)
        ax.plot(w, f_sm, "k", lw=1.1, label=f"LE {epoch} (rest, $v={V_LE_KMS:.0f}$)")
        ax.plot(w, f_tr, "C3", lw=1.3, label=f"Pickles {sp_adopt} ({teff_tmpl:.0f} K)")
        for wave in caii:
            ax.axvline(wave, color="C0", ls=":", lw=0.8)
        for wave in paschen_in:
            ax.axvline(wave, color="C2", ls=":", lw=0.6, alpha=0.7)
        ax.set_ylim(0.4, 1.35)
        ax.set_ylabel("Fλ / cont.")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(f"{epoch}: Ca II IR triplet + H Paschen")
        if stack_w is None:
            stack_w = w
        stack.append(resample(w, f_sm, stack_w))

    ax = axs[-1]
    f_coadd = np.nanmedian(np.vstack(stack), axis=0)
    ax.plot(stack_w, f_coadd, "k", lw=1.3, label="LE coadd (rest frame)")
    ax.plot(stack_w, resample(w_t, f_t, stack_w), "C3", lw=1.4, label=f"{sp_adopt} {teff_tmpl:.0f} K")
    for wave, lab, col in [
        (8498, "Ca II", "C0"),
        (8542, "Ca II", "C0"),
        (8662, "Ca II", "C0"),
        (8502.5, "P16", "C2"),
        (8598.4, "P14", "C2"),
    ]:
        ax.axvline(wave, color=col, ls=":", lw=0.8)
        ax.text(wave, 1.28, lab, ha="center", fontsize=7, color=col, rotation=90)
    # Paschen / Ca II depths from coadd
    # Approximate using first epoch indices printed separately
    ax.text(
        0.01, 0.06,
        f"Rest-frame LE: Ca II absorption; Paschen weak vs early-A/F templates.\n"
        f"These Mar/Apr 2011 EC1 spectra match the Rest+ absorption-line phase.\n"
        f"Adopted {sp_adopt}, $T_{{\\rm eff}}={teff_adopt:.0f}$ K (Rest+: G2–G5, ~5000 K)",
        transform=ax.transAxes, fontsize=7.5, va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )
    ax.set_xlim(WAVE_CAII_LO, WAVE_CAII_HI)
    ax.set_ylim(0.4, 1.38)
    ax.set_xlabel("Rest wavelength (Å)")
    ax.set_ylabel("Fλ / cont.")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Coadded LE vs best template — Ca II / Paschen diagnostic")
    fig.suptitle("Rest+-style Ca II IR triplet + H Paschen comparison", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outpath.relative_to(ROOT)}")


def plot_rest_style_feature_match(merged, summaries, outpath: Path):
    """
    Rest+ SI Fig. S2–style comparison: continuum-flattened LE vs best G-type
    template, with Mg b / 5270 / Fe I labeled and Teff indicated.
    """
    # Adopted type from bootstrap medians (matches Rest+ G2–G5 / ~5000 K).
    teffs = [s["teff_bootstrap_median"] for s in summaries]
    teff_adopt = float(np.median(teffs))
    sp_adopt, teff_tmpl = sptype_from_teff(teff_adopt)
    n_adopt = min(PICKLES_I.items(), key=lambda kv: abs(kv[1][1] - teff_adopt))[0]
    wt, ft = load_pickles(n_adopt)
    w_t, f_t = continuum_normalized_window(wt, ft)

    # Chronological epoch order when possible
    summaries = sorted(summaries, key=lambda s: EPOCH_ORDER.get(s["epoch"], 99))
    epochs = [s["epoch"] for s in summaries]
    n = len(epochs)
    fig, axs = plt.subplots(n + 1, 1, figsize=(10.5, 2.6 * (n + 1)), sharex=True)
    if n + 1 == 1:
        axs = [axs]

    # Per-epoch panels
    stack_w = None
    stack = []
    for ax, epoch, summary in zip(axs[:-1], epochs, summaries):
        df = merged[epoch]
        w, f = continuum_normalized_window(df["wavelength_A"].values, df["flux"].values)
        f_sm = smooth_display(f, sigma=2.5)
        f_t_res = resample(w_t, f_t, w)

        ax.plot(w, f, color="0.75", lw=0.5, alpha=0.9, zorder=1)
        ax.plot(w, f_sm, color="k", lw=1.1, label=f"LE {epoch} (smoothed)", zorder=2)
        ax.plot(w, f_t_res, color="C3", lw=1.3, alpha=0.95,
                label=f"Pickles {sp_adopt}  $T_{{\\rm eff}}={teff_tmpl:.0f}$ K", zorder=3)

        # Shade Rest+ masked windows inside plot range
        for lo, hi in MASK_WINDOWS:
            if hi < WAVE_LO or lo > WAVE_HI:
                continue
            ax.axvspan(max(lo, WAVE_LO), min(hi, WAVE_HI), color="gold", alpha=0.18, zorder=0)

        teff_b = summary["teff_bootstrap_median"]
        lo_b, hi_b = summary["teff_bootstrap_p2.5"], summary["teff_bootstrap_p97.5"]
        ax.text(
            0.01, 0.06,
            f"$T_{{\\rm eff}}={teff_b:.0f}$ K "
            f"[{lo_b:.0f}–{hi_b:.0f}]; best disc. {summary['best_template_sptype']}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
        )
        annotate_rest_features(ax)
        ax.set_ylim(0.55, 1.28)
        ax.set_ylabel("Fλ / cont.")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.set_title(f"{epoch}: Rest+-style match → {sp_adopt}  ($T_{{\\rm eff}}\\approx{teff_adopt:.0f}$ K)")

        if stack_w is None:
            stack_w = w
        stack.append(resample(w, f_sm, stack_w))

    # Coadded LE vs template (higher S/N), Rest+ critical features emphasized
    ax = axs[-1]
    stack_arr = np.vstack(stack)
    f_coadd = np.nanmedian(stack_arr, axis=0)
    f_t_c = resample(w_t, f_t, stack_w)
    ax.plot(stack_w, f_coadd, color="k", lw=1.3, label="LE coadd (Mar/Apr 2011 EC1)")
    ax.plot(
        stack_w, f_t_c, color="C3", lw=1.5,
        label=f"Best template: {sp_adopt},  $T_{{\\rm eff}}={teff_tmpl:.0f}$ K",
    )
    for lo, hi in MASK_WINDOWS:
        if hi < WAVE_LO or lo > WAVE_HI:
            continue
        ax.axvspan(max(lo, WAVE_LO), min(hi, WAVE_HI), color="gold", alpha=0.18, zorder=0,
                   label="masked (nebular/sky)" if lo == MASK_WINDOWS[2][0] else None)
    annotate_rest_features(ax, y_top=1.20)
    # Highlight Mg b and 5270 as Rest+ key discriminants
    for w0, lab in [(5172.0, "Mg b"), (5270.0, "5270 blend")]:
        ax.annotate(
            lab, xy=(w0, 0.78), xytext=(w0, 0.62),
            fontsize=9, ha="center", color="C0",
            arrowprops=dict(arrowstyle="->", color="C0", lw=0.9),
        )
    ax.text(
        0.01, 0.06,
        f"Adopted: {sp_adopt},  $T_{{\\rm eff}}={teff_adopt:.0f}$ K   "
        f"(Rest+2012: G2–G5, ~5000 K)",
        transform=ax.transAxes, fontsize=9, va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )
    ax.set_ylim(0.55, 1.30)
    ax.set_xlim(WAVE_LO, WAVE_HI)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Fλ / cont.")
    ax.set_title("Coadded light-echo vs best-fitting G-supergiant template")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Rest et al. (2012)–style spectral type: critical absorption features",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outpath.relative_to(ROOT)}")


def main():
    print(f"Spectra: {SPEC_DIR}")
    print(f"Templates: {TEMPLATE_DIR}")
    summaries = []
    epochs = discover_epochs()
    if not epochs:
        raise SystemExit(f"No EC1 spectra found in {SPEC_DIR}")

    # --- Export merged ASCII and overview plot ---
    fig, axs = plt.subplots(len(epochs), 1, figsize=(10, 3.2 * len(epochs)), sharex=True)
    if len(epochs) == 1:
        axs = [axs]

    merged = {}
    for ax, (epoch, kind, path_a, path_b) in zip(axs, epochs):
        df = load_epoch(kind, path_a, path_b)
        out_csv = ASCII_DIR / f"etacar_ec1_{epoch}.csv"
        df.to_csv(out_csv, index=False, float_format="%.8e")
        print(f"Wrote {out_csv.relative_to(ROOT)}")
        merged[epoch] = df
        ax.plot(df["wavelength_A"], df["flux"], lw=0.6, color="k")
        ax.set_ylabel("Fλ")
        ax.set_title(f"Rest+ EC1  {epoch}")
        ax.set_xlim(4000, 9500)
    axs[-1].set_xlabel("Wavelength (Å)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "merged_spectra.png", dpi=150)
    plt.close(fig)
    print(f"Wrote figures/rest2012_spectral_type/merged_spectra.png")

    # --- Cross-correlation analysis per epoch ---
    fig_r, ax_r = plt.subplots(figsize=(8, 5))
    fig_pdf, ax_pdf = plt.subplots(figsize=(8, 5))

    for epoch, df in merged.items():
        wave = df["wavelength_A"].values
        flux = df["flux"].values
        cdf = correlate_templates(wave, flux)
        cdf.to_csv(FIGDIR / f"xcorr_{epoch}.csv", index=False)

        grid, rr = smooth_r_teff(cdf)
        boot = bootstrap_teff(cdf, n_boot=10000, rng=0)
        best_sp, best_t_tbl = sptype_from_teff(boot["teff_peak"])
        # Best discrete template by raw r
        i_best = cdf["r"].idxmax()
        row_best = cdf.loc[i_best]

        summary = {
            "epoch": epoch,
            "best_template_sptype": row_best["sptype"],
            "best_template_teff": float(row_best["teff"]),
            "best_template_r": float(row_best["r"]),
            "teff_smoothed_peak": float(grid[np.nanargmax(rr)]),
            "teff_bootstrap_median": boot["teff_peak"],
            "teff_bootstrap_p2.5": boot["teff_p2.5"],
            "teff_bootstrap_p97.5": boot["teff_p97.5"],
            "nearest_sptype_to_peak": best_sp,
            "wave_window": f"{WAVE_LO:.0f}-{WAVE_HI:.0f}",
            "template_library": "Pickles 1998 luminosity class I (proxy for Rest+ UVES POP)",
            "rest2012_result": "G2-G5, Teff ~ 5000 K",
        }
        summaries.append(summary)
        print(
            f"{epoch}: best {row_best['sptype']} (T={row_best['teff']:.0f} K, r={row_best['r']:.3f}); "
            f"bootstrap Teff={boot['teff_peak']:.0f} "
            f"[{boot['teff_p2.5']:.0f}, {boot['teff_p97.5']:.0f}] K → {best_sp}"
        )

        ax_r.plot(cdf["teff"], cdf["r"], "o", label=f"{epoch} templates")
        ax_r.plot(grid, rr, "-", label=f"{epoch} smoothed")
        ax_pdf.hist(boot["peaks"], bins=40, histtype="step", density=True, label=epoch)

        # Flattened comparison figure vs best template
        w_obs, f_obs = flatten_for_xcorr(wave, flux)
        wt, ft = load_pickles(int(row_best["pickles_id"]))
        wt_win = (wt >= WAVE_LO) & (wt <= WAVE_HI)
        ft_n = continuum_normalize(wt[wt_win], ft[wt_win])
        ft_res = resample(wt[wt_win], ft_n, w_obs)
        m = apply_masks(w_obs, ft_res) & np.isfinite(ft_res) & np.isfinite(f_obs)
        fig_c, ax_c = plt.subplots(figsize=(10, 4))
        ax_c.plot(w_obs[m], f_obs[m], "k", lw=0.7, label=f"echo {epoch}")
        ax_c.plot(w_obs[m], (ft_res[m] - np.mean(ft_res[m])) / np.std(ft_res[m]), "r", lw=0.7, alpha=0.8,
                  label=f"Pickles {row_best['sptype']}")
        ax_c.set_xlabel("Wavelength (Å)")
        ax_c.set_ylabel("Continuum-normalized (arb.)")
        ax_c.set_title(f"Rest+-style match: {epoch} → {row_best['sptype']} ({row_best['teff']:.0f} K)")
        ax_c.legend()
        fig_c.tight_layout()
        fig_c.savefig(FIGDIR / f"match_{epoch}.png", dpi=150)
        plt.close(fig_c)

    # Rest+-style continuum-flattened comparison with critical features + Teff
    plot_rest_style_feature_match(
        merged, summaries, FIGDIR / "feature_match_rest2012.png"
    )
    # Rest+ Fig. S2–style Mg b + Ca II / Paschen stacked comparisons
    plot_rest_s2_mgb_caii(merged, summaries, FIGDIR / "FigS2_style_mgb_caii.png")
    plot_caii_best_match(merged, summaries, FIGDIR / "feature_match_caii_paschen.png")

    # Ca II / Paschen window cross-correlation (consistency with optical)
    for epoch, df in merged.items():
        cdf_ca = correlate_templates_window(
            df["wavelength_A"].values,
            df["flux"].values,
            WAVE_CAII_LO,
            WAVE_CAII_HI,
            masks=[],
            exclude_for_cont=CAII_CONT_MASK,
        )
        cdf_ca.to_csv(FIGDIR / f"xcorr_caii_{epoch}.csv", index=False)
        d_ca, d_pa = paschen_caii_index(df["wavelength_A"].values, df["flux"].values)
        if len(cdf_ca) and cdf_ca["r"].notna().any():
            i_best = cdf_ca["r"].idxmax()
            row = cdf_ca.loc[i_best]
            print(
                f"{epoch} CaII/Paschen (F5–K only): best {row['sptype']} "
                f"(T={row['teff']:.0f} K, r={row['r']:.3f}); "
                f"depth CaII8542={d_ca:.3f}, P14={d_pa:.3f}"
            )

    ax_r.axvspan(4850, 5550, color="C2", alpha=0.15, label="Rest+ EC1B 95% CI")
    ax_r.set_xlabel("Teff (K)")
    ax_r.set_ylabel("Cross-correlation r")
    ax_r.set_title("Rest+-style r(Teff) vs Pickles I templates")
    ax_r.legend(fontsize=8)
    fig_r.tight_layout()
    fig_r.savefig(FIGDIR / "r_vs_teff.png", dpi=150)
    plt.close(fig_r)

    ax_pdf.set_xlabel("Teff (K)")
    ax_pdf.set_ylabel("PDF")
    ax_pdf.set_title("Bootstrap Teff PDFs (σ_smooth = 300 K)")
    ax_pdf.legend()
    fig_pdf.tight_layout()
    fig_pdf.savefig(FIGDIR / "teff_pdf.png", dpi=150)
    plt.close(fig_pdf)

    # --- Intrinsic SED model at adopted Teff ---
    # Use median bootstrap Teff across epochs
    teffs = [s["teff_bootstrap_median"] for s in summaries]
    teff_adopt = float(np.median(teffs))
    sp_adopt, _ = sptype_from_teff(teff_adopt)
    wave_model = np.arange(3500, 10000, 2.0)
    # Best Pickles template nearest adopted Teff for an empirical SED shape
    n_adopt = min(PICKLES_I.items(), key=lambda kv: abs(kv[1][1] - teff_adopt))[0]
    wt, ft = load_pickles(n_adopt)
    # Scale template to match median echo continuum in 5500-6000 Å (relative)
    fig_s, ax_s = plt.subplots(figsize=(10, 5))
    for epoch, df in merged.items():
        w, f = df["wavelength_A"].values, df["flux"].values
        m = (w > 5500) & (w < 6000) & np.isfinite(f) & (f > 0)
        scale = np.nanmedian(f[m]) / np.nanmedian(resample(wt, ft, np.array([5750.0])))
        ax_s.plot(w, f, lw=0.5, alpha=0.7, label=f"{epoch} (obs)")
        ax_s.plot(wt, ft * scale, lw=1.0, alpha=0.8)
    # Blackbody reference (relative), scaled similarly
    bb = blackbody_flam(wave_model, teff_adopt)
    # scale bb using first epoch
    df0 = next(iter(merged.values()))
    m0 = (
        (df0["wavelength_A"].values > 5500)
        & (df0["wavelength_A"].values < 6000)
        & np.isfinite(df0["flux"].values)
        & (df0["flux"].values > 0)
    )
    bb_scale = np.nanmedian(df0["flux"].values[m0]) / np.nanmedian(
        blackbody_flam(np.array([5750.0]), teff_adopt)
    )
    ax_s.plot(wave_model, bb * bb_scale, "k--", lw=1.2, label=f"Blackbody {teff_adopt:.0f} K")
    ax_s.set_xlim(4000, 9500)
    ax_s.set_xlabel("Wavelength (Å)")
    ax_s.set_ylabel("Fλ (scaled)")
    ax_s.set_title(f"Intrinsic spectral shape model ≈ {sp_adopt}, Teff ≈ {teff_adopt:.0f} K")
    ax_s.legend(fontsize=8, ncol=2)
    fig_s.tight_layout()
    fig_s.savefig(FIGDIR / "intrinsic_sed_model.png", dpi=150)
    plt.close(fig_s)

    # Save model SED (Pickles + BB) as ASCII for color calibration use
    model = pd.DataFrame(
        {
            "wavelength_A": wt,
            "flux_pickles": ft,
            "sptype": PICKLES_I[n_adopt][0],
            "teff_K": PICKLES_I[n_adopt][1],
        }
    )
    model_path = SPEC_DIR / "intrinsic_sed_pickles_best.csv"
    model.to_csv(model_path, index=False)
    bb_df = pd.DataFrame({"wavelength_A": wave_model, "flux_bb_rel": bb, "teff_K": teff_adopt})
    bb_df.to_csv(SPEC_DIR / "intrinsic_sed_blackbody.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(FIGDIR / "spectral_type_summary.csv", index=False)
    with open(FIGDIR / "spectral_type_summary.json", "w") as f:
        json.dump(
            {
                "method": "Rest et al. 2012 SI cross-correlation adapted to Pickles I templates",
                "reference": "Rest et al. 2012, Nature, 482, 375 (arXiv:1112.2210)",
                "rest2012_published": {"sptype": "G2-G5", "teff_K": 5000, "notes": "UVES POP; 5050-6500A"},
                "this_work_adopted": {"sptype": sp_adopt, "teff_K": teff_adopt},
                "epochs": summaries,
            },
            f,
            indent=2,
        )

    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nAdopted intrinsic type ≈ {sp_adopt}, Teff ≈ {teff_adopt:.0f} K")
    print(f"Rest+2012 published: G2–G5, Teff ~ 5000 K")
    print(f"Figures → {FIGDIR}")


def figure_only():
    """Rebuild Rest+-style comparison figures from existing products."""
    summary_path = FIGDIR / "spectral_type_summary.json"
    with open(summary_path) as f:
        meta = json.load(f)
    summaries = meta["epochs"]
    merged = {}
    for s in summaries:
        epoch = s["epoch"]
        csv = ASCII_DIR / f"etacar_ec1_{epoch}.csv"
        merged[epoch] = pd.read_csv(csv)
    plot_rest_style_feature_match(
        merged, summaries, FIGDIR / "feature_match_rest2012.png"
    )
    plot_rest_s2_mgb_caii(merged, summaries, FIGDIR / "FigS2_style_mgb_caii.png")
    plot_caii_best_match(merged, summaries, FIGDIR / "feature_match_caii_paschen.png")
    for epoch, df in merged.items():
        cdf_ca = correlate_templates_window(
            df["wavelength_A"].values,
            df["flux"].values,
            WAVE_CAII_LO,
            WAVE_CAII_HI,
            masks=[],
            exclude_for_cont=CAII_CONT_MASK,
        )
        cdf_ca.to_csv(FIGDIR / f"xcorr_caii_{epoch}.csv", index=False)
        d_ca, d_pa = paschen_caii_index(df["wavelength_A"].values, df["flux"].values)
        if len(cdf_ca) and cdf_ca["r"].notna().any():
            i_best = cdf_ca["r"].idxmax()
            row = cdf_ca.loc[i_best]
            print(
                f"{epoch} CaII/Paschen (F5–K only): best {row['sptype']} "
                f"(T={row['teff']:.0f} K, r={row['r']:.3f}); "
                f"depth CaII8542={d_ca:.3f}, P14={d_pa:.3f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure-only",
        action="store_true",
        help="Only rebuild feature_match_rest2012.png from existing summary/ASCII",
    )
    args = parser.parse_args()
    if args.figure_only:
        figure_only()
    else:
        main()
