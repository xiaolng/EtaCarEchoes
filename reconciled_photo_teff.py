#!/usr/bin/env python3
"""
Photometric Teff reconciled with the Rest+2012 spectral line temperature.

Why the earlier photometry disagreed
------------------------------------
Xiaolong's notebook (and color_to_Teff_mc.py defaults) used E(B-V)=1 with
R_V=3.1, Alonso dwarf (V-I) calibrations, and mixed colors including i-z.
The EC1 SED is not a CCM89 curve at one E(B-V) from g through z: the blue
continuum matches E(B-V)~0.9 while i-z is too blue for that extrapolation
(scattering; see sed_shape_comparison.png). i-z therefore cannot return the
line temperature under the spectral reddening.

What this script does instead
-----------------------------
1. Fix reddening to the joint Mg+Ca continuum result: E(B-V)=0.909, R_V=4.8.
2. Deredden DECam g,r,i,z with filter-convolved CCM89 (not SDSS λ_eff).
3. Convert dereddened g-i to Teff with the same Pickles I library used in the
   spectral fit. g-i spans the wavelength region where the March continuum
   and the spectral E(B-V) agree; i and z are shown for the light curve but
   are not used for Teff.
4. Propagate photometric errors and the MCMC E(B-V) uncertainty into Teff.
5. Overlay the spectral line Teff at the 2011 epochs.

Outputs under figures/rest2012_spectral_type/:
  photo_teff_comp.png
  reconciled_photo_teff_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splrep

import color_to_Teff_mc as ct
import photo_spec_teff as ps
import rest2012_joint_mcmc as jm
import rest2012_reddening_mcmc as rm
import rest2012_spectral_type as r12

ROOT = Path(__file__).resolve().parent
FIGDIR = r12.FIGDIR
FIGDIR.mkdir(parents=True, exist_ok=True)

# Joint Mg+Ca reddening MCMC (rest2012_reddening_mcmc / joint summary)
EBV = 0.909
EBV_ERR = 0.018
RV = 4.8
LINE_TEFF = 5411.0
LINE_TEFF_ERR = 9.0
# Spectral nights (Rest+2012 EC1)
SPECTRAL_EPOCHS = {
    "Mar IMACS": 55635.0,
    "Apr IMACS": 55657.0,
    "Apr WFCCD": 55658.0,
}
# Residual g-band floor after difference imaging (nebular / calibration).
# Smaller than the full on-sky Hβ+[O III] bias (~0.2–0.5 mag) because the
# light curve is difference-imaged; large enough that Teff errors are not
# photon-noise limited.
GI_SYS = 0.05


def band_extinctions(filters, teffs, wave, flux, ebv: float, rv: float = RV) -> dict[str, float]:
    """A_λ through each DECam band for a 5411 K Pickles I continuum."""
    tmpl = jm.interp_template(LINE_TEFF, teffs, flux)
    out = {}
    for band in "griz":
        s0, _ = ps.band_signal(wave, tmpl, filters[band], min_cov=0.7)
        s1, _ = ps.band_signal(
            wave, tmpl * rm.extinction_factor(wave, ebv, rv=rv), filters[band], min_cov=0.7
        )
        out[band] = float(-2.5 * np.log10(s1 / s0))
    return out


def pickles_gi_calibration(filters, teffs, wave, flux):
    """Intrinsic (g-i)_0 vs Teff for Pickles I (monotonic, cooler = redder)."""
    tgrid = np.linspace(4000.0, 9000.0, 251)
    gi0 = np.empty_like(tgrid)
    for i, t in enumerate(tgrid):
        tmpl = jm.interp_template(float(t), teffs, flux)
        sg, _ = ps.band_signal(wave, tmpl, filters["g"], min_cov=0.7)
        si, _ = ps.band_signal(wave, tmpl, filters["i"], min_cov=0.7)
        gi0[i] = -2.5 * np.log10(sg / si)
    if not np.all(np.diff(gi0) < 0):
        raise RuntimeError("Pickles g-i is not monotonic in Teff")
    return tgrid, gi0


def teff_from_gi(gi, tgrid, gi0):
    """Invert (g-i)_0 → Teff. gi0 decreases with Teff."""
    gi = np.asarray(gi, dtype=float)
    return np.interp(gi, gi0[::-1], tgrid[::-1], left=np.nan, right=np.nan)


def load_light_curves():
    frames = ct.load_models()
    inds = ct.add_ab_mags(*frames)
    t, i, ie, g, ge, r, re, z = ct.interpolate_mags(*frames, inds)
    LCm2_z = frames[3]
    wz = inds[3]
    ze = splev(t, splrep(LCm2_z.loc[wz, "mjd"], LCm2_z.loc[wz, "AB_mag_err"], k=1))
    return t, g, ge, r, re, i, ie, z, ze


def main():
    filters = ps.load_filters()
    teffs, _, wave, flux = ps.template_library(filters)
    tgrid, gi0 = pickles_gi_calibration(filters, teffs, wave, flux)

    A = band_extinctions(filters, teffs, wave, flux, EBV, RV)
    A_hi = band_extinctions(filters, teffs, wave, flux, EBV + EBV_ERR, RV)
    A_lo = band_extinctions(filters, teffs, wave, flux, EBV - EBV_ERR, RV)
    e_gi = (A["g"] - A["i"])
    de_gi = 0.5 * abs((A_hi["g"] - A_hi["i"]) - (A_lo["g"] - A_lo["i"]))

    t, g, ge, r, re, i, ie, z, ze = load_light_curves()
    g0, r0, i0, z0 = g - A["g"], r - A["r"], i - A["i"], z - A["z"]

    gi = g0 - i0
    gr = g0 - r0
    ri = r0 - i0
    iz = i0 - z0
    gi_err_phot = np.sqrt(ge**2 + ie**2)
    gi_err = np.sqrt(gi_err_phot**2 + de_gi**2 + GI_SYS**2)
    gr_err = np.sqrt(ge**2 + re**2 + GI_SYS**2)
    ri_err = np.sqrt(re**2 + ie**2)
    iz_err = np.sqrt(ie**2 + ze**2)

    teff = teff_from_gi(gi, tgrid, gi0)
    teff_lo = teff_from_gi(gi + gi_err, tgrid, gi0)  # redder → cooler
    teff_hi = teff_from_gi(gi - gi_err, tgrid, gi0)

    j = int(np.argmin(np.abs(t - SPECTRAL_EPOCHS["Apr IMACS"])))
    summary = {
        "ebv": EBV,
        "ebv_err": EBV_ERR,
        "rv": RV,
        "A_band": A,
        "E_g_i": e_gi,
        "color_for_teff": "g-i",
        "gi_sys_mag": GI_SYS,
        "calibration": "Pickles luminosity class I through DECam g,i",
        "spectral_line_teff": LINE_TEFF,
        "spectral_line_teff_err": LINE_TEFF_ERR,
        "photo_epoch_mjd": float(t[j]),
        "observed_g_i": float(g[j] - i[j]),
        "dereddened_g_i": float(gi[j]),
        "dereddened_g_i_err": float(gi_err[j]),
        "photo_teff": float(teff[j]),
        "photo_teff_lo": float(teff_lo[j]),
        "photo_teff_hi": float(teff_hi[j]),
    }
    out_json = FIGDIR / "reconciled_photo_teff_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # ---- figure: LC / colors / Teff ----
    fig, axs = plt.subplots(
        3, 1, sharex=True, figsize=(10.0, 9.4), gridspec_kw={"hspace": 0.06}
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.07)
    ylab_kw = dict(fontsize=14)

    ax = axs[0]
    for y, ye, c, lab in (
        (g0, ge, "C0", "g"),
        (r0, re, "C3", "r"),
        (i0, ie, "C1", "i"),
        (z0, ze, "C4", "z"),
    ):
        ax.fill_between(t, y - ye, y + ye, color=c, alpha=0.2, linewidth=0)
        ax.plot(t, y, color=c, lw=1.2, label=lab)
    ax.invert_yaxis()
    ax.set_ylabel("AB mag", **ylab_kw)
    ax.legend(loc="upper right", ncol=4, frameon=False, fontsize=9)
    ax.set_title(
        rf"EC1 photometry (dereddened), CCM89 $E(B-V)={EBV:.3f}\pm{EBV_ERR:.3f}$, $R_V={RV}$"
    )

    ax = axs[1]
    for y, ye, c, lab in (
        (gr, gr_err, "C0", "g-r"),
        (gi, gi_err, "C2", "g-i"),
        (ri, ri_err, "C3", "r-i"),
        (iz, iz_err, "C4", "i-z"),
    ):
        ax.fill_between(t, y - ye, y + ye, color=c, alpha=0.18, linewidth=0)
        ax.plot(t, y, color=c, lw=1.2, label=lab)
    ax.axhline(0, color="0.75", lw=0.6)
    ax.set_ylabel("color", **ylab_kw)
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)

    ax = axs[2]
    ax.fill_between(
        t, teff_lo, teff_hi, color="C2", alpha=0.25, linewidth=0, label=r"$g-i$ $T_{\rm eff}$ 68%"
    )
    ax.plot(t, teff, color="C2", lw=1.4, label=r"$g-i$ Pickles I")
    spec_mjds = list(SPECTRAL_EPOCHS.values())
    for mjd in spec_mjds:
        ax.errorbar(
            mjd,
            LINE_TEFF,
            yerr=LINE_TEFF_ERR,
            fmt="o",
            color="k",
            ms=7,
            zorder=5,
        )
    ax.axhline(LINE_TEFF, color="k", ls=":", lw=0.9, alpha=0.7)
    ax.text(
        float(np.mean(spec_mjds)),
        LINE_TEFF - 280,
        r"LE spectral $T_{\rm eff}$",
        ha="center",
        va="top",
        fontsize=9,
        color="0.1",
        zorder=10,
        bbox=dict(
            boxstyle="square,pad=0.35",
            facecolor="white",
            edgecolor="black",
            linewidth=1.2,
        ),
    )
    ax.set_ylabel(r"$T_{\rm eff}$ (K)", fontsize=14)
    ax.set_xlabel("MJD", fontsize=12)
    ax.set_ylim(4200, 7000)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    for ax in axs:
        for mjd in spec_mjds:
            ax.axvline(mjd, color="0.5", ls="--", lw=0.7, alpha=0.6)
        ax.set_xlim(t.min(), t.max())

    # Centered over the Teff minimum, after the curve has dropped below ~5200 K.
    axins = axs[2].inset_axes([0.40, 0.52, 0.24, 0.28])
    m = (t > 55200) & (t < 56200)
    axins.fill_between(t[m], teff_lo[m], teff_hi[m], color="C2", alpha=0.25, linewidth=0)
    axins.plot(t[m], teff[m], color="C2", lw=1.3)
    for mjd in spec_mjds:
        axins.errorbar(mjd, LINE_TEFF, yerr=LINE_TEFF_ERR, fmt="o", color="k", ms=6)
        axins.axvline(mjd, color="0.5", ls="--", lw=0.6, alpha=0.6)
    axins.axhline(LINE_TEFF, color="k", ls=":", lw=0.8, alpha=0.7)
    axins.set_ylim(5200, 5800)
    axins.set_xlim(55200, 56200)
    axins.tick_params(labelsize=7)
    axins.set_title("2011 spectral epoch", fontsize=8, pad=2)
    axins.patch.set_facecolor("white")
    axins.patch.set_alpha(1.0)
    axins.set_facecolor("white")

    out = FIGDIR / "photo_teff_comp.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    old = FIGDIR / "reconciled_photo_teff.png"
    if old.exists():
        old.unlink()
    print(f"Wrote {out.relative_to(ROOT)}")
    print(f"Wrote {out_json.relative_to(ROOT)}")
    print(
        f"At MJD {t[j]:.0f}: photo Teff = {teff[j]:.0f} "
        f"[{teff_lo[j]:.0f}, {teff_hi[j]:.0f}] K vs spectral {LINE_TEFF:.0f}±{LINE_TEFF_ERR:.0f} K"
    )


if __name__ == "__main__":
    main()
