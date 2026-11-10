#!/usr/bin/env python3
"""
Convert MCMC light-curve models to colors, Teff, radius, and HR diagram.

Converted from old_vers1/color_to_Teff_mc.ipynb. Figures are written to figures/color_to_Teff/.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import splev, splrep
from scipy.optimize import fsolve

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "figures" / "color_to_Teff"
LCPATH = ROOT / "emcee_fitting" / "LC_tables" / "models"
GPMODEL_DIR = ROOT / "data" / "other_transients" / "gpmodels_for_publication"
UGC_PATH = ROOT / "data" / "other_transients" / "UGC2773_unfiltered.csv"

FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["axes.labelsize"] = 26
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 24


def convert_Jy2mag(x_jy, x_jy_err=None):
    """Convert micro-Janskys to AB mag (and optional error)."""
    mag = 23.9 - 2.5 * np.log10(x_jy)
    if x_jy_err is not None:
        mag_err = 1.0857 * (x_jy_err / x_jy)
        return mag, mag_err
    return mag


def savefig(name: str, tight: bool = True) -> Path:
    path = FIGDIR / name
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path.relative_to(ROOT)}")
    return path


def data_limits(*arrays, pad=0.05, invert=False):
    """
    Return (lo, hi) spanning all finite values in arrays, with fractional padding.

    If invert=True, return (hi, lo) for inverted axes (e.g. magnitudes).
    """
    chunks = []
    for a in arrays:
        if a is None:
            continue
        v = np.asarray(a, dtype=float).ravel()
        v = v[np.isfinite(v)]
        if v.size:
            chunks.append(v)
    if not chunks:
        return (1.0, 0.0) if invert else (0.0, 1.0)
    vals = np.concatenate(chunks)
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        span = abs(lo) * 0.05 if lo != 0 else 1.0
        lo, hi = lo - span, hi + span
    else:
        span = hi - lo
        lo -= pad * span
        hi += pad * span
    return (hi, lo) if invert else (lo, hi)


# ---------------------------------------------------------------------------
# Empirical Teff calibrations (Alonso et al.)
# ---------------------------------------------------------------------------
def VI_to_Teff(VI):
    """Convert (V-I) color to Teff [K] = 5040 / Theta."""
    VI = np.asarray(VI)
    a3, a2, a1, a0 = -0.0192951, 0.0197577, 0.41923694, 0.53356974
    Theta = a0 + a1 * VI + a2 * VI**2 + a3 * VI**3
    return 5040 / Theta


def VR_to_Teff(VR):
    """Convert (V-R) color to Teff [K] = 5040 / Theta."""
    VR = np.asarray(VR)
    a3, a2, a1, a0 = -2.28030203e-04, -1.88669354e-01, 8.81425556e-01, 4.98517999e-01
    Theta = a0 + a1 * VR + a2 * VR**2 + a3 * VR**3
    return 5040 / Theta


def RI_to_Teff(RI, model="poly"):
    """Convert (R-I) color to Teff [K]."""
    RI = np.asarray(RI)
    if model == "poly":
        a3, a2, a1, a0 = -0.00372186, -0.48393689, 1.32916366, 0.50060817
        Theta = a0 + a1 * RI + a2 * RI**2 + a3 * RI**3
        return 5040 / Theta
    if model == "exp":
        a3, a2, a1, a0 = -0.2610645, 0.78625533, -0.93975579, 3.98867204
        logTeff = a0 + a1 * RI + a2 * RI**2 + a3 * RI**3
        return 10**logTeff
    raise ValueError(f"Unknown model: {model}")


# Jordi et al. (2005) SDSS <-> VRI transforms
# https://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php#Jordi2006
def gi_to_VI(gi):
    return 0.675 * gi + 0.364


def gr_to_VR(gr):
    return gr / 1.646 + 0.139 / 1.646


def ri_to_RI(ri):
    return 0.930 * ri + 0.259


def rz_to_RI(rz):
    return rz / 1.584 + 0.386 / 1.584


def iz_to_RI(iz):
    return 1 / (1.584 - 1 / 0.930) * iz + (0.386 - 0.259 / 0.930) / (1.584 - 1 / 0.930)


# ---------------------------------------------------------------------------
# Blackbody color -> Teff
# ---------------------------------------------------------------------------
h = 6.62607015e-27  # erg·s
c = 2.99792458e18  # Å/s
k = 1.380649e-16  # erg/K

lambda_g = 4770.0
lambda_r = 6370.0
lambda_i = 7785.0
lambda_z = 9155.0

# Default reddening (Smith et al. light-echo value); MW-like R_V
DEFAULT_EBV = 1.0
DEFAULT_RV = 3.1


def ccm89_alambda_av(wave_angstrom, rv=DEFAULT_RV):
    """
    A(λ)/A(V) from Cardelli, Clayton & Mathis (1989) for optical/NIR.

    wave_angstrom : wavelength in Å
    rv : total-to-selective extinction R_V = A_V / E(B-V)
    """
    wave_angstrom = np.asarray(wave_angstrom, dtype=float)
    x = 1.0 / (wave_angstrom * 1e-4)  # μm^-1
    if np.any(x < 0.3) or np.any(x > 3.3):
        raise ValueError("CCM89 implementation here covers 0.3 < x < 3.3 μm^-1 only")

    a = np.empty_like(x)
    b = np.empty_like(x)

    # Infrared: 0.3 <= x <= 1.1
    ir = x <= 1.1
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61

    # Optical/NIR: 1.1 < x <= 3.3
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


def extinction_sdss_griz(ebv=DEFAULT_EBV, rv=DEFAULT_RV):
    """
    Return A_g, A_r, A_i, A_z for given E(B-V) and R_V.

    Uses CCM89 at the SDSS effective wavelengths used elsewhere in this script.
    A_V = R_V * E(B-V); A_λ = [A_λ/A_V] * A_V.
    """
    av = rv * ebv
    waves = {"g": lambda_g, "r": lambda_r, "i": lambda_i, "z": lambda_z}
    return {band: float(ccm89_alambda_av(wave, rv=rv) * av) for band, wave in waves.items()}


def deredden_mags(g, r, i, z, ebv=DEFAULT_EBV, rv=DEFAULT_RV):
    """
    Correct observed AB magnitudes for extinction: m_0 = m_obs - A_λ.

    Applied to each band light curve before colors / Teff are computed.
    """
    a = extinction_sdss_griz(ebv=ebv, rv=rv)
    return g - a["g"], r - a["r"], i - a["i"], z - a["z"], a


def color_eq(T, color_obs, lambda_1, lambda_2):
    B_ratio = (lambda_2**5 / lambda_1**5) * (
        (np.exp(h * c / (lambda_2 * k * T)) - 1)
        / (np.exp(h * c / (lambda_1 * k * T)) - 1)
    )
    return -2.5 * np.log10(B_ratio) - color_obs


def teff_from_color(color, lambda_1, lambda_2, T0=4000.0):
    teff = []
    for ci in color:
        T_eff = fsolve(color_eq, T0, args=(ci, lambda_1, lambda_2))[0]
        teff.append(T_eff)
    return np.asarray(teff)


# ---------------------------------------------------------------------------
# Radius from magnitude + Teff
# ---------------------------------------------------------------------------
R_sun = 6.957e10
T_sun = 5772
M_bol_sun = 4.74


def BC_i(T_eff):
    return 0


def radius_from_Mi_Teff(M_i, T_eff):
    """Stellar radius in solar radii from i-band abs mag and Teff."""
    BC = BC_i(T_eff)
    log_R_Rsun = (M_bol_sun - M_i - BC) / 5 - 2 * np.log10(T_eff / T_sun)
    return 10**log_R_Rsun


def load_models():
    LCm2_i = pd.read_csv(LCPATH / "iLC_mod2.csv")
    LCm2_g = pd.read_csv(LCPATH / "gLC_mod2.csv")
    LCm2_r = pd.read_csv(LCPATH / "rLC_mod2.csv")
    LCm2_z = pd.read_csv(LCPATH / "zLC_mod2.csv")
    return LCm2_i, LCm2_g, LCm2_r, LCm2_z


def add_ab_mags(LCm2_i, LCm2_g, LCm2_r, LCm2_z):
    working_inds_i = np.where(LCm2_i["muJyas2"] > 0.0)[0]
    working_inds_g = np.where(LCm2_g["muJyas2"] > 0.0)[0]
    working_inds_r = np.where(LCm2_r["muJyas2"] > 0.0)[0]
    working_inds_z = np.where(LCm2_z["muJyas2"] > 0.0)[0]

    mag_i, mag_err_i = convert_Jy2mag(
        LCm2_i.loc[working_inds_i, "muJyas2"].values,
        LCm2_i.loc[working_inds_i, "muJyas2_err"].values,
    )
    mag_g, mag_err_g = convert_Jy2mag(
        LCm2_g.loc[working_inds_g, "muJyas2"].values,
        LCm2_g.loc[working_inds_g, "muJyas2_err"].values,
    )
    mag_r, mag_err_r = convert_Jy2mag(
        LCm2_r.loc[working_inds_r, "muJyas2"].values,
        LCm2_r.loc[working_inds_r, "muJyas2_err"].values,
    )
    mag_z, mag_err_z = convert_Jy2mag(
        LCm2_z.loc[working_inds_z, "muJyas2"].values,
        LCm2_z.loc[working_inds_z, "muJyas2_err"].values,
    )

    LCm2_i.loc[working_inds_i, "AB_mag"] = np.array(mag_i)
    LCm2_i.loc[working_inds_i, "AB_mag_err"] = np.array(mag_err_i)
    LCm2_g.loc[working_inds_g, "AB_mag"] = np.array(mag_g)
    LCm2_g.loc[working_inds_g, "AB_mag_err"] = np.array(mag_err_g)
    LCm2_r.loc[working_inds_r, "AB_mag"] = np.array(mag_r)
    LCm2_r.loc[working_inds_r, "AB_mag_err"] = np.array(mag_err_r)
    LCm2_z.loc[working_inds_z, "AB_mag"] = np.array(mag_z)
    LCm2_z.loc[working_inds_z, "AB_mag_err"] = np.array(mag_err_z)

    return working_inds_i, working_inds_g, working_inds_r, working_inds_z


def interpolate_mags(LCm2_i, LCm2_g, LCm2_r, LCm2_z, inds):
    working_inds_i, working_inds_g, working_inds_r, working_inds_z = inds
    t_ran = np.arange(54000, 62800)

    i_new = splev(
        t_ran,
        splrep(LCm2_i.loc[working_inds_i, "mjd"], LCm2_i.loc[working_inds_i, "AB_mag"], k=1),
    )
    i_new_err = splev(
        t_ran,
        splrep(
            LCm2_i.loc[working_inds_i, "mjd"],
            LCm2_i.loc[working_inds_i, "AB_mag_err"],
            k=1,
        ),
    )
    g_new = splev(
        t_ran,
        splrep(LCm2_g.loc[working_inds_g, "mjd"], LCm2_g.loc[working_inds_g, "AB_mag"], k=1),
    )
    g_new_err = splev(
        t_ran,
        splrep(
            LCm2_g.loc[working_inds_g, "mjd"],
            LCm2_g.loc[working_inds_g, "AB_mag_err"],
            k=1,
        ),
    )
    r_new = splev(
        t_ran,
        splrep(LCm2_r.loc[working_inds_r, "mjd"], LCm2_r.loc[working_inds_r, "AB_mag"], k=1),
    )
    r_new_err = splev(
        t_ran,
        splrep(
            LCm2_r.loc[working_inds_r, "mjd"],
            LCm2_r.loc[working_inds_r, "AB_mag_err"],
            k=1,
        ),
    )
    z_new = splev(
        t_ran,
        splrep(LCm2_z.loc[working_inds_z, "mjd"], LCm2_z.loc[working_inds_z, "AB_mag"], k=1),
    )
    return t_ran, i_new, i_new_err, g_new, g_new_err, r_new, r_new_err, z_new


def plot_flux_panels(LCm2_i, LCm2_g, LCm2_r, LCm2_z):
    fig, axs = plt.subplots(4, 1, dpi=100, figsize=[8, 8], sharex=True)
    axs = axs.flatten()
    mjd_lim = data_limits(
        LCm2_g["mjd"], LCm2_i["mjd"], LCm2_r["mjd"], LCm2_z["mjd"], pad=0.02
    )

    for ax, lc, color, label, with_err in [
        (axs[0], LCm2_g, "blue", "g-band", True),
        (axs[1], LCm2_i, "darkred", "i-band", True),
        (axs[2], LCm2_r, "red", "r-band", False),
        (axs[3], LCm2_z, "chocolate", "z-band", False),
    ]:
        ax.minorticks_on()
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.tick_params(which="major", bottom="on", top="on", left="on", right="on", length=12)
        ax.tick_params(which="minor", bottom="on", top="on", left="on", right="on", length=5)
        ax.set_ylabel(r"$\mu$Jy/as$^2$", fontsize=14)
        ax.ticklabel_format(axis="both", style="plain")
        if with_err:
            ax.errorbar(
                lc["mjd"],
                lc["muJyas2"],
                lc["muJyas2_err"],
                fmt="-",
                ecolor="black",
                color=color,
                label=label,
            )
            y_lim = data_limits(lc["muJyas2"], pad=0.08)
        else:
            ax.errorbar(lc["mjd"], lc["muJyas2"], fmt="-", ecolor="black", color=color, label=label)
            y_lim = data_limits(lc["muJyas2"], pad=0.08)
        ax.axhline(y=0, linestyle="--", color="black")
        ax.set_xlim(*mjd_lim)
        ax.set_ylim(*y_lim)
        ax.legend(fontsize=14)

    axs[2].set_xlabel("MJD (days)", fontsize=14)
    axs[3].set_xlabel("MJD (days)", fontsize=14)
    axs[2].set_ylabel(r"$\mu$Jy/as$^2$", fontsize=16)
    axs[3].set_ylabel(r"$\mu$Jy/as$^2$", fontsize=16)
    savefig("flux_lightcurves.png")


def plot_calibration_curves():
    t_effs_VI = np.array(
        [
            3500,
            3750,
            4000,
            4250,
            4500,
            4750,
            5000,
            5250,
            5500,
            5750,
            6000,
            6250,
            6500,
            6750,
            7000,
            7250,
            7500,
            7750,
            8000,
        ]
    )
    V_Is = np.array(
        [
            2.800,
            2.210,
            1.865,
            1.625,
            1.435,
            1.275,
            1.135,
            1.015,
            0.910,
            0.815,
            0.725,
            0.645,
            0.575,
            0.505,
            0.440,
            0.380,
            0.325,
            0.275,
            0.225,
        ]
    )
    V_I_test = np.linspace(0.1, 3)
    plt.figure()
    plt.plot(V_Is, t_effs_VI, ".", label="data")
    plt.plot(V_I_test, VI_to_Teff(V_I_test), label="model")
    plt.legend()
    plt.xlabel("V-I")
    plt.ylabel("T eff")
    savefig("teff_calibration_VI.png")

    t_effs = np.array(
        [
            3750,
            4000,
            4250,
            4500,
            4750,
            5000,
            5250,
            5500,
            5750,
            6000,
            6250,
            6500,
            6750,
            7000,
            7250,
            7500,
            7750,
            8000,
        ]
    )
    V_Rs = np.array(
        [
            1.350,
            1.145,
            0.990,
            0.865,
            0.760,
            0.680,
            0.6,
            0.535,
            0.480,
            0.425,
            0.380,
            0.340,
            0.300,
            0.265,
            0.235,
            0.205,
            0.180,
            0.155,
        ]
    )
    V_R_test = np.linspace(0.1, 2)
    plt.figure()
    plt.plot(V_Rs, t_effs, ".", label="data")
    plt.plot(V_R_test, VR_to_Teff(V_R_test), label="model")
    plt.legend()
    plt.xlabel("V-R")
    plt.ylabel("T eff")
    savefig("teff_calibration_VR.png")

    R_Is = np.array(
        [
            1.005,
            0.815,
            0.690,
            0.595,
            0.520,
            0.460,
            0.405,
            0.360,
            0.320,
            0.285,
            0.255,
            0.225,
            0.2,
            0.175,
            0.155,
            0.135,
            0.120,
            0.100,
        ]
    )
    R_I_test = np.linspace(0.1, 1.2)
    Teff_RI_test = RI_to_Teff(R_I_test, model="poly")
    plt.figure()
    plt.plot(R_Is, t_effs, ".", label="data")
    plt.plot(R_I_test, Teff_RI_test, label="model")
    plt.legend()
    plt.xlabel("R-I")
    plt.ylabel("T eff")
    savefig("teff_calibration_RI.png")

    plt.figure()
    plt.plot(V_I_test, VI_to_Teff(V_I_test), label="V-I")
    plt.plot(V_R_test, VR_to_Teff(V_R_test), label="V-R")
    plt.plot(R_I_test, Teff_RI_test, label="R-I")
    plt.legend()
    plt.xlabel("color")
    plt.ylabel("T eff")
    savefig("teff_calibration_compare.png")


def plot_hr_diagram(t_ran, g_new, i_new):
    plt.figure(figsize=(12, 8))
    idx = t_ran < 62900
    step = 1

    color_gi = g_new[idx][::step] - i_new[idx][::step]
    mag_i = g_new[idx][::step]
    abs_mag_i = mag_i - 32.65
    t = t_ran[idx][::step]
    T = VI_to_Teff(gi_to_VI(color_gi))
    R_ec_bk = radius_from_Mi_Teff(abs_mag_i, T)
    R_ec = R_ec_bk**2 / 615.132 / 2

    plt.plot(color_gi, abs_mag_i, "-", alpha=0.5, c="k", lw=0.3)

    color_gi_show = [color_gi[0], color_gi[-1]]
    m_show = [abs_mag_i[0], abs_mag_i[-1]]
    T_show = [T[0], T[-1]]
    R_show = [R_ec[0], R_ec[-1]]

    values = np.array(T)
    norm = (values - values.min()) / (values.max() - values.min())
    colors = plt.cm.Spectral(norm)

    x_first, y_first = color_gi[0], abs_mag_i[0]
    plt.annotate(
        "1835",
        fontsize=16,
        xy=(x_first, y_first),
        xycoords="data",
        xytext=(-40, 10),
        textcoords="offset points",
        arrowprops=dict(
            facecolor="black",
            arrowstyle="simple, head_length=1, head_width=1",
            mutation_scale=5,
        ),
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    annotations = [
        (54890 - 60, "1st peak(1838)", (-80, 10)),
        (56690 + 50, "2nd peak(1843)", (-10, 30)),
        (57388, "3rd peak(1845)", (-80, 20)),
        (59360, "4th peak \n(1850)", (30, 20)),
    ]
    for mjd_peak, label, xytext in annotations:
        idx_peak = np.argmin(np.abs(t_ran[::step] - mjd_peak))
        x_peak, y_peak = color_gi[idx_peak], abs_mag_i[idx_peak]
        color_gi_show.append(x_peak)
        m_show.append(y_peak)
        T_show.append(T[idx_peak])
        R_show.append(R_ec[idx_peak])
        plt.annotate(
            label,
            fontsize=16,
            xy=(x_peak, y_peak),
            xycoords="data",
            xytext=xytext,
            textcoords="offset points",
            arrowprops=dict(
                facecolor="black",
                arrowstyle="simple, head_length=1, head_width=1",
                mutation_scale=5,
            ),
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    x_last, y_last = color_gi[-1], abs_mag_i[-1]
    plt.annotate(
        "1859",
        fontsize=16,
        xy=(x_last, y_last),
        xycoords="data",
        xytext=(50, -5),
        textcoords="offset points",
        arrowprops=dict(
            facecolor="black",
            arrowstyle="simple, head_length=1, head_width=1",
            mutation_scale=5,
        ),
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    for color, color_gi_i, m_i, R_i, T_i in zip(
        colors[1:], color_gi[1:], abs_mag_i[1:], R_ec[1:], T[1:]
    ):
        color_gi_show_arr = np.array(color_gi_show)
        m_show_arr = np.array(m_show)
        dist = (color_gi_i - color_gi_show_arr) ** 2 + (m_i - m_show_arr) ** 2
        if min(dist[dist > 0]) > 0.02:
            color_gi_show.append(color_gi_i)
            m_show.append(m_i)
            T_show.append(T_i)
            R_show.append(R_i)

    sc = plt.scatter(
        np.array(color_gi_show),
        np.array(m_show),
        s=np.array(R_show) / 5,
        c=np.array(T_show),
        cmap=plt.cm.Spectral,
    )
    cbar = plt.colorbar(sc, ax=plt.gca())
    cbar.set_label("Temperature (K)")
    plt.xlabel("g-i")
    plt.ylabel("absolute magnitude g")
    plt.xlim(*data_limits(color_gi, np.array(color_gi_show), pad=0.08))
    plt.ylim(*data_limits(abs_mag_i, np.array(m_show), pad=0.08, invert=True))
    savefig("hr_diagram_color_teff.png")

    plt.figure()
    plt.plot(t, T)
    plt.xlabel("mjd")
    plt.ylabel("T (K)")
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    for mjd in (54850, 56740, 57430, 59300):
        plt.axvline(mjd, color="k", ls="--", alpha=0.5)
    savefig("teff_vs_mjd_hr.png")


def load_gp_models():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(GPMODEL_DIR / "lrn_gmodel.pickle", "rb") as f:
            lrn_gmodel = pickle.load(f)
        with open(GPMODEL_DIR / "lrn_rmodel.pickle", "rb") as f:
            lrn_rmodel = pickle.load(f)
        with open(GPMODEL_DIR / "ilrt_gmodel.pickle", "rb") as f:
            ilrt_gmodel = pickle.load(f)
        with open(GPMODEL_DIR / "ilrt_rmodel.pickle", "rb") as f:
            ilrt_rmodel = pickle.load(f)
    return lrn_gmodel, lrn_rmodel, ilrt_gmodel, ilrt_rmodel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Color → Teff / radius analysis from MCMC light-curve models."
    )
    parser.add_argument(
        "--ebv",
        type=float,
        default=DEFAULT_EBV,
        help=f"Color excess E(B-V) used to deredden griz LCs before colors/Teff (default: {DEFAULT_EBV})",
    )
    parser.add_argument(
        "--rv",
        type=float,
        default=DEFAULT_RV,
        help=f"Total-to-selective extinction R_V for CCM89 MW-like law (default: {DEFAULT_RV})",
    )
    return parser.parse_args()


def main(ebv=DEFAULT_EBV, rv=DEFAULT_RV):
    print(f"Project root: {ROOT}")
    print(f"Figures -> {FIGDIR}")
    print(f"Dereddening with E(B-V)={ebv:.3f}, R_V={rv:.3f} (CCM89)")

    LCm2_i, LCm2_g, LCm2_r, LCm2_z = load_models()
    plot_flux_panels(LCm2_i, LCm2_g, LCm2_r, LCm2_z)

    inds = add_ab_mags(LCm2_i, LCm2_g, LCm2_r, LCm2_z)
    t_ran, i_new, i_new_err, g_new, g_new_err, r_new, r_new_err, z_new = interpolate_mags(
        LCm2_i, LCm2_g, LCm2_r, LCm2_z, inds
    )

    # Observed (reddened) interpolated AB mags
    plt.figure()
    plt.errorbar(t_ran, i_new, i_new_err, ecolor="r", label="i")
    plt.errorbar(t_ran, g_new, g_new_err, ecolor="r", label="g")
    plt.errorbar(t_ran, r_new, r_new_err, ecolor="r", label="r")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title("Observed (before dereddening)")
    savefig("ab_mag_interp.png")

    # Deredden band light curves before forming colors / Teff
    g_new, r_new, i_new, z_new, a_bands = deredden_mags(
        g_new, r_new, i_new, z_new, ebv=ebv, rv=rv
    )
    print(
        "Extinction applied (mag): "
        + ", ".join(f"A_{b}={a_bands[b]:.3f}" for b in ("g", "r", "i", "z"))
    )

    # Colors from dereddened light curves
    g_i = g_new - i_new
    g_i_err = np.sqrt(g_new_err**2 + i_new_err**2)
    g_r = g_new - r_new
    g_r_err = np.sqrt(g_new_err**2 + r_new_err**2)
    plt.figure()
    plt.errorbar(t_ran, g_i, g_i_err, ecolor="r", label="g-i")
    plt.errorbar(t_ran, g_r, g_r_err, ecolor="r", label="g-r")
    plt.legend()
    plt.title(f"Dereddened colors (E(B-V)={ebv:.2f}, R_V={rv:.1f})")
    savefig("colors_gi_gr.png")

    # Dereddened AB mag light curves
    plt.subplots(dpi=100, figsize=[12, 6])
    plt.minorticks_on()
    plt.tick_params(axis="x", labelsize=12)
    plt.tick_params(axis="y", labelsize=12)
    plt.tick_params(which="major", bottom="on", top="on", left="on", right="on", length=12)
    plt.tick_params(which="minor", bottom="on", top="on", left="on", right="on", length=5)
    plt.gca().set_xlabel("mjd", fontsize=14)
    plt.gca().set_ylabel("AB mag (dereddened)", fontsize=14)
    plt.ticklabel_format(axis="both", style="plain")
    plt.errorbar(t_ran, g_new, ecolor="black", color="blue")
    plt.errorbar(t_ran, i_new, ecolor="black", color="darkred")
    plt.errorbar(t_ran, r_new, ecolor="black", color="red")
    plt.errorbar(t_ran, z_new, ecolor="black", color="chocolate")
    plt.ylim(*data_limits(g_new, r_new, i_new, z_new, pad=0.08, invert=True))
    savefig("ab_mag_lightcurves.png")

    plot_calibration_curves()

    # Empirical Teff from colors
    color_gr = g_new - r_new
    color_ri = r_new - i_new
    color_iz = i_new - z_new
    color_gi = g_new - i_new
    color_rz = r_new - z_new

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
    mjd_lim = data_limits(t_ran, pad=0.02)

    ax = axs[0]
    ax.plot(t_ran, g_new)
    ax.plot(t_ran, r_new)
    ax.plot(t_ran, i_new)
    ax.plot(t_ran, z_new)
    ax.set_ylim(*data_limits(g_new, r_new, i_new, z_new, pad=0.08, invert=True))
    ax.set_ylabel("mag (dereddened)")

    ax = axs[1]
    ax.plot(t_ran, color_gr, label="g-r")
    ax.plot(t_ran, color_ri, label="r-i")
    ax.plot(t_ran, color_gi, label="g-i")
    ax.set_ylim(*data_limits(color_gr, color_ri, color_gi, pad=0.08))
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_ylabel("color (dereddened)")

    ax = axs[2]
    T_gr = VR_to_Teff(gr_to_VR(color_gr))
    T_ri = RI_to_Teff(ri_to_RI(color_ri))
    T_gi = VI_to_Teff(gi_to_VI(color_gi))
    ax.plot(t_ran, T_gr, label="g-r")
    ax.plot(t_ran, T_ri, label="r-i")
    ax.plot(t_ran, T_gi, label="g-i")
    ax.set_ylim(*data_limits(T_gr, T_ri, T_gi, pad=0.08))
    ax.set_xlabel("mjd")
    ax.set_ylabel("T (K)")
    ax.set_title("T from emperical relation", fontsize=22)

    for ax in axs:
        for mjd in (54850, 56740, 57430, 59300):
            ax.axvline(mjd, color="k", ls="--", alpha=0.5)
        ax.set_xlim(*mjd_lim)
    fig.tight_layout()
    savefig("empirical_teff.png", tight=False)

    # Blackbody Teff
    teff_bb_gr = teff_from_color(color_gr, lambda_g, lambda_r)
    teff_bb_ri = teff_from_color(color_ri, lambda_r, lambda_i)
    teff_bb_iz = teff_from_color(color_iz, lambda_i, lambda_z)
    teff_bb_gi = teff_from_color(color_gi, lambda_g, lambda_i)
    teff_bb_rz = teff_from_color(color_rz, lambda_r, lambda_z)
    plt.figure(figsize=(8, 4))
    plt.plot(t_ran, teff_bb_gr, label="gr")
    plt.plot(t_ran, teff_bb_ri, label="ri")
    plt.plot(t_ran, teff_bb_iz, label="iz")
    plt.plot(t_ran, teff_bb_gi, label="gi")
    plt.plot(t_ran, teff_bb_rz, label="rz")
    plt.xlim(*mjd_lim)
    plt.ylim(*data_limits(teff_bb_gr, teff_bb_ri, teff_bb_iz, teff_bb_gi, teff_bb_rz, pad=0.08))
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("mjd")
    plt.ylabel("T(K)")
    plt.title("T from blackbody", fontsize=22)
    savefig("blackbody_teff.png")

    # Radius–mag relation
    mags = np.arange(-15, -5, 0.5)
    plt.figure()
    for ti in [4000, 5000, 6055]:
        plt.plot(radius_from_Mi_Teff(mags, ti), mags, label=f"T={ti} K")
    plt.axhline(y=-12, c="k")
    plt.xlabel("radius")
    plt.ylabel("abs mag")
    plt.gca().invert_yaxis()
    plt.legend()
    savefig("radius_mag_relation.png")

    V_new = g_new - 0.03 - 0.42 * (g_new - r_new)
    print(f"V_new.min() + 12 = {V_new.min() + 12:.4f}")

    T = VI_to_Teff(gi_to_VI(color_gi))
    R_ec = radius_from_Mi_Teff(g_new - 32.65, T)
    plt.figure()
    plt.plot(t_ran, R_ec)
    plt.xlabel("mjd")
    plt.ylabel("")
    savefig("radius_vs_mjd.png")

    plot_hr_diagram(t_ran, g_new, i_new)

    # Transient comparison (LRN / ILRT GP models + UGC 2773)
    lrn_gmodel, lrn_rmodel, ilrt_gmodel, ilrt_rmodel = load_gp_models()
    x = np.atleast_2d(np.linspace(-20, 200, 100)).T
    y_pred_g_lrn, _ = lrn_gmodel.predict(x, return_std=True)
    y_pred_r_lrn, _ = lrn_rmodel.predict(x, return_std=True)
    y_pred_g_ilrt, _ = ilrt_gmodel.predict(x, return_std=True)
    y_pred_r_ilrt, _ = ilrt_rmodel.predict(x, return_std=True)

    plt.figure()
    plt.plot(x, y_pred_g_lrn, c=plt.cm.tab10(0), label="g Prediction lrn")
    plt.plot(x, y_pred_r_lrn, c=plt.cm.tab10(1), label="r Prediction lrn")
    plt.plot(x, y_pred_g_ilrt, c=plt.cm.tab10(2), label="g Prediction ilrt")
    plt.plot(x, y_pred_r_ilrt, c=plt.cm.tab10(3), label="r Prediction ilrt")
    plt.gca().invert_yaxis()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("time")
    plt.ylabel("mag")
    plt.title("Luminous Red Novae model")
    savefig("lrn_ilrt_models.png")

    lc_ugc = pd.read_csv(UGC_PATH, sep=r"\s+")
    t_ugc = lc_ugc["MJD"].values - 55100
    mag_ugc = lc_ugc["mag"].values

    plt.figure()
    plt.plot(t_ugc, mag_ugc, ".-", label="unfiltered UGC 2773-OT")
    plt.gca().invert_yaxis()
    plt.legend(loc="upper left", bbox_to_anchor=(0.5, 1))
    savefig("ugc2773.png")

    phase_eta = t_ran - 57400
    mag_eta_rel = g_new - 20.5
    mag_ugc_rel = mag_ugc - 16.5
    plt.figure(figsize=(10, 6))
    plt.plot(phase_eta, mag_eta_rel, label="g eta car")
    plt.plot(x, y_pred_g_lrn, label="g lrn")
    plt.plot(x, y_pred_g_ilrt, label="g ilrt")
    plt.plot(t_ugc, mag_ugc_rel, ".-", label="unfiltered UGC 2773-OT")
    plt.xlim(*data_limits(phase_eta, x, t_ugc, pad=0.05))
    plt.ylim(
        *data_limits(mag_eta_rel, y_pred_g_lrn, y_pred_g_ilrt, mag_ugc_rel, pad=0.08, invert=True)
    )
    plt.xlabel("time (days)")
    plt.ylabel("relative mag")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    savefig("transient_comparison.png")

    color_eta = g_new - r_new
    color_ilrt = y_pred_g_ilrt - y_pred_r_ilrt
    color_lrn = y_pred_g_lrn - y_pred_r_lrn
    color_ylim = data_limits(color_eta, color_ilrt, color_lrn, pad=0.08)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    axs[0].plot(phase_eta, color_eta, label="Eta Car", c=plt.cm.tab10(0))
    axs[0].set_xlim(*data_limits(phase_eta, pad=0.05))
    axs[0].set_ylim(*color_ylim)
    axs[0].set_ylabel("g-r")
    axs[0].set_xlabel("phase (days)")
    axs[0].legend()
    axs[1].plot(x, color_ilrt, label="ILRT", c=plt.cm.tab10(1))
    axs[1].set_xlim(*data_limits(x, pad=0.05))
    axs[1].set_xlabel("phase (days)")
    axs[1].legend()
    axs[2].plot(x, color_lrn, label="LRN", c=plt.cm.tab10(2))
    axs[2].set_xlim(*data_limits(x, pad=0.05))
    axs[2].set_xlabel("phase (days)")
    axs[2].legend()
    savefig("color_comparison.png")

    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(ebv=args.ebv, rv=args.rv)
