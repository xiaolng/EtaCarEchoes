#!/usr/bin/env python3
"""
Joint line + photometry temperature.

Spectroscopy enters only through the Mg b and Ca II line shapes. The IMACS
spectra are not spectrophotometric across r, i, and z, so their broadband
colors are not a second continuum measurement.

Photometry is the DECam r-i and i-z colors at the 2011 spectral epoch.
A reddened Pickles supergiant cannot match both colors at once: r-i is too
red for i-z. The fit therefore includes one nuisance, delta_i, an additive
offset to the i-band magnitude (positive: model i too bright). It is the
smallest extra term that lets one Teff satisfy both datasets.

R_V is free on 3–6. Velocities and line-scatter terms are fixed at the
previous joint-line posterior.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import photo_spec_teff as ps
import rest2012_joint_mcmc as jm
import rest2012_reddening_mcmc as rm

FIGDIR = rm.r12.FIGDIR
RV_FIXED_LINES = 4.8
V_EPOCHS = np.array([-380.9, -330.5, -380.1])
LN_SIG_MG = -0.34
LN_SIG_CA = -0.64
# i-band offset prior. 0.10 mag is the Hα / Ca II budget already measured.
# The data need more than that; the prior is wide enough to see where they pull.
DELTA_SIGMA = 0.25
COLOR_SIGMA = {"r-i": 0.05, "i-z": 0.04}


def precompute_colors(teff_grid, ebv_grid, rv_grid, teffs, wave, flux, filters):
    """Model r-i, i-z on the grid, shape (nT, nE, nR)."""
    nT, nE, nR = len(teff_grid), len(ebv_grid), len(rv_grid)
    ri = np.full((nT, nE, nR), np.nan)
    iz = np.full((nT, nE, nR), np.nan)
    for i, teff in enumerate(teff_grid):
        tmpl = jm.interp_template(float(teff), teffs, flux)
        for k, rv in enumerate(rv_grid):
            # extinction_factor is linear in ebv only through the exponent, so
            # recompute per ebv.
            for j, ebv in enumerate(ebv_grid):
                reddened = tmpl * rm.extinction_factor(wave, float(ebv), rv=float(rv))
                c = ps.colors_from_spectrum(wave, reddened, filters)
                ri[i, j, k] = c["r-i"]
                iz[i, j, k] = c["i-z"]
    return ri, iz


def line_loglike(teff_grid):
    data = jm.JointDataset()
    out = np.empty(len(teff_grid))
    for i, teff in enumerate(teff_grid):
        out[i] = jm.ln_likelihood_mgb_v(float(teff), LN_SIG_MG, V_EPOCHS, data)
        out[i] += jm.ln_likelihood_caii_v(float(teff), LN_SIG_CA, V_EPOCHS, data)
    return out


def logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    s = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(s, axis=axis)


def marginal_teff(lnL):
    """lnL shape (nT, ...). Return normalized p(Teff) and quantiles."""
    flat = logsumexp(lnL.reshape(lnL.shape[0], -1), axis=1)
    flat = flat - np.max(flat)
    p = np.exp(flat)
    return p / np.trapezoid(p, dx=1.0)  # caller renormalizes on the Teff axis


def quantiles(teff, prob):
    prob = np.asarray(prob, dtype=float)
    prob = np.clip(prob, 0, None)
    total = prob.sum()
    if total <= 0:
        return (np.nan, np.nan, np.nan, np.nan)
    cdf = np.cumsum(prob) / total

    def q(frac):
        i = int(np.searchsorted(cdf, frac, side="left"))
        i = min(max(i, 0), len(teff) - 1)
        return float(teff[i])

    return q(0.16), q(0.50), q(0.84), float(teff[int(np.argmax(prob))])


def main():
    filters = ps.load_filters()
    teffs, _, wave, flux = ps.template_library(filters)
    obs = ps.photometric_colors()
    print(
        f"DECam at MJD {ps.PHOTO_MJD:.0f}: "
        f"r-i={obs['r-i']:.3f}  i-z={obs['i-z']:.3f}"
    )

    teff_grid = np.linspace(4200.0, 7000.0, 57)
    ebv_grid = np.linspace(0.0, 1.3, 27)
    rv_grid = np.linspace(3.0, 6.0, 7)
    delta_grid = np.linspace(-0.6, 0.6, 49)

    print("Line likelihood …")
    ll_line = line_loglike(teff_grid)
    print("Model colors …")
    ri, iz = precompute_colors(teff_grid, ebv_grid, rv_grid, teffs, wave, flux, filters)

    # Photometry, no band offset. delta=0.
    # pred matches obs.
    sig_ri = COLOR_SIGMA["r-i"]
    sig_iz = COLOR_SIGMA["i-z"]
    chi_photo = ((ri - obs["r-i"]) / sig_ri) ** 2 + ((iz - obs["i-z"]) / sig_iz) ** 2
    lnL_photo = -0.5 * chi_photo  # (nT, nE, nR)

    # With delta_i: i_model_mag += delta.
    # r-i decreases by delta, i-z increases by delta.
    # Broadcast delta over the color grid.
    d = delta_grid[None, None, None, :]
    chi_d = ((ri[..., None] - d - obs["r-i"]) / sig_ri) ** 2
    chi_d += ((iz[..., None] + d - obs["i-z"]) / sig_iz) ** 2
    ln_prior_d = -0.5 * (delta_grid / DELTA_SIGMA) ** 2
    lnL_delta = -0.5 * chi_d + ln_prior_d  # (nT, nE, nR, nD)

    lnL_joint = ll_line[:, None, None, None] + lnL_delta

    p_line = np.exp(ll_line - ll_line.max())
    p_line /= np.trapezoid(p_line, teff_grid)
    log_p_photo = logsumexp(lnL_photo.reshape(len(teff_grid), -1), axis=1)
    log_p_photo -= log_p_photo.max()
    p_photo = np.exp(log_p_photo)
    p_photo /= np.trapezoid(p_photo, teff_grid)

    log_p_joint = logsumexp(lnL_joint.reshape(len(teff_grid), -1), axis=1)
    log_p_joint -= log_p_joint.max()
    p_joint = np.exp(log_p_joint)
    p_joint /= np.trapezoid(p_joint, teff_grid)

    def report(name, prob):
        a, b, c, peak = quantiles(teff_grid, prob)
        print(f"{name:22s}  peak {peak:.0f}   {a:.0f}–{c:.0f}   median {b:.0f}")

    print(f"\nTeff posteriors (R_V in [{rv_grid[0]:.0f}, {rv_grid[-1]:.0f}], E(B-V) marginalized)")
    report("lines only", p_line)
    report("photometry only", p_photo)
    report("joint, with δ_i", p_joint)

    # At the joint median Teff, posterior of EBV and delta.
    med = quantiles(teff_grid, p_joint)[1]
    it = int(np.argmin(np.abs(teff_grid - med)))
    slice_ln = lnL_joint[it]
    # marginal delta
    log_d = logsumexp(slice_ln.reshape(len(ebv_grid) * len(rv_grid), len(delta_grid)), axis=0)
    log_d -= log_d.max()
    p_d = np.exp(log_d)
    p_d /= np.trapezoid(p_d, delta_grid)
    d16, d50, d84, dpeak = quantiles(delta_grid, p_d)
    log_e = logsumexp(slice_ln.reshape(len(ebv_grid), -1), axis=1)
    log_e -= log_e.max()
    p_e = np.exp(log_e)
    p_e /= np.trapezoid(p_e, ebv_grid)
    e16, e50, e84, epeak = quantiles(ebv_grid, p_e)
    print(f"\nAt Teff = {teff_grid[it]:.0f} K")
    print(f"  E(B-V) = {e50:.2f}  [{e16:.2f}, {e84:.2f}]   peak {epeak:.2f}")
    print(f"  δ_i    = {d50:+.2f}  [{d16:+.2f}, {d84:+.2f}] mag   peak {dpeak:+.2f}")
    print(f"  (positive δ_i: i-band data fainter than the reddened supergiant)")

    # chi2 of the no-offset model at this Teff, best EBV, RV
    chi_best = np.nanmin(chi_photo[it])
    print(f"  photometry χ² without δ_i, best dust law: {chi_best:.1f} for 2 colors")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    ax = axes[0]
    ax.plot(teff_grid, p_line, color="C0", lw=1.8, label="lines only")
    ax.plot(teff_grid, p_photo, color="k", lw=1.8, label="photometry only")
    ax.plot(teff_grid, p_joint, color="C3", lw=1.8, label=r"joint, $\delta_i$ free")
    ax.set_xlabel(r"$T_{\rm eff}$ (K)")
    ax.set_ylabel(r"$p(T_{\rm eff})$")
    ax.set_xlim(teff_grid[0], teff_grid[-1])
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Temperature")

    ax = axes[1]
    ax.plot(delta_grid, p_d, color="C3", lw=1.8)
    ax.axvline(0.0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel(r"$\delta_i$ (mag)")
    ax.set_ylabel(r"$p(\delta_i)$")
    ax.set_title(rf"i-band offset at $T_{{\rm eff}}={teff_grid[it]:.0f}$ K")
    fig.tight_layout()
    out = FIGDIR / "joint_line_photo_teff.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
