#!/usr/bin/env python3
"""
Spectral typing of Rest+2012 EC1 light-echo spectra vs UVES POP templates.

Method (Rest et al. 2012, Nature 482, 375, SI):
  - continuum-flatten (Gaussian FWHM ~200 A)
  - degrade templates to ~7 A resolution
  - Pearson / cross-correlation in the Mg b window (default 5100-5400 A)
  - smooth r(Teff) with Gaussian sigma=300 K
  - map Teff -> intrinsic (V-I) via Alonso et al. calibration

Data:
  Preferred: author-reduced ASCII in
    data/spectra/rest2012_nature482_375/EC1B.txt  (wavelength_AA flux)
    data/spectra/rest2012_nature482_375/EC1C.txt
  Fallback: digitized SI Fig.S2 traces
    data/spectra/rest2012_nature482_375/EC1B_S2flat.txt
    data/spectra/rest2012_nature482_375/EC1C_S2flat.txt

Templates:
  data/spectra/templates/uves_pop/hd*.dat.gz  (ESO UVES POP; Rest Table S2)

No public machine-readable EC1 release was found (WISeREP/OSC/Zenodo/Nature).
Digitized figure spectra do NOT recover Rest's G2-G5 result quantitatively;
request native 1D products from A. Rest / J. L. Prieto for a definitive run.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent
SPEC_DIR = ROOT / "data" / "spectra" / "rest2012_nature482_375"
TMPL_DIR = ROOT / "data" / "spectra" / "templates" / "uves_pop"
FIG_DIR = ROOT / "figures" / "spectral_type"
OUT_DIR = SPEC_DIR / "analysis"

WAVE_LO, WAVE_HI = 5100.0, 5400.0
OBS_FWHM_AA = 7.0
CONT_FWHM_AA = 200.0
SMOOTH_TEFF_K = 300.0

TEMPLATES = [
    (47306, "A0 II", 9500),
    (102878, "A2 Iab", 9100),
    (34295, "A4 II", 8800),
    (74272, "A5 II", 8500),
    (97534, "A6 Ia", 8400),
    (81471, "A7 Iab", 8300),
    (80404, "A8 Ib", 8200),
    (104111, "A8 II", 8200),
    (90772, "A9 Ia", 8000),
    (75276, "F2 Iab", 7480),
    (74180, "F3 Ia", 7320),
    (115778, "F4 II", 7160),
    (67523, "F6 II", 6600),
    (108968, "F7 Ib/II", 6400),
    (210848, "F7 II", 6400),
    (54605, "F8 Iab", 6200),
    (101947, "F9 Iab", 5900),
    (146143, "F9 Ia", 5900),
    (174383, "G0 Ib", 5500),
    (97082, "G1 Iab/Ib", 5300),
    (136537, "G2 II", 5100),
    (109379, "G5 II", 4830),
    (125809, "G5/G6 Ib", 4790),
    (79698, "G6 II", 4750),
    (99648, "G8 Iab", 4590),
    (117440, "G9 Ib", 4500),
    (77020, "G9 II", 4500),
    (11643, "K1 II", 4400),
    (206778, "K2 Ib", 4300),
    (225212, "K3 Iab", 4000),
    (12642, "K5 Iab", 3750),
    (49331, "M1 Iab", 3450),
    (95950, "M2 Ib", 3350),
    (131217, "M2/M3 II", 3300),
]


def resolve_echo_path(name: str) -> Path:
    for cand in (SPEC_DIR / f"{name}.txt", SPEC_DIR / f"{name}_S2flat.txt", SPEC_DIR / f"{name}_digitized.txt"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"No spectrum found for {name} under {SPEC_DIR}")


def load_ascii(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def load_uves(hd: int, wlo: float = 5000.0, whi: float = 5500.0) -> tuple[np.ndarray, np.ndarray]:
    path = TMPL_DIR / f"hd{hd}.dat.gz"
    waves, fluxes = [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            w = float(parts[0])
            if w < wlo:
                continue
            if w > whi:
                break
            waves.append(w)
            fluxes.append(float(parts[1]))
    if len(waves) < 50:
        raise RuntimeError(f"UVES template HD{hd} has too few points in window")
    return np.asarray(waves), np.asarray(fluxes)


def flatten(wave: np.ndarray, flux: np.ndarray, fwhm_aa: float = CONT_FWHM_AA) -> np.ndarray:
    dw = np.median(np.diff(wave))
    sigma = max((fwhm_aa / 2.355) / dw, 1.0)
    cont = gaussian_filter1d(flux, sigma=sigma, mode="nearest")
    return flux / np.clip(cont, 1e-30, None)


def prep_template(wave: np.ndarray, flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dw = np.median(np.diff(wave))
    flux = gaussian_filter1d(flux, sigma=(OBS_FWHM_AA / 2.355) / dw, mode="nearest")
    flat = flatten(wave, flux)
    grid = np.arange(WAVE_LO, WAVE_HI + 0.5, 0.5)
    y = np.interp(grid, wave, flat)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return grid, y


def prep_echo(wave: np.ndarray, flux: np.ndarray, already_flat: bool, v_blue_kms: float = 210.0):
    # Rest: absorption blueshifted by ~-210 km/s
    wave = wave * (1.0 + v_blue_kms / 2.99792458e5)
    if already_flat:
        y = flux - gaussian_filter1d(flux, sigma=max(3.0, 40.0 / np.median(np.diff(wave))))
    else:
        y = flatten(wave, flux)
    grid = np.arange(WAVE_LO, WAVE_HI + 0.5, 0.5)
    y = np.interp(grid, wave, y)
    y = (y - y.mean()) / (y.std() + 1e-12)
    return grid, y


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float(np.mean(a * b))


def smooth_peak(teffs: np.ndarray, r: np.ndarray, sigma_k: float = SMOOTH_TEFF_K):
    order = np.argsort(teffs)
    t, rr = teffs[order], r[order]
    tu = np.unique(t)
    ru = np.array([rr[t == u].mean() for u in tu])
    tg = np.linspace(tu.min(), tu.max(), 800)
    rg = np.interp(tg, tu, ru)
    rgs = gaussian_filter1d(rg, sigma=sigma_k / (tg[1] - tg[0]))
    return tg[np.argmax(rgs)], tg, rgs


def bootstrap_teff(teffs: np.ndarray, r: np.ndarray, n: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(teffs))
    peaks = []
    for _ in range(n):
        draw = rng.choice(idx, size=len(idx), replace=True)
        peak, _, _ = smooth_peak(teffs[draw], r[draw])
        peaks.append(peak)
    return np.percentile(peaks, [2.5, 50, 97.5])


def alonso_VI_from_teff(teff: float) -> float:
    xs = np.linspace(0.2, 2.5, 5000)
    theta = 0.4967 + 0.5408 * xs - 0.0279 * xs**2
    t = 5040.0 / theta
    return float(np.interp(teff, t[::-1], xs[::-1]))


def nearest_sptype(teff: float) -> str:
    return min(TEMPLATES, key=lambda x: abs(x[2] - teff))[1]


def analyze(name: str):
    path = resolve_echo_path(name)
    already_flat = "S2flat" in path.name
    wave, flux = load_ascii(path)
    grid, obs = prep_echo(wave, flux, already_flat=already_flat)

    rows = []
    for hd, sptype, teff in TEMPLATES:
        tw, tf = load_uves(hd)
        _, tmpl = prep_template(tw, tf)
        r = pearson_r(obs, tmpl)
        rows.append((hd, sptype, teff, r))
        print(f"  {name}  {path.name:20s}  HD{hd:<6d} {sptype:<10s} Teff={teff:5d}  r={r:+.3f}")

    teffs = np.array([x[2] for x in rows], dtype=float)
    rvals = np.array([x[3] for x in rows], dtype=float)
    peak, tg, rg = smooth_peak(teffs, rvals)
    p025, p50, p975 = bootstrap_teff(teffs, rvals)
    return {
        "name": name,
        "path": path,
        "rows": rows,
        "teffs": teffs,
        "rvals": rvals,
        "peak": peak,
        "tg": tg,
        "rg": rg,
        "p025": p025,
        "p50": p50,
        "p975": p975,
        "sptype": nearest_sptype(peak),
        "VI": alonso_VI_from_teff(peak),
        "grid": grid,
        "obs": obs,
        "already_flat": already_flat,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--echoes", nargs="+", default=["EC1B", "EC1C"])
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for name in args.echoes:
        print(f"\n=== {name} ===")
        res = analyze(name)
        results.append(res)
        print(
            f"{name}: Teff≈{res['peak']:.0f} K (~{res['sptype']}), "
            f"95% CI {res['p025']:.0f}-{res['p975']:.0f} K, (V-I)_0≈{res['VI']:.3f}"
        )
        with (OUT_DIR / f"{name}_xcorr_uves.csv").open("w") as f:
            f.write("hd,spectral_type,teff,r\n")
            for hd, sptype, teff, r in res["rows"]:
                f.write(f"{hd},{sptype},{teff},{r:.6f}\n")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"EC1B": "crimson", "EC1C": "royalblue"}
    for res in results:
        c = colors.get(res["name"], "k")
        ax.scatter(res["teffs"], res["rvals"], s=32, c=c, label=res["name"], zorder=3)
        ax.plot(res["tg"], res["rg"], c=c, lw=1.8)
        ax.axvline(res["peak"], c=c, ls="--", lw=1)
    ax.axvspan(4450, 5550, color="0.88", zorder=0, label="Rest+2012 95% band")
    ax.axvline(5000, color="k", ls=":", lw=1, label="Rest ~5000 K")
    ax.set_xlabel(r"$T_{\rm eff}$ (K)")
    ax.set_ylabel(rf"Pearson $r$ ({WAVE_LO:.0f}–{WAVE_HI:.0f} Å)")
    ax.set_title("EC1 vs UVES POP supergiants")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG_DIR / "ec1_xcorr_teff.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(results), 1, figsize=(8, 3 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        ax.plot(res["grid"], res["obs"], "k", lw=1.1, label=res["name"])
        for hd, lab, c, off in [
            (136537, "G2 II", "tab:orange", 0.0),
            (109379, "G5 II", "tab:green", -1.2),
            (54605, "F8 Iab", "tab:purple", -2.4),
            (80404, "A8 Ib", "0.5", -3.6),
        ]:
            tw, tf = load_uves(hd)
            g, t = prep_template(tw, tf)
            ax.plot(g, 0.8 * t + off, color=c, lw=0.9, label=lab)
        ax.legend(fontsize=8, frameon=False, ncol=2)
        ax.set_ylabel("flattened")
        ax.set_title(
            f"{res['name']}: Teff≈{res['peak']:.0f} K (~{res['sptype']}), (V−I)_0≈{res['VI']:.2f}  [{res['path'].name}]"
        )
    axes[-1].set_xlabel("Rest wavelength (Å)")
    axes[-1].set_xlim(WAVE_LO, WAVE_HI)
    fig.savefig(FIG_DIR / "ec1_mgb_compare.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    with (OUT_DIR / "summary.txt").open("w") as f:
        f.write("Rest+2012 EC1 spectral typing\n")
        f.write(f"Window: {WAVE_LO:.0f}-{WAVE_HI:.0f} A; templates: UVES POP (Rest Table S2)\n\n")
        for res in results:
            f.write(
                f"{res['name']} ({res['path'].name}): Teff≈{res['peak']:.0f} K (~{res['sptype']}), "
                f"95% CI {res['p025']:.0f}-{res['p975']:.0f} K, (V-I)_0≈{res['VI']:.3f}\n"
            )
        f.write("\nRest+2012 published: G2-G5, Teff~5000 K (EC1B 4850-5550; EC1C 4450-5400).\n")
        f.write(
            "If inputs are *_S2flat.txt / *_digitized.txt, results are limited by figure digitization;\n"
            "drop native EC1B.txt / EC1C.txt into this folder and re-run for a definitive analysis.\n"
        )
    print(f"\nWrote {OUT_DIR / 'summary.txt'}")
    print(f"Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
