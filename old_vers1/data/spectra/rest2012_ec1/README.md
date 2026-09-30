# Rest+2012 EC1 light-echo 1D spectra

Correct Magellan/IMACS and du Pont/WFCCD spectra of the Eta Carinae EC1 light
echo used for Rest et al. (2012) spectral-type / \(T_{\rm eff}\) calibration
(Mar–Apr 2011 absorption-line phase).

| Epoch label | File(s) | Instrument | DATE-OBS |
|-------------|---------|------------|----------|
| Mar2011_IMACS200 | `*.imacs200.Mar2011.fits` | Magellan/IMACS (200 l/mm) | 2011-03 |
| Apr2011_IMACS300 | `*.imacs300.Apr2011.{blue,red}.fits` | Magellan/IMACS (300 l/mm) | 2011-04-06 |
| Apr2011_WFCCD | `*.wfccd.Apr2011.tell.fits` | du Pont/WFCCD | 2011-04-07 |

Object ID in filenames: `eta3_grp10_id134_x1138_y706` (Rest+ EC1 pointing).

## Analysis

```bash
conda activate etacar-color-teff
python rest2012_spectral_type.py
```

Paper comparison figures (from arXiv:1112.2210) are in `rest2012_figures/`.

## References

- Rest et al. 2012, Nature, 482, 375 ([arXiv:1112.2210](https://arxiv.org/abs/1112.2210))
