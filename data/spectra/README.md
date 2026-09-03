# Eta Carinae light-echo spectra

## Rest+2012 EC1 spectra — for spectral type / color / Teff calibration

`rest2012_ec1/` contains the Mar–Apr 2011 Magellan/IMACS and du Pont/WFCCD
1D spectra of the EC1 light echo (Rest et al. 2012). Run:

```bash
python rest2012_spectral_type.py
```

## Smith+2018 folders

| Folder | Paper |
|--------|-------|
| `smith2018_mnras480_1457/` | Smith et al. 2018, MNRAS, 480, 1457 |
| `smith2018_mnras480_1466/` | Smith et al. 2018, MNRAS, 480, 1466 |

Place additional author-provided 1D spectra here when available.
`smith2018_mnras480_1466/observing_log.csv` lists the spectroscopic epochs.

## Templates

`templates/pickles_I/` — Pickles (1998) luminosity-class I spectra used as a
public stand-in for the UVES POP library in the Rest+-style cross-correlation.
