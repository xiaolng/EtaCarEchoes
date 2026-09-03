# Rest et al. 2012 — EC1 light-echo spectra

**Paper:** Rest et al. 2012, Nature 482, 375 ([arXiv:1112.2210](https://arxiv.org/abs/1112.2210))

Pure-absorption Great Eruption light-echo spectra used for spectral typing (**G2–G5**, Teff ∼ **5000 K**).

## Target

| Field | Value |
|-------|-------|
| Echo | EC1 |
| RA, Dec (J2000) | 10:44:12.127, −60:16:01.69 |
| Epoch (source) | ~1843 peak (most likely) |
| Observed | 2011 March–April |

| ID | UT date | Telescope / instrument |
|----|---------|------------------------|
| EC1A | 2011-04-06 | Magellan/IMACS 300 l/mm |
| EC1B | 2011-03-07 | Magellan/IMACS 200 l/mm |
| EC1C | 2011-04-08 | du Pont/WFCCD |

## Data availability

**No public machine-readable 1D release was found** (checked WISeREP, Open Supernova Catalog / AstroCats, Zenodo, Figshare, Nature SI, author pages). Nature provides only a PDF supplement and PowerPoint figure slides.

### What is in this folder

| File | Provenance |
|------|------------|
| `EC1A_digitized.txt` | Vector-digitized from arXiv Fig. 3 (`eta03_spec.pdf`) |
| `EC1B_digitized.txt` | same |
| `EC1C_digitized.txt` | same |
| `EC1B_S2flat.txt` | Vector-digitized from SI Fig. S2 (`lines_3.pdf`; already flattened) |
| `EC1C_S2flat.txt` | same |
| `analysis/` | Cross-correlation tables + summary |

Flux scales are arbitrary (figure offsets). **Use for pipeline development only.** Digitization does **not** recover Rest’s published G2–G5 Teff quantitatively.

### Preferred inputs (not yet available)

Place author-reduced ASCII here as:

```text
EC1B.txt   # columns: wavelength_AA  flux
EC1C.txt
```

Then re-run:

```bash
python spectral_type_ec1.py
```

Request spectra from **A. Rest** / **J. L. Prieto** (corresponding author email in the Nature paper).

## Templates

UVES POP merged ASCII for Rest Table S2 stars:

`../templates/uves_pop/hd*.dat.gz`

Source: [ESO UVES POP](https://www.eso.org/sci/observing/tools/uvespop.html) (Bagnulo et al. 2003). Large `.dat.gz` files are gitignored; re-download from ESO if needed.
