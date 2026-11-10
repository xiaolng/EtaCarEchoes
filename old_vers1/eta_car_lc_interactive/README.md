# Eta Car interactive notebook

Open `eta_car_interactive.ipynb` in VS Code with the Python and Jupyter extensions. Choose a Python environment with the dependencies in `requirements.txt` installed. Run All: the long Dynesty fit is OFF by default (`RUN_FIT = False`). Widgets become live when their cells run; release sliders to update. Saved static plots provide previews.

Keep this folder together when sharing: it bundles echo/historical data, the named baseline, and the original posterior in a portable NPZ file. Full sampler diagnostics are omitted from this share package; that section skips gracefully. Dependencies: numpy, pandas, scipy, matplotlib, ipywidgets, ipykernel, nbformat, nbclient, corner, dynesty.

The default vector is the maximum stored log-likelihood sample from Notebook 6's `dynesty_results_three_peak_convolution.pkl`; provenance and source hash are in `baseline.json`. It is not a new fit. Convolution uses a corrected uniform padded grid, independent of plotting view; a new fit may shift the parameters. Background is restored in echo and historical model overlays. Historical alignment is one illustrative unweighted global scale and time shift, never an independent alignment per peak.

The historical background is restricted to source years 1835–1849, shifted with the complete model. Input flux errors are used with an independent Gaussian likelihood, conditional on a fixed background; correlation and background uncertainty are not modeled. Third-peak rise time is near its original prior ceiling.

Setup: install the packages in `requirements.txt` (for example, `python -m pip install -r requirements.txt`), open this folder in VS Code, open the notebook, select that Python environment, and Run All.

Fit status: the bundled baseline and posterior are the original reference results. The ongoing refit is not included. Posterior plots evaluated with the corrected convolution are illustrative and are not a posterior from a new fit of that corrected model.
