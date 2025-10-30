The following data are included in this repository - 

1. lrn_ilrt_gpmodels.zip
Contains gaussian process models for templates of r-band and g-band lightcurves of luminous red novae (LRNe) and intermediate luminosity red transients (ILRTs) as pickle files.

2. plot_gpmodel.py
An example python script to load the Gaussian Process models and plots the median and variance of the template lightcurve.

3. lrn_ilrt_lbv_lightcurves.zip
Multiband lightcurves of LRNe, ILRTs and possible LBV outbursts, as described in Karambelkar et al. (https://arxiv.org/abs/2211.05141). 
The lightcurves are presented as a table with JD, Filter, magnitudes (mag) and magnitude uncertainties (magunc). All magnitudes are in the AB system. 
Entries where magunc == -99 represent non-detections, and the 5-sigma limiting magnitudes are quoted in the "mag" column.

4. lrn_ilrt_lbv_spectra.zip
Spectra of LRNe, ILRTs and possible LBV outbursts, as described in Karambelkar et al. (https://arxiv.org/abs/2211.05141). 
Spectra are presented as tables with two columns each - Wavelength in Angstrom, and the scaled flux.
The full of spectroscopic observations is presented in Table 2 of the paper.