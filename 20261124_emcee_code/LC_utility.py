#!/usr/bin/env python
"""
Author: Rodrigo Angulo   Email: rangulo1@jhu.edu

Utility python file.
Import all the modules needed.
Definitions used in the notebooks combining light curves.

Code helped by Claude.

"""

import re
import glob
import requests
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

import h5py
from natsort import natsorted

import scipy.optimize as opt
from scipy.interpolate import interp1d, splev, splrep

import emcee
import corner

# Gaussian process for curve fitting 
from celerite2 import GaussianProcess, terms


### Get and read in light echo data from websites

  # Avaiable keys in data; seperate keys with arrays for proper data handling
singular_keys = ['lc_key', 'group', 'ID', 'ra', 'dec', 'color', 'x', 'y', 'LC_url']
other_keys = ['bsize', 'mjd', 'zpt', 'fluxADU', 'flux_err', 'X2norm', 'Npix', 'Nmask', 'Jyas2', 'Jyas2_err', 'SB', 'SB_err', 
              'fitsfile', 'skip', 'imID', 'filter', 'CONVOL00', 'PHOTNORM', 'KSUM00', 'FSIG00', 'FSCAT00', 'FWHM', 'M5SIGMA', 'SKYADU']

str_dtypes = ['skip', 'im_instrum', 'im_expnum', 'fitsfile', 'filter', 'PHOTNORM', 'CONVOL00']
ordered_keys = ['mjd', 'muJyas2', 'muJyas2_err', 'fitsfile', 'im_expnum', 'im_instrum', 'imID', 'skip', 'CONVOL00', 'PHOTNORM', 
                'filter', 'zpt', 'X2norm', 'FSCAT00', 'FSIG00', 'KSUM00', 'SKYADU', 'FWHM', 'M5SIGMA', 'Nmask', 'Npix', 'bsize', 
                'fluxADU', 'flux_err', 'Jyas2', 'Jyas2_err', 'SB', 'SB_err']


def get_LCdata(url, output_file=None, b_instrum=False, group=None, indices=None, excl_inds=None):
    '''
    Read in txt files from html containing LC info
    '''

    html = requests.get(url).content
    table = pd.read_html(html)[-1]
    for i in range(len(table)):
        ID_i = table.loc[i,'ID']
        lctxt_fi = f"{url.rpartition('.')[0]}_ID{ID_i}_lc.txt"
        table.loc[i,'LC_url'] = lctxt_fi

        grp_i = table.loc[i, 'group']
        lc_key = f"lc_{grp_i}_{ID_i}"
        table.loc[i,'lc_key'] = lc_key

    if indices is None:
        inds_t = table.index.values
    elif indices is not None:
        inds_t = indices
        
    if excl_inds is not None:
        inds_t = np.setdiff1d(inds_t, excl_inds)

    if group is not None:
        grp_inds = np.where(table['group'].eq(group))[0]
        inds_t = np.intersect1d(inds_t, grp_inds)
        
    lc_list = []
    for i in inds_t:
        tab = table.loc[i,:]
        obj_dict = {sk: tab[sk] for sk in singular_keys}

        temp_tab = pd.read_csv(tab['LC_url'], sep='\s+')

        for ok in other_keys:
            obj_dict[ok] = temp_tab[ok].values

        obj_dict['muJyas2'] = temp_tab['Jyas2'].values*1e6          # Want to work with micro-Jansky
        obj_dict['muJyas2_err'] = temp_tab['Jyas2_err'].values*1e6  
        
        ### Get expnum from fitsfile name
        if not b_instrum:
            expnums = temp_tab['fitsfile'].str.extract('\.(\d+)_ooi_')[0].values.astype(object)     # file pattern for pure DECam images (finds first instance of file pattern)
            instrums = np.array(['DECam']*len(expnums)).astype(object)
        elif b_instrum:
            expnums = temp_tab['fitsfile'].str.extract('\.x(\d+\.a\d+\_)')[0].values.astype(object)      # file pattern for mix of other instruments
            dates = temp_tab['fitsfile'].str.extract('\..\.(\d+)\.')[0].values.astype(object)            # add date to differentiate between instruments expnums
            expnums = expnums + dates
            instrums = temp_tab['fitsfile'].str.extract('\.\d+\.(.*)\.x')[0].values.astype(object)
                
        obj_dict['im_expnum'] = expnums
        obj_dict['im_instrum'] = instrums
    
        lc_list.append(obj_dict)
    
    if output_file is not None:
        with h5py.File(output_file, 'w') as f:
            for i, obj in enumerate(lc_list):
                label = f"lc_{obj['group']}_{obj['ID']}"
                grp = f.create_group(f"{label}")
    
                for sk in singular_keys:
                    grp.attrs[sk] = obj[sk]
        
                for ork in ordered_keys:
                    grp.create_dataset(ork, data=obj[ork])
        print(f"Saved to {output_file}")
        
    return(lc_list)



### Read in data and flag indices

def readin_LCdfs(filepath):
    le_list = []
    lc_data = {}
    with h5py.File(filepath, 'r') as f:
        for i, nm in enumerate(list(f)):
            df = pd.DataFrame([dict(f[nm].attrs)])
            le_list.append(df)
            
            data = {}
            extra_keys1 = []
            for key in f[nm].keys():
                if key in str_dtypes:
                    data[key] = f[nm][key][:].astype(str)
                elif key not in str_dtypes:
                    data[key] = f[nm][key][:]
                if key not in ordered_keys:             # want to have set order of columns, but also not miss added columns along the analysis
                    extra_keys1.append(key)
                    
            ddf = pd.DataFrame(data)
            ddf = ddf[ordered_keys+extra_keys1]
            lc_data[nm] = ddf
            
    les_df = pd.concat(le_list, ignore_index=True)
    les_df = les_df.sort_values(by=['ID']).reset_index(drop=True)
    extra_keys2 = [col for col in les_df.keys() if col not in singular_keys]      # want to have set order of columns, but also not miss added columns along the analysis
    les_df = les_df[singular_keys+extra_keys2]

    lc_data = {k: lc_data[k] for k in natsorted(lc_data.keys())}
    
    return(les_df, lc_data)


def swap_badband(dfs, bb_dfs, bb_expnums):

    new_dfs = {}

    for key in dfs.keys():
        df = dfs[key]
        bb_df = bb_dfs[key]

        matching_expnums = [exp for exp in bb_expnums if exp in df['im_expnum'].values and exp in bb_df['im_expnum'].values]
        g_inds = [np.where(df['im_expnum'] == exp)[0][0] for exp in matching_expnums]
        b_inds = [np.where(bb_df['im_expnum'] == exp)[0][0] for exp in matching_expnums]
        
        df_c = df.copy()
        df_c.loc[g_inds,'muJyas2'] = bb_df.loc[b_inds,'muJyas2']
        df_c.loc[g_inds,'muJyas2_err'] = bb_df.loc[b_inds,'muJyas2_err']
        df_c.loc[g_inds,'im_instrum'] = 'DECam(badband-corrected)'

        new_dfs[key] = df_c
    
    return(new_dfs)


def add_otherinstrums(dfs, oth_dfs):

    new_dfs = {}

    for key in dfs.keys():
        df = dfs[key]
        oth_df = oth_dfs[key]
    
        combined_df = pd.concat([df, oth_df], ignore_index=True)
        combined_df = combined_df.sort_values(by=['mjd']).reset_index(drop=True)

        new_dfs[key] = combined_df
    
    return(new_dfs)


def flag_inds(dfs, f_lm=None, ferr_lm=None, diffstats=None, bad_expnums=None):

    for key in dfs.keys():
        df = dfs[key]
        
        df['g_ixs_flag'] = False
        working_inds = df.index.values
        
        ## Exclude bad images
        if bad_expnums is not None:
            ginds = np.where(~df['im_expnum'].isin(bad_expnums))[0]
            df['g_ims_flag'] = False
            df.loc[ginds, 'g_ims_flag'] = True
            working_inds = ginds
    
        ## Exclude nans
        working_inds = working_inds[np.where(pd.notnull(df.loc[working_inds, 'muJyas2']))[0]]
    
        ## Kick out clear outliers in flux and error
        # sig2noi = abs(df.loc[working_inds,'muJyas2'])/df.loc[working_inds,'muJyas2_err']
        # working_inds = working_inds[np.where(sig2noi >= 1.0)[0]]
    
        if ferr_lm == 'med_cut':
            med_err = np.median(df.loc[working_inds,'muJyas2_err'])
            std_err = np.std(df.loc[working_inds,'muJyas2_err'])
            working_inds = working_inds[np.where(df.loc[working_inds, 'muJyas2_err'].lt(med_err + 1.0*std_err))[0]]
        elif ferr_lm is not None:
            working_inds = working_inds[np.where(df.loc[working_inds, 'muJyas2_err'].lt(ferr_lm))[0]]
        if f_lm is not None:
            working_inds = working_inds[np.where(df.loc[working_inds, 'muJyas2'].lt(f_lm))[0]]
    
        ## Exclude images based off of chi2, depth, and seeing
        if diffstats is not None:
            med_X2 = np.nanmedian(diffstats['X2NRM00'])
            std_X2 = np.nanstd(diffstats['X2NRM00'])
            med_M5 = np.nanmedian(diffstats['IM_M5SIGMA'])
            std_M5 = np.nanstd(diffstats['IM_M5SIGMA'])
            med_FW = np.nanmedian(diffstats['IM_FWHM'])
            std_FW = np.nanstd(diffstats['IM_FWHM'])
            temp_inds = np.where((diffstats['X2NRM00'] <= med_X2 + 3.0*std_X2) & 
                                 (diffstats['IM_M5SIGMA'] >= med_M5 - 3.0*std_M5) & 
                                 (diffstats['IM_FWHM'] <= med_FW + 3.0*std_FW))[0]
            gstats_expnums = diffstats.loc[temp_inds,'im_expnum'].values
            temp2_inds = np.where(df['im_expnum'].isin(gstats_expnums))[0]
            working_inds = np.intersect1d(working_inds, temp2_inds)
        elif diffstats is None:
            med_X2 = np.nanmedian(df.loc[working_inds, 'X2norm'])
            std_X2 = np.nanstd(df.loc[working_inds, 'X2norm'])
            med_M5 = np.nanmedian(df.loc[working_inds, 'M5SIGMA'])
            std_M5 = np.nanstd(df.loc[working_inds, 'M5SIGMA'])
            med_FW = np.nanmedian(df.loc[working_inds, 'FWHM'])
            std_FW = np.nanstd(df.loc[working_inds, 'FWHM'])
            temp_inds = np.where((df['X2norm'] <= med_X2 + 3.0*std_X2) & 
                                 (df['M5SIGMA'] >= med_M5 - 3.0*std_M5) & 
                                 (df['FWHM'] <= med_FW + 3.0*std_FW))[0]
            working_inds = np.intersect1d(working_inds, temp_inds)
    
        df.loc[working_inds, 'g_ixs_flag'] = True
    
    return('Updated LC dataframe -- flagged working indices')


def save_modifieddata(les_df, lc_dfs, outputfile):

    with h5py.File(outputfile, 'w') as f:
        for key in lc_dfs.keys():
            grp = f.create_group(key)
    
            le_tab = les_df.loc[np.where(les_df['lc_key'].eq(key))[0][0], :]
            for col in le_tab.keys():
                grp.attrs[col] = le_tab[col]
                
            lc_tab = lc_dfs[key]
            for col in lc_tab.keys():
                grp.create_dataset(col, data=lc_tab[col].values)
    print(f"Saved to {outputfile}")


### Methods for selecting light curves and indices, and getting values
def getLC_grp(dfs, le_group):
    grp_keys = []
    for key in list(dfs.keys()):
        group_k = key.partition('_')[-1].partition('_')[0]
        if int(group_k) == le_group:
            grp_keys.append(key)
    return(grp_keys)


def getLC_inds(dfs, lc_key, indices=None, exclude_inds=None, flag=None):

    if lc_key is not None:
        df = dfs[lc_key]
    elif lc_key is None:
        df = dfs
    
    if indices is None:
        indices = df.index.values
        
    if flag is None:
        s_inds = indices
    elif flag == 'sigmaclip':
        s_inds = np.where(df[f"{flag}_flag"].eq(False))[0]
    else:
        s_inds = np.where(df[f"{flag}_flag"].eq(True))[0]
            
    sel_inds = np.intersect1d(indices, s_inds)
        
    if exclude_inds is not None:
        sel_inds = np.setdiff1d(sel_inds, exclude_inds)

    return(sel_inds)


def getLC_vals(dfs, lc_key, peak_range=False, indices=None, exclude_inds=None, flag=None):

    inds = getLC_inds(dfs=dfs, lc_key=lc_key, indices=indices, exclude_inds=exclude_inds, flag=flag)

    if lc_key is not None:
        df = dfs[lc_key]
    elif lc_key is None:
        df = dfs
    
    t = df.loc[inds,'mjd'].values
    f = df.loc[inds,'muJyas2'].values
    err = df.loc[inds,'muJyas2_err'].values

    if peak_range is None:
        peak_loc = t[-1]
    elif peak_range == 'full':
        peak_loc = t[np.argmax(f)]
    elif peak_range is not False:
        new_t = t[(t>peak_range[0]) & (t<peak_range[1])]
        new_f = f[(t>peak_range[0]) & (t<peak_range[1])]
        peak_loc = new_t[np.argmax(new_f)]
            
    if peak_range is False:
        return(t, f, err)
    else:
        return(t, f, err, peak_loc)


def optimizeGP_celer(dfs, lc_key, indices=None, plot=False, smoother=False, flag=None, exclude_inds=None):

    t_i, f_i, err_i = getLC_vals(dfs=dfs, lc_key=lc_key, indices=indices, flag=flag, exclude_inds=exclude_inds)

    def build_gp(params):
        sigma, rho = np.exp(params)
        kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=0.5)

        gp = GaussianProcess(kernel)
        gp.compute(t_i, yerr=err_i)

        return(gp)

    def neg_log_like(params):
        gp = build_gp(params)
        return(-gp.log_likelihood(f_i))

    if smoother:
        p0 = np.log([np.std(f_i), 30000])
        bounds0 = [(np.log(1e-3), np.log(10*np.std(f_i))), (np.log(25000), np.log(35000))]
    elif not smoother:
        p0 = np.log([np.std(f_i), 20000])
        bounds0 = [(np.log(1e-3), np.log(10*np.std(f_i))), (np.log(15000), np.log(25000))]
    
    result = opt.minimize(neg_log_like, p0, method="L-BFGS-B", bounds=bounds0)
    gpr = build_gp(result.x)
    
    t_range = np.linspace(min(t_i), max(t_i), 1000)
    mu, var = gpr.predict(f_i, t_range, return_var=True)
    std = np.sqrt(var)

    gp_df = pd.DataFrame({'mjd':t_range, 'muJyas2':mu, 'muJyas2_err':std})

    if plot:
        plt.errorbar(t_i, f_i, yerr=err_i, fmt='.', ecolor='black', color='blue')
        plt.errorbar(gp_df['mjd'], gp_df['muJyas2'], yerr=gp_df['muJyas2_err'], fmt='.', ecolor='k', color='orange')
        plt.axhline(0, ls='--', color='k')
        plt.show()

    return(gp_df)


def sigmacut_df(dfs, lc_key=None, gpdf=None, indices=None, sigma=5.0, flag=None, return_inds=False, exclude_inds=None):
    inds_i = getLC_inds(dfs=dfs, lc_key=lc_key, indices=indices, flag=flag, exclude_inds=exclude_inds)
    t_i, f_i, err_i = getLC_vals(dfs=dfs, lc_key=lc_key, indices=indices, flag=flag, exclude_inds=exclude_inds)

    if gpdf is None:
        gpdf = optimizeGP_celer(dfs=dfs, lc_key=lc_key, indices=indices, exclude_inds=exclude_inds)
        
    mf = interp1d(gpdf['mjd'], gpdf['muJyas2'])
    merr = interp1d(gpdf['mjd'], gpdf['muJyas2_err'])

    resid = f_i - mf(t_i)
    err_tot = np.sqrt(err_i**2 + merr(t_i)**2)
    sigma_inds = np.where(abs(resid/err_tot) < sigma)[0]

    if lc_key is not None:
        df = dfs[lc_key]
    elif lc_key is None:
        df = dfs
    df['sigmaclip_flag'] = True
    df.loc[inds_i[sigma_inds], 'sigmaclip_flag'] = False
    # print('Updating dfs -- added sigmaclip_flag')
    if exclude_inds is not None:
        df.loc[exclude_inds, 'sigmaclip_flag'] = False

    if return_inds:
        return(inds_i[sigma_inds])



### Define function for plotting
def plot_LCs(dfs, lc_keys=None, plot_orig=False, plt_xl=None):

    keys_to_plot = lc_keys if lc_keys is not None else dfs.keys()

    for key in keys_to_plot:
        print(key)
        df = dfs[key]
    
        if plot_orig:
            i_inds = getLC_inds(dfs, key, flag=None)
            plt.errorbar(df.loc[i_inds,'mjd'], df.loc[i_inds,'muJyas2'], yerr=df.loc[i_inds,'muJyas2_err'], fmt='o', ecolor='black', color='blue')

        inds_sig = getLC_inds(dfs, key, flag='g_ixs')
        plt.errorbar(df.loc[inds_sig,'mjd'], df.loc[inds_sig,'muJyas2'], yerr=df.loc[inds_sig,'muJyas2_err'], fmt='o', ecolor='black', color='orange')
        
        plt.axhline(y=0, linestyle='--', color='black')

        if plt_xl is not None:
            plt.axvline(x=plt_xl)

        plt.xlabel('MJD')
        plt.ylabel('$\mu$Jy/as$^2$')
        plt.show()

    return('Done plotting')



# =============================================================================
# Reddening law utilities
# =============================================================================
# DECam effective wavelengths (Angstroms) — adjust if using different filter curves
FILTER_WAVE_EFF = {
    'g': 4770.0,
    'r': 6231.0,
    'i': 7625.0,
    'z': 9134.0,
    'Y': 10305.0,
}

def fitzpatrick99_Rlam(wave_eff, Rv=3.1):
    """
    R(lambda) = A(lambda)/E(B-V) using Fitzpatrick (1999).
    Uses dust_extinction if available, otherwise falls back to CCM89 optical approximation.
    
    Parameters
    ----------
    wave_eff : float or array-like
        Effective wavelength(s) in Angstroms.
    Rv : float
        Total-to-selective extinction ratio.
    
    Returns
    -------
    R_lam : float or ndarray
    """
    try:
        from dust_extinction.parameter_averages import F99
        import astropy.units as u
        ext_model = F99(Rv=Rv)
        R_lam = ext_model(np.atleast_1d(wave_eff) * u.AA) * Rv
        return float(np.squeeze(R_lam))
    except ImportError:
        pass
    
    # Fallback: CCM89 optical (1.1 <= x <= 3.3 inverse microns)
    x = 1e4 / np.atleast_1d(wave_eff).astype(float)
    y = x - 1.82
    a = (1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4
         + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
    b = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4
         - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)
    R_lam = Rv * a + b
    return float(np.squeeze(R_lam))


def compute_klam(filt_bands, Rv=3.1, ref_blue='g', ref_red='i'):
    """
    Compute k(lambda) = A(lambda)/E(ref_blue - ref_red) = R(lambda) / [R(blue) - R(red)]
    for a set of filter bands.
    
    Parameters
    ----------
    filt_bands : list of str
        Filter names (must be keys in FILTER_WAVE_EFF).
    Rv : float
        Total-to-selective extinction ratio.
    ref_blue, ref_red : str
        Reference filters defining the color index (default: g-i).
    
    Returns
    -------
    k_dict : dict
        k_dict[filter_name] = k(lambda).
    E_color_per_EBV : float
        E(blue-red)/E(B-V), useful for converting between parameterizations.
    """
    R_lam = {fb: fitzpatrick99_Rlam(FILTER_WAVE_EFF[fb], Rv=Rv) for fb in filt_bands}
    R_blue = fitzpatrick99_Rlam(FILTER_WAVE_EFF[ref_blue], Rv=Rv)
    R_red  = fitzpatrick99_Rlam(FILTER_WAVE_EFF[ref_red], Rv=Rv)
    
    E_color_per_EBV = R_blue - R_red  # E(g-i)/E(B-V)
    
    if E_color_per_EBV == 0:
        raise ValueError(f"R({ref_blue}) == R({ref_red}); choose different reference filters")
    
    k_dict = {fb: R_lam[fb] / E_color_per_EBV for fb in filt_bands}
    return k_dict, E_color_per_EBV


def reddening_ratio(filt_band, dEgi, k_dict):
    """
    Compute flux ratio from differential reddening:
        ared_tmpl / ared_sci = 10^(-0.4 * dEgi * k(lambda))
    
    Parameters
    ----------
    filt_band : str
        Filter name.
    dEgi : float
        Differential color excess E(g-i)_tmpl - E(g-i)_sci.
    k_dict : dict
        Precomputed k(lambda) values from compute_klam().
    
    Returns
    -------
    ratio : float
        Multiplicative flux ratio.
    """
    return 10**(-0.4 * dEgi * k_dict[filt_band])


### Fitting process for light curves using emcee
def emcee_fit(dfs_dict, le_group, lc_keys=None, tmpl_key=None, LCmodels=None, 
              peak_range=None, mod_peak_range=None, peak_filt=None, sel_filters=None,
              fixed_params=None, verbose=False, chain_sz=10000, return_samps=False, outputfile=None,
              fit_Rv=False, Rv_init=3.1, per_key_fit=False, tshift_i=None):
    '''
    Fitting LCs of LEs to a selected LC or to a model, using per-key MCMC.
    
    Model equation:
        f_tmpl = anorm_tmpl/anorm_sci * ared_ratio(lambda, dEgi, Rv) * [f_sci(t+tshift) + fshift_sci] + fshift_tmpl
    
    where:
        ared_ratio(lambda) = 10^(-0.4 * dEgi * k(lambda))
        k(lambda) = R(lambda) / [R(g) - R(i)]   from a reddening law (Fitzpatrick99)
    
    Parameters
    ----------
    dfs_dict : dict of dict of DataFrames
        dfs_dict[filter_band][lc_key] = DataFrame.
        can also be: dfs_dict[filter_band][segment] for fitting segments together
    lc_keys : list, optional
        LC keys to fit. If None, all keys (excluding tmpl_key) are used.
    tmpl_key : str, optional
        Key of template LC. Mutually exclusive with LCmodels.
    LCmodels : dict, optional
        LCmodels[filter_band] = DataFrame. Mutually exclusive with tmpl_key.
    peak_range : tuple, optional  -- or dict for fitting segments
        (t_lo, t_hi) to find peak in science LCs.
        or dict of above ^ to specify lc_key
    mod_peak_range : tuple, optional
        (t_lo, t_hi) to find peak in model LCs.
    tmpl_peak_range : tuple or dict, optional  -- only needed for fitting segments, the time axes are significantly shifted
        (t_lo, t_hi) to find peak in tmpl LCs.
        or dict of above ^ to specify lc_key
    peak_filt : str, optional
        Filter for peak-based tshift initialization.
    sel_filters : list, optional
        Subset of filters to fit.
    fixed_params : dict, optional
        Parameters to fix (name -> value). Applied to all keys.
    verbose : bool
        Print progress and diagnostics.
    chain_sz : int
        MCMC steps per walker.
    return_samps : bool
        If True, also return samples dict.
    outputfile : str, optional
        HDF5 file to save samples.
    fit_Rv : bool
        If True, fit Rv per key. If False, fix Rv=Rv_init.
    Rv_init : float
        Initial (or fixed) value of Rv.
    
    Returns
    -------
    combined_res : dict
        Combined result dict with median values and errors for all keys.
    samples_all : dict (only if return_samps=True)
        samples_all[lc_key] = DataFrame of posterior samples.
    '''

    # =========================================================================
    # 1. Setup: filters, keys, validation
    # =========================================================================
    if sel_filters is not None:
        filt_bands = sel_filters
    else:
        filt_bands = list(dfs_dict.keys())

    if lc_keys is not None: sel_keys = lc_keys
    elif le_group is not None: sel_keys = getLC_grp(dfs=dfs_dict[filt_bands[0]], le_group=le_group)
    else: sel_keys = dfs_dict[filt_bands[0]].keys()
    sel_keys = [k for k in sel_keys if k != tmpl_key]
    
    if peak_filt is None:
        peak_filt = filt_bands[0]

    if fixed_params is None:
        fixed_params = {}

    if LCmodels is None and tmpl_key is None:
        raise ValueError("Need to define LCmodels or a tmpl_key")
    if LCmodels is not None and tmpl_key is not None:
        raise ValueError("Provide either LCmodels or tmpl_key, not both")
    if LCmodels is not None and not set(filt_bands) <= set(LCmodels.keys()):
        raise ValueError("LCmodels must have the same filter bands as dfs_dict")
    if not set(filt_bands) <= set(dfs_dict.keys()):
        raise ValueError("sel_filters must be a subset of dfs_dict keys")

    # =========================================================================
    # 2. Precompute reddening law k(lambda) values
    # =========================================================================
    k_dict_cache = {}  # Rv -> k_dict, computed on demand

    def get_k_dict(Rv):
        if Rv not in k_dict_cache:
            k_dict_cache[Rv], _ = compute_klam(filt_bands, Rv=Rv, ref_blue='g', ref_red='i')
        return k_dict_cache[Rv]

    # Precompute for initial Rv
    get_k_dict(Rv_init)

    # =========================================================================
    # 3. Load template data (shared across all keys)
    # =========================================================================
     ## different data flag for when fitting segments
    if le_group is not None: 
        dflag = 'g_ixs'
    elif le_group is None: 
        dflag = 'sigmaclip'

    t_tmpl, f_tmpl, err_tmpl, peak_tmpl = {}, {}, {}, {}
    for fband in filt_bands:
        if tmpl_key is not None:
            if isinstance(peak_range, dict): peak_range1 = peak_range[tmpl_key]
            else: peak_range1 = peak_range
            t1, f1, err1, peak1 = getLC_vals(dfs_dict[fband], lc_key=tmpl_key, 
                                              peak_range=peak_range1, flag=dflag)
        elif LCmodels is not None:
            t1, f1, err1, peak1 = getLC_vals(LCmodels[fband], lc_key=None, 
                                              peak_range=mod_peak_range, flag=None)
        t_tmpl[fband] = t1
        f_tmpl[fband] = f1
        err_tmpl[fband] = err1
        peak_tmpl[fband] = peak1

    # =========================================================================
    # 4. Load science data + build interpolators (done once for all keys)
    # =========================================================================
    t_sci, f_sci, err_sci, peak_sci = {}, {}, {}, {}
    sci_flux, sci_error = {}, {}
    for fband in filt_bands:
        t_sci[fband], f_sci[fband], err_sci[fband], peak_sci[fband] = {}, {}, {}, {}
        sci_flux[fband], sci_error[fband] = {}, {}
        for key in sel_keys:
            if isinstance(peak_range, dict): peak_range2 = peak_range[key]
            else: peak_range2 = peak_range
            t2, f2, err2, peak2 = getLC_vals(dfs_dict[fband], lc_key=key, peak_range=peak_range2, flag=dflag)
            t_sci[fband][key] = t2
            f_sci[fband][key] = f2
            err_sci[fband][key] = err2
            peak_sci[fband][key] = peak2
            sci_flux[fband][key] = interp1d(t2, f2, bounds_error=False, fill_value=np.nan)
            sci_error[fband][key] = interp1d(t2, err2, bounds_error=False, fill_value=np.nan)

    # =========================================================================
    # 5. Define compute_chi2 (closed over template & science data)
    # =========================================================================
    def compute_chi2(params, filt_band, lckey, res_d=None):
        """
        Compute chi-squared for one filter + one LC key.
        
        Model: f_model = (anorm_tmpl/anorm) * ared_ratio(lambda) * (f_sci(t+tshift) + fshift) + fshift_tmpl
        where ared_ratio = 10^(-0.4 * dEgi * k(lambda))
        """
        tshift      = params[f'tshift_{lckey}']
        anorm_tmpl  = params['anorm_tmpl']
        anorm       = params[f'anorm_{lckey}']
        fshift_tmpl = params[f'fshift_{filt_band}_tmpl']
        fshift      = params[f'fshift_{filt_band}_{lckey}']
        dEgi        = params[f'dEgi_{lckey}']
        Rv          = params.get(f'Rv_{lckey}', params.get('Rv', Rv_init))

        # Get k(lambda) for this Rv
        k_dict = get_k_dict(Rv)
        ared_ratio = reddening_ratio(filt_band, dEgi, k_dict)

        # Interpolate science LC onto template time grid
        fshifted = sci_flux[filt_band][lckey](t_tmpl[filt_band] + tshift)
        ferr     = sci_error[filt_band][lckey](t_tmpl[filt_band] + tshift)
        mask     = np.isfinite(fshifted) & np.isfinite(ferr)

        if np.sum(mask) == 0:
            if res_d is not None:
                res_d[f"chi2_{filt_band}_{lckey}"] = np.nan
                res_d[f"npoints_{filt_band}_{lckey}"] = 0
            return np.inf

        amp     = anorm_tmpl / anorm * ared_ratio
        fnew    = amp * (fshifted[mask] + fshift) + fshift_tmpl
        ferrnew = amp * ferr[mask]

        resid   = fnew - f_tmpl[filt_band][mask]
        err_tot = np.sqrt(ferrnew**2 + err_tmpl[filt_band][mask]**2)
        chi2    = np.sum(resid**2 / err_tot**2)
        npts    = int(np.sum(mask))

        if res_d is not None:
            res_d[f"chi2_{filt_band}_{lckey}"] = chi2
            res_d[f"npoints_{filt_band}_{lckey}"] = npts

        log_norm = np.sum(np.log(2 * np.pi * err_tot**2))
        return chi2 + log_norm

    # =========================================================================
    # 6. Template params — fixed by default (breaks degeneracies)
    # =========================================================================
    tmpl_defaults = {'anorm_tmpl': 1.0}
    for fband in filt_bands:
        tmpl_defaults[f'fshift_{fband}_tmpl'] = 0.0

    # Merge: user fixed_params override defaults
    global_fixed = {**tmpl_defaults, **fixed_params}

    # If not fitting Rv, fix it globally
    if not fit_Rv:
        global_fixed.setdefault('Rv', Rv_init)

    # =========================================================================
    # 7. MCMC fit -- per-key loop or all together
    # =========================================================================
    combined_res = {}
    samples_all = {}

    if per_key_fit:
        for ki, key in enumerate(sel_keys):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Fitting key {key}  ({ki+1}/{len(sel_keys)})")
                print(f"{'='*60}")

            # ----- Build parameter dict for this key -----
            tshift_tol = 730
            if isinstance(tshift_i, dict):
                tshift_init = tshift_i[key]
            elif tshift_i is not None:
                tshift_init = tshift_i
            else:
                tshift_init = peak_sci[peak_filt][key] - peak_tmpl[peak_filt]

            all_params = {}

            # Science params (per-key)
            all_params[f'tshift_{key}'] = {
                'init': tshift_init,
                'bounds': (tshift_init - tshift_tol, tshift_init + tshift_tol)
            }
            all_params[f'anorm_{key}'] = {'init': 1.0, 'bounds': (1e-4, 20.0)}
            all_params[f'dEgi_{key}']  = {'init': 0.0, 'bounds': (-3.0, 3.0)}

            if fit_Rv:
                all_params[f'Rv_{key}'] = {'init': Rv_init, 'bounds': (1.0, 6.0)}

            for fband in filt_bands:
                all_params[f'fshift_{fband}_{key}'] = {'init': 0.0, 'bounds': (-100, 100)}

            # Template params (included so they can be freed via fixed_params if desired)
            all_params['anorm_tmpl'] = {'init': 1.0, 'bounds': (1e-4, 20.0)}
            for fband in filt_bands:
                all_params[f'fshift_{fband}_tmpl'] = {'init': 0.0, 'bounds': (-100, 100)}

            # Separate free vs fixed
            free_params = [k for k in all_params.keys() if k not in global_fixed]
            initial     = np.array([all_params[k]['init'] for k in free_params])
            bounds      = [all_params[k]['bounds'] for k in free_params]

            if len(free_params) == 0:
                raise ValueError(f"No free parameters for key {key}. Check fixed_params.")

            if verbose:
                print(f"  Free params ({len(free_params)}): {free_params}")
                print(f"  Fixed: { {k: v for k, v in global_fixed.items() if k in all_params} }")

            # ----- Inner functions (capture per-key values via defaults) -----
            def unpack_params(params, _fp=free_params, _gf=global_fixed):
                full = _gf.copy()
                for i, name in enumerate(_fp):
                    full[name] = params[i]
                return full

            def log_like(params, _key=key):
                ps = unpack_params(params)
                total = sum(compute_chi2(ps, filt_band=fband, lckey=_key) 
                            for fband in filt_bands)
                return -0.5 * total

            def log_prior(params, _bounds=bounds):
                for i in range(len(params)):
                    lo, hi = _bounds[i]
                    if not (lo < params[i] < hi):
                        return -np.inf
                return 0.0

            def log_probability(params):
                lp = log_prior(params)
                if not np.isfinite(lp):
                    return -np.inf
                return lp + log_like(params)

            # ----- Run emcee -----
            ndim     = len(free_params)
            nwalkers = max(32, ndim * 5)

            # Initialize walkers, clipped to stay within bounds
            lo_arr = np.array([b[0] for b in bounds])
            hi_arr = np.array([b[1] for b in bounds])
            pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)
            pos = np.clip(pos, lo_arr + 1e-10, hi_arr - 1e-10)

            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
            sampler.run_mcmc(pos, chain_sz, progress=verbose)

            if verbose:
                try:
                    tau = sampler.get_autocorr_time(quiet=True)
                    print(f"  Autocorrelation time: {np.round(tau, 1)}")
                    print(f"  Chain length / max(tau): {chain_sz / np.max(tau):.1f}")
                except Exception as e:
                    print(f"  Autocorrelation warning: {e}")

            # ----- Extract samples -----
            discard = int(chain_sz * 3. / 10)
            thin = 10
            flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
            samples = pd.DataFrame(flat_samples, columns=free_params)

            samples_all[key] = samples

            if outputfile is not None:
                save_samples(samples=samples, outputfile=outputfile, 
                            full_key=f'{key}', mode='w' if ki == 0 else 'a')

            # ----- Extract medians + uncertainties -----
            for p in free_params:
                samps  = samples[p].values
                ptiles = np.percentile(samps, [16, 50, 84])
                combined_res[p] = ptiles[1]
                combined_res[f"{p}_err"] = [ptiles[2] - ptiles[1], ptiles[1] - ptiles[0]]

            # ----- Compute chi2 at best-fit -----
            bestfit_params = {**global_fixed}
            for p in free_params:
                bestfit_params[p] = combined_res[p]

            for fband in filt_bands:
                compute_chi2(bestfit_params, filt_band=fband, lckey=key, res_d=combined_res)

            if verbose:
                total_chi2 = sum(combined_res.get(f"chi2_{fband}_{key}", 0) for fband in filt_bands)
                total_npts = sum(combined_res.get(f"npoints_{fband}_{key}", 0) for fband in filt_bands)
                dEgi_val = combined_res[f'dEgi_{key}']
                anorm_val = combined_res[f'anorm_{key}']
                tshift_val = combined_res[f'tshift_{key}']
                print(f"  Results: tshift={tshift_val:.1f}, anorm={anorm_val:.3f}, dEgi={dEgi_val:.3f}")
                print(f"  Chi2/npts = {total_chi2:.1f}/{total_npts} = {total_chi2/max(total_npts,1):.2f}")

    else:
        all_params = {}
        for key in sel_keys:
            tshift_tol = 730
            if key not in tshift_i.keys():
                tshift_init = peak_sci[peak_filt][key] - peak_tmpl[peak_filt]
            else:
                tshift_init = tshift_i[key]
            tshift_init = peak_sci[peak_filt][key] - peak_tmpl[peak_filt]

            # Science params (per-key)
            all_params[f'tshift_{key}'] = {
                'init': tshift_init,
                'bounds': (tshift_init - tshift_tol, tshift_init + tshift_tol)
            }
            all_params[f'anorm_{key}'] = {'init': 1.0, 'bounds': (1e-4, 20.0)}
            all_params[f'dEgi_{key}']  = {'init': 0.0, 'bounds': (-3.0, 3.0)}

            if fit_Rv:
                all_params[f'Rv_{key}'] = {'init': Rv_init, 'bounds': (1.0, 6.0)}

            for fband in filt_bands:
                all_params[f'fshift_{fband}_{key}'] = {'init': 0.0, 'bounds': (-100, 100)}

        # Template params (included so they can be freed via fixed_params if desired)
        all_params['anorm_tmpl'] = {'init': 1.0, 'bounds': (1e-4, 20.0)}
        for fband in filt_bands:
            all_params[f'fshift_{fband}_tmpl'] = {'init': 0.0, 'bounds': (-100, 100)}

        # Separate free vs fixed
        free_params = [k for k in all_params.keys() if k not in global_fixed]
        initial     = np.array([all_params[k]['init'] for k in free_params])
        bounds      = [all_params[k]['bounds'] for k in free_params]

        if len(free_params) == 0:
            raise ValueError(f"No free parameters. Check fixed_params.")

        if verbose:
            print(f"  Free params ({len(free_params)}): {free_params}")
            print(f"  Fixed: { {k: v for k, v in global_fixed.items() if k in all_params} }")

        # ----- Inner functions (capture per-key values via defaults) -----
        def unpack_params(params, _fp=free_params, _gf=global_fixed):
            full = _gf.copy()
            for i, name in enumerate(_fp):
                full[name] = params[i]
            return full

        def log_like(params):
            ps = unpack_params(params)
            total = sum(compute_chi2(ps, filt_band=fband, lckey=key) for fband in filt_bands for key in sel_keys)
            return -0.5 * total

        def log_prior(params, _bounds=bounds):
            for i in range(len(params)):
                lo, hi = _bounds[i]
                if not (lo < params[i] < hi):
                    return -np.inf
            return 0.0

        def log_probability(params):
            lp = log_prior(params)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_like(params)

        # ----- Run emcee -----
        ndim     = len(free_params)
        nwalkers = max(32, ndim * 5)

        # Initialize walkers, clipped to stay within bounds
        lo_arr = np.array([b[0] for b in bounds])
        hi_arr = np.array([b[1] for b in bounds])
        pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)
        pos = np.clip(pos, lo_arr + 1e-10, hi_arr - 1e-10)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
        sampler.run_mcmc(pos, chain_sz, progress=verbose)

        if verbose:
            try:
                tau = sampler.get_autocorr_time(quiet=True)
                print(f"  Autocorrelation time: {np.round(tau, 1)}")
                print(f"  Chain length / max(tau): {chain_sz / np.max(tau):.1f}")
            except Exception as e:
                print(f"  Autocorrelation warning: {e}")

        # ----- Extract samples -----
        discard = int(chain_sz * 3. / 10)
        thin = 10
        flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        samples = pd.DataFrame(flat_samples, columns=free_params)
        samples_all=samples

        if outputfile is not None:
            save_samples(samples=samples, outputfile=outputfile, 
                        full_key='samps', mode='w')

        # ----- Extract medians + uncertainties -----
        for p in free_params:
            samps  = samples[p].values
            ptiles = np.percentile(samps, [16, 50, 84])
            combined_res[p] = ptiles[1]
            combined_res[f"{p}_err"] = [ptiles[2] - ptiles[1], ptiles[1] - ptiles[0]]

        # ----- Compute chi2 at best-fit -----
        bestfit_params = {**global_fixed}
        for p in free_params:
            bestfit_params[p] = combined_res[p]

        for fband in filt_bands:
            for key in sel_keys:
                compute_chi2(bestfit_params, filt_band=fband, lckey=key, res_d=combined_res)

        if verbose:
            total_chi2 = sum(combined_res.get(f"chi2_{fband}_{key}", 0) for fband in filt_bands for key in sel_keys)
            total_npts = sum(combined_res.get(f"npoints_{fband}_{key}", 0) for fband in filt_bands for key in sel_keys)
            dEgi_val = np.average(combined_res[f'dEgi_{key}'] for key in sel_keys)
            anorm_val = np.average(combined_res[f'anorm_{key}'] for key in sel_keys)
            tshift_val = np.average(combined_res[f'tshift_{key}'] for key in sel_keys)
            print(f"  Results: tshift={tshift_val:.1f}, anorm={anorm_val:.3f}, dEgi={dEgi_val:.3f}")
            print(f"  Chi2/npts = {total_chi2:.1f}/{total_npts} = {total_chi2/max(total_npts,1):.2f}")

    # =========================================================================
    # 8. Finalize combined result dict
    # =========================================================================
    combined_res.update(global_fixed)

    # Metadata
    combined_res['sel_keys'] = sel_keys
    combined_res['filt_bands'] = filt_bands
    combined_res['fit_Rv'] = fit_Rv
    combined_res['Rv_init'] = Rv_init
    if LCmodels is not None:
        combined_res['tmpl_key'] = None
    else:
        combined_res['tmpl_key'] = tmpl_key

    # Lists of free/fixed param names (across all keys)
    all_free = [p for key in sel_keys for p in 
                [f'tshift_{key}', f'anorm_{key}', f'dEgi_{key}'] +
                ([f'Rv_{key}'] if fit_Rv else []) +
                [f'fshift_{fb}_{key}' for fb in filt_bands]
                if p not in global_fixed]
    combined_res['free_params'] = all_free
    combined_res['fixed_params'] = list(global_fixed.keys())

    if verbose:
        print(f"\n{'='*60}")
        print(f"Done. Fit {len(sel_keys)} keys x {len(filt_bands)} filters.")
        print(f"{'='*60}")

    if return_samps:
        return samples_all, combined_res
    else:
        return combined_res


def save_samples(samples, outputfile, full_key=None, mode='a'):
    with pd.HDFStore(outputfile, mode) as store:
        if full_key is None:
            for key in samples.keys():
                store.put(key, samples[key])
        else:
            store.put(full_key, samples)

def load_samples(filename):
    samples_ls = {}
    with pd.HDFStore(filename, mode='r') as store:
        for full_key in store.keys():
            full_key = full_key.lstrip('/')
            keys = full_key.split('/')
            # Navigate to the correct nested level, creating dicts as needed
            current = samples_ls
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            # Store the DataFrame at the final key
            current[keys[-1]] = store.get('/' + full_key)
    return samples_ls


def save_fitparams(mcres_dict, outputfile, mode='w'):
    """
    Save fit parameters to HDF5 file (possibly nested dict)
    """
    def _save_recursive(parent, data):
        for key, value in data.items():
            if isinstance(value, dict):
                # Nested dict: create group and recurse
                grp = parent.create_group(key)
                _save_recursive(grp, value)
            elif isinstance(value, (list,np.ndarray)):
                # List: save as dataset
                # Check if it's a string array
                arr = np.array(value)
                if arr.dtype.kind in ('U', 'S', 'O'):  # Unicode, byte string, or object
                    parent.create_dataset(key, data=value, dtype=h5py.string_dtype(encoding='utf-8'))
                else:
                    parent.create_dataset(key, data=value)
            else:
                # Scalar: save as attribute
                if value is None: value = 'None'
                parent.attrs[key] = value

    with h5py.File(outputfile, mode) as f:
        _save_recursive(f, mcres_dict)
    print(f"Saved to {outputfile}")


def load_fitparams(inputfile):
    """
    Load fit parameters from HDF5 file.
    """
    def _load_recursive(parent):
        result = {}
        # Load attributes (scalars)
        for key, value in parent.attrs.items():
            if value == 'None': value = None
            result[key] = value
        # Load datasets and groups
        for key in parent.keys():
            item = parent[key]
            if isinstance(item, h5py.Group):
                result[key] = _load_recursive(item)
            else:  # Dataset
                data = item[()]
                result[key] = np.array([x.decode('utf-8') if isinstance(x, bytes) else x for x in data.flat]).reshape(data.shape)
        return result
    with h5py.File(inputfile, 'r') as f:
        return _load_recursive(f)


def apply_fparams(params, filt_band, lckey, f_sci, err_sci):
    """
    Apply fit parameters to transform a science LC to the template frame.
    Uses E(g-i) reddening parameterization.
    
    Parameters
    ----------
    params : dict
        Full parameter dictionary (from emcee_fit result).
    filt_band : str
        Filter band name.
    lckey : str
        LC key name.
    f_sci : array
        Science flux values.
    err_sci : array
        Science flux error values.
    
    Returns
    -------
    fnew : array
        Transformed flux.
    ferrnew : array
        Transformed flux error.
    """
    anorm_tmpl  = params['anorm_tmpl']
    anorm       = params[f'anorm_{lckey}']
    fshift_tmpl = params[f'fshift_{filt_band}_tmpl']
    fshift      = params[f'fshift_{filt_band}_{lckey}']
    dEgi        = params[f'dEgi_{lckey}']
    Rv          = params.get(f'Rv_{lckey}', params.get('Rv', params.get('Rv_init', 3.1)))

    # Compute reddening ratio from physical parameters
    filt_bands = params.get('filt_bands', list(FILTER_WAVE_EFF.keys()))
    k_dict, _ = compute_klam(filt_bands, Rv=Rv, ref_blue='g', ref_red='i')
    ared_ratio = reddening_ratio(filt_band, dEgi, k_dict)

    amp = anorm_tmpl / anorm * ared_ratio
    fnew    = amp * (f_sci + fshift) + fshift_tmpl
    ferrnew = amp * err_sci

    return fnew, ferrnew


def plot_LCfits(dfs_dict, fitparams_dict, lckeys=None, LCmodels=None):

    filt_bands = list(dfs_dict.keys())
    if lckeys is None: lckeys = fitparams_dict['sel_keys']

    tmpl_key = fitparams_dict.get('tmpl_key', None)

    if LCmodels is not None and not set(filt_bands) <= set(LCmodels.keys()):
        raise ValueError("dfs_dict must have same filters as LCmodels")

    ## Initialize template data
    if 'seg1' in dfs_dict[filt_bands[0]].keys(): dflag = 'sigma_clip'
    else: dflag = 'g_ixs'
    t_tmpl, f_tmpl, err_tmpl = {}, {}, {}
    for fband in filt_bands:
        if tmpl_key is not None:
            t1, f1, err1 = getLC_vals(dfs_dict[fband], lc_key=tmpl_key, flag=dflag)
        elif LCmodels is not None:
            t1, f1, err1 = getLC_vals(LCmodels[fband], lc_key=None, flag=None)
        t_tmpl[fband], f_tmpl[fband], err_tmpl[fband] = t1, f1, err1

    ## Plot
    n_lcs = len(lckeys)
    fig_l = int(np.round(n_lcs + 0.49))
    fig, axs = plt.subplots(fig_l, len(filt_bands), figsize=(5*len(filt_bands), 5*fig_l))
    
    f_ps = fitparams_dict
    for i, key in enumerate(lckeys):
        for j, fband in enumerate(filt_bands):
            if len(filt_bands) == 1 and len(lckeys) > 1:
                ax = axs[i]
            elif len(filt_bands) == 1 and len(lckeys) == 1:
                ax = axs
            elif len(filt_bands) > 1 and len(lckeys) == 1:
                ax = axs[j]
            elif len(filt_bands) > 1 and len(lckeys) > 1:
                ax = axs[i][j]
            
            t2, f2, err2 = getLC_vals(dfs=dfs_dict[fband], lc_key=key, flag=dflag)
            fnew, ferrnew = apply_fparams(params=f_ps, filt_band=fband, lckey=key, f_sci=f2, err_sci=err2)
            tnew = t2 - f_ps[f'tshift_{key}']

            ax.errorbar(t2, f2, label='LC', fmt='.-', color='orange')
            ax.errorbar(t_tmpl[fband], f_tmpl[fband], label='LC tmpl', color='blue', fmt='.-')
            ax.errorbar(tnew, fnew, alpha=.5, label='LC aligned', fmt='.-', color='green')

            # Fixed: use per-key, per-filter chi2 keys
            chi2_val  = f_ps.get(f"chi2_{fband}_{key}", np.nan)
            npts_val  = f_ps.get(f"npoints_{fband}_{key}", 1)
            chi2_red  = chi2_val / max(npts_val, 1)

            tshift_val = f_ps[f'tshift_{key}']
            anorm_val  = f_ps[f'anorm_{key}']
            dEgi_val   = f_ps[f'dEgi_{key}']
            fshift_val = f_ps[f'fshift_{fband}_{key}']

            ax.legend()
            if j == 0:
                ax.set_title(f"{key} -- {fband}-band\n"
                             f"(tshift={tshift_val:.1f}, anorm={anorm_val:.2f}, "
                             f"dEgi={dEgi_val:.3f}, fshift={fshift_val:.1f}, "
                             f"chi2r={chi2_red:.1f})")
            else:
                ax.set_title(f"{fband}-band\n"
                             f"fshift={fshift_val:.1f}, chi2r={chi2_red:.1f})")

    fig.subplots_adjust(hspace=0.3)
    return('Plotting LC fits')


def get_combLCs(dfs_dict, fitparams_dict, LCmodels=None, flag='g_ixs'):

    filt_bands = list(dfs_dict.keys())
    
    if LCmodels is not None and not set(filt_bands) <= set(LCmodels.keys()):
        raise ValueError("dfs_dict must have same filters as LCmodels")

    tmpl_key = fitparams_dict.get('tmpl_key', None)
    sel_keys = fitparams_dict['sel_keys']
    if tmpl_key is not None:
        tot_keys = np.concatenate([[tmpl_key], sel_keys])
    else: tot_keys = sel_keys

    comb_dict = {}
    f_ps = fitparams_dict
    for fband in filt_bands:
        dfs = dfs_dict[fband]
        
        tnew_ls = []
        fnew_ls = []
        errnew_ls = []
        orig_keys = []
        orig_inds = []
        im_instrums = []
        fits_files = []
        
        for key in tot_keys:
            inds = getLC_inds(dfs, lc_key=key, flag=flag)
            t, f, err = getLC_vals(dfs, lc_key=key, flag=flag)

            if key == tmpl_key:
                fnew = f
                ferrnew = err
                tnew = t
            else:
                fnew, ferrnew = apply_fparams(params=f_ps, filt_band=fband, lckey=key, f_sci=f, err_sci=err)
                tnew = t - f_ps[f'tshift_{key}']

            tnew_ls.append(tnew)
            fnew_ls.append(fnew)
            errnew_ls.append(ferrnew)
            orig_keys.append([key]*len(inds))
            orig_inds.append(inds)
            im_instrums.append(dfs[key].loc[inds, 'im_instrum'].values)
            fits_files.append(dfs[key].loc[inds, 'fitsfile'].values)

            ### Update table
            dfs[key].loc[inds,'new_mjd'] = tnew
            dfs[key].loc[inds,'new_muJyas2'] = fnew
            dfs[key].loc[inds,'new_muJyas2_err'] = ferrnew

        ### Put combined data into a data frame
        new_t = np.concatenate(tnew_ls)
        new_f = np.concatenate(fnew_ls)
        new_err = np.concatenate(errnew_ls)
        orig_keys = np.concatenate(orig_keys)
        orig_inds = np.concatenate(orig_inds)
        im_instrums = np.concatenate(im_instrums)
        fits_files = np.concatenate(fits_files)
        
        comb_df = pd.DataFrame({
            'mjd': new_t,
            'muJyas2': new_f,
            'muJyas2_err': new_err,
            'orig_key': orig_keys,
            'orig_ind': orig_inds,
            'im_instrum': im_instrums,
            'fitsfile': fits_files
        })
        comb_df = comb_df.sort_values(by=['mjd']).reset_index(drop=True)
        comb_dict[fband] = comb_df

    return comb_dict


def comb_dicts(dict_ls, labels=None):
    combs_dict = {}
    for fband in dict_ls[0].keys():
        combs_dict[fband] = {}
        for i, cb in enumerate(dict_ls):
            if labels is None:
                combs_dict[fband][f'seg{i+1}'] = cb[fband]
            else:
                combs_dict[fband][f'{labels[0]}'] = cb[fband]
    return(combs_dict)



def set_peak(findat_dict, finmod_dict, peak_filt='i', flag=None):
    
    t_i, f_i, err_i, peak_i = getLC_vals(dfs=finmod_dict[peak_filt], lc_key=None, peak_range='full', flag=flag)

    for fband in findat_dict.keys():
        df_dat = findat_dict[fband]
        df_dat['phase'] = df_dat['mjd'] - peak_i

        df_mod = finmod_dict[fband]
        df_mod['phase'] = df_mod['mjd'] - peak_i
        
    return('Updated phase')


def save_finaldata(fin_data, outputfile):
    with pd.HDFStore(outputfile, mode='w') as store:
        for key, df in fin_data.items():
            store.put(key, df)
    print(f"Saving to {outputfile}")

def load_finaldata(filename, filt_key=None):
    with pd.HDFStore(filename, mode='r') as store:
        if filt_key is None:
            table = {}
            for key in list(store.keys()):
                key = key.strip('/')
                table[key] = store.get(key)
        elif filt_key is not None:
            table = store.get(filt_key)
    return(table)


def convert_muJy2mag(df, lc_key=None, flag=None):

    t, f, err = getLC_vals(df, lc_key=lc_key, flag=flag)
    wf = np.where(f <= 0, 1e-3, f)     # Replace negative values with small numbers (assuming those are non-detections)
    
    df['ABmag'] = 23.9 - (2.5 * np.log10(wf))     # AB magnitude conversion
    df['ABmag_err'] = 1.086 * (err/wf)            # AB mag error of propagation
    
    return('AB mag calculated')