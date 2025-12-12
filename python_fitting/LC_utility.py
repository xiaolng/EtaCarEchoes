"""

Utility python file.
Import all the modules needed. This includes pdastro python file.
Definitions used in the notebooks combining light curves.

"""


import requests
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from scipy.interpolate import splev, splrep, interp1d
import scipy.optimize as opt
from scipy.signal import find_peaks, medfilt

# Gaussian process for curve fitting
import george
from george.kernels import ExpSquaredKernel

import glob

from pdastro import *



def get_LCtable(url):
    html = requests.get(url).content
    df_table = pd.read_html(html)[-1]
    for i in range(len(df_table)):
        ID_i = df_table['ID'][i]
        lctxt_fi = url.rpartition('.')[0]+'_ID'+str(ID_i)+'_lc.txt'
        df_table.loc[i,'LC_url'] = lctxt_fi
    
    return(df_table)


def get_table_inds(table, group):
    inds = []
    for i in range(len(table)):
        group_i = table['group'][i]
        if group_i == group:
            inds.append(i)
    return(inds)



def get_LCdata(table, group=None, indices=None):
    '''
    Read in txt files from html containing LC info
    '''

    if indices is None:
        inds_t = range(len(table))
    elif indices is not None:
        inds_t = indices
    
    lc_list = []
    for i in inds_t:
        group_i = table['group'][i]
        url_i = table['LC_url'][i]

        if group is None:
            temp_tab = pd.read_csv(url_i, sep='\s+')
            lc_list.append(temp_tab)
        elif group_i == group:
            temp_tab = pd.read_csv(url_i, sep='\s+')
            lc_list.append(temp_tab)

    ### Want to work with micro-Jansky
    for df in lc_list:
        df['mJyas2'] = df['Jyas2']*1e6
        df['mJyas2_err'] = df['Jyas2_err']*1e6

    return(pd.concat(lc_list))


def readin_LCdf(filepath):
    lc_data = pd.read_csv(filepath)
    lc_df = pdastrostatsclass()
    lc_df.t = lc_df.t.assign(**lc_data)
    lc_df.t.reset_index(drop=True, inplace=True)
    return(lc_df)



def get_LCbins(df, bin_size, by_mjd=False, indices=None):
    '''
    Get the bin indices for error-cut loop
    '''
    
    bins_inds = []
    bin_ts = []    

    sort_inds = df.ix_sort_by_cols('mjd', indices=indices)
    sort_inds = df.ix_not_null('mJyas2', indices=sort_inds)
    
    if by_mjd is False:
        for i in range(0, len(sort_inds), bin_size):
            if len(sort_inds) - (i+bin_size) < bin_size or len(sort_inds) - i < bin_size:    # if point is near end or at end of list
                bins_inds.append(sort_inds[i:])
                break
            else:
                bins_inds.append(sort_inds[i:i + bin_size])
    
    elif by_mjd is True:
        t_range = np.arange(min(df.t.loc[sort_inds,'mjd']),max(df.t.loc[sort_inds,'mjd']),bin_size)
        skip_inds = []
        bin_ts = [t_range[0]]
        for i in range(len(t_range)-1):
            if i not in skip_inds:
                lo_i = i
                skip_inds.append(i)
                up_i = i+1
                bin_i = df.ix_inrange('mjd', lowlim=t_range[lo_i], uplim=t_range[up_i], indices=sort_inds)
                while len(bin_i)<30 and up_i!=len(t_range)-1:
                    skip_inds.append(up_i)
                    up_i += 1
                    bin_i = df.ix_inrange('mjd', lowlim=t_range[lo_i], uplim=t_range[up_i], indices=sort_inds)
                    
                bins_inds.append(bin_i)
                bin_ts.append(t_range[up_i])

    return(bins_inds, bin_ts)



def get_LCinds(df, lc_ID, f_lm=None, ferr_lm=None, bad_expnums=None, bin_size=None, by_mjd=False):
    """
    Return the indices of the table of usable values for a given LC position
    # f_lm and ferr_lm is the upper limit of flux and its error in mJyas2 -- micro-Jansky / arcsec^2
    """
    id_inds = np.where(df.t['ID'].eq(lc_ID))[0]
    
    if bad_expnums != None:
        working_inds = []
        for i in id_inds:
            diff_nm = df.t.loc[i,'fitsfile']
            expnum_se = re.search('\.(\d+)_ooi_',os.path.basename(diff_nm))
            expnum = float(expnum_se.groups()[0])
            if expnum not in bad_expnums:
                working_inds.append(i)
    elif bad_expnums == None:
        working_inds = id_inds

    if ferr_lm is not None:
        working_inds = df.ix_inrange('mJyas2_err', uplim=ferr_lm, indices=working_inds)
    if f_lm is not None:
        working_inds = df.ix_inrange('mJyas2', uplim=f_lm, indices=working_inds)

    if bin_size is None:
        g_ixs = working_inds
    # elif bin_size is not None:
    #     bins, bints = get_LCbins(df, bin_size=bin_size, by_mjd=by_mjd, indices=working_inds)
    #     g_ixs = []
        
    #     for i in range(len(bins)):
    #         bin_i = bins[i]
    #         df.calcaverage_sigmacutloop('mJyas2', indices=bin_i, noisecol='mJyas2_err', percentile_cut_firstiteration=70, Nsigma=3.0, verbose=0)
    #         g_ixs.append(df.statparams['ix_good'])

    #     g_ixs = np.concatenate(g_ixs)

    # median_filter = medfilt(df.t.loc[working_inds,'mJyas2'])
    # diff_med = abs(df.t.loc[working_inds,'mJyas2'] - median_filter)
    # thresh_inds = np.where(diff_med < 2.0)[0]
    # g_ixs = working_inds[thresh_inds]
        
    return(g_ixs)



def flag_inds(df, f_lm=None, ferr_lm=None, bad_expnums=None, bin_size=None, by_mjd=False):
    lc_ids = np.unique(df.t['ID'])
    g_ixs = [get_LCinds(df, lc_ID=i, f_lm=f_lm, ferr_lm=ferr_lm, bad_expnums=bad_expnums, bin_size=bin_size, by_mjd=by_mjd) for i in lc_ids]

    df.t['g_ixs_flag'] = False
    
    for i in np.concatenate(g_ixs):
        df.t.loc[i,'g_ixs_flag'] = True

    return('Updated LC dataframe -- flagged working indices')
        


def plot_lcs(df, lc_ID=None, plot_orig=False, plt_xl=None):

    lc_ids = np.unique(df.t['ID'])
    
    if lc_ID is None:
        for n, i in enumerate(lc_ids):
            print(i)
            
            if plot_orig is True:
                i_inds = np.where(df.t['ID'].eq(i))[0]
                plt.errorbar(df.t.loc[i_inds,'mjd'], df.t.loc[i_inds,'mJyas2'], yerr=df.t.loc[i_inds,'mJyas2_err'], fmt='o', ecolor='black', color='blue')

            inds_sig = np.where(df.t['ID'].eq(i) & df.t['g_ixs_flag'].eq(True))[0]
            plt.errorbar(df.t.loc[inds_sig,'mjd'], df.t.loc[inds_sig,'mJyas2'], yerr=df.t.loc[inds_sig,'mJyas2_err'], fmt='o', ecolor='black', color='orange')
            
            plt.axhline(y=0, linestyle='--', color='black')
    
            if plt_xl is not None:
                plt.axvline(x=plt_xl)
    
            plt.xlabel('MJD')
            plt.ylabel('$\mu$Jy/as$^2$')
            plt.show()
            
    elif lc_ID is not None:
        print(lc_ID)
        
        if plot_orig is True:
            i_inds = np.where(df.t['ID'].eq(lc_ID))[0]
            plt.errorbar(df.t.loc[i_inds,'mjd'], df.t.loc[i_inds,'mJyas2'], yerr=df.t.loc[i_inds,'mJyas2_err'], fmt='o', ecolor='black', color='blue')
        
        inds_sig = np.where(df.t['ID'].eq(lc_ID) & df.t['g_ixs_flag'].eq(True))[0]
        plt.errorbar(df.t.loc[inds_sig,'mjd'], df.t.loc[inds_sig,'mJyas2'], yerr=df.t.loc[inds_sig,'mJyas2_err'], fmt='o', ecolor='black', color='orange')
        
        plt.axhline(y=0, linestyle='--', color='black')

        if plt_xl is not None:
            plt.axvline(x=plt_xl)

        plt.xlabel('MJD')
        plt.ylabel('$\mu$Jy/as$^2$')

    return('Done plotting')



def diff(params, f_tmp, f_tmp_err, f_obs, f_obs_err):
    """calculate the difference between two light curves"""
    a, b = params
    f_obs_new = a * f_obs + b
    err = np.sqrt(f_tmp_err**2+f_obs_err**2)
    chi2 = (f_obs_new - f_tmp)**2 / err**2
    
    return np.sum(chi2)


def find_t_peak(t, f, ferr, t_range=None, metric=None):

    if metric is not None:
        kernel = ExpSquaredKernel(metric=metric)
        gp = george.GP(kernel)
        gp.compute(t, ferr)
    
        new_t = np.linspace(min(t), max(t), 5000)
        mu, cov = gp.predict(f, new_t)
        std = np.sqrt(np.diag(cov))
    elif metric is None:
        new_t = t
        mu = f
    
    if t_range is None:
        t_peak = new_t[np.argmax(mu)]
    else:
        t_min = t_range[0]
        t_max = t_range[1]
        new_mu = mu[(new_t<t_max) & (new_t>t_min)]
        new_t2 = new_t[(new_t<t_max) & (new_t>t_min)]
        t_peak = new_t2[np.argmax(new_mu)]
    
    return t_peak


def find_t_shift(t1, f1, f1_err, t2, f2, f2_err, t_range=None, metric=None):
    t_peak1 = find_t_peak(t1, f1, f1_err, t_range=t_range, metric=metric)
    t_peak2 = find_t_peak(t2, f2, f2_err, t_range=t_range, metric=metric)
    
    return t_peak2 - t_peak1


def get_tshift_scale_offset(t1, f1, f1_err, t2, f2, f2_err, t_range=None, t_peaks=None, tshift_d=None, metric=None):
    """get optimized scale factor tshift, a, b
    t1, f1: template 
    t2, f2: f2_new = a * f2 + b
    """

    if tshift_d is None:
        if t_peaks is None:
            if t_range is None:
                tshift = find_t_shift(t1, f1, f1_err, t2, f2, f2_err, metric=metric)
            elif t_range is not None:
                tshift = find_t_shift(t1, f1, f1_err, t2, f2, f2_err, t_range=t_range, metric=metric)
        elif t_peaks is not None:
            tshift = t_peaks[1] - t_peaks[0]
    elif tshift_d is not None:
        tshift = tshift_d
    
    t2_shifted = t2 - tshift

    tmin = max(t1.min(), t2_shifted.min())
    tmax = min(t1.max(), t2_shifted.max())
    
    t_range = t2_shifted[(t2_shifted>tmin) & ((t2_shifted<=tmax))]
    
    f1_new = splev(t_range, splrep(t1[::2], f1[::2], k=1))
    f1_new_err = splev(t_range, splrep(t1[::2], f1_err[::2], k=1))
    f2_new = f2[(t2_shifted>tmin) & ((t2_shifted<=tmax))]
    f2_new_err = f2_err[(t2_shifted>tmin) & ((t2_shifted<=tmax))]
    
    # get optimzied scale f2_new = a * f2 + b
    guess = (1, 0)
    res = opt.minimize(diff, guess, args=(f1_new, f1_new_err, f2_new, f2_new_err))
    a, b = res['x'][0], res['x'][1]
    
    return(tshift, a, b)



def fit_LCs(df, templ_ID=0, LCmodel=None, t_range=None, t_peaks=None, tshift_d=None, tshift_ls=None, metric=None):
    '''
    Fit light curves to a template
    Template can be another individual LC -- default to fit to first in list
    Or can be a LC model -- after iteration 1 we have a model
    '''
    
    # Template choice
    if LCmodel is None:
        g1_inds = np.where((df.t['ID'].eq(templ_ID)) & (df.t['g_ixs_flag'].eq(True)))[0]
        t1 = df.t.loc[g1_inds,'mjd'].values
        f1 = df.t.loc[g1_inds,'mJyas2'].values
        f1_err = df.t.loc[g1_inds,'mJyas2_err'].values
    elif LCmodel is not None:
        t1 = LCmodel['mjd'].values
        f1 = LCmodel['mJyas2'].values
        f1_err = LCmodel['mJyas2_err'].values
        
    # LCs
    tshifts = []
    a_norms = []
    b_offsets = []

    lc_ids = np.unique(df.t['ID'])
    for i, iid in enumerate(lc_ids):
        g2_inds = np.where((df.t['ID'].eq(iid)) & (df.t['g_ixs_flag'].eq(True)))[0]
        t2 = df.t.loc[g2_inds,'mjd'].values
        f2 = df.t.loc[g2_inds,'mJyas2'].values
        f2_err = df.t.loc[g2_inds,'mJyas2_err'].values
        
        if tshift_ls is not None:
            tshift_d = tshift_ls[i]
        if t_peaks is not None:
            if t_peaks != 'ends':
                t_pks = [t_peaks[-1], t_peaks[i]]
            elif t_peaks == 'ends':
                t_pks = [t1[-1], t2[-1]]
        elif t_peaks is None:
            t_pks = None

        if LCmodel is None:
            tshift, a, b = get_tshift_scale_offset(t1, f1, f1_err, t2, f2, f2_err, t_range=t_range, t_peaks=t_pks, tshift_d=tshift_d, metric=metric)
        elif LCmodel is not None:
            if t_peaks is not None:
                t_pks = t_pks[::-1]
            tshift, a, b = get_tshift_scale_offset(t2, f2, f2_err, t1, f1, f1_err, t_range=t_range, t_peaks=t_pks, tshift_d=tshift_d, metric=metric)
    
        tshifts.append(tshift)
        a_norms.append(a)
        b_offsets.append(b)

    return(lc_ids, tshifts, a_norms, b_offsets)



def plot_LCfits(df, fit_params, templ_ID=0, LCmodel=None):
    
    ## Template choice
    if LCmodel is None:
        g1_inds = np.where(df.t['ID'].eq(templ_ID) & df.t['g_ixs_flag'].eq(True))[0]
        t1 = df.t.loc[g1_inds,'mjd'].values
        f1 = df.t.loc[g1_inds,'mJyas2'].values
        f1_err = df.t.loc[g1_inds,'mJyas2_err'].values
    elif LCmodel is not None:
        t1 = LCmodel['mjd'].values
        f1 = LCmodel['mJyas2'].values
        f1_err = LCmodel['mJyas2_err'].values
 
    ## LCs
    lc_ids, tshifts, a_norms, b_offsets = fit_params
    n_lcs = len(lc_ids)
    fig_l = int(np.round((n_lcs/5) + 0.49))
    fig, axs = plt.subplots(fig_l, 5, figsize=(20, fig_l*4))
    # fig, axs = plt.subplots(fig_l, 5, figsize=(40, fig_l*8))
    axs = axs.flatten()

    for i, iid in enumerate(lc_ids):
        g2_inds = np.where(df.t['ID'].eq(iid) & df.t['g_ixs_flag'].eq(True))[0]
        t2 = df.t.loc[g2_inds,'mjd'].values
        f2 = df.t.loc[g2_inds,'mJyas2'].values
        f2_err = df.t.loc[g2_inds,'mJyas2_err'].values

        tshift =tshifts[i]
        a = a_norms[i]
        b = b_offsets[i]
    
        ax = axs[i]
        ax.plot(t1, f1, label='LC tmpl')
        ax.errorbar(t2, f2, label='LC', fmt='.-')
        
        if LCmodel is None:
            ax.errorbar(t2-tshift, f2*a + b, alpha=.5, label='LC aligned', fmt='.-')
        elif LCmodel is not None:
            # ax.errorbar(t2+tshift, (f2 - b)/a, alpha=.5, label='LC aligned', fmt='.-')
            ax.errorbar(t1-tshift, f1*a + b, alpha=.5, label='Model aligned', fmt='.-', zorder=0)
            
        ax.legend()
        ax.set_title(f'ID={i}')

    return('Plotting LC fits')



def get_combLCs(df, fit_params, LCmodel=None):

    new_t_ls = []
    new_f_ls = []
    new_err_ls = []

    lc_ids, tshifts, a_norms, b_offsets = fit_params
    
    for i, iid in enumerate(lc_ids):
        g_inds = np.where(df.t['ID'].eq(iid) & df.t['g_ixs_flag'].eq(True))[0]
        
        tshift_i = tshifts[i]
        a_i = a_norms[i]
        b_i = b_offsets[i]

        # Define new values -- from fitting
        if LCmodel is None:
            t_new = df.t.loc[g_inds,'mjd'] - tshift_i
            f_new = a_i*df.t.loc[g_inds,'mJyas2'] + b_i
            ferr_new = a_i*df.t.loc[g_inds,'mJyas2_err']
        elif LCmodel is not None:
            t_new = df.t.loc[g_inds,'mjd'] + tshift_i
            f_new = (df.t.loc[g_inds,'mJyas2'] - b_i)/a_i
            ferr_new = (df.t.loc[g_inds,'mJyas2_err'])/a_i

        new_t_ls.append(t_new)
        new_f_ls.append(f_new)
        new_err_ls.append(ferr_new)
        
        df.t.loc[g_inds,'new_mjd'] = t_new
        df.t.loc[g_inds,'new_mJyas2'] = f_new
        df.t.loc[g_inds,'new_mJyas2_err'] = ferr_new

        
    ### Put combined data into a data frame
    tls = np.concatenate(new_t_ls)
    fls = np.concatenate(new_f_ls)
    ferrls = np.concatenate(new_err_ls)
    
    ord_inds = tls.argsort()
    tls = tls[ord_inds]
    fls = fls[ord_inds]
    ferrls = ferrls[ord_inds]
    
    comb_data = {'mjd': tls, 'mJyas2': fls, 'mJyas2_err': ferrls}
    comb = pdastrostatsclass()
    comb.t = comb.t.assign(**comb_data) 
        
    return(comb)



def get_LCfit(comb_df, metric=1e4, indices=None):
    inds = comb_df.getindices(indices=indices)
    x = comb_df.t.loc[inds,'mjd']
    y = comb_df.t.loc[inds,'mJyas2']
    yerr = comb_df.t.loc[inds,'mJyas2_err']
    
    kernel = ExpSquaredKernel(metric=metric)
    gp = george.GP(kernel)
    gp.compute(x, yerr)

    t = np.linspace(min(x), max(x), 5000)
    mu, cov = gp.predict(y, t)
    std = np.sqrt(np.diag(cov))

    return(t, mu, std)

# def get_LCfit(comb_df, noise_std=0.75, indices=None, verbose=0):
#     if indices is None:
#         indices = range(len(comb_df.t))

#     X = comb_df.t.loc[indices,'mjd'].values.reshape(-1,1)
#     y = comb_df.t.loc[indices,'mJyas2'].values
    
#     kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
    
#     gaussian_process = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=9)
#     gaussian_process.fit(X, y)
#     if verbose > 0:
#         gaussian_process.kernel_
#     mean_prediction, std_prediction = gaussian_process.predict(X, return_std=True)

#     return(mean_prediction, std_prediction)
    


def convert_Jy2mag(x_jy, err_jy):
    # x_jy in micro-janskys
    mag = 23.9 - 2.5*np.log10(x_jy)
    if err_jy is not None:
        mag_err = 1.086 * (err_jy/x_jy)    # error of propagation
        return(mag, mag_err)
    else:
        return(mag)



def get_mean_pix(img, x, y):
    """ get mean pixel values over 9 nearby pixels centered at (x, y)
        img: array
        x, y center"""
    mean = (img[y, x] + img[y, x-1] + img[y, x+1]
          + img[y-1,x]+ img[y-1,x-1]+ img[y-1,x+1]
          + img[y+1,x]+ img[y+1,x-1]+ img[y+1,x+1]).astype(np.float32)/9
    return mean