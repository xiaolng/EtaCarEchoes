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
from scipy.signal import find_peaks

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



def get_LCcurves(table, group=None, indices=None):
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



def get_table_inds(table, group):
    inds = []
    for i in range(len(table)):
        group_i = table['group'][i]
        if group_i == group:
            inds.append(i)
            
    return(inds)



def get_LCinds(df, lc_ID, n_sigmas=None, mJyas2_lms=None, bad_expnums=None):
    """
    Return the indices of the table of usable values

    n_sigmas = [3.0, 1.0]  -- n times standard deviation away from median value of mJyas2, and its error
    mJyas2_lms = [20.0,1.0] -- upperlimits for mJyas2, and its error
    """

    id_inds = np.where(df['ID'].eq(lc_ID))[0]
    
    if bad_expnums != None:
        working_inds = []
        for i in id_inds:
            diff_nm = df.loc[i,'fitsfile']
            expnum_se = re.search('\.(\d+)_ooi_',os.path.basename(diff_nm))
            expnum = float(expnum_se.groups()[0])
            if expnum not in bad_expnums:
                working_inds.append(i)
    elif bad_expnums == None:
        working_inds = id_inds
    
    f = df.loc[working_inds,'mJyas2']
    ferr = df.loc[working_inds,'mJyas2_err']
    med_f = np.nanmedian(f)
    std_f = np.nanstd(f)
    med_ferr = np.nanmedian(ferr)
    std_ferr = np.nanstd(ferr)

    if n_sigmas!=None:
        n = n_sigmas[0]
        nerr = n_sigmas[1]
        inds = np.where((ferr < med_ferr + nerr*std_ferr) & (abs(f) < med_f + n*std_f))[0]
    elif mJyas2_lms!=None:
        f_lm = mJyas2_lms[0]
        ferr_lm = mJyas2_lms[1]
        inds = np.where((f < f_lm) & (ferr < ferr_lm))[0]
    else:
        inds = range(len(working_inds))
        
    return(np.array(working_inds)[np.array(inds)])



def plot_lcs(df, n_sigmas=None, mJyas2_lms=None, bad_expnums=None, plt_xl=None):
    lc_ids = np.unique(df['ID'])
    for i in lc_ids:
        print(i)
        i_inds = np.where(df['ID'].eq(i))[0]
        # plt.errorbar(df.loc[i_inds,'mjd'], df.loc[i_inds,'mJyas2'], yerr=df.loc[i_inds,'mJyas2_err'], fmt='o', ecolor='black', color='blue')
        
        inds_sig = get_LCinds(df, lc_ID=i, n_sigmas=n_sigmas, mJyas2_lms=mJyas2_lms, bad_expnums=bad_expnums)
        plt.errorbar(df.loc[inds_sig,'mjd'], df.loc[inds_sig,'mJyas2'], yerr=df.loc[inds_sig,'mJyas2_err'], fmt='--o', ecolor='black', color='orange')
        
        plt.axhline(y=0, linestyle='--', color='black')

        if plt_xl is not None:
            plt.axvline(x=plt_xl)

        plt.xlabel('MJD')
        plt.ylabel('$\mu$Jy/as$^2$')
        
        plt.show()

    return()



def diff(params, f_tmp, f_tmp_err, f_obs, f_obs_err):
    """calculate the difference between two light curves"""
    a, b = params
    f_obs_new = a * f_obs + b
    err = np.sqrt(f_tmp_err**2+f_obs_err**2)
    chi2 = (f_obs_new - f_tmp)**2 / err**2
    
    return np.sum(chi2)



def find_t_peak(t, f, t_range=None):
    if t_range is None:
        t_peak = t[np.argmax(f)]
    else:
        t_min = t_range[0]
        t_max = t_range[1]
        f = f[(t<t_max) & (t>t_min)]
        t = t[(t<t_max) & (t>t_min)]
        t_peak = t[np.argmax(f)]
    
    return t_peak



def find_t_shift(t1, f1, t2, f2, t_range=None):
    t_peak1 = find_t_peak(t1, f1, t_range=t_range)
    t_peak2 = find_t_peak(t2, f2, t_range=t_range)
    
    return t_peak2 - t_peak1



def get_tshift_scale_offset(t1, f1, f1_err, t2, f2, f2_err, t_range=None, t_peaks=None, tshift_d=None):
    """get optimized scale factor tshift, a, b
    t1, f1: template 
    t2, f2: f2_new = a * f2 + b
    """

    if tshift_d is None:
        if t_peaks is None:
            if t_range is None:
                tshift = find_t_shift(t1, f1, t2, f2)
            elif t_range is not None:
                tshift = find_t_shift(t1, f1, t2, f2, t_range=t_range)
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



def fit_LCs(df, templ_ID=0, LCmodel=None, n_sigmas=None, mJyas2_lms=None, bad_expnums=None, t_range=None, t_peaks=None, tshift_d=None, tshift_ls=None):
    '''
    Fit light curves to a template
    Template can be another individual LC -- default to fit to first in list
    Or can be a LC model -- after iteration 1 we have a model
    '''
    
    # Template choice
    if LCmodel is None:
        # id_grps = np.unique(df['ID'])
        # temp_i = np.where(LCid_ls==templ_ID)[0][0]
        g1_inds = get_LCinds(df, lc_ID=templ_ID, n_sigmas=n_sigmas, mJyas2_lms=mJyas2_lms, bad_expnums=bad_expnums)
        t1 = df.loc[g1_inds,'mjd'].values
        f1 = df.loc[g1_inds,'mJyas2'].values
        f1_err = df.loc[g1_inds,'mJyas2_err'].values
    elif LCmodel is not None:
        t1 = LCmodel['mjd'].values
        f1 = LCmodel['mJyas2'].values
        f1_err = LCmodel['mJyas2_err'].values
        
    # LCs
    tshifts = []
    a_norms = []
    b_offsets = []

    lc_ids = np.unique(df['ID'])
    for i, iid in enumerate(lc_ids):
        g2_inds = get_LCinds(df, lc_ID=iid, n_sigmas=n_sigmas, mJyas2_lms=mJyas2_lms, bad_expnums=bad_expnums)
        t2 = df.loc[g2_inds,'mjd'].values
        f2 = df.loc[g2_inds,'mJyas2'].values
        f2_err = df.loc[g2_inds,'mJyas2_err'].values
        
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
            tshift, a, b = get_tshift_scale_offset(t1, f1, f1_err, t2, f2, f2_err, t_range=t_range, t_peaks=t_pks, tshift_d=tshift_d)
        elif LCmodel is not None:
            if t_peaks is not None:
                t_pks = t_pks[::-1]
            tshift, a, b = get_tshift_scale_offset(t2, f2, f2_err, t1, f1, f1_err, t_range=t_range, t_peaks=t_pks, tshift_d=tshift_d)
    
        tshifts.append(tshift)
        a_norms.append(a)
        b_offsets.append(b)

    return(lc_ids, tshifts, a_norms, b_offsets)



def plot_LCfits(df, fit_params, templ_ID=0, LCmodel=None, n_sigmas=None, mJyas2_lms=None, bad_expnums=None):
    
    ## Template choice
    if LCmodel is None:
        g1_inds = get_LCinds(df, lc_ID=templ_ID, n_sigmas=n_sigmas, mJyas2_lms=mJyas2_lms, bad_expnums=bad_expnums)
        t1 = df.loc[g1_inds,'mjd'].values
        f1 = df.loc[g1_inds,'mJyas2'].values
        f1_err = df.loc[g1_inds,'mJyas2_err'].values
    elif LCmodel is not None:
        t1 = LCmodel['mjd'].values
        f1 = LCmodel['mJyas2'].values
        f1_err = LCmodel['mJyas2_err'].values
 
    ## LCs
    lc_ids, tshifts, a_norms, b_offsets = fit_params
    n_lcs = len(lc_ids)
    fig_l = int(np.round((n_lcs/5) + 0.49))
    fig, axs = plt.subplots(fig_l, 5, figsize=(20, fig_l*4))
    axs = axs.flatten()

    for i, iid in enumerate(lc_ids):
        g2_inds = get_LCinds(df, lc_ID=iid, n_sigmas=n_sigmas, mJyas2_lms=mJyas2_lms, bad_expnums=bad_expnums)
        t2 = df.loc[g2_inds,'mjd'].values
        f2 = df.loc[g2_inds,'mJyas2'].values
        f2_err = df.loc[g2_inds,'mJyas2_err'].values

        tshift =tshifts[i]
        a = a_norms[i]
        b = b_offsets[i]
    
        ax = axs[i]
        ax.plot(t1, f1, label='LC tmpl')
        ax.errorbar(t2, f2, label='LC', fmt='.-')
        
        if LCmodel is None:
            ax.errorbar(t2-tshift, f2*a + b, alpha=.5, label='LC aligned', fmt='.-')
        elif LCmodel is not None:
            ax.errorbar(t2+tshift, (f2 - b)/a, alpha=.5, label='LC aligned', fmt='.-')
            # ax.errorbar(t1-tshift, f1*a + b, alpha=.5, label='Model aligned', fmt='.-')
            
        ax.legend()
        ax.set_title(f'ID={i}')

    return()



def get_alignedLCs(df, fit_params, LCmodel=None, n_sigmas=None, mJyas2_lms=None, bad_expnums=None):

    new_t_ls = []
    new_f_ls = []
    new_err_ls = []

    lc_ids, tshifts, a_norms, b_offsets = fit_params
    
    for i, iid in enumerate(lc_ids):
        i_inds = get_LCinds(df, lc_ID=iid, n_sigmas=n_sigmas, mJyas2_lms=mJyas2_lms, bad_expnums=bad_expnums)
        
        tshift_i = tshifts[i]
        a_i = a_norms[i]
        b_i = b_offsets[i]

        # Define new values -- from fitting
        # I may want to test whether I add to all data (not just subset)
        if LCmodel is None:
            t_new = df['mjd'][i_inds] - tshift_i
            f_new = a_i*df['mJyas2'][i_inds] + b_i
            ferr_new = a_i*df['mJyas2_err'][i_inds]
        elif LCmodel is not None:
            t_new = df['mjd'][i_inds] + tshift_i
            f_new = (df['mJyas2'][i_inds] - b_i)/a_i
            ferr_new = (df['mJyas2_err'][i_inds])/a_i

        new_t_ls.append(t_new)
        new_f_ls.append(f_new)
        new_err_ls.append(ferr_new)
        
        df.loc[i_inds,'new_mjd'] = t_new
        df.loc[i_inds,'new_mJyas2'] = f_new
        df.loc[i_inds,'new_mJyas2_err'] = ferr_new

        
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



def get_bins(df, t):
    t_range = np.linspace(min(t),max(t),100)
    skip_inds = []
    bins_t = []
    t_bvals = [t_range[0]]
    for i in range(len(t_range)-1):
        # if i not in skip_inds:
            lo_i = i
            skip_inds.append(i)
            up_i = i+1
            # bin_i = np.where((df.t['mjd']>t_range[lo_i]) & (df.t['mjd']<t_range[up_i]))[0]
            bin_i = df.ix_inrange('mjd', lowlim=t_range[lo_i], uplim=t_range[up_i])
            while len(bin_i)<50 and up_i!=len(t_range)-1:
                skip_inds.append(up_i)
                up_i += 1
                bin_i = df.ix_inrange('mjd', lowlim=t_range[lo_i], uplim=t_range[up_i])
                
            bins_t.append(bin_i)
            t_bvals.append(t_range[up_i])
    return(bins_t, t_bvals)



def get_LCbins(df, bin_sz):

    sort_inds = df.ix_sort_by_cols('mjd')

    bins_inds = []
    
    for i in range(0, len(sort_inds), bin_sz):
        if len(sort_inds) - (i+bin_sz) < bin_sz or len(sort_inds) - i < bin_sz:
            bins_inds.append(sort_inds[i:])
            break
        else:
            bins_inds.append(sort_inds[i:i + bin_sz])

    return(bins_inds)



def get_combinedLC(comb_df, bn_sz = 30):
    
    tmeans = []
    fmeans = []
    fmerrs = []
    stdevs = []
    stdev_errs = []
    chi2s = []
    g_ixs = []
    b_ixs = []
    
    bins = get_LCbins(comb_df,bn_sz)
    
    for i in range(len(bins)):
        bin_i = bins[i]
        comb_df.calcaverage_sigmacutloop('mJyas2', indices=bin_i, noisecol='mJyas2_err', percentile_cut_firstiteration=70, Nsigma=3.0)
        tmeans.append(np.average(comb_df.t.loc[bin_i,'mjd'], weights=1-comb_df.t.loc[bin_i,'mJyas2_err']/max(comb_df.t.loc[bin_i,'mJyas2_err'])))
        fmeans.append(comb_df.statparams['mean'])
        fmerrs.append(comb_df.statparams['mean_err'])
        stdevs.append(comb_df.statparams['stdev'])
        stdev_errs.append(comb_df.statparams['stdev_err'])
        chi2s.append(comb_df.statparams['X2norm'])
        g_ixs.append(comb_df.statparams['ix_good'])
        b_ixs.append(comb_df.statparams['ix_clip'])
    
    
    tmeans = np.array(tmeans)
    fmeans = np.array(fmeans)
    fmerrs = np.array(fmerrs)
    stdevs = np.array(stdevs)
    stdev_errs = np.array(stdev_errs)
    chi2s = np.array(chi2s)

    return(tmeans, fmeans, fmerrs, [stdevs, stdev_errs, chi2s, g_ixs, b_ixs])



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