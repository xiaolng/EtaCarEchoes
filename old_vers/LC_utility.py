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

import pickle


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



def get_tshift_scale_offset(t1, f1, f1_err, t2, f2, f2_err, t_range=None, t_peaks=None):
    """get optimized scale factor tshift, a, b
    t1, f1: template 
    t2, f2: f2_new = a * f2 + b
    """

    if t_peaks is None:
        if t_range is None:
            tshift = find_t_shift(t1, f1, t2, f2)
        elif t_range is not None:
            tshift = find_t_shift(t1, f1, t2, f2, t_range=t_range)
    elif t_peaks is not None:
        tshift = t_peaks[1] - t_peaks[0]
    
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
    
    return tshift, a, b



def lc_indices(df, n_sigmas=None, mJyas2_lms=None, bad_expnums=None):
    """
    Return the indices of the table of usable values

    n_sigmas = [3.0, 1.0]  -- n times standard deviation away from median value of mJyas2, and its error
    mJyas2_lms = [20.0,1.0] -- upperlimits for mJyas2, and its error
    """

    if bad_expnums != None:
        working_inds = []
        for i in range(len(df)):
            diff_nm = df.loc[i,'fitsfile']
            expnum_se = re.search('\.(\d+)_ooi_',os.path.basename(diff_nm))
            expnum = float(expnum_se.groups()[0])
            if expnum not in bad_expnums:
                working_inds.append(i)
    elif bad_expnums == None:
        working_inds = range(len(df))
    
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



def convert_Jy2mag(x_jy):
    # x_jy in micro-janskys
    mag = 23.9 - 2.5*np.log10(x_jy)
    return(np.round(mag, 2))

def convert_Jy2mag(x_jy, x_jy_err=None):
    # x_jy in micro-janskys
    mag = 23.9 - 2.5*np.log10(x_jy)
    if x_jy_err is not None:
        mag_err = 1.0857 * ( x_jy_err / x_jy )
        return mag, mag_err
    return mag



def get_mean_pix(img, x, y):
    """ get mean pixel values over 9 nearby pixels centered at (x, y)
        img: array
        x, y center"""
    mean = (img[y, x] + img[y, x-1] + img[y, x+1]
          + img[y-1,x]+ img[y-1,x-1]+ img[y-1,x+1]
          + img[y+1,x]+ img[y+1,x-1]+ img[y+1,x+1]).astype(np.float32)/9
    return mean