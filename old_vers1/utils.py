import numpy as np
import pandas as pd

from scipy.interpolate import splrep, splev
import scipy.optimize as opt


def get_median(T_obs, F_obs, Nbins=80):
    '''get median '''
    t_master_bin = np.linspace(T_obs.min(), T_obs.max(), Nbins)
    
    t_master = []
    f_master = []
    for i in range(len(t_master_bin)-1):
        
        fi = F_obs[(T_obs>t_master_bin[i]) & (T_obs<t_master_bin[i+1])]
        
        #print(len(fi))
        if len(fi)!=0:
            t_master.append(t_master_bin[i])
            f_master.append(np.median(fi))
    return np.array(t_master), np.array(f_master)


def get_imgarr(fitsfile, idx=0, return_header=False):
    """get image array from fits
    Parameters:
        fitsfile: FITS file
        idx: hdul index in FITS
        return_header: boolean, return FITS header
    """
    from astropy.io import fits
    hdul = fits.open(fitsfile)
    item = hdul[idx]
    if item.is_image:
        header = item.header
        # invert y to be same as jpg
        #imgarr = item.data[::-1, :] 
        imgarr = item.data
        hdul.close()
        if return_header:
            return imgarr, header
        else:
            return imgarr
    else:
        print('not image file') 
        

def get_vmin_vmax(img, low=5, high=95):
    """return the percentile value at low and high"""
    img_nonan = img[~np.isnan(img)]
    vmin = np.percentile(img_nonan, low)
    vmax = np.percentile(img_nonan, high)
    return vmin, vmax
        

def get_mean_pix(img, x, y):
    """ get mean pixel values over 9 nearby pixels centered at (x, y)
        img: array
        x, y center"""
    mean = (img[y, x] + img[y, x-1] + img[y, x+1]
          + img[y-1,x]+ img[y-1,x-1]+ img[y-1,x+1]
          + img[y+1,x]+ img[y+1,x-1]+ img[y+1,x+1]).astype(np.float32)/9
    return mean


# used for align light curves

def find_t_peak(t, f):
    t_peak = t[np.argmax(f)]

    return t_peak

def find_t_shift(t1, f1, t2, f2):
    t_peak1 = find_t_peak(t1, f1)
    t_peak2 = find_t_peak(t2, f2)
    
    return t_peak2 - t_peak1

def find_offset_b(f1, f2):
    """find offset from peak"""
    return np.max(f1) - np.max(f2)


def diff(params, f_tmp, f_obs):
    """calculate the difference between two light curves"""
    a, b = params
    f_obs_new = a * f_obs + b
    diff = (f_obs_new - f_tmp)**2 
    
    return np.sum(diff)

def diff_scale(params, f_tmp, f_obs):
    """only scale a"""
    a, = params
    b = find_offset_b(f_tmp, f_obs)
    f_obs_new = a * (f_obs + b)
    diff = (f_obs_new - f_tmp)**2
    
    return np.sum(diff)


def get_tshift_scale_offset(t1, f1, t2, f2):
    """get optimized scale factor tshift, a, b
    t1, f1: template 
    t2, f2: f2_new = a * f2 + b
    """
    tshift = find_t_shift(t1, f1, t2, f2)
    
    t2_shifted = t2 - tshift

    tmin = max(t1.min(), t2.min())
    tmax = min(t2_shifted.max(), t2_shifted.max())
    
    #t_range = np.arange(tmin, tmax, 200)
    t_range = t2_shifted[(t2_shifted>tmin) & ((t2_shifted<=tmax))]
    
    f1_new = splev(t_range, splrep(t1[::2], f1[::2], k=1, ))
    #f2_new = splev(t_range, splrep(t2_shifted, f2_scaled, k=1, ))
    f2_new = f2[(t2_shifted>tmin) & ((t2_shifted<=tmax))]
    
    # get optimzied scale f2_new = a * f2 + b
    guess = (1, 0)
    res = opt.minimize(diff, guess, args=(f1_new, f2_new))
    a, b = res['x'][0], res['x'][1]
    #
    #guess = (1, )
    
    #res = opt.minimize(diff_scale, guess, args=(f1_new, f2_new))
    #a, = res['x'][0],
    
    #b = find_offset_b(f1_new, f2_new)
    
    return tshift, a, b


# -------------------------

def load_lc_from_url(tmpID, posID, field=54):
    '''load light curve from url 
    https://stsci-transients.stsci.edu/eta/etalc/results
    return: DataFrame'''
    #txtfile = f'https://stsci-transients.stsci.edu/eta/etalc/results/rod_test/ec0915/54/ec0915_54_poly1pos_i_tmpl{tmpID}_ID{posID}_lc.txt'
    if field==54:
        txtfile =f'https://stsci-transients.stsci.edu/eta/etalc/results/rod_test/ec0915i_poly1/ec0915/54/ec0915_54_poly1pos_i_tmpl{tmpID}_ID{posID}_lc.txt'
    elif field==57:
        txtfile=f'https://stsci-transients.stsci.edu/eta/etalc/results/rod_test/ec0915/57/ec0915_57_poly1pos_i_tmpl{tmpID}_ID{posID}_lc.txt'
    
    #df = pd.read_csv(txtfile, delim_whitespace=True)
    df = pd.read_csv(txtfile, sep='\s+')
    
    # remove nan 
    df = df[df['Jyas2'].notna()]
    
    df['Jyas2'] *= 1e6
    df['Jyas2_err'] *= 1e6

    #idx_drop = df['ID'][df['Jyas2_err']>=1].index.to_list()
    #idx_drop = [58, 59, 60, 61, 180, 199]
    #print('drop', idx_drop)
    #df.drop(idx_drop, inplace=True)   
    # reset index
    #df.reset_index() 
    df['good'] = df['Jyas2_err']<=1
    
    return df


def get_mean_pix(img, x, y):
    """ get mean pixel values over 9 nearby pixels
        img: array
        x, y center"""
    mean = (img[y, x] + img[y, x-1] + img[y, x+1]
          + img[y-1,x]+ img[y-1,x-1]+ img[y-1,x+1]
          + img[y+1,x]+ img[y+1,x-1]+ img[y+1,x+1]).astype(np.float32)/9
    return mean
    
        