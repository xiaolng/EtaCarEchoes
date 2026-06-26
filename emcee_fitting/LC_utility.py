"""
Author: Rodrigo Angulo   Email: rangulo1@jhu.edu

Utility python file.
Import all the modules needed. This includes pdastro python file.
Definitions used in the notebooks combining light curves.

"""


import requests
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

import scipy.optimize as opt
from scipy.interpolate import splev, splrep, interp1d

import glob

import emcee
import corner

# Gaussian process for curve fitting
import george
from celerite2 import GaussianProcess
from celerite2 import terms

from pdastro import *



### Get and read in light echo data from websites

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


def get_LCdata(table, group=None, indices=None, excl_inds=None):
    '''
    Read in txt files from html containing LC info
    '''

    if indices is None:
        inds_t = range(len(table))
    elif indices is not None:
        inds_t = indices
        
    if excl_inds is not None:
        inds_t = AnotB(inds_t, excl_inds)
        
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
        df['muJyas2'] = df['Jyas2']*1e6
        df['muJyas2_err'] = df['Jyas2_err']*1e6

        ### Get expnum from fitsfile name
        expnums = [int(re.search('\.(\d+)_ooi_',os.path.basename(diff_nm)).groups()[0]) for diff_nm in df['fitsfile']]
        df['im_expnum'] = expnums

    return(pd.concat(lc_list))


### Read in data and flag indices

def readin_LCdf(filepath):
    lc_data = pd.read_csv(filepath)
    lc_df = pdastrostatsclass()
    lc_df.t = lc_df.t.assign(**lc_data)
    lc_df.t.reset_index(drop=True, inplace=True)
    return(lc_df)


def swap_badband(df, bb_df, bb_expnums):
    g_inds = np.where(df.t['im_expnum'].isin(bb_expnums))[0]
    b_inds = np.where(bb_df.t['im_expnum'].isin(bb_expnums))[0]

    df.t.loc[g_inds,'muJyas2'] = bb_df.t.loc[b_inds,'muJyas2']
    df.t.loc[g_inds,'muJyas2_err'] = bb_df.t.loc[b_inds,'muJyas2_err']

    return('Swapped bad band corrections')


def flag_inds(df, f_lm=None, ferr_lm=None, diffstats=None, bad_expnums=None):

    df.t['g_ixs_flag'] = False
    working_inds = df.getindices()
    
    ## Exclude bad images
    if bad_expnums is not None:
        ginds = np.where(~df.t['im_expnum'].isin(bad_expnums))[0]
        df.t['g_ims_flag'] = False
        df.t.loc[ginds, 'g_ims_flag'] = True
        working_inds = ginds

    ## Exclude nans
    working_inds = df.ix_not_null(colnames='muJyas2', indices=working_inds)

    ## Kick out clear outliers in flux and error
    # sig2noi = abs(df.t.loc[working_inds,'muJyas2'])/df.t.loc[working_inds,'muJyas2_err']
    # working_inds = working_inds[np.where(sig2noi >= 1.0)[0]]
    
    if ferr_lm is not None:
        working_inds = df.ix_inrange('muJyas2_err', uplim=ferr_lm, indices=working_inds)
    if f_lm is not None:
        working_inds = df.ix_inrange('muJyas2', uplim=f_lm, indices=working_inds)

    ## Exclude images based off of chi2, depth, and seeing
    if diffstats is not None:
        temp_inds = np.where((diffstats.t['X2NRM00'] <= 15.0) & (diffstats.t['IM_M5SIGMA'] >= 21.0) & (diffstats.t['IM_FWHM'] <= 7.0))[0]
        gstats_expnums = diffstats.t.loc[temp_inds,'im_expnum'].values
        temp2_inds = np.where(df.t['im_expnum'].isin(gstats_expnums))[0]
        working_inds = AandB(working_inds,temp2_inds)

    ## Exclude duplicates and close observations (same day observations)
     # Some images have two reduction versions giving multiple flux for same mjd -- vx vs v1
    # excl_vers = df.ix_matchregex('fitsfile', '_vx_')
    # working_inds = AnotB(working_inds, excl_vers)

    # keep_inds = []
    # for i in np.unique(df.t['ID']):
    #     tskip = []
    #     tkeep = []
    #     i_inds = np.where(df.t['ID'].eq(i))[0]
    #     id_inds = AandB(working_inds,i_inds)
    #     for i in id_inds:
    #         if i not in tskip:
    #             dup_inds = np.where(abs(df.t.loc[id_inds,'mjd'].values - df.t.loc[i,'mjd']) < 0.1)[0]
    #             if len(dup_inds) > 1:
    #                 grp_inds = id_inds[dup_inds]
    #                 minchi2_i = df.t.loc[grp_inds,'FWHM'].idxmin()
    #                 tskip.extend(grp_inds)
    #                 tkeep.append(minchi2_i)
    #             else:
    #                 tkeep.append(i)
    #     keep_inds.extend(tkeep)
    # working_inds = np.array(keep_inds)

    df.t.loc[working_inds, 'g_ixs_flag'] = True
    
    return('Updated LC dataframe -- flagged working indices')


def sel_LC(df, lc_ID=None, flag='g_ixs', exclude_inds=None, indices=None):
    all_inds = df.getindices(indices=indices)
    if lc_ID is not None:
        if flag is None:
            s_inds = np.where(df.t['ID'].eq(lc_ID))[0]
        elif flag == 'g_ims':
            s_inds = np.where(df.t['ID'].eq(lc_ID) & df.t['g_ims_flag'].eq(True))[0]
        elif flag == 'g_ixs':
            s_inds = np.where(df.t['ID'].eq(lc_ID) & df.t['g_ixs_flag'].eq(True))[0]
        sel_inds = AandB(all_inds, s_inds)
    else:
        sel_inds = all_inds
        
    if exclude_inds is not None:
        sel_inds = AnotB(sel_inds, exclude_inds)

    return(sel_inds)


def get_tferr(df, lc_ID=None, peak_l=False, indices=None):

    inds = sel_LC(df=df, lc_ID=lc_ID, indices=indices)
    t = df.t.loc[inds,'mjd'].values
    f = df.t.loc[inds,'muJyas2'].values
    err = df.t.loc[inds,'muJyas2_err'].values

    if peak_l is True:
        peak_loc = df.t.loc[inds,'peak_loc'].values[0]
    elif peak_l == 'main':
        peak_loc = t[np.argmax(f)]
    elif peak_l == 'plat':
        if lc_ID is not None:
            new_t = t[(t>57300) & (t<58000)]
            new_f = f[(t>57300) & (t<58000)]
        elif lc_ID is None:
            new_t = t[(t>59000) & (t<59700)]
            new_f = f[(t>59000) & (t<59700)]
        peak_loc = new_t[np.argmax(new_f)]
    elif peak_l is None:
        peak_loc = t[-1]
            
    if peak_l is False:
        return(t, f, err)
    else:
        return(t, f, err, peak_loc)


def optimizeGP_george(df, lc_ID, kernel='exps', verbose=False, excl_dups=False, indices=None):

    t_i, f_i, err_i = get_tferr(df=df, lc_ID=lc_ID, indices=indices)

    if excl_dups is True:
        inds_i = sel_LC(df=df, lc_ID=lc_ID, indices=indices)
        tskip = []
        tkeep = []
        for i in inds_i:
            if i not in tskip:
                dup_inds = np.where(abs(df.t.loc[inds_i,'mjd'].values - df.t.loc[i,'mjd']) < 1.0)[0]
                if len(dup_inds) > 1:
                    grp_inds = inds_i[dup_inds]
                    minchi2_i = df.t.loc[grp_inds,'FWHM'].idxmin()
                    tskip.extend(grp_inds)
                    tkeep.append(minchi2_i)
                else:
                    tkeep.append(i)
        t_i = df.t.loc[tkeep,'mjd'].values
        f_i = df.t.loc[tkeep,'muJyas2'].values
        err_i = df.t.loc[tkeep,'muJyas2_err'].values
    
    if kernel=='exps':
        kern = 1.0 * george.kernels.ExpSquaredKernel(1e3)
        # bounds_k = [(np.log(0.1),np.log(1500)),(np.log(20000),np.log(100000))]
    elif kernel=='mater':
        kern = 1.0 * george.kernels.Matern32Kernel(1e3)
        # bounds_k = [(np.log(0.1),np.log(1500)),(np.log(20000),np.log(100000))]
    elif kernel=='mix':
        kern = 1.0 * george.kernels.Matern32Kernel(1e2)+ 1.0 * george.kernels.ExpSquaredKernel(1e3)
        # bounds_k = [(np.log(0.1),np.log(1500)),(np.log(30),np.log(200)), (np.log(0.1),np.log(1500)), (np.log(10000),np.log(100000))]
    bounds_k = None
    
    gp = george.GP(kern, fit_mean=False, fit_white_noise=False)
    
    gp.compute(t_i, err_i)
    def nll(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(f_i)

    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(f_i)
    
    result = opt.minimize(nll, gp.get_parameter_vector(), jac=grad_nll, method="L-BFGS-B", bounds=bounds_k)
    gp.set_parameter_vector(result.x)

    # def nll(p):
    #     gp_ps = p[:-1]
    #     ljitter = p[-1]
        
    #     gp.set_parameter_vector(gp_ps)
    #     gp.compute(t_i, np.sqrt(err_i**2 + np.exp(ljitter)**2))
        
    #     return -gp.log_likelihood(f_i)
    
    # p0 = np.append(gp.get_parameter_vector(), 1e-6)
    # result = opt.minimize(nll, p0, method="L-BFGS-B", bounds=bounds_k)
    # gp.set_parameter_vector(result.x[:-1])
    # gp.compute(t_i, np.sqrt(err_i**2 + np.exp(result.x[-1])**2))
    # if verbose is True:
    #     print(np.exp(result.x[-1]))

    # t_range = np.linspace(min(t_i), max(t_i), 5000)
    # mu, var = gp.predict(f_i, t_range, return_var=True)
    # std = np.sqrt(var)

    # gp_df = pdastrostatsclass()
    # gp_df.t = gp_df.t.assign(**{'mjd':t_range, 'muJyas2':mu, 'muJyas2_err':std})

    return(gp)


def optimizeGP_celer(df, lc_ID=None, indices=None, plot=False, smoother=False):

    t_i, f_i, err_i = get_tferr(df=df, lc_ID=lc_ID, indices=indices)

    def build_gp(params):
        sigma, rho = np.exp(params)
        kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=0.5)

        gp = GaussianProcess(kernel)
        gp.compute(t_i, yerr=err_i)

        return gp

    def neg_log_like(params):
        gp = build_gp(params)
        return -gp.log_likelihood(f_i)

    if smoother is True:
        p0 = np.log([np.std(f_i), 30000])
        bounds0 = [(np.log(1e-3), np.log(10*np.std(f_i))), (np.log(25000), np.log(35000))]
    elif smoother is False:
        p0 = np.log([np.std(f_i), 20000])
        bounds0 = [(np.log(1e-3), np.log(10*np.std(f_i))), (np.log(15000), np.log(25000))]
    
    result = opt.minimize(neg_log_like, p0, method="L-BFGS-B", bounds=bounds0)
    gpr = build_gp(result.x)
    
    t_range = np.linspace(min(t_i), max(t_i), 5000)
    mu, var = gpr.predict(f_i, t_range, return_var=True)
    std = np.sqrt(var)

    gp_df = pdastrostatsclass()
    gp_df.t = gp_df.t.assign(**{'mjd':t_range, 'muJyas2':mu, 'muJyas2_err':std})

    if plot is True:
        plt.errorbar(t_i, f_i, yerr=err_i, fmt='.', ecolor='black', color='blue')
        plt.errorbar(gp_df.t['mjd'], gp_df.t['muJyas2'], yerr=gp_df.t['muJyas2_err'], fmt='.', ecolor='k', color='orange')
        plt.axhline(0, ls='--', color='k')
        plt.show()

    return(gp_df)


def sigmacut_df(df, gpdf, lc_ID=None, sigma=5.0):
    inds_i = sel_LC(df=df, lc_ID=lc_ID)
    t_i, f_i, err_i = get_tferr(df=df, lc_ID=lc_ID)
    
    mf = interp1d(gpdf.t['mjd'], gpdf.t['muJyas2'])
    merr = interp1d(gpdf.t['mjd'], gpdf.t['muJyas2_err'])

    resid = f_i - mf(t_i)
    err_tot = np.sqrt(err_i**2 + merr(t_i)**2)
    sigma_inds = np.where(abs(resid/err_tot) < sigma)[0]
        
    return(inds_i[sigma_inds])


def update_gixs(df, ginds):
    df.t['g_ixs_flag'] = False
    df.t.loc[ginds, 'g_ixs_flag'] = True
    return('Updated g_ixs')


### Define functions for plotting and fitting

def plot_LCs(df, lc_ID=None, plot_orig=False, plt_xl=None, plt_peak=False):

    lc_ids = np.unique(df.t['ID'])
    
    if lc_ID is None:
        for n, i in enumerate(lc_ids):
            print(i)
            
            if plot_orig is True:
                i_inds = sel_LC(df, i, flag=None)
                plt.errorbar(df.t.loc[i_inds,'mjd'], df.t.loc[i_inds,'muJyas2'], yerr=df.t.loc[i_inds,'muJyas2_err'], fmt='o', ecolor='black', color='blue')

            inds_sig = sel_LC(df, i, flag='g_ixs')
            plt.errorbar(df.t.loc[inds_sig,'mjd'], df.t.loc[inds_sig,'muJyas2'], yerr=df.t.loc[inds_sig,'muJyas2_err'], fmt='o', ecolor='black', color='orange')
            
            plt.axhline(y=0, linestyle='--', color='black')
    
            if plt_xl is not None:
                plt.axvline(x=plt_xl)
            if plt_peak is True:
                plt.axvline(x=df.t.loc[inds_sig,'peak_loc'].values[0])
    
            plt.xlabel('MJD')
            plt.ylabel('$\mu$Jy/as$^2$')
            plt.show()
            
    elif lc_ID is not None:
        print(lc_ID)
        
        if plot_orig is True:
            i_inds = sel_LC(df, lc_ID, flag=None)
            plt.errorbar(df.t.loc[i_inds,'mjd'], df.t.loc[i_inds,'muJyas2'], yerr=df.t.loc[i_inds,'muJyas2_err'], fmt='o', ecolor='black', color='blue')

        inds_sig = sel_LC(df, lc_ID, flag='g_ixs')
        plt.errorbar(df.t.loc[inds_sig,'mjd'], df.t.loc[inds_sig,'muJyas2'], yerr=df.t.loc[inds_sig,'muJyas2_err'], fmt='o', ecolor='black', color='orange')
        
        plt.axhline(y=0, linestyle='--', color='black')

        if plt_xl is not None:
            plt.axvline(x=plt_xl)
        if plt_peak is True:
            plt.axvline(x=df.t.loc[inds_sig,'peak_loc'][0])

        plt.xlabel('MJD')
        plt.ylabel('$\mu$Jy/as$^2$')
        plt.show()

    return('Done plotting')



def emcee_fit(df, lc_ID, temp_ID=None, LCmodel=None, peak_l=None, tshift=None, anorm=None):

    if temp_ID is not None:
        t_temp, f_temp, err_temp, peak_temp = get_tferr(df=df, lc_ID=temp_ID, peak_l=peak_l)
        t_sci, f_sci, err_sci, peak_sci = get_tferr(df=df, lc_ID=lc_ID, peak_l=peak_l)
        
        # inds_temp = sel_LC(df, temp_ID, flag='g_ixs')
        # t_temp = df.t.loc[inds_temp,'mjd'].values
        # f_temp = df.t.loc[inds_temp,'muJyas2'].values
        # err_temp = df.t.loc[inds_temp,'muJyas2_err'].values

        # inds_sci = sel_LC(df, lc_ID, flag='g_ixs')
        # t_sci = df.t.loc[inds_sci,'mjd'].values
        # f_sci = df.t.loc[inds_sci,'muJyas2'].values
        # err_sci = df.t.loc[inds_sci,'muJyas2_err'].values

        # if tshift is None:
            # peak_temp = df.t.loc[inds_temp,'peak_loc'].values[0]
            # peak_sci = df.t.loc[inds_sci,'peak_loc'].values[0]
        
    if LCmodel is not None:
        ## switch temp and sci -- fit model to data
        t_temp, f_temp, err_temp, peak_temp = get_tferr(df=df, lc_ID=lc_ID, peak_l=peak_l)
        t_sci, f_sci, err_sci, peak_sci = get_tferr(df=LCmodel, lc_ID=None, peak_l=peak_l)

        # inds_temp = sel_LC(df, lc_ID, flag='g_ixs')
        # t_temp = df.t.loc[inds_temp,'mjd'].values
        # f_temp = df.t.loc[inds_temp,'muJyas2'].values
        # err_temp = df.t.loc[inds_temp,'muJyas2_err'].values

        # t_sci = LCmodel.t['mjd'].values
        # f_sci = LCmodel.t['muJyas2'].values
        # err_sci = LCmodel.t['muJyas2_err'].values

        # if tshift is None:
            # peak_temp = df.t.loc[inds_temp,'peak_loc'].values[0]
            # if peak_l == 'main':
            #     peak_sci = t_sci[np.argmax(f_sci)]
            # elif peak_l == 'plat':
            #     new_t = t_sci[(t_sci>59000) & (t_sci<59700)]
            #     new_f = f_sci[(t_sci>59000) & (t_sci<59700)]
            #     peak_sci = new_t[np.argmax(new_f)]
            # elif peak_l is None:
            #     peak_temp = t_temp[-1]
            #     peak_sci = t_sci[-1]
    
    sci_flux = interp1d(t_sci, f_sci, bounds_error=False, fill_value=np.nan)
    sci_error = interp1d(t_sci, err_sci, bounds_error=False, fill_value=np.nan)

    if tshift is None:
        tshift = peak_sci - peak_temp
        peak_tol = 180.0
    else:
        peak_tol = 1e-3

    if anorm is None:
        anorm = 1.0
        lo_a = 0
        up_a = 10.0
    else:
        anorm_tol = 1e-3
        lo_a = anorm - anorm_tol
        up_a = anorm + anorm_tol
    
    initial = np.array([tshift, anorm, 0])
    
    def log_like(params):
        t0, a, b = params
        
        f2_shifted = sci_flux(t_temp+t0)
        f2_err = sci_error(t_temp+t0)
        mask = np.isfinite(f2_shifted)
        
        f2_new = a * f2_shifted[mask] + b
        f2_new_err = a * f2_err[mask]
    
        resid = f2_new - f_temp[mask]
        err_tot = np.sqrt(f2_new_err**2 + err_temp[mask]**2)
        chi2 = resid**2/err_tot**2
    
        llike = -0.5 * np.sum(chi2 + np.log(2*np.pi*err_tot**2))
        
        return(llike)
    
    def log_prior(params):
        t0, a, b = params
        
        if (tshift-peak_tol < t0 < tshift+peak_tol) and (lo_a < a < up_a) and (-100 < b < 100):
            return 0.0
        else:
            return -np.inf
    
    def log_probability(params):
        lp = log_prior(params)
    
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + log_like(params)

    ndim = 3
    nwalkers = 32
    
    pos = (initial + 1e-4 * np.random.randn(nwalkers, ndim))
    
    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_probability)
    
    sampler.run_mcmc(pos, 5000, progress=False)

    samples = sampler.get_chain(discard=1000, thin=10, flat=True)

    # return([t_temp,f_temp,err_temp], [t_sci,f_sci,err_sci],[trange_temp, mu_temp, std_temp], [trange_sci, mu_sci, std_sci], [sci_flux,sci_error], [temp_flux,temp_error])

    return(samples)


def loop_emcee(df, temp_ID=None, LCmodel=None, peak_l=None, tshifts=None, anorms=None):
    params_results = []
    for i, val in enumerate(np.unique(df.t['ID'])):
        if tshifts is not None: tshift = tshifts[i]
        else: tshift = None
        if anorms is not None: anorm = anorms[i]
        else: anorm = None
        samples_i = emcee_fit(df=df, lc_ID=val, temp_ID=temp_ID, LCmodel=LCmodel, peak_l=peak_l, tshift=tshift, anorm=anorm)
        t0_samps = samples_i[:,0]
        a_samps = samples_i[:,1]
        b_samps = samples_i[:,2]

        t0_qs = np.percentile(t0_samps, [16, 50, 84])
        a_qs = np.percentile(a_samps, [16, 50, 84])
        b_qs = np.percentile(b_samps, [16, 50, 84])

        t0_results = [t0_qs[1], t0_qs[2]-t0_qs[1], t0_qs[1]-t0_qs[0]]
        a_results = [a_qs[1], a_qs[2]-a_qs[1], a_qs[1]-a_qs[0]]
        b_results = [b_qs[1], b_qs[2]-b_qs[1], b_qs[1]-b_qs[0]]
        
        params_results.append([val,t0_results, a_results, b_results])

    return(params_results)


def plot_LCfits(df, fit_params, templ_ID=None, LCmodel=None):
    
    ## Template choice
    if LCmodel is None:
        t1, f1, f1_err = get_tferr(df=df, lc_ID=templ_ID)
        # g1_inds = sel_LC(df, templ_ID, flag='g_ixs')
        # t1 = df.t.loc[g1_inds,'mjd'].values
        # f1 = df.t.loc[g1_inds,'muJyas2'].values
        # f1_err = df.t.loc[g1_inds,'muJyas2_err'].values
    elif LCmodel is not None:
        t1, f1, f1_err = get_tferr(df=LCmodel, lc_ID=None)
        # t1 = LCmodel.t['mjd'].values
        # f1 = LCmodel.t['muJyas2'].values
        # f1_err = LCmodel.t['muJyas2_err'].values
 
    ## LCs
    n_lcs = len(fit_params)
    fig_l = int(np.round((n_lcs/5) + 0.49))
    fig, axs = plt.subplots(fig_l, 5, figsize=(20, fig_l*4))
    axs = axs.flatten()

    for i in range(n_lcs):
        LC_id, t_shift, a_norm, b_offset = fit_params[i]
        
        # g2_inds = sel_LC(df=df, lc_ID=LC_id, flag='g_ixs')
        tshift =t_shift[0]
        a = a_norm[0]
        b = b_offset[0]

        t2, f2, f2_err = get_tferr(df=df, lc_ID=LC_id)
        # t2 = df.t.loc[g2_inds,'mjd'].values
        # f2 = df.t.loc[g2_inds,'muJyas2'].values
        # f2_err = df.t.loc[g2_inds,'muJyas2_err'].values
    
        ax = axs[i]
        ax.errorbar(t2, f2, label='LC', fmt='.-', color='orange')
        
        if LCmodel is None:
            ax.errorbar(t1, f1, label='LC tmpl', color='blue', fmt='.-')
            ax.errorbar(t2-tshift, f2*a + b, alpha=.5, label='LC aligned', fmt='.-', color='green')
        elif LCmodel is not None:
            # ax.errorbar(t2+tshift, (f2 - b)/a, alpha=.5, label='LC aligned', fmt='.-')
            ax.errorbar(t1-tshift, f1*a + b, alpha=.5, label='Model aligned', fmt='.-', zorder=0, color='green')
            
        ax.legend()
        ax.set_title(f'ID={LC_id}')

    return('Plotting LC fits')


### Functions for combining data and fitting segments

def get_combLCs(df, fit_params, LCmodel=None):

    new_t_ls = []
    new_f_ls = []
    new_err_ls = []
    old_inds = []
    
    for i in range(len(fit_params)):
        LC_id, t_shift, a_norm, b_offset = fit_params[i]
        
        g_inds = sel_LC(df, LC_id, flag='g_ixs')
        tshift =t_shift[0]
        a = a_norm[0]
        b = b_offset[0]

        # Define new values -- from fitting
        if LCmodel is None:
            t_new = df.t.loc[g_inds,'mjd'] - tshift
            f_new = a*df.t.loc[g_inds,'muJyas2'] + b
            ferr_new = a*df.t.loc[g_inds,'muJyas2_err']
        elif LCmodel is not None:
            t_new = df.t.loc[g_inds,'mjd'] + tshift
            f_new = (df.t.loc[g_inds,'muJyas2'] - b)/a
            ferr_new = (df.t.loc[g_inds,'muJyas2_err'])/a

        new_t_ls.append(t_new)
        new_f_ls.append(f_new)
        new_err_ls.append(ferr_new)
        old_inds.append(g_inds)
        
        df.t.loc[g_inds,'new_mjd'] = t_new
        df.t.loc[g_inds,'new_muJyas2'] = f_new
        df.t.loc[g_inds,'new_muJyas2_err'] = ferr_new

    ### Put combined data into a data frame
    tls = np.concatenate(new_t_ls)
    fls = np.concatenate(new_f_ls)
    ferrls = np.concatenate(new_err_ls)
    orig_inds = np.concatenate(old_inds)
    
    ord_inds = tls.argsort()
    tls = tls[ord_inds]
    fls = fls[ord_inds]
    ferrls = ferrls[ord_inds]
    orig_inds = orig_inds[ord_inds]
    
    comb_data = {'mjd': tls, 'muJyas2': fls, 'muJyas2_err': ferrls, 'orig_inds': orig_inds}
    comb = pdastrostatsclass()
    comb.t = comb.t.assign(**comb_data) 
        
    return(comb)


def fit_segdata(comb1, comb2, tshift_i, shift_tol=180.0):

    flux2 = interp1d(comb2.t['mjd'].values, comb2.t['muJyas2'].values, bounds_error=False, fill_value=np.nan)
    error2 = interp1d(comb2.t['mjd'].values, comb2.t['muJyas2_err'].values, bounds_error=False, fill_value=np.nan)
    
    initial = np.array([tshift_i, 1, 0])
    
    def log_like(params):
        t0, a, b = params
        
        f2_shifted = flux2(comb1.t['mjd'].values+t0)
        f2_err = error2(comb1.t['mjd'].values+t0)
        mask = np.isfinite(f2_shifted)
        
        f2_new = a * f2_shifted[mask] + b
        f2_new_err = a * f2_err[mask]
    
        resid = f2_new - comb1.t['muJyas2'].values[mask]
        err_tot = np.sqrt(f2_new_err**2 + comb1.t['muJyas2_err'].values[mask]**2)
        chi2 = resid**2/err_tot**2
    
        llike = -0.5 * np.sum(chi2 + np.log(2*np.pi*err_tot**2))
        
        return(llike)
    
    def log_prior(params):
        t0, a, b = params
        
        if (tshift_i-shift_tol < t0 < tshift_i+shift_tol) and (0 < a < 20) and (-60 < b < 60):
            return 0.0
        else:
            return -np.inf
    
    def log_probability(params):
        lp = log_prior(params)
    
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + log_like(params)

    ndim = 3
    nwalkers = 32
    
    pos = (initial + 1e-4 * np.random.randn(nwalkers, ndim))
    
    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_probability)
    
    sampler.run_mcmc(pos, 5000, progress=False)

    samples = sampler.get_chain(discard=1000, thin=10, flat=True)

    t0_samps = samples[:,0]
    a_samps = samples[:,1]
    b_samps = samples[:,2]

    t0_qs = np.percentile(t0_samps, [16, 50, 84])
    a_qs = np.percentile(a_samps, [16, 50, 84])
    b_qs = np.percentile(b_samps, [16, 50, 84])

    t0_results = [t0_qs[1], t0_qs[2]-t0_qs[1], t0_qs[1]-t0_qs[0]]
    a_results = [a_qs[1], a_qs[2]-a_qs[1], a_qs[1]-a_qs[0]]
    b_results = [b_qs[1], b_qs[2]-b_qs[1], b_qs[1]-b_qs[0]]


    return(t0_results, a_results, b_results)


def fit_tail(comb, mc_res, gp_df=None, plot=False):
    new_comb = update_comb(comb, mc_res)
    cb_inds = new_comb.ix_inrange('mjd', lowlim=63500)
    ctail_ave = np.average(new_comb.t.loc[cb_inds,'muJyas2'].values,weights=1/new_comb.t.loc[cb_inds,'muJyas2_err'].values**2)
    
    if gp_df is not None:
        new_gpdf = update_comb(gp_df, mc_res)
        gp_inds = new_gpdf.ix_inrange('mjd', lowlim=63500)
        gtail_ave = np.average(new_gpdf.t.loc[gp_inds,'muJyas2'].values,weights=1/new_gpdf.t.loc[gp_inds,'muJyas2_err'].values**2)
        tail_ave = np.average([ctail_ave,gtail_ave])
        if plot==True:
            print(ctail_ave, gtail_ave, tail_ave)
            plt.scatter(new_comb.t['mjd'], new_comb.t['muJyas2'])
            plt.scatter(new_gpdf.t.loc[gp_inds,'mjd'], new_gpdf.t.loc[gp_inds,'muJyas2'])
            plt.axhline(tail_ave,color='k')
            plt.show()
    else:
        tail_ave=ctail_ave

        if plot==True:
            print(tail_ave)
            plt.scatter(new_comb.t['mjd'], new_comb.t['muJyas2'])
            plt.axhline(tail_ave,color='k')
            plt.show()
        
    return(-1*tail_ave)


def update_comb(comb, mc_res, offset_f=0.0):
    if mc_res is None:
        new_t = comb.t['mjd'].values
        new_f = comb.t['muJyas2'].values + offset_f
        new_err = comb.t['muJyas2_err'].values
    else:
        new_t = comb.t['mjd'].values - mc_res[0][0]
        new_f = mc_res[1][0] * comb.t['muJyas2'].values  + mc_res[2][0] + offset_f
        new_err = mc_res[1][0] * comb.t['muJyas2_err'].values
        
    new_comb = pdastrostatsclass()
    new_comb.t = new_comb.t.assign(**{'mjd':new_t, 'muJyas2':new_f, 'muJyas2_err':new_err})

    return(new_comb)


def comb_comb(comb_ls, mc_res_ls, offset_f):
    new_ls = []
    for i, cb in enumerate(comb_ls):
        new_comb = update_comb(cb, mc_res_ls[i], offset_f=offset_f)
        new_ls.append(new_comb.t)
        
    f_comb = pd.concat(new_ls, ignore_index=True)
    f_comb = f_comb.sort_values('mjd')

    fin_comb = pdastrostatsclass()
    fin_comb.t = fin_comb.t.assign(**f_comb)

    return(fin_comb)



### For simultaneous fitting across filters

def emcee_allfilts(df_ls, lc_ID, temp_ID=None, LCmodels=None, peak_l=None, tshift=None, anorm=None):

    df_g, df_r, df_i, df_z = df_ls
    
    if temp_ID is not None:
        t_temp_g, f_temp_g, err_temp_g = get_tferr(df_g, temp_ID)
        t_sci_g, f_sci_g, err_sci_g = get_tferr(df_g, lc_ID)

        t_temp_r, f_temp_r, err_temp_r = get_tferr(df_r, temp_ID)
        t_sci_r, f_sci_r, err_sci_r = get_tferr(df_r, lc_ID)

        t_temp_i, f_temp_i, err_temp_i, peak_temp_i = get_tferr(df_i, temp_ID, peak_l=peak_l)
        t_sci_i, f_sci_i, err_sci_i, peak_sci_i = get_tferr(df_i, lc_ID, peak_l=peak_l)

        t_temp_z, f_temp_z, err_temp_z = get_tferr(df_z, temp_ID)
        t_sci_z, f_sci_z, err_sci_z = get_tferr(df_z, lc_ID)
        
    if LCmodels is not None:
        LCmodel_g, LCmodel_r, LCmodel_i, LCmodel_z = LCmodels
        ## switch temp and sci -- fit model to data
        t_temp_g, f_temp_g, err_temp_g = get_tferr(df_g, lc_ID)
        t_sci_g, f_sci_g, err_sci_g = get_tferr(LCmodel_g, None)

        t_temp_r, f_temp_r, err_temp_r = get_tferr(df_r, lc_ID)
        t_sci_r, f_sci_r, err_sci_r = get_tferr(LCmodel_r, None)

        t_temp_i, f_temp_i, err_temp_i, peak_temp_i = get_tferr(df_i, lc_ID, peak_l=peak_l)
        t_sci_i, f_sci_i, err_sci_i, peak_sci_i = get_tferr(LCmodel_i, None, peak_l=peak_l)

        t_temp_z, f_temp_z, err_temp_z = get_tferr(df_z, lc_ID)
        t_sci_z, f_sci_z, err_sci_z = get_tferr(LCmodel_z, None)
    
    sci_flux_g = interp1d(t_sci_g, f_sci_g, bounds_error=False, fill_value=np.nan)
    sci_error_g = interp1d(t_sci_g, err_sci_g, bounds_error=False, fill_value=np.nan)

    sci_flux_r = interp1d(t_sci_r, f_sci_r, bounds_error=False, fill_value=np.nan)
    sci_error_r = interp1d(t_sci_r, err_sci_r, bounds_error=False, fill_value=np.nan)

    sci_flux_i = interp1d(t_sci_i, f_sci_i, bounds_error=False, fill_value=np.nan)
    sci_error_i = interp1d(t_sci_i, err_sci_i, bounds_error=False, fill_value=np.nan)

    sci_flux_z = interp1d(t_sci_z, f_sci_z, bounds_error=False, fill_value=np.nan)
    sci_error_z = interp1d(t_sci_z, err_sci_z, bounds_error=False, fill_value=np.nan)

    if tshift is None:
        tshift = peak_sci_i - peak_temp_i
        peak_tol = 180.0
    else:
        peak_tol = 1e-3

    if anorm is None:
        anorm = 1.0
        lo_a = 0
        up_a = 10.0
    else:
        anorm_tol = 1e-3
        lo_a = anorm - anorm_tol
        up_a = anorm + anorm_tol
    
    initial = np.array([tshift, anorm, 0, 0, 0, 0])
    
    def log_like(params):
        t0, a, b_g, b_r, b_i, b_z = params
        
        fshifted_g = sci_flux_g(t_temp_g+t0)
        ferr_g = sci_error_g(t_temp_g+t0)
        mask_g = np.isfinite(fshifted_g)
        fnew_g = a * fshifted_g[mask_g] + b_g
        ferrnew_g = a * ferr_g[mask_g]
        resid_g = fnew_g - f_temp_g[mask_g]
        err_tot_g = np.sqrt(ferrnew_g**2 + err_temp_g[mask_g]**2)
        chi2_g = resid_g**2/err_tot_g**2
        chi2_g = np.sum(chi2_g + np.log(2*np.pi*err_tot_g**2))

        fshifted_r = sci_flux_r(t_temp_r+t0)
        ferr_r = sci_error_r(t_temp_r+t0)
        mask_r = np.isfinite(fshifted_r)
        fnew_r = a * fshifted_r[mask_r] + b_r
        ferrnew_r = a * ferr_r[mask_r]
        resid_r = fnew_r - f_temp_r[mask_r]
        err_tot_r = np.sqrt(ferrnew_r**2 + err_temp_r[mask_r]**2)
        chi2_r = resid_r**2/err_tot_r**2
        chi2_r = np.sum(chi2_r + np.log(2*np.pi*err_tot_r**2))

        fshifted_i = sci_flux_i(t_temp_i+t0)
        ferr_i = sci_error_i(t_temp_i+t0)
        mask_i = np.isfinite(fshifted_i)
        fnew_i = a * fshifted_i[mask_i] + b_i
        ferrnew_i = a * ferr_i[mask_i]
        resid_i = fnew_i - f_temp_i[mask_i]
        err_tot_i = np.sqrt(ferrnew_i**2 + err_temp_i[mask_i]**2)
        chi2_i = resid_i**2/err_tot_i**2
        chi2_i = np.sum(chi2_i + np.log(2*np.pi*err_tot_i**2))

        fshifted_z = sci_flux_z(t_temp_z+t0)
        ferr_z = sci_error_z(t_temp_z+t0)
        mask_z = np.isfinite(fshifted_z)
        fnew_z = a * fshifted_z[mask_z] + b_z
        ferrnew_z = a * ferr_z[mask_z]
        resid_z = fnew_z - f_temp_z[mask_z]
        err_tot_z = np.sqrt(ferrnew_z**2 + err_temp_z[mask_z]**2)
        chi2_z = resid_z**2/err_tot_z**2
        chi2_z = np.sum(chi2_z + np.log(2*np.pi*err_tot_z**2))
    
        llike = -0.5 * (chi2_g + chi2_r + chi2_i + chi2_z)
    
        return(llike)
    
    def log_prior(params):
        t0, a, b_g, b_r, b_i, b_z = params
        
        if (tshift-peak_tol < t0 < tshift+peak_tol) and (lo_a < a < up_a) and (-100 < b_g < 100) and (-100 < b_r < 100) and (-100 < b_i < 100) and (-100 < b_z < 100):
            return 0.0
        else:
            return -np.inf
    
    def log_probability(params):
        lp = log_prior(params)
    
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + log_like(params)

    ndim = 6
    nwalkers = 32
    
    pos = (initial + 1e-4 * np.random.randn(nwalkers, ndim))
    
    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_probability)
    
    sampler.run_mcmc(pos, 5000, progress=False)

    samples = sampler.get_chain(discard=1000, thin=10, flat=True)

    return(samples)


def loop_allemcee(df_ls, temp_ID=None, LCmodels=None, peak_l=None, tshifts=None, anorms=None):
    params_results = []
    for i, val in enumerate(np.unique(df_ls[0].t['ID'])):
        if tshifts is not None: tshift = tshifts[i]
        else: tshift = None
        if anorms is not None: anorm = anorms[i]
        else: anorm = None
        samples_i = emcee_allfilts(df_ls, val, temp_ID=temp_ID, LCmodels=LCmodels, peak_l=peak_l, tshift=tshift, anorm=anorm)
        t0_samps = samples_i[:,0]
        a_samps = samples_i[:,1]
        b_g_samps = samples_i[:,2]
        b_r_samps = samples_i[:,3]
        b_i_samps = samples_i[:,4]
        b_z_samps = samples_i[:,5]

        t0_qs = np.percentile(t0_samps, [16, 50, 84])
        a_qs = np.percentile(a_samps, [16, 50, 84])
        b_g_qs = np.percentile(b_g_samps, [16, 50, 84])
        b_r_qs = np.percentile(b_r_samps, [16, 50, 84])
        b_i_qs = np.percentile(b_i_samps, [16, 50, 84])
        b_z_qs = np.percentile(b_z_samps, [16, 50, 84])

        t0_results = [t0_qs[1], t0_qs[2]-t0_qs[1], t0_qs[1]-t0_qs[0]]
        a_results = [a_qs[1], a_qs[2]-a_qs[1], a_qs[1]-a_qs[0]]
        b_g_results = [b_g_qs[1], b_g_qs[2]-b_g_qs[1], b_g_qs[1]-b_g_qs[0]]
        b_r_results = [b_r_qs[1], b_r_qs[2]-b_r_qs[1], b_r_qs[1]-b_r_qs[0]]
        b_i_results = [b_i_qs[1], b_i_qs[2]-b_i_qs[1], b_i_qs[1]-b_i_qs[0]]
        b_z_results = [b_z_qs[1], b_z_qs[2]-b_z_qs[1], b_z_qs[1]-b_z_qs[0]]
        
        params_results.append([val,t0_results, a_results, b_g_results, b_r_results, b_i_results, b_z_results])

    return(params_results)


def plot_allLCfits(df_ls, fit_params, templ_ID=None, LCmodels=None):

    df_g, df_r, df_i, df_z = df_ls
    
    ## Template choice
    if LCmodels is None:
        t1_g, f1_g, f1err_g = get_tferr(df=df_g, lc_ID=templ_ID)
        t1_r, f1_r, f1err_r = get_tferr(df=df_r, lc_ID=templ_ID)
        t1_i, f1_i, f1err_i = get_tferr(df=df_i, lc_ID=templ_ID)
        t1_z, f1_z, f1err_z = get_tferr(df=df_z, lc_ID=templ_ID)
    elif LCmodels is not None:
        LCmodel_g, LCmodel_r, LCmodel_i, LCmodel_z = LCmodels
        t1_g, f1_g, f1err_g = get_tferr(df=LCmodel_g, lc_ID=None)
        t1_r, f1_r, f1err_r = get_tferr(df=LCmodel_r, lc_ID=None)
        t1_i, f1_i, f1err_i = get_tferr(df=LCmodel_i, lc_ID=None)
        t1_z, f1_z, f1err_z = get_tferr(df=LCmodel_z, lc_ID=None)

    ## LCs
    n_lcs = len(fit_params)*4
    fig_l = int(np.round((n_lcs/4) + 0.49))
    fig, axs = plt.subplots(fig_l, 4, figsize=(20, fig_l*5))
    axs = axs.flatten()

    for i, ii in enumerate(range(0, n_lcs, 4)):
        LC_id, t_shift, a_norm, b_g_offset, b_r_offset, b_i_offset, b_z_offset = fit_params[i]
        
        tshift =t_shift[0]
        a = a_norm[0]
        b_g = b_g_offset[0]
        b_r = b_r_offset[0]
        b_i = b_i_offset[0]
        b_z = b_z_offset[0]

        t2_g, f2_g, f2err_g = get_tferr(df=df_g, lc_ID=LC_id)
        t2_r, f2_r, f2err_r = get_tferr(df=df_r, lc_ID=LC_id)
        t2_i, f2_i, f2err_i = get_tferr(df=df_i, lc_ID=LC_id)
        t2_z, f2_z, f2err_z = get_tferr(df=df_z, lc_ID=LC_id)
    
        ax_g = axs[ii]
        ax_r = axs[ii+1]
        ax_i = axs[ii+2]
        ax_z = axs[ii+3]
        
        ax_g.errorbar(t2_g, f2_g, label='LC', fmt='.-', color='orange')
        ax_r.errorbar(t2_r, f2_r, label='LC', fmt='.-', color='orange')
        ax_i.errorbar(t2_i, f2_i, label='LC', fmt='.-', color='orange')
        ax_z.errorbar(t2_z, f2_z, label='LC', fmt='.-', color='orange')
        
        if LCmodels is None:
            ax_g.errorbar(t1_g, f1_g, label='LC tmpl', color='blue', fmt='.-')
            ax_r.errorbar(t1_r, f1_r, label='LC tmpl', color='blue', fmt='.-')
            ax_i.errorbar(t1_i, f1_i, label='LC tmpl', color='blue', fmt='.-')
            ax_z.errorbar(t1_z, f1_z, label='LC tmpl', color='blue', fmt='.-')
            ax_g.errorbar(t2_g-tshift, f2_g*a + b_g, alpha=.5, label='LC aligned', fmt='.-', color='green')
            ax_r.errorbar(t2_r-tshift, f2_r*a + b_r, alpha=.5, label='LC aligned', fmt='.-', color='green')
            ax_i.errorbar(t2_i-tshift, f2_i*a + b_i, alpha=.5, label='LC aligned', fmt='.-', color='green')
            ax_z.errorbar(t2_z-tshift, f2_z*a + b_z, alpha=.5, label='LC aligned', fmt='.-', color='green')
        elif LCmodels is not None:
            # ax_g.errorbar(t2_g+tshift, (f2_g - b_g)/a, alpha=.5, label='LC aligned', fmt='.-')
            # ax_r.errorbar(t2_r+tshift, (f2_r - b_r)/a, alpha=.5, label='LC aligned', fmt='.-')
            # ax_i.errorbar(t2_i+tshift, (f2_i - b_i)/a, alpha=.5, label='LC aligned', fmt='.-')
            # ax_z.errorbar(t2_z+tshift, (f2_z - b_z)/a, alpha=.5, label='LC aligned', fmt='.-')
            ax_g.errorbar(t1_g-tshift, f1_g*a + b_g, alpha=.5, label='Model aligned', fmt='.-', zorder=0, color='green')
            ax_r.errorbar(t1_r-tshift, f1_r*a + b_r, alpha=.5, label='Model aligned', fmt='.-', zorder=0, color='green')
            ax_i.errorbar(t1_i-tshift, f1_i*a + b_i, alpha=.5, label='Model aligned', fmt='.-', zorder=0, color='green')
            ax_z.errorbar(t1_z-tshift, f1_z*a + b_z, alpha=.5, label='Model aligned', fmt='.-', zorder=0, color='green')
            
        ax_g.legend()
        ax_r.legend()
        ax_i.legend()
        ax_z.legend()
        ax_g.set_title(f'g - ID={LC_id}')
        ax_r.set_title(f'r - ID={LC_id}')
        ax_i.set_title(f'i - ID={LC_id}')
        ax_z.set_title(f'z - ID={LC_id}')

    return('Plotting LC fits')


def get_allcombLCs(df_ls, fit_params, LCmodel=None):

    df_g, df_r, df_i, df_z = df_ls
    
    tnewls_g = []
    fnewls_g = []
    errnewls_g = []
    oinds_g = []

    tnewls_r = []
    fnewls_r = []
    errnewls_r = []
    oinds_r = []

    tnewls_i = []
    fnewls_i = []
    errnewls_i = []
    oinds_i = []

    tnewls_z = []
    fnewls_z = []
    errnewls_z = []
    oinds_z = []
    
    for i in range(len(fit_params)):
        LC_id, t_shift, a_norm, b_g_offset, b_r_offset, b_i_offset, b_z_offset = fit_params[i]

        g_inds = sel_LC(df_g, LC_id, flag='g_ixs')
        r_inds = sel_LC(df_r, LC_id, flag='g_ixs')
        i_inds = sel_LC(df_i, LC_id, flag='g_ixs')
        z_inds = sel_LC(df_z, LC_id, flag='g_ixs')
        
        t_g, f_g, err_g = get_tferr(df_g, LC_id)
        t_r, f_r, err_r = get_tferr(df_r, LC_id)
        t_i, f_i, err_i = get_tferr(df_i, LC_id)
        t_z, f_z, err_z = get_tferr(df_z, LC_id)
        
        tshift =t_shift[0]
        a = a_norm[0]
        b_g = b_g_offset[0]
        b_r = b_r_offset[0]
        b_i = b_i_offset[0]
        b_z = b_z_offset[0]

        # Define new values -- apply fitting parameters
        if LCmodel is None:
            tnew_g = t_g - tshift
            fnew_g = a*f_g + b_g
            errnew_g = a*err_g

            tnew_r = t_r - tshift
            fnew_r = a*f_r + b_r
            errnew_r = a*err_r

            tnew_i = t_i - tshift
            fnew_i = a*f_i + b_i
            errnew_i = a*err_i

            tnew_z = t_z - tshift
            fnew_z = a*f_z + b_z
            errnew_z = a*err_z
        elif LCmodel is not None:
            tnew_g = t_g + tshift
            fnew_g = (f_g - b_g)/a
            errnew_g = err_g/a

            tnew_r = t_r + tshift
            fnew_r = (f_r - b_r)/a
            errnew_r = err_r/a

            tnew_i = t_i + tshift
            fnew_i = (f_i - b_i)/a
            errnew_i = err_i/a

            tnew_z = t_z + tshift
            fnew_z = (f_z - b_z)/a
            errnew_z = err_z/a

        tnewls_g.append(tnew_g)
        fnewls_g.append(fnew_g)
        errnewls_g.append(errnew_g)
        oinds_g.append(g_inds)

        tnewls_r.append(tnew_r)
        fnewls_r.append(fnew_r)
        errnewls_r.append(errnew_r)
        oinds_r.append(r_inds)

        tnewls_i.append(tnew_i)
        fnewls_i.append(fnew_i)
        errnewls_i.append(errnew_i)
        oinds_i.append(i_inds)

        tnewls_z.append(tnew_z)
        fnewls_z.append(fnew_z)
        errnewls_z.append(errnew_z)
        oinds_z.append(z_inds)
        
        df_g.t.loc[g_inds,'new_mjd'] = tnew_g
        df_g.t.loc[g_inds,'new_muJyas2'] = fnew_g
        df_g.t.loc[g_inds,'new_muJyas2_err'] = errnew_g

        df_r.t.loc[r_inds,'new_mjd'] = tnew_r
        df_r.t.loc[r_inds,'new_muJyas2'] = fnew_r
        df_r.t.loc[r_inds,'new_muJyas2_err'] = errnew_r

        df_i.t.loc[i_inds,'new_mjd'] = tnew_i
        df_i.t.loc[i_inds,'new_muJyas2'] = fnew_i
        df_i.t.loc[i_inds,'new_muJyas2_err'] = errnew_i

        df_z.t.loc[z_inds,'new_mjd'] = tnew_z
        df_z.t.loc[z_inds,'new_muJyas2'] = fnew_z
        df_z.t.loc[z_inds,'new_muJyas2_err'] = errnew_z

    ### Put combined data into a data frame
    tls_g = np.concatenate(tnewls_g)
    fls_g = np.concatenate(fnewls_g)
    errls_g = np.concatenate(errnewls_g)
    originds_g = np.concatenate(oinds_g)
    ordinds_g = tls_g.argsort()
    tls_g = tls_g[ordinds_g]
    fls_g = fls_g[ordinds_g]
    errls_g = errls_g[ordinds_g]
    originds_g = originds_g[ordinds_g]
    combdata_g = {'mjd': tls_g, 'muJyas2': fls_g, 'muJyas2_err': errls_g, 'orig_inds': originds_g}
    comb_g = pdastrostatsclass()
    comb_g.t = comb_g.t.assign(**combdata_g)

    tls_r = np.concatenate(tnewls_r)
    fls_r = np.concatenate(fnewls_r)
    errls_r = np.concatenate(errnewls_r)
    originds_r = np.concatenate(oinds_r)
    ordinds_r = tls_r.argsort()
    tls_r = tls_r[ordinds_r]
    fls_r = fls_r[ordinds_r]
    errls_r = errls_r[ordinds_r]
    originds_r = originds_r[ordinds_r]
    combdata_r = {'mjd': tls_r, 'muJyas2': fls_r, 'muJyas2_err': errls_r, 'orig_inds': originds_r}
    comb_r = pdastrostatsclass()
    comb_r.t = comb_r.t.assign(**combdata_r) 

    tls_i = np.concatenate(tnewls_i)
    fls_i = np.concatenate(fnewls_i)
    errls_i = np.concatenate(errnewls_i)
    originds_i = np.concatenate(oinds_i)
    ordinds_i = tls_i.argsort()
    tls_i = tls_i[ordinds_i]
    fls_i = fls_i[ordinds_i]
    errls_i = errls_i[ordinds_i]
    originds_i = originds_i[ordinds_i]
    combdata_i = {'mjd': tls_i, 'muJyas2': fls_i, 'muJyas2_err': errls_i, 'orig_inds': originds_i}
    comb_i = pdastrostatsclass()
    comb_i.t = comb_i.t.assign(**combdata_i)

    tls_z = np.concatenate(tnewls_z)
    fls_z = np.concatenate(fnewls_z)
    errls_z = np.concatenate(errnewls_z)
    originds_z = np.concatenate(oinds_z)
    ordinds_z = tls_z.argsort()
    tls_z = tls_z[ordinds_z]
    fls_z = fls_z[ordinds_z]
    errls_z = errls_z[ordinds_z]
    originds_z = originds_z[ordinds_z]
    combdata_z = {'mjd': tls_z, 'muJyas2': fls_z, 'muJyas2_err': errls_z, 'orig_znds': originds_z}
    comb_z = pdastrostatsclass()
    comb_z.t = comb_z.t.assign(**combdata_z)
        
    return(comb_g, comb_r, comb_i, comb_z)


def fit_allsegdata(comb_ls1, comb_ls2, tshift_i, shift_tol=180.0):

    comb1g, comb1r, comb1i, comb1z = comb_ls1
    comb2g, comb2r, comb2i, comb2z = comb_ls2

    t1_g, f1_g, err1_g = get_tferr(comb1g, None)
    t2_g, f2_g, err2_g = get_tferr(comb2g, None)

    t1_r, f1_r, err1_r = get_tferr(comb1r, None)
    t2_r, f2_r, err2_r = get_tferr(comb2r, None)

    t1_i, f1_i, err1_i = get_tferr(comb1i, None)
    t2_i, f2_i, err2_i = get_tferr(comb2i, None)

    t1_z, f1_z, err1_z = get_tferr(comb1z, None)
    t2_z, f2_z, err2_z = get_tferr(comb2z, None)

    flux2g = interp1d(t2_g, f2_g, bounds_error=False, fill_value=np.nan)
    error2g = interp1d(t2_g, err2_g, bounds_error=False, fill_value=np.nan)

    flux2r = interp1d(t2_r, f2_r, bounds_error=False, fill_value=np.nan)
    error2r = interp1d(t2_r, err2_r, bounds_error=False, fill_value=np.nan)

    flux2i = interp1d(t2_i, f2_i, bounds_error=False, fill_value=np.nan)
    error2i = interp1d(t2_i, err2_i, bounds_error=False, fill_value=np.nan)

    flux2z = interp1d(t2_z, f2_z, bounds_error=False, fill_value=np.nan)
    error2z = interp1d(t2_z, err2_z, bounds_error=False, fill_value=np.nan)
    
    initial = np.array([tshift_i, 1, 0, 0, 0, 0])
    
    def log_like(params):
        t0, a, b_g, b_r, b_i, b_z = params

        fshifted_g = flux2g(t1_g+t0)
        ferr_g = error2g(t1_g+t0)
        mask_g = np.isfinite(fshifted_g)
        fnew_g = a * fshifted_g[mask_g] + b_g
        ferrnew_g = a * ferr_g[mask_g]
        resid_g = fnew_g - f1_g[mask_g]
        err_tot_g = np.sqrt(ferrnew_g**2 + err1_g[mask_g]**2)
        chi2_g = resid_g**2/err_tot_g**2
        chi2_g = np.sum(chi2_g + np.log(2*np.pi*err_tot_g**2))

        fshifted_r = flux2r(t1_r+t0)
        ferr_r = error2r(t1_r+t0)
        mask_r = np.isfinite(fshifted_r)
        fnew_r = a * fshifted_r[mask_r] + b_r
        ferrnew_r = a * ferr_r[mask_r]
        resid_r = fnew_r - f1_r[mask_r]
        err_tot_r = np.sqrt(ferrnew_r**2 + err1_r[mask_r]**2)
        chi2_r = resid_r**2/err_tot_r**2
        chi2_r = np.sum(chi2_r + np.log(2*np.pi*err_tot_r**2))

        fshifted_i = flux2i(t1_i+t0)
        ferr_i = error2i(t1_i+t0)
        mask_i = np.isfinite(fshifted_i)
        fnew_i = a * fshifted_i[mask_i] + b_i
        ferrnew_i = a * ferr_i[mask_i]
        resid_i = fnew_i - f1_i[mask_i]
        err_tot_i = np.sqrt(ferrnew_i**2 + err1_i[mask_i]**2)
        chi2_i = resid_i**2/err_tot_i**2
        chi2_i = np.sum(chi2_i + np.log(2*np.pi*err_tot_i**2))

        fshifted_z = flux2z(t1_z+t0)
        ferr_z = error2z(t1_z+t0)
        mask_z = np.isfinite(fshifted_z)
        fnew_z = a * fshifted_z[mask_z] + b_z
        ferrnew_z = a * ferr_z[mask_z]
        resid_z = fnew_z - f1_z[mask_z]
        err_tot_z = np.sqrt(ferrnew_z**2 + err1_z[mask_z]**2)
        chi2_z = resid_z**2/err_tot_z**2
        chi2_z = np.sum(chi2_z + np.log(2*np.pi*err_tot_z**2))
    
        llike = -0.5 * (chi2_g + chi2_r + chi2_i + chi2_z)
        
        return(llike)
    
    def log_prior(params):
        t0, a, b_g, b_r, b_i, b_z = params
        
        if (tshift_i-shift_tol < t0 < tshift_i+shift_tol) and (0 < a < 20) and (-60 < b_g < 60) and (-60 < b_r < 60) and (-60 < b_i < 60) and (-60 < b_z < 60):
            return 0.0
        else:
            return -np.inf
    
    def log_probability(params):
        lp = log_prior(params)
    
        if not np.isfinite(lp):
            return -np.inf
        else:
            return lp + log_like(params)

    ndim = 6
    nwalkers = 32
    
    pos = (initial + 1e-4 * np.random.randn(nwalkers, ndim))
    
    sampler = emcee.EnsembleSampler(nwalkers,ndim,log_probability)
    
    sampler.run_mcmc(pos, 5000, progress=False)

    samples = sampler.get_chain(discard=1000, thin=10, flat=True)

    t0_samps = samples[:,0]
    a_samps = samples[:,1]
    b_g_samps = samples[:,2]
    b_r_samps = samples[:,3]
    b_i_samps = samples[:,4]
    b_z_samps = samples[:,5]

    t0_qs = np.percentile(t0_samps, [16, 50, 84])
    a_qs = np.percentile(a_samps, [16, 50, 84])
    b_g_qs = np.percentile(b_g_samps, [16, 50, 84])
    b_r_qs = np.percentile(b_r_samps, [16, 50, 84])
    b_i_qs = np.percentile(b_i_samps, [16, 50, 84])
    b_z_qs = np.percentile(b_z_samps, [16, 50, 84])

    t0_results = [t0_qs[1], t0_qs[2]-t0_qs[1], t0_qs[1]-t0_qs[0]]
    a_results = [a_qs[1], a_qs[2]-a_qs[1], a_qs[1]-a_qs[0]]
    b_g_results = [b_g_qs[1], b_g_qs[2]-b_g_qs[1], b_g_qs[1]-b_g_qs[0]]
    b_r_results = [b_r_qs[1], b_r_qs[2]-b_r_qs[1], b_r_qs[1]-b_r_qs[0]]
    b_i_results = [b_i_qs[1], b_i_qs[2]-b_i_qs[1], b_i_qs[1]-b_i_qs[0]]
    b_z_results = [b_z_qs[1], b_z_qs[2]-b_z_qs[1], b_z_qs[1]-b_z_qs[0]]

    return(t0_results, a_results, b_g_results, b_r_results, b_i_results, b_z_results)


def fit_alltail(comb_ls, mc_res, plot=False):
    newcomb_g, newcomb_r, newcomb_i, newcomb_z = update_allcomb(comb_ls, mc_res)

    cbinds_g = newcomb_g.ix_inrange('mjd', lowlim=63500)
    tailave_g = np.average(newcomb_g.t.loc[cbinds_g,'muJyas2'].values,weights=1/newcomb_g.t.loc[cbinds_g,'muJyas2_err'].values**2)

    cbinds_r = newcomb_r.ix_inrange('mjd', lowlim=63500)
    tailave_r = np.average(newcomb_r.t.loc[cbinds_r,'muJyas2'].values,weights=1/newcomb_r.t.loc[cbinds_r,'muJyas2_err'].values**2)

    cbinds_i = newcomb_i.ix_inrange('mjd', lowlim=63500)
    tailave_i = np.average(newcomb_i.t.loc[cbinds_i,'muJyas2'].values,weights=1/newcomb_i.t.loc[cbinds_i,'muJyas2_err'].values**2)

    cbinds_z = newcomb_z.ix_inrange('mjd', lowlim=63500)
    tailave_z = np.average(newcomb_z.t.loc[cbinds_z,'muJyas2'].values,weights=1/newcomb_z.t.loc[cbinds_z,'muJyas2_err'].values**2)


    if plot==True:
        plt.scatter(newcomb_g.t['mjd'], newcomb_g.t['muJyas2'])
        plt.axhline(tailave_g,color='k', label=f'{tailave_g:3f}')
        plt.legend()
        plt.show()

        plt.scatter(newcomb_r.t['mjd'], newcomb_r.t['muJyas2'])
        plt.axhline(tailave_r,color='k', label=f'{tailave_r:3f}')
        plt.legend()
        plt.show()

        plt.scatter(newcomb_i.t['mjd'], newcomb_i.t['muJyas2'])
        plt.axhline(tailave_i,color='k', label=f'{tailave_i:3f}')
        plt.legend()
        plt.show()

        plt.scatter(newcomb_z.t['mjd'], newcomb_z.t['muJyas2'])
        plt.axhline(tailave_z,color='k', label=f'{tailave_z:3f}')
        plt.legend()
        plt.show()
        
    return(-1*tailave_g, -1*tailave_r, -1*tailave_i, -1*tailave_z)


def update_allcomb(comb_ls, mc_res, offset_fs=0.0):
    comb_g, comb_r, comb_i, comb_z = comb_ls
    if mc_res is None:
        t0, a, b_g, b_r, b_i, b_z = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    else:
        tshift, anorm, boff_g, boff_r, boff_i, boff_z = mc_res
    
        t0 =tshift[0]
        a = anorm[0]
        b_g = boff_g[0]
        b_r = boff_r[0]
        b_i = boff_i[0]
        b_z = boff_z[0]

    if offset_fs == 0.0:
        offset_g, offset_r, offset_i, offset_z = [0.0]*4
    else:
        offset_g, offset_r, offset_i, offset_z = offset_fs

    tnew_g = comb_g.t['mjd'].values - t0
    fnew_g = a * comb_g.t['muJyas2'].values + b_g + offset_g
    errnew_g = a * comb_g.t['muJyas2_err'].values

    tnew_r = comb_r.t['mjd'].values - t0
    fnew_r = a * comb_r.t['muJyas2'].values + b_r + offset_r
    errnew_r = a * comb_r.t['muJyas2_err'].values

    tnew_i = comb_i.t['mjd'].values - t0
    fnew_i = a * comb_i.t['muJyas2'].values + b_i + offset_i
    errnew_i = a * comb_i.t['muJyas2_err'].values

    tnew_z = comb_z.t['mjd'].values - t0
    fnew_z = a * comb_z.t['muJyas2'].values + b_z + offset_z
    errnew_z = a * comb_z.t['muJyas2_err'].values
        
    newcomb_g = pdastrostatsclass()
    newcomb_g.t = newcomb_g.t.assign(**{'mjd':tnew_g, 'muJyas2':fnew_g, 'muJyas2_err':errnew_g})

    newcomb_r = pdastrostatsclass()
    newcomb_r.t = newcomb_r.t.assign(**{'mjd':tnew_r, 'muJyas2':fnew_r, 'muJyas2_err':errnew_r})

    newcomb_i = pdastrostatsclass()
    newcomb_i.t = newcomb_i.t.assign(**{'mjd':tnew_i, 'muJyas2':fnew_i, 'muJyas2_err':errnew_i})

    newcomb_z = pdastrostatsclass()
    newcomb_z.t = newcomb_z.t.assign(**{'mjd':tnew_z, 'muJyas2':fnew_z, 'muJyas2_err':errnew_z})

    return(newcomb_g, newcomb_r, newcomb_i, newcomb_z)


def comb_allcomb(combls_ls, mcres_ls, offset_fs):

    combls1 = update_allcomb(combls_ls[0], mcres_ls[0], offset_fs=offset_fs)
    combls2 = update_allcomb(combls_ls[1], mcres_ls[1], offset_fs=offset_fs)
    combls3 = update_allcomb(combls_ls[2], mcres_ls[2], offset_fs=offset_fs)
    combls4 = update_allcomb(combls_ls[3], mcres_ls[3], offset_fs=offset_fs)
    
    combls_g = [combls1[0].t, combls2[0].t, combls3[0].t, combls4[0].t]
    combls_r = [combls1[1].t, combls2[1].t, combls3[1].t, combls4[1].t]
    combls_i = [combls1[2].t, combls2[2].t, combls3[2].t, combls4[2].t]
    combls_z = [combls1[3].t, combls2[3].t, combls3[3].t, combls4[3].t]
        
    fcomb_g = pd.concat(combls_g, ignore_index=True)
    fcomb_g = fcomb_g.sort_values('mjd')
    fincomb_g = pdastrostatsclass()
    fincomb_g.t = fincomb_g.t.assign(**fcomb_g)

    fcomb_r = pd.concat(combls_r, ignore_index=True)
    fcomb_r = fcomb_r.sort_values('mjd')
    fincomb_r = pdastrostatsclass()
    fincomb_r.t = fincomb_r.t.assign(**fcomb_r)

    fcomb_i = pd.concat(combls_i, ignore_index=True)
    fcomb_i = fcomb_i.sort_values('mjd')
    fincomb_i = pdastrostatsclass()
    fincomb_i.t = fincomb_i.t.assign(**fcomb_i)

    fcomb_z = pd.concat(combls_z, ignore_index=True)
    fcomb_z = fcomb_z.sort_values('mjd')
    fincomb_z = pdastrostatsclass()
    fincomb_z.t = fincomb_z.t.assign(**fcomb_z)

    return(fincomb_g, fincomb_r, fincomb_i, fincomb_z)


def set_peak(findat_ls, finmod_ls):
    
    t_i, f_i, err_i, peak_i = get_tferr(finmod_ls[2], peak_l='main')

    for dfd in findat_ls:
        dfd.t['phase'] = dfd.t['mjd'] - peak_i
    for dfm in finmod_ls:
        dfm.t['phase'] = dfm.t['mjd'] - peak_i
        
    return('Updated phase')


def convert_muJy2mag(df):

    t, f, err = get_tferr(df)
    wf = np.where(f <= 0, 1e-2, f)     # Replace negative values with small numbers (assuming those are non-detections)
    
    df.t['ABmag'] = 23.9 - (2.5 * np.log10(wf))     # AB magnitude conversion
    df.t['ABmag_err'] = 1.086 * (err/wf)            # AB mag error of propagation
    
    return('AB mag calculated')
