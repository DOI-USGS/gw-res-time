
# coding: utf-8

# In[ ]:

__author__ = 'Jeff Starn'
# get_ipython().magic('matplotlib notebook')

# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('png', 'pdf')
# from IPython.display import Image
# from IPython.display import Math
# from ipywidgets import interact, Dropdown
# from IPython.display import display

import os
import shelve
import pickle
from scipy.optimize import OptimizeWarning
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import pandas as pd
import geopandas as gp
import datetime as dt
import gdal, osr
gdal.UseExceptions()
import flopy as fp

import scipy.stats as ss
import scipy.optimize as so
from scipy import interpolate as si
from sklearn import linear_model

import sklearn.cluster as cluster
import sklearn.linear_model as linear_model
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import learning_curve, validation_curve, ShuffleSplit, cross_val_score, cross_val_predict, KFold 
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.pipeline import make_pipeline


# In[ ]:

homes = ['../Models']
fig_dir = '../Figures'

mfpth = '../executables/MODFLOW-NWT_1.0.9/bin/MODFLOW-NWT_64.exe'
mp_exe_name = '../executables/modpath.6_0/bin/mp6.exe' 

mf_start_date_str = '01/01/1900' 
mp_release_date_str = '01/01/2020' 

num_surf_layers = 3
num_depth_groups = 5

por = 0.20


# In[ ]:

dir_list = []
pfile = 'fit_dict_res_all_layers.pickle'

for home in homes:
    if os.path.exists(home):
        for dirpath, dirnames, filenames in os.walk(home):
            for f in filenames:
                if f == pfile:
                    pth = os.path.join(dirpath, f)
                    model = os.path.normpath(pth).split(os.sep)[2]
                    dir_list.append(pth)
                    mto = os.path.getmtime(pth)
                    d = dt.datetime.fromtimestamp(mto)
                    mt = dt.datetime.strftime(d, '%b. %d, %Y at %I:%M %p')
                    print('{:50s} {}'.format(model, mt))
print('\n    {} models read'.format(len(dir_list)))


# In[ ]:

shp = gp.read_file('../Data/Watersheds/watersheds.shp')

shp['model_num'] = shp.model_num.astype(np.int32())


# In[ ]:

arr = np.zeros((len(dir_list), 12))

mod_list = list()
dist = ss.weibull_min

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 8,
        'sans-serif' : 'Arial'}
plt.rc('font', **font)

fig, axs = plt.subplots(5, 6, sharex=True, sharey=True, figsize=(7.4, 8), 
                    gridspec_kw={'wspace':0.05, 'hspace':0.05})
axs = axs.ravel()

uname = 'uni_{}'.format(dist.name)
aname = 'add_{}'.format(dist.name)
iname = 'imp_{}'.format(dist.name)

diff_step = 0.0001
sigma = 1.
kwargs = {'ftol':1E-07, 'xtol':1E-07, 'max_nfev':5000, 'loss':'soft_l1', 'diff_step':diff_step, 'f_scale':1.0}

for i, j in shp.iterrows():
    model = j.model_name
    pth = [item for item in dir_list if model in item][0]
    plot = j.model_num - 1
    
    mod_list.append(model)
    model_ws = os.path.dirname(pth)
    
    # read the dictionary with fitted parameters and particle travel times 
    with open(pth, 'rb') as f:
        pick = pickle.load(f)
    x = pick[model]['tt']['rt']
    y = pick[model]['tt']['rt_cdf']
    uni_cdf = pick[model]['cdf'][uname]
    add_cdf = pick[model]['cdf'][aname]

    # read the text file with mean age information (tau)
    src = os.path.join(model_ws, 'tau.txt')
    with open(src) as f:
        lines = f.readlines()
    items = [item.split()[6] for item in lines]
    tau_glacial = np.float(items [0])
    tau_bedrock = np.float(items [1])
    tau_total = np.float(items [2])
    items = [item.split()[4] for item in lines]
    frac = np.float(items[4])

    # compute the exponential distribution using the calculated mean age (tau)
    expon = ss.expon(np.exp(x.min()), tau_glacial)
    exy = expon.cdf(np.exp(x))
    
    # start plotting stuff ...
    ax = axs[plot]
    # ... the calculated exponential distribution CDF
    ax.semilogx(np.exp(x) / por, exy, label='Calculated', color='k', lw=1, ls='dotted')

    # ... the CDFs from particle and parametric distributions
    try:
        ax.semilogx(np.exp(x), y, label='Particle', lw=5, color='r', alpha=0.4)
        ax.semilogx(np.exp(x), uni_cdf, label='Univariate', color='k', lw=1.5, ls='solid')
        ax.semilogx(np.exp(x), add_cdf, label='Explicitly mixed', color='b', lw=1, ls='dashed')
        ax.set_xlim(1, 10000)
    except:
        pass
    
    subtitle = '{}'.format(plot + 1, fontsize=8)
    ax.text(2, 0.85, subtitle, fontsize=8)

    # calculate error measures by interpolating the particle travel times and 
    # calculated exponential distribution to the same x coordinates
    newx = np.logspace(np.log10(np.exp(x)).min(), np.log10(np.exp(x)).max(), 1000)
    newy1 = np.interp(newx, np.exp(x) / por, exy)
    newy2 = np.interp(newx, np.exp(x), uni_cdf)
    
    resid = newy1 - newy2
    err = (resid).T.dot(resid)
    rmse = np.sqrt(err / newx.shape[0])

    pars = pick[model]['par'][uname]
    arr[plot, 0] = plot + 1
    arr[plot, 1:4] = pars
    arr[plot, 4] = rmse
    arr[plot, 5] = frac
    arr[plot, 6] = tau_glacial
    arr[plot, 7] = tau_bedrock
    arr[plot, 8] = tau_total
    
    def efunc(x, scale):
        return 1 - np.exp(-scale * x)
    def wfunc(x, shape, scale):
        return 1 - np.exp(-(x / scale) ** shape)

    bnds = (0.000001, 0.2)  
    pe_opt, pe_cov = so.curve_fit(efunc, newx, newy1, 
                                   bounds = bnds, method='dogbox', sigma=sigma, **kwargs)
    bnds = ((0.00001, 0.00001), (1000, 3000))    
    pw_opt, pw_cov = so.curve_fit(wfunc, newx, newy2, 
                                   bounds = bnds, method='trf', sigma=sigma, **kwargs)
    
    arr[plot, 9] = pe_opt
    arr[plot, 10:] = pw_opt    

ax.legend(bbox_to_anchor=[0.45, -0.4], ncol=4, frameon=False)
fig.set_tight_layout(True)

fig.text(0.05, 0.55, 'Cumulative frequency', rotation=90)
fig.text(0.4, 0.05, 'Residence time / porosity in years')
form_list = ['png', 'pdf', 'tif']
for form in form_list:
    line = 'Paper #2017WR021531-f05.{}'.format(form)
    fig_name = os.path.join(fig_dir, line)
    plt.savefig(fig_name, dpi=600)
plt.close()


# In[ ]:

cols = ['model_num', 'shape', 'location', 'scale', 'rmse', 'frac', 'tau_glacial', 
        'tau_bedrock', 'tau_total', 'exp_scale', 'wei_shape', 'wei_scale']
df = pd.DataFrame(arr, columns=cols, index=mod_list)
df['model_num'] = df.model_num.astype(np.int32).astype('str')
df.loc[:, 'inv_exp_scale'] = 1 / df['exp_scale']
df.loc[:, 'tau_div_por'] = df['tau_glacial'] / por
dffile_name = os.path.join(fig_dir, 'weib_and_exp_fit.csv')
df.to_csv(dffile_name)

