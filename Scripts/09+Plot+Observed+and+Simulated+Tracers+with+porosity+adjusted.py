
# coding: utf-8

# In[ ]:

# get_ipython().magic('matplotlib inline')
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gp
import scipy.stats as ss


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

dir_list = []
mod_list = []
i = 0

for home in homes:
    if os.path.exists(home):
        for dirpath, dirnames, filenames in os.walk(home):
            for f in filenames:
                if os.path.splitext(f)[-1] == '.nam':
                    mod = os.path.splitext(f)[0]
                    mod_list.append(mod)
                    dir_list.append(dirpath)
                    i += 1
print('    {} models read'.format(i))


# In[ ]:

samp_df = pd.DataFrame()
for model_ws in dir_list:
    model = os.path.normpath(model_ws).split(os.sep)[2]
    src = os.path.join(model_ws, 'sample_dict_wells.csv')
    if os.path.exists(src):
        data = pd.read_csv(src)
        data['model'] = model
        samp_df = samp_df.append(data)
dst = os.path.join(fig_dir, 'master_sample_fit.csv')
samp_df.to_csv(dst)


# In[ ]:

samp_df.loc[samp_df.Trit < 0.01, 'Trit'] = 0.01
samp_df['3H residual'] = (samp_df.Trit - samp_df.calc_3H_) 
samp_df['3H relative residual'] = (samp_df.Trit - samp_df.calc_3H_) / samp_df.Trit


# In[ ]:

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 8,
        'sans-serif' : 'Arial'}

plt.rc('font', **font)

TritFit_df = pd.Series()
df = samp_df.loc[:, ['model', 'por', 'Trit', 'NetworkTyp', 'SuCode', 'calc_3H_']]

df_sub = df.copy()

mslope, minter, lslope, uslope = ss.theilslopes(y=df_sub['calc_3H_'].values, 
                                                x=df_sub.Trit.values, alpha=0.95)
ktau, kalpha = ss.kendalltau(df_sub.Trit.values, df_sub['calc_3H_'].values, nan_policy='omit')

TritFit_df.loc['Theil-Sen slope'] = mslope
TritFit_df.loc["Kendall's tau"] = ktau
TritFit_df.loc["Kendall's tau alpha"] = kalpha
TritFit_df.loc['N'] = df_sub.shape[0]

fig, ax = plt.subplots(1, 1, figsize=(3.8, 4));
ax.plot(df_sub.Trit.values, df_sub['calc_3H_'].values, marker='o', 
        ls='none', ms=3, alpha=0.3, mec='k', mfc='k');

quantiles = 4
df_sub.loc[:, 'quant'], bins = pd.qcut(df_sub.Trit, quantiles, retbins=True)
xplot = df_sub.loc[:, ['Trit', 'quant']].groupby('quant').agg([min, np.median, max, np.std])
yplot = df_sub.loc[:, ['calc_3H_', 'quant']].groupby('quant').agg([min, np.median, max, np.std])
xmed_ar = xplot.loc[:, ('Trit', 'median')]
ymed_ar = yplot.loc[:, ('calc_3H_', 'median')]
ystd_ar = yplot.loc[:, ('calc_3H_', 'std')]

yerr = pd.DataFrame()
yerr['yhi'] = (yplot.loc[:, ('calc_3H_', 'median')] - yplot.loc[:, ('calc_3H_', 'max')]).abs()
yerr['ylo'] = (yplot.loc[:, ('calc_3H_', 'median')] - yplot.loc[:, ('calc_3H_', 'min')]).abs()
yerr = yerr.T
yerr = yerr[::-1]

x = np.arange(25)
y = mslope * x + minter
ylo = lslope * x + minter
yup = uslope * x + minter
ax.plot(xmed_ar, ymed_ar, marker='^', ls='none', ms=8, mfc='r', mec='r')
ax.plot((0, df_sub.Trit.max()), (0, df_sub.Trit.max()), color='k', linestyle='dashed', alpha=0.50)

ax.set_xlabel('Measured tritium concentration');
ax.set_ylabel('Calculated tritium concentration');
fig.set_tight_layout(True)
form_list = ['png', 'pdf', 'tif']
for form in form_list:
    line = 'Paper #2017WR021531-f7.{}'.format(form)
    fig_name = os.path.join(fig_dir, line)
    plt.savefig(fig_name, dpi=300)
    
plt.close()

dst = os.path.join(fig_dir, 'trit_fit_df.csv')
TritFit_df.to_csv(dst)


# In[ ]:

TritFit_df

