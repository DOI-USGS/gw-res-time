
# coding: utf-8

# In[ ]:

# get_ipython().magic('matplotlib notebook')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


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
file_name = 'fit_dict_res_all_layers.pickle'

i = 0

for home in homes:
    if os.path.exists(home):
        for dirpath, dirnames, filenames in os.walk(home):
            for f in filenames:
                if file_name in f:
                    mod = os.path.splitext(f)[0]
                    mod_list.append(mod)
                    dir_list.append(dirpath)
                    i += 1
print('    {} models read'.format(i))



# ### Form the data sets

# The files **```fit_params_```** *model name* **```.csv```** contain the results of a unimodal Weibull fit to the ensemble of residence times in the glacial sediments. 

# In[ ]:

par_dict = {}

for pth in dir_list:
    src = os.path.join(pth, file_name)
    if os.path.exists(src):
        with open(src, 'rb') as f:
            data = pickle.load(f)
        err = {}
        for well, df in data.items():
            err[well] = df['err']
            par_dict.update(err)


# This cells parses the 3 Weibull parameter values into a data frame.  The parameters are
# * **shape** (= 1 is an exponential distribution, which is the theoretical simple aquifer distribution)
# * **loc** (first arrival of a particle or smallest residence time)
# * **scale** (the spread of the data; may correspond to the range of sediment thickness)

# In[ ]:

df = pd.DataFrame(par_dict)

dfm = df.T
dfm['best'] = dfm.idxmin(axis=1)
dfm['worst'] = dfm.iloc[:, :9].idxmax(axis=1)
dfm.loc['N', :] = dfm.count(axis=0)

dst = os.path.join(fig_dir, 'rtd_error_all_layers_summary.csv')
dfm.to_csv(dst)


# In[ ]:

dfm.best.value_counts()


# In[ ]:

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 8,
        'sans-serif' : 'Arial'}

plt.rc('font', **font)


# In[ ]:

i = dfm.columns


# In[ ]:

newcol = dict(zip(i[:9],['Ge', 'Ie', 'We', 'Gi',
       'Ii', 'Wi', 'G', 'I',
       'W'] ))
dfm = dfm.rename(columns=newcol)


# In[ ]:

fig, ax = plt.subplots(1, 1, figsize=(4.6, 4.6))
boxprops = {'showfliers':True, 'showcaps':False, 'color':'black'}
cax = dfm.loc[:, :'W'].plot( kind='box', logy=False, ax=ax, **boxprops)
ymin, ymax = cax.get_ylim()
ax.set_ylabel('Root mean squared error', fontsize=12)
ax.set_ylim(0, 0.06)
ax.set_xticklabels(['G', 'I', 'W', 'G', 'I', 'W', 'G', 'I', 'W'])
ax.text(1, 0.057, 'a', fontsize=8, fontdict={'weight':'bold'})
ax.text(4, 0.057, 'b', fontsize=8, fontdict={'weight':'bold'})
ax.text(7, 0.057, 'c', fontsize=8, fontdict={'weight':'bold'})
ax.tick_params(axis='x', length=0)
ax.axvline(3.5, color='k', lw=1, ls='dashed')
ax.axvline(6.5, color='k', lw=1, ls='dashed')
fig.set_tight_layout(True)

form_list = ['png', 'pdf', 'tif']
for form in form_list:
    line = 'Paper #2017WR021531-f04.{}'.format(form)
    fig_name = os.path.join(fig_dir, line)
    plt.savefig(fig_name, dpi=300)
plt.close()

# In[ ]:



