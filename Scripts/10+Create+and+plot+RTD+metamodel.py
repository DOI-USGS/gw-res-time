
# coding: utf-8

# In[ ]:

# get_ipython().magic('matplotlib notebook')
import os
import pandas as pd
import numpy as np
import geopandas as gp
import matplotlib.pyplot as plt
import scipy.stats as ss
import sklearn.cluster as cluster
import sklearn.linear_model as linear_model
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import learning_curve, validation_curve, ShuffleSplit, cross_val_score, cross_val_predict, KFold 
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.pipeline import make_pipeline


# ## Can the parametric distribution be regionalized?

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

src = '../Data/Watersheds/watersheds.shp'
shp = gp.read_file(src)
shp['model'] = shp.model_name.str.lower()


# Read explanatory variables derived from MODFLOW simulation

# In[ ]:

src = os.path.join(fig_dir, 'master_modflow_table.csv')
area_summary = pd.read_csv(src)


# Read response variables from summary of all models fits

# In[ ]:

src = os.path.join(fig_dir, 'weib_and_exp_fit.csv')
parameter_summary = pd.read_csv(src, index_col=0)


# Select one parametric RTD model and merge the response and explanatory variables

# In[ ]:

model_list = ['add', 'imp', 'uni']
dist_list = ['invgauss', 'gamma', 'weibull_min']

ml = 2
dl = 2

dist = '{}_{}'.format(model_list[ml], dist_list[dl])


# In[ ]:

indy = ['Assabet', 'Board2', 'CONN', 'IA_Willow_02', 'IL_West_Fork_Mazon_03',
       'Kala2', 'MN_Talcot_Lake-Des_Moines_04', 
       'MO_U16_Mississippi', 'MO_Wildcat_02',  
       'NE_Upper_Logan_Creek_02', 'NY_Ramapo_02', 'NorthSkunk',
       'OH_Three_Brothers_Creek-Grand_05', 'Oconto', 'Racoon', 'SD_Willow_07',
       'SugarCreek', 'Tomorrow', 'Upper_fox', 
       'WI_Waumaundee_04', 'Whitedam3', 'huc_07030001_domain',
       'huc_07050002_domain', 'huc_07070001_domain', 'huc_07070003_domain',
       'huc_07080205_domain', 'huc_07090006_domain', 'huc_07120003_domain',
       'huc_07120004_domain', 'huc_07130009_domain']

par_sum = parameter_summary.loc[indy, :]
par_sum['model'] = par_sum.index.str.lower()
print('number of model = {}'.format(par_sum.shape[0]))


# In[ ]:

data = par_sum.merge(area_summary, left_on='model', right_on='model', how='inner')
data = data.merge(shp, left_on='model', right_on='model')

data.dropna(axis=0, how='all', inplace=True)
data.dropna(axis=1, how='all', inplace=True)
data.set_index('model', drop=True, inplace=True)
print('number of models = {}'.format(data.shape[0]))

data['wei_location'] = np.exp(data.location)


# In[ ]:

feacols = ['inv_exp_scale', 
           'rech_vol', 'rech_frac', 
           'Kf', 'Kc', 'Kb', 
           'vani', 
           'fKf_vol', 'fKc_vol', 'fKlake_vol', 'fKb_vol', 
           'drn_dens_area']
labcols = ['wei_shape', 'wei_location', 'wei_scale'] 

sel_ind = data.rech_frac <= 1
features = data.loc[sel_ind, feacols]
labels = data.loc[sel_ind, labcols]
names = feacols

targets = labels.columns


# In[ ]:

shp_used_4_meta = gp.GeoDataFrame(data, geometry='geometry', crs=shp.crs)


# In[ ]:

dst = os.path.join(fig_dir, 'shp_used_4_meta.shp')


# In[ ]:

shp_used_4_meta.to_file(dst)


# In[ ]:

poly = False
# poly = True

if poly:
    pp = preprocessing.PolynomialFeatures()
    features = pp.fit_transform(features)
    names = pp.get_feature_names(names)


# In[ ]:

#Standardizes labels
yscaler = preprocessing.StandardScaler().fit(labels)
std_Y = yscaler.transform(labels)
# std_Y = labels

#Standardizes features
xscaler = preprocessing.StandardScaler().fit(features)
std_X = xscaler.transform(features)
# std_X = features


# In[ ]:

test_size = 0.25
num_perm = 1000
num_alpha = 50

cv = ShuffleSplit(n_splits=num_perm, test_size=test_size, random_state=2909591)

lrp = linear_model.MultiTaskLasso(max_iter=10000, normalize=False)
param_range = np.logspace(-3, 1, num_alpha)

path = np.zeros((std_Y.shape[1], num_alpha, std_X.shape[1]))
train_scores_mean = np.zeros((num_alpha))
train_scores_std = np.zeros((num_alpha))

for i, v in enumerate(param_range):
    lrp.set_params(alpha=v)
    lrp.fit(std_X, std_Y)
    path[:, i, :] = lrp.coef_
    cvs = cross_val_score(lrp, std_X, std_Y, cv=cv, scoring='neg_mean_squared_error')
    train_scores_mean[i] = cvs.mean()
    train_scores_std[i] = cvs.std()
    
se = train_scores_std / np.sqrt(num_perm)
best = np.argmax(train_scores_mean)
best_alpha = param_range[best]
best_mse = train_scores_mean[best]
best_se = se[best]
simple_mse = best_mse - best_se
simple = np.argmin((train_scores_mean <= simple_mse)[::-1])
simple_alpha = param_range[::-1][simple]


# In[ ]:

lrp.set_params(alpha=simple_alpha)
lrp.fit(std_X, std_Y)
predicted = lrp.predict(std_X)
r2 = lrp.score(std_X, std_Y)
coef_df = pd.DataFrame(lrp.coef_.T, columns=labcols, index=names)

num_active_pars = (np.abs(path[0, :, :] ) > 0).sum(axis=1)


# In[ ]:

print(r2)
print(best_alpha)
print(best_mse)
print(simple_mse)
print(simple_alpha)


# In[ ]:

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 8,
        'sans-serif' : 'Arial'}

plt.rc('font', **font)

fig, ax = plt.subplots(1, 1, figsize=(3.8, 4))
lw = 1

ax.set_xlabel(r'$\alpha_r $')

ax.set_ylabel(r"Negative mean squared error")
ax.semilogx(param_range, train_scores_mean, 
             color='black', lw=lw, zorder=5)
ax.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.4,
                 color="darkorange", lw=lw, label='$\overline {MSE}\pm\sigma$ across folds', 
                zorder=1)

ax.errorbar(param_range, train_scores_mean, yerr=se, linestyle='None',
            elinewidth=lw, ecolor='k', capthick=lw, capsize=lw, 
            zorder=4, label=r'$\overline {MSE}\pm 1$ standard error')

ax.axvline(best_alpha, linestyle='--',
       label=r'Best $\alpha$ = {:2.4f}'.format(best_alpha), zorder=3)
ax.axvline(simple_alpha, linestyle=':',
       label=r'Simple $\alpha$ = {:2.4f}'.format(simple_alpha), zorder=2)
ax.set_ylim(-2.5, 0.5)
ax.invert_xaxis();

fig.text(0, 0.95, 'a', fontdict={'weight':'bold'})
fig.set_tight_layout(True)
form_list = ['png', 'pdf', 'tif']
for form in form_list:
    line = 'Paper #2017WR021531-f9a.{}'.format(form)
    fig_name = os.path.join(fig_dir, line)
    plt.savefig(fig_name, dpi=300)
plt.close()

# In[ ]:

greekcols = [r'$\gamma$', r'$\mu$', r'$\alpha$', ]
newcolumns = dict(zip(labcols, zip(greekcols)))

greeknames = [r'$\tau^{-1}$', r'$R_{bedrock}$', r'$R_{bedrock} / R_{surface}$', r'$K_{fine}$',
          r'$K_{coarse}$', r'$K_{bedrock}$', r'$K_{h}/K_{v}$', r'$fv_{fine}$', r'$fv_{coarse}$',
          r'$fv_{lakes}$', r'$fv_{bedrock}$', r'$fa_{drains}$']
newnames = dict(zip(feacols, greeknames))

coef_df = coef_df.rename(index=newnames, columns=newcolumns)


# In[ ]:

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 8,
        'sans-serif' : 'Arial'}

plt.rc('font', **font)

num_plots = std_Y.shape[1]
fig, ax_grid = plt.subplots(1, num_plots, figsize=(7.4, 4.6), sharex=True, sharey=True, 
                            gridspec_kw={'wspace':0.0, 'hspace':0.05})
ax = ax_grid.ravel()
for p in range(num_plots):
    ax[p].scatter(std_Y[:, p], predicted[:, p], label=greekcols[p], c='k', s=5)
    ax[p].plot([std_Y[:, p].min(), std_Y[:, p].max()], 
               [std_Y[:, p].min(), std_Y[:, p].max()], 'k--', lw=1, alpha=0.50)
    if p == 1:
        ax[p].set_xlabel( 'Standardized Weibull parameters fit to particle distributions')
    leg = ax[p].legend(handlelength=0, handletextpad=0, frameon=False, loc=2)
    for item in leg.legendHandles:
        item.set_visible(False)

fig.text(0, 0.95, 'b', fontdict={'weight':'bold'})
ax[0].set_ylabel('Standardized Weibull parameters predicted by LASSO')
fig.set_tight_layout(True)
form_list = ['png', 'pdf', 'tif']
for form in form_list:
    line = 'Paper #2017WR021531-f9b.{}'.format(form)
    fig_name = os.path.join(fig_dir, line)
    plt.savefig(fig_name, dpi=300)
plt.close()

# In[ ]:

coef_df_bar = coef_df.loc[(coef_df[r'$\alpha$'] != 0), :]

fig, ax = plt.subplots(1, 1, figsize=(3.8, 4.6))
coef_df_bar.T.plot(kind='barh', stacked=True, ax=ax,
      colormap=plt.cm.spectral, legend=True,
      table=False,
      sort_columns=True)
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0*2.8, box.width, box.height*.8])
ax.set_xlabel('Value of standardized coefficient')
ax.legend(loc='center left', bbox_to_anchor=(0.0, -0.3), ncol=3, frameon=False)
fig.text(0, 0.95, 'c', fontdict={'weight':'bold'})
form_list = ['png', 'pdf', 'tif']
for form in form_list:
    line = 'Paper #2017WR021531-f9c.{}'.format(form)
    fig_name = os.path.join(fig_dir, line)
    plt.savefig(fig_name, dpi=300)
plt.close()

# In[ ]:

print("Attributes Ordered by How Early They Enter the Model")

path_df = pd.DataFrame(path[0,::-1,:], index=param_range[::-1], columns=names)
tmp = path_df != 0
tmp.idxmax(axis=0).sort_values(ascending=False)


# In[ ]:

a = pd.DataFrame(lrp.coef_.T, index=names, columns=targets)
b = a.abs()
b[b == 0] = np.nan
b.dropna(axis=0, how='all', inplace=True)


# In[ ]:

b.sort_values('wei_shape', ascending=False)


# In[ ]:

b.sort_values('wei_scale', ascending=False)


# In[ ]:

b.sort_values('wei_location', ascending=False)

