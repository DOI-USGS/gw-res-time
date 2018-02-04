
# coding: utf-8

# In[ ]:

# get_ipython().magic('matplotlib inline')
import os
import numpy as np
import flopy as fp
import pandas as pd


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


# Read individual `tau.txt` files and make a summary data frame. This data frame gets augmented in the following cell with MODFLOW information. 

# In[ ]:

tau_dict = {}

# loop through each model area
for pth in dir_list:
    model = os.path.normpath(pth).split(os.path.sep)[2]    

    tau = []
    src = os.path.join(pth, 'tau.txt')
    with open(src) as f:
        line = f.readlines()
    tau.append(np.float32(line[0].split(' ')[6]))
    tau.append(np.float32(line[1].split(' ')[6]))
    tau.append(np.float32(line[2].split(' ')[6]))
    tau.append(np.float32(line[3].split(' ')[4]))
    tau.append(np.float32(line[4].split(' ')[4]))
    
    tau_dict[model] = tau

columns = ['tau_glac', 'tau_bed', 'tau_tot', 'rech_vol', 'rech_frac']

tau = pd.DataFrame(tau_dict, index=columns).T

dst = os.path.join(fig_dir, 'tau_summary.csv')
tau.to_csv(dst)


# In[ ]:

for model_ws in dir_list:
    model = os.path.normpath(model_ws).split(os.sep)[2]
    model_name = '{}.nam'.format(model)

    mf = fp.modflow.Modflow.load(model_name, model_ws=model_ws, version='mfnwt', check=False)
    dis = mf.get_package('DIS')
    bas = mf.get_package('BAS6')
    upw = mf.get_package('UPW')
    drn = mf.get_package('DRN')

    src = os.path.join(model_ws, 'par.csv')
    par = pd.read_csv(src)
    if 'low_diff' not in par.columns:
        target_magnitude = par.sum_error.min() * 1.10
        par.loc[par.sum_error <= target_magnitude, 'Best'] = 'low_sum'
        index = par.loc[par.Best == 'low_sum', :].diff_error.argmin()
        par.loc[index, 'Best'] = 'low_diff'
    Kf = par.loc[par.Best == 'low_diff', 'Kf'].values
    Kc = par.loc[par.Best == 'low_diff', 'Kc'].values
    Kb = par.loc[par.Best == 'low_diff', 'Kb'].values
    
    if 'vani' not in par.columns:
        vani = upw.vka.get_value()[0][0,0]
    else:
        vani = par.loc[par.Best == 'low_diff', 'vani'].values
        
    hydro = par.loc[par.Best == 'low_diff', 'hydro'].values
    topo = par.loc[par.Best == 'low_diff', 'topo'].values
    
    tau.loc[model, 'Kf'] = Kf
    tau.loc[model, 'Kc'] = Kc
    tau.loc[model, 'Kb'] = Kb
    tau.loc[model, 'vani'] = vani
    tau.loc[model, 'hydro'] = hydro
    tau.loc[model, 'topo'] = topo
    
    ibound = np.array(bas.ibound.get_value())
    num_active = ibound[ibound > 0].sum()
    num_active_surf = ibound[0, :, :][ibound[0, :, :] > 0].sum()
    tau.loc[model, 'num_active'] = num_active
    tau.loc[model, 'num_active_surf'] = num_active_surf
    
    hnoflo = bas.hnoflo
    hdry = upw.hdry
    src = os.path.join(model_ws, '{}.hds'.format(model))
    hd_obj = fp.utils.HeadFile(src)
    heads = hd_obj.get_data((0, 0))
    heads[heads == hnoflo] = np.nan
    heads[heads <= hdry] = np.nan
    heads[heads > 1.E+28] = np.nan

    grid = np.zeros((dis.nlay+1, dis.nrow, dis.ncol))
    grid[0, :, :] = dis.gettop()
    grid[1:, :, :] = dis.getbotm()

    sat_top = np.minimum(heads, grid[:-1, :, :])

    sat_thk_cell = (sat_top - grid[1:, :, :]) 
    sat_thk_cell[sat_thk_cell < 0] = 0
    sat_thk_cell[np.isnan(sat_thk_cell)] = 0

    vfrac = sat_thk_cell / -np.diff(grid, axis=0)

    delc_ar, dum, delr_ar = np.meshgrid(dis.delc, np.arange(dis.nlay), dis.delr)

    sat_vol_cell = sat_thk_cell * delc_ar * delr_ar
    sat_vol_cell_glac = sat_vol_cell[0:num_surf_layers, :, :]
    sat_vol_cell_bedr = sat_vol_cell[num_surf_layers:, :, :]

    src = os.path.join(model_ws, 'zone_array.npz')
    zones = np.load(src)
    zones = zones['zone']
    
    act_vol = sat_vol_cell[ibound > 0]
    act_zone = zones[ibound > 0]
    
    aq_vol = act_vol.sum()
    tau.loc[model, 'aq_vol'] = aq_vol
    
    Kf_vol = act_vol[act_zone == 0].sum()
    tau.loc[model, 'fKf_vol'] = Kf_vol / aq_vol
    
    Kc_vol = act_vol[act_zone == 1].sum()
    tau.loc[model, 'fKc_vol'] = Kc_vol / aq_vol
    
    Klake_vol = act_vol[act_zone == 2].sum()
    tau.loc[model, 'fKlake_vol'] = Klake_vol / aq_vol
    
    Kb_vol = act_vol[act_zone == 3].sum()
    tau.loc[model, 'fKb_vol'] = Kb_vol / aq_vol
        
    num_drn = drn.ncells()
    tau.loc[model, 'drn_dens_area'] = num_drn / num_active_surf

tau['model'] = tau.index.str.lower()

dst = os.path.join(fig_dir, 'master_modflow_table.csv')
tau.to_csv(dst)


# In[ ]:



