
# coding: utf-8

# This notebook is used to get residence-time distribution (RTD) for individual wells from an existing MODFLOW model. It is possible to read in any group or label from a 3D array and make RTDs for those groups. The approach is to 
# * read an existing model
# * create flux-weighted particle starting locations in every cell
# * run MODPATH and read endpoints
# * fit parametric distributions

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
import sys
import shutil
import pickle
import numpy as np
import datetime as dt
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import flopy as fp
import imeth
import fit_parametric_distributions
import pandas as pd
import scipy.stats as ss
import scipy.optimize as so
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from scipy.interpolate import UnivariateSpline


# # Preliminary stuff

# ## Set user-defined variables
# 
# MODFLOW and MODPATH use elapsed time and are not aware of calendar time. To place MODFLOW/MODPATH elapsed time on the calendar, two calendar dates were specified at the top of the notebook: the beginning of the first stress period (`mf_start_date`) and when particles are to be released (`mp_release_date`). The latter date could be used in many ways, for example to represent a sampling date, or it could be looped over to create a time-lapse set of ages. 
# 
# `num_surf_layers` is an arbitrary layer number on which to divide the model domain for calculating RTDs. For example, in glacial aquifers it could represent the layer number of the bottom of unconsolidated deposits. In that case, anything below this layer could be considered bedrock.
# 
# `num_depth_groups` is an arbitrary number of equally groups starting from the water table to the bottom of the lowest model layer.

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

for model_ws in dir_list:
    model = os.path.normpath(model_ws).split(os.sep)[2]
    nam_file = '{}.nam'.format(model)
    new_ws = os.path.join(model_ws, 'WEL')
    geo_ws = os.path.dirname(model_ws)

    print("working model is {}".format(model_ws))

    # Load an existing model

    print ('Reading model information')

    fpmg = fp.modflow.Modflow.load(nam_file, model_ws=model_ws, exe_name=mfpth, version='mfnwt', 
                                   load_only=['DIS', 'BAS6', 'UPW', 'OC'], check=False)

    dis = fpmg.get_package('DIS')
    bas = fpmg.get_package('BAS6')
    upw = fpmg.get_package('UPW')
    oc = fpmg.get_package('OC')

    delr = dis.delr
    delc = dis.delc
    nlay = dis.nlay
    nrow = dis.nrow
    ncol = dis.ncol
    bot = dis.getbotm()
#     top = dis.gettop()

    hnoflo = bas.hnoflo
    ibound = np.asarray(bas.ibound.get_value())
    hdry = upw.hdry

    print ('   ... done') 

    ## Specification of time in MODFLOW/MODPATH

#     There are several time-related concepts used in MODPATH.
#     * `simulation time` is the elapsed time in model time units from the beginning of the first stress period
#     * `reference time` is an arbitrary value of `simulation time` that is between the beginning and ending of `simulation time`
#     * `tracking time` is the elapsed time relative to `reference time`. It is always positive regardless of whether particles are tracked forward or backward
#     * `release time` is when a particle is released and is specified in `tracking time`

    # setup dictionaries of the MODFLOW units for proper labeling of figures.
    lenunit = {0:'undefined units', 1:'feet', 2:'meters', 3:'centimeters'}
    timeunit = {0:'undefined', 1:'second', 2:'minute', 3:'hour', 4:'day', 5:'year'}

    # Create dictionary of multipliers for converting model time units to days
    time_dict = dict()
    time_dict[0] = 1.0 # undefined assumes days, so enter conversion to days
    time_dict[1] = 24 * 60 * 60
    time_dict[2] = 24 * 60
    time_dict[3] = 24
    time_dict[4] = 1.0
    time_dict[5] = 1.0

    # convert string representation of dates into Python datetime objects
    mf_start_date = dt.datetime.strptime(mf_start_date_str , '%m/%d/%Y')
    mp_release_date = dt.datetime.strptime(mp_release_date_str , '%m/%d/%Y')

    # convert simulation time to days from the units specified in the MODFLOW DIS file
    sim_time = np.append(0, dis.get_totim())
    sim_time /= time_dict[dis.itmuni]

    # make a list of simulation time formatted as calendar dates
    date_list = [mf_start_date + dt.timedelta(days = item) for item in sim_time]

    # reference time and date are set to the end of the last stress period
    ref_time = sim_time[-1]
    ref_date = date_list[-1]

    # release time is calculated in tracking time (for particle release) and 
    # in simulation time (for identifying head and budget components)
    release_time_trk = np.abs((ref_date - mp_release_date).days)
    release_time_sim = (mp_release_date - mf_start_date).days

    # Fit parametric distributions

    try:
        src = os.path.join(model_ws, 'WEL', 'node_df.csv')
        node_df = pd.read_csv(src)

        src = os.path.join(model_ws, 'WEL', 'well_gdf.shp')
        well_shp = gp.read_file(src)

        src = os.path.join(model_ws, 'WEL', 'sample_gdf.shp')
        sample_shp = gp.read_file(src)
        sample_shp['STAID'] = sample_shp.STAID.astype(np.int64())
        sample_shp['DATES'] = pd.to_datetime(sample_shp['DATES'])

        # Process endpoint information

        ## Read endpoint file

        # form the path to the endpoint file
        mpname = '{}_flux'.format(fpmg.name)
        endpoint_file = '{}.{}'.format(mpname, 'mpend')
        endpoint_file = os.path.join(model_ws, endpoint_file)
        
        ep_data = fit_parametric_distributions.read_endpoints(endpoint_file, dis, time_dict)
        ep_data['initial_node_num'] = ep_data.index

        dist_list = [ss.weibull_min]
        fit_dict = dict()
        method = 'add_weibull_min'
        # group nodes by station ID
        ng = node_df.groupby('staid')
        fit_dict = {}

        # loop through station ID groups
        for staid, nodes in ng:
            # start dictionary for this well
            rt = list()
            # append particles rt's for all nodes for each well
            for k, m in nodes.iterrows():
                rt.extend(ep_data.loc[ep_data.initial_node_num == m.seqnum, 'rt'])

            # # sort rt's
            rt.sort()
            trav_time_raw = np.array(rt)

            # create arrays of CDF value between 1/x and 1
            # number of particles above num_surf_layers
            n = trav_time_raw.shape[0]

            # number of particles desired to approximate the particle CDF
            s = 1000
            ly = np.linspace(1. / s, 1., s, endpoint=True)
            tt_cdf = np.linspace(1. / n, 1., n, endpoint=True)

            # log transform the travel times and normalize to porosity
            tt = np.log(trav_time_raw / por)

            # interpolate at equally spaced points to reduce the number of particles
            lprt = np.interp(ly, tt_cdf , tt)
            first = lprt.min()

            fit_dict[staid] = fit_parametric_distributions.fit_dists(ly, lprt, dist_list)

        dst = os.path.join(model_ws, 'fit_dict_wells_{}.pickle'.format(model))
        with open(dst, 'wb') as f:
            pickle.dump(fit_dict, f)
    except FileNotFoundError:
        print('Sample and wells files not found')


