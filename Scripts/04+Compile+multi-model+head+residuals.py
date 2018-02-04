
# coding: utf-8

# In[ ]:

__author__ = 'Jeff Starn'
# get_ipython().magic('matplotlib notebook')

# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('png', 'pdf')
# from IPython.display import Image
# from IPython.display import Math

import os
import shelve
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gp
import gdal, osr
gdal.UseExceptions()
import flopy as fp
import seaborn as sns
import scipy.interpolate as si


# Read groundwater point ("GWP") locations with depth to water and land surface altitude fom Terri Arnold

# In[ ]:

src = os.path.join('../Data/GWSW_points', 'GWSW_points.shp')
all_wells = gp.read_file(src)


# Loop through all the general models that were created for this study. Read the heads and land surface altitude for them. Sites selected to encompass the highest fraction of Terri's data.

# Read head output from MODFLOW and write the head in the upper-most active cell to a 2D array and GeoTiff.

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


# Make geotiffs of calibrated heads

# In[ ]:

df = pd.DataFrame()
for model_ws in dir_list:
    model = os.path.normpath(model_ws).split(os.sep)[2]
    nam_file = '{}.nam'.format(model)
    new_ws = os.path.join(model_ws, 'WEL')
    geo_ws = os.path.dirname(model_ws)
    
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

    # create a 2D surface of the simulated head in the highest active cell ("water table")
    src = os.path.join(model_ws, '{}.hds'.format(model))
    headobj = fp.utils.HeadFile(src)
    heads = headobj.get_data(kstpkper=(0, 0))
    heads[heads == hnoflo] = np.nan
    heads[heads <= hdry] = np.nan
    heads[heads > 1E+29] = np.nan
    hin = np.argmax(np.isfinite(heads), axis=0)
    row, col = np.indices((hin.shape))
    water_table = heads[hin, row, col]

    src = os.path.join(geo_ws, 'top.tif')
    ph = gdal.Open(src)

    band = ph.GetRasterBand(1)
    top = band.ReadAsArray()
    gt = ph.GetGeoTransform()

    ph = None
    band = None   
        
    domain_file = os.path.join(geo_ws, 'domain_outline.shp')
    basin = gp.read_file(domain_file)
    
    # intersect the watershed (domain_diss) and well shapefiles
    # and find the rows where the intersection is not null
    mp = basin.geometry[0]
    in_area_index = ~all_wells['geometry'].intersection(mp).isnull()

    # create a geodataframe (sample_gdf) with all the well attributes
    wells = all_wells.loc[in_area_index].copy()

    if wells.shape[0] != 0:

        # Transform GWPs into fractional row, column coordinates for each model

        # format the geotransformation list into an affine transformation matrix
        forward_transform = np.array(gt).reshape(2, -1)
        # add a row to get homogeneous coodinates (offsets are in the first column)
        forward_transform = np.vstack((forward_transform, [1, 0, 0]))
        # invert the forward transform
        reverse_transform = np.linalg.inv(forward_transform)

        x = wells.geometry.apply(lambda p: p.x)
        y = wells.geometry.apply(lambda p: p.y)
        one = np.ones_like(x)

        wpts = np.column_stack((x, y, one))

        # reverse transform the real-world coordinate to pixel coordinates (row, column)
        wpp = reverse_transform.dot(wpts.T)

        r, c = np.indices(water_table.shape)

        # interpolate water table from model to GWPs
        sim_heads = si.griddata((c.ravel(), r.ravel()), water_table.ravel(), wpp.T[:, 1:], method='linear')
        sim_top = si.griddata((c.ravel(), r.ravel()), top.ravel(), wpp.T[:, 1:], method='linear')

        # convert model values to feet
        wells['x'] = x
        wells['y'] = y
        wells['sim_top'] = sim_top 
        wells['sim_heads'] = sim_heads   
        wells['model'] = model    
        wells['WLm'] = wells['WLElevFt'] * 0.3048
        wells['DTWm'] = wells['DTWFt'] * 0.3048
        wells['LSDm'] = wells['LSElevFt'] * 0.3048
        wells['sim_dtw_top'] = wells['sim_top'] - wells['sim_heads']
        wells['sim_dtw_lsd'] = wells['LSElevFt'] - wells['sim_heads']
        wells['dtw_res_top'] = wells['DTWm'] - wells['sim_dtw_top']
        wells['dtw_res_lsd'] = wells['DTWm'] - wells['sim_dtw_lsd']
        wells['res_wl_el'] = wells['WLm'] - wells['sim_heads']
        wells['res_lsd'] = wells['LSElevFt'] - wells['sim_top']
        wells['swgw'] = pd.factorize(wells.SiteType)[0]

        # save the data  
        df = df.append(wells)


# In[ ]:

model_data = gp.read_file('../Data/Watersheds/watersheds.shp')


# In[ ]:

# newdf.columns


# In[ ]:

newdf = df.merge(model_data, left_on='model', right_on='model_name')
newdf.loc[:, 'model_num'] = newdf.model_num.astype(np.int32())
newdf = newdf.loc[:, ['OBJECTID', 'SITE_NO', 'DEC_LAT_VA', 'DEC_LONG_V', 'LSElevFt',
       'NHDStrmOrd', 'SiteType', 'WLElevFt', 'DTWFt', 'dtwBin', 'location',
       'geometry_x', 'x', 'y', 'sim_top', 'sim_heads', 'model', 'WLm', 'DTWm',
       'LSDm', 'sim_dtw_top', 'sim_dtw_lsd', 'dtw_res_top', 'dtw_res_lsd',
       'res_wl_el', 'res_lsd', 'swgw', 'model_num', 'model_name']]
dst = os.path.join(fig_dir, 'head_resid_df.csv')
newdf.to_csv(dst)


# In[ ]:

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 8,
        'sans-serif' : 'Arial'}

plt.rc('font', **font)

fig, ax = plt.subplots(1, 1, figsize=(4.6, 4.6))
sns.set_style("ticks",  {'axes.facecolor':'white'})
ax.grid(False)
ax.axhspan(-5, 5, color='red', alpha=0.2)
ax.set_yticks(np.arange(-30, 40, 10))
ax.set_ylim(-30, 30)
ax.tick_params(axis='x', length=0)

ax = sns.swarmplot(x="model_num", y="res_wl_el", data=newdf, color='k', 
                   size=3, alpha=0.40)

ax = sns.boxplot(x="model_num", y="res_wl_el", data=newdf, whis=1.5,
        showcaps=False, boxprops={'facecolor':'None', 'linewidth':0.5, 'edgecolor':'k', 'alpha':1.0},
        showfliers=False, whiskerprops={'linewidth':0, 'color':'k'},
                medianprops={'linewidth':0.5})

ax.set_xlabel('Model number')
ax.set_ylabel('Measured - simulated water-table elevation in meters')
fig.set_tight_layout(True)

forms = ['png', 'tif', 'pdf']

for f in forms:
    dst = os.path.join(fig_dir, 'Paper #2017WR021531-f02.{}'.format(f))
    plt.savefig(dst, dpi=300)
plt.close()

# In[ ]:



