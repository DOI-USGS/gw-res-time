
# coding: utf-8

# # Create a general MODFLOW model from the NHDPlus dataset--calculate age distributions and tracer concentrations at pumping wells 

# In[ ]:

__author__ = 'Jeff Starn'   
# get_ipython().magic('matplotlib notebook')
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('png', 'pdf')
# from IPython.display import Image
# from IPython.display import Math
import os
import shutil
import gdal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import flopy as fp
import pandas as pd
import geopandas as gp

# from ipywidgets import interact, Dropdown
# from IPython.display import display

ft2m = 0.3048006096012192
add_bedrock = True
well_pth = '../Data/Wells/all_glac_wells.shp'


# Set some constants

# In[ ]:

# read all_glac_wells shapefile of points from ChemTool
well_shp = gp.read_file(well_pth)


# The next cell doesn't do anything in this notebook, but it can used as a template for creating a batch python script. The rest of the notebook should placed in the try/except statement (with the except moved to the end of the script). This loops through the default and the calibration directories for the scenario selected in gen_mod_dict.

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


# Read existing general model MODFLOW packages

# In[ ]:

for model_ws in dir_list:
    model = os.path.normpath(model_ws).split(os.sep)[2]
    nam_file = '{}.nam'.format(model)
    new_ws = os.path.join(model_ws, 'WEL')
    geo_ws = os.path.dirname(model_ws)

    print("working model is {}".format(model_ws))
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
    # hk = upw.hk.get_value()

    hnoflo = bas.hnoflo
    ibound = np.asarray(bas.ibound.get_value())
    hdry = upw.hdry

    print ('   ... done') 

    # FloPy loads MODFLOW packages but not their name-file unit numbers, so these have to be read separately.

    src = os.path.join(model_ws, fpmg.namefile)
    name_file_df = pd.read_table(src, header=None, comment='#', delim_whitespace=True, 
                  names=['package', 'unit', 'filename', 'type'])

    name_file_df['package'] = name_file_df.package.str.lower()
    name_file_df['ext'] = name_file_df.filename.str.split('.').str.get(1)
    name_file_df.set_index('package', inplace=True)

    # Compute the top of the saturated zone by taking the minimum of the aquifer top elevation and the simulated head in the top layer. Concatenate the saturated top onto the bottom elevations resulting in a 3D grid of vertical cell boundary elevations. Take the difference between layers to get a 3D grid of layer thicknesses. NaNs are propagated through the minimum function so that dry cells are not used to compute thickness. Once the thickness is computed the NaNs are converted back to 0 because a cell with a NaN has no saturation. Save open raster_src for use later.

    src = os.path.join(model_ws, fpmg.namefile)
    name_file_df = pd.read_table(src, header=None, comment='#', delim_whitespace=True, 
                  names=['package', 'unit', 'filename', 'type'])

    name_file_df['package'] = name_file_df.package.str.lower()
    name_file_df['ext'] = name_file_df.filename.str.split('.').str.get(1)
    name_file_df.set_index('package', inplace=True)

    disnum = name_file_df.loc['dis', 'unit']
    disnam = name_file_df.loc['dis', 'filename']

    if 'mnw2' not in name_file_df.index:
        new_ws = os.path.abspath(os.path.join(model_ws, 'WEL'))
        if not os.path.exists(new_ws):
            os.makedirs(new_ws)
    else:
        new_ws = model_ws

    head_file_name = '{}.hds'.format(model)
    src = os.path.join(model_ws, head_file_name)
    hd_obj = fp.utils.HeadFile(src)
    heads = hd_obj.get_data((0, 0))
    heads[heads == hnoflo] = np.nan
    heads[heads <= hdry] = np.nan
    hin = np.argmax(np.isfinite(heads), axis=0)
    row, col = np.indices((hin.shape))
    water_table = heads[hin, row, col]

    water_table[ibound[0,:,:] == 0] = np.nan
    
    src = os.path.join(geo_ws, 'top.tif')
    ph = gdal.Open(src)

    band = ph.GetRasterBand(1)
    top = band.ReadAsArray()
    gt = ph.GetGeoTransform()

    ph = None
    band = None   

    grid = np.zeros((nlay + 1, nrow, ncol))
    grid[0, :, :] = top
    grid[1:, :, :] = dis.getbotm()

#     Read shapefile created from chemtool pull (done in a separate notebook). Clip it to the general model watershed shapefile. Extract spatial coordinates and save it to the new working directory as sample_gdf.shp.

    print ('   Reading sample data')

    # read the watershed and well shapefiles
    domain_file = os.path.join(geo_ws, 'domain_outline.shp')
    domain_diss = gp.read_file(domain_file)

    # intersect the watershed (domain_diss) and well shapefiles
    # and find the rows where the intersection is not null
    mp = domain_diss.geometry[0]
    in_area_index = ~well_shp['geometry'].intersection(mp).isnull()

    # create a geodataframe (sample_gdf) with all the well attributes
    sample_gdf = well_shp.loc[in_area_index].copy()

    # extract the sample point coordinates
    sample_gdf['xg'] = sample_gdf.geometry.apply(lambda p : p.x)
    sample_gdf['yg'] = sample_gdf.geometry.apply(lambda p : p.y)

    # save
    if sample_gdf.shape[0] > 0:
        sample_gdf.to_file(os.path.join(new_ws, 'sample_gdf.shp'))

    print ('   ... done')

#     Next cell is for batch mode.

    try:

    #     Apply an affine transform using linear algebra. The transformation matrix is created from the geotransform data in one of the geotiff files created in NB1.

        print ('   Creating well data frame')

        # eliminate duplicate wells that have more than one sample
        ob_well_unq, ob_well_unq_ind = np.unique(sample_gdf.STAID.values, return_index=True)
        cl = ['ALT_VA', 'DEC_LAT_VA', 'DEC_LONG_V', 'HOLE_DEPTH', 'MAXOFOPEN_', 'MINOFOPEN_', 
              'NetworkTyp', 'STAID', 'SuCode', 'WELL_DEPTH', 'xg', 'yg', 'geometry']
        well_gdf = sample_gdf.iloc[ob_well_unq_ind, :].copy()
        well_gdf = well_gdf.loc[:, cl]
        
        # read geotransform list from a geotiff
        src_filename = os.path.join(os.path.dirname(model_ws), 'top.tif')
        data_source = gdal.Open(src_filename)
        ncol, nrow = data_source.RasterXSize, data_source.RasterYSize
        gt = data_source.GetGeoTransform()
        data_source = None

        # format the geotransformation list into an affine transformation matrix
        forward_transform = np.array(gt).reshape(2, -1)
        # add a row to get homogeneous coodinates (offsets are in the first column)
        forward_transform = np.vstack((forward_transform, [1, 0, 0]))
        # invert the forward transform
        reverse_transform = np.linalg.inv(forward_transform)

        # reverse transform the real-world coordinate to pixel coordinates (row, column)
        well_gdf['one'] = 1
        wpts = well_gdf[['xg', 'yg', 'one']]
        wpp = reverse_transform.dot(wpts.T)
        well_gdf['xm'] = wpp[1, :]
        well_gdf['ym'] = wpp[2, :]

        # row and column (and layer) are zero based
        well_gdf['col'] = np.int32(well_gdf.xm)
        well_gdf['row'] = np.int32(well_gdf.ym)

        # add local coordinates of well within cell
        well_gdf['welx'] = well_gdf.xm - well_gdf.col
        well_gdf['wely'] = 1 - (well_gdf.ym - well_gdf.row)

        # a little db checking:
        # elim rows with no altitude
        # elimn rows where screen top and bot are reversed
        well_gdf = well_gdf.loc[well_gdf.ALT_VA.notnull(), :]
        # well_gdf = well_gdf.loc[well_gdf.MAXOFOPEN_ > well_gdf.MINOFOPEN_, :]

        # convert to meters and to float 
        well_gdf['ALT_VA'] =  well_gdf['ALT_VA'].astype(np.float32()) * ft2m
        well_gdf['MAXOFOPEN_'] = well_gdf['MAXOFOPEN_'].astype(np.float32()) * ft2m
        well_gdf['MINOFOPEN_'] = well_gdf['MINOFOPEN_'].astype(np.float32()) * ft2m
        well_gdf['WELL_DEPTH'] = well_gdf['WELL_DEPTH'].astype(np.float32()) * ft2m
        well_gdf['HOLE_DEPTH'] = well_gdf['HOLE_DEPTH'].astype(np.float32()) * ft2m

        _r, _c = well_gdf['row'].values, well_gdf['col'].values

        # get bedrock altitude
        if add_bedrock:
            well_gdf['bedrock'] = grid[dis.nlay - 1, _r, _c]
        else:
            well_gdf['bedrock'] = grid[dis.nlay, _r, _c]   

        # get bedrock depth
        well_gdf['bedrock_depth'] = grid[0, _r, _c] - well_gdf['bedrock']

        # estimate depth to screen bottom as, in order
        # stated depth, well depth, hole depth, bedrock depth 
        well_gdf['alt_depth'] = np.where(well_gdf.MAXOFOPEN_.notnull(), well_gdf.MAXOFOPEN_, well_gdf.WELL_DEPTH)
        well_gdf['alt_depth'] = np.where(well_gdf.alt_depth.notnull(), well_gdf.alt_depth, well_gdf.HOLE_DEPTH)
        well_gdf['alt_depth'] = np.where(well_gdf.alt_depth.notnull(), well_gdf.alt_depth, well_gdf.bedrock_depth)

        # get estimated altitude of screen bottom
        well_gdf['bot'] = well_gdf.ALT_VA - well_gdf.alt_depth
        # set screen bottom to maximum of bedrock or estimated screen bottom
        well_gdf['screenbot'] = well_gdf.loc[:, ['bot', 'bedrock']].max(axis=1)

        # calculate screen length; set to 5 meters if it is missing
        well_gdf['screenlen'] = well_gdf.MAXOFOPEN_ - well_gdf.MINOFOPEN_
        well_gdf['screenlen'].fillna(5, inplace=True)

        # get the water table altitude
        well_gdf['watertable'] = water_table[well_gdf.row, well_gdf.col]
        # elim wells if the water table is below the screen bottom
        well_gdf = well_gdf.loc[well_gdf.watertable > well_gdf.screenbot]

        # estimate the screen top, keeping the screen length accurate
        well_gdf['top'] = well_gdf.screenbot + well_gdf.screenlen
        # set screen top to minimum of water table or estimated screen top
        well_gdf['screentop'] = well_gdf.loc[:, ['watertable', 'top']].min(axis=1)
        
        num_wells = well_gdf.shape[0]
        well_gdf.to_file(os.path.join(new_ws, 'well_gdf.shp'))
        
        # get layer boundary elevations for well cells (num layers + 1 incl top)
        # this information is needed for the well diagrams plotted below
        # however, shapefiles will not accept an array in a field, therefore
        # this operation has to be done after saving the shapefile
        # the layer elevation data are saved in node_df
        lays = grid[:, well_gdf.row.values, well_gdf.col.values]
        well_gdf['layer_el'] = tuple(lays.T)

    #     Read file of actual average pumping rates for NAWQA PAS wells from "PumpingRate.txt". Location of that file is specified in model_spec.py. If there is no pumping rate, use the default (pas_q could be 0).

        print ('   Creating nodes')

        # decide if each layer has part of the well screen in it
        # --the well screen bottom has to be below the top of the layer (tmp1)
        # AND the well screen top has to be above the bottom of the layer (tmp2)
        tmp1 = lays[:-1].T > well_gdf.screenbot.values.reshape(-1, 1)
        tmp2 = lays[1:].T < well_gdf.screentop.values.reshape(-1, 1)
        # islayer is a boolean array (num_nodes, nlay); 
        # True, if well screen is present in that layer;
        # can be used to pull well cell layer boundaries
        islayer = np.logical_and(tmp1, tmp2)
        num_nodes = islayer.sum()
        # make groupnum, which is an integer index to go from well_gdf to node_df
        # a, b = np.mgrid[0:num_nodes, 0:dis.nlay]
        a, b = np.indices((islayer.shape))
        groupnum = a[islayer]

        node_df = pd.DataFrame(index = np.arange(num_nodes))
        node_df['lay'] = b[islayer].astype(np.int32)
        node_df['row'] = well_gdf.iloc[groupnum, :].row.values
        node_df['col'] = well_gdf.iloc[groupnum, :].col.values
        node_df['screentop'] = well_gdf.iloc[groupnum, :].screentop.values
        node_df['screenbot'] = well_gdf.iloc[groupnum, :].screenbot.values
        node_df['screen_length'] = node_df.screentop - node_df.screenbot
        node_df['cell_top_elev'] = lays[0:-1].T[islayer]
        node_df['cell_bot_elev'] = lays[1:].T[islayer]
        node_df['wel_x'] = well_gdf.iloc[groupnum, :].welx.values
        node_df['wel_y'] = well_gdf.iloc[groupnum, :].wely.values
        node_df['cellx'] = dis.delr[node_df.col]
        node_df['celly'] = dis.delc[node_df.row]
        node_df['cellz'] = node_df.cell_top_elev - node_df.cell_bot_elev
        node_df['wel_z_top'] = np.minimum(node_df.cellz, node_df.screentop - node_df.cell_bot_elev)
        node_df['wel_z_bot'] = np.maximum(0, node_df.screenbot - node_df.cell_bot_elev)
        node_df['screen_length_in_cell'] = node_df.wel_z_top - node_df.wel_z_bot
        node_df['scrn_frac'] = node_df.screen_length_in_cell / node_df.screen_length
        # node_df['pumping_rate'] = well_gdf.iloc[groupnum, :].PumpRate.values * node_df['scrn_frac']
        node_df['layer_el'] = well_gdf.iloc[groupnum, :].layer_el.values

        # Add sequence, group numbers, and station id (staid)
        def t_func(x):
            return x.lay * nrow * ncol + x.row * ncol + x.col
        node_df['seqnum'] = node_df.apply(t_func, axis = 1).astype(np.int32)

        node_df['group'] = groupnum
        node_df['group'] = np.arange(node_df.shape[0])

        node_df['staid'] = well_gdf.STAID.values[groupnum]
        node_df['node_id'] = node_df.staid.astype(str) + '_' + node_df.lay.astype(str)
        node_df.set_index('node_id', inplace=True)

        # save
        node_df.to_csv(os.path.join(new_ws, 'node_df.csv'))

        print ('   ... done')
        

        for i, well in well_gdf.iterrows():
            fig, ax = plt.subplots(1, 1, figsize=(2,5), sharey=True)

            row, col = well.row, well.col
            wt = well.watertable
            wtop = well.top
            lays = well.layer_el
            stop = well.screentop
            sbot = well.screenbot
            K = np.array(upw.hk.get_value())[:, row, col]

            colors = ['b', 'g', 'r']

            ax.axhline(wt, c='b')
            ax.axhline(wtop, c='g', lw=2)
            for lay in range(num_surf_layers):
                ax.axhspan(lays[lay], lays[lay + 1], color=colors[lay], alpha=0.2)
                ax.annotate('Layer {}'.format(lay + 1),
                    xy=(0.5, (lays[lay] + lays[lay + 1]) / 2), #fontweight='bold',
                    size=8, ha='center', va='bottom')
                ax.annotate('{:3.2f} m/d'.format(K[lay]),
                    xy=(0.5, (lays[lay] + lays[lay + 1]) / 2), #fontweight='bold',
                    size=8, ha='center', va='top')

            ax.set_title('{}\n{}:{}'.format(well.STAID, well.NetworkTyp, well.SuCode), {'fontsize': 8})
            ax.annotate('Water table',
                        xy=(0.50, wt), 
                        size=8, ha='center', va='top')

            ax.annotate('Land surface',
                        xy=(0.50, wtop), 
                        size=8, ha='center', va='bottom')

            tnode = node_df.loc[node_df.staid == well.STAID, :]
            stop = tnode.screentop.max()
            sbot = tnode.screenbot.min()

            # Create a Rectangle patch for the well screen
            rect = patches.Rectangle((0, sbot), 1.0, stop-sbot, linewidth=1, 
                                     edgecolor='k', facecolor='k', alpha=0.2, hatch='/')

            # Add the patch to the Axes
            ax.add_patch(rect)

            ax.set_ylabel('Altitude in meters', fontsize=8)
            ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') # labels along the bottom edge are off

            ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            labelsize=8)

            fig.set_tight_layout(True)
            dst = os.path.join(model_ws, 'well_diagrams')
            if not os.path.exists(dst):
                print ('Making directory {}'.format(dst))
                os.mkdir(dst)

            form_list = ['png', 'pdf', 'tif']
            for form in form_list:
                fig_name = os.path.join(dst, '{}.{}'.format(well.STAID, form))
                plt.savefig(fig_name)
            plt.close()
    except ValueError:
        pass

