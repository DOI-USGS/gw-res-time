
# coding: utf-8

# In[ ]:

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
# import subprocess as sp
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geopandas as gp
import flopy as fp
import fit_parametric_distributions
import imeth
import pandas as pd
import gdal
import scipy.stats as ss
import scipy.optimize as so
from scipy.interpolate import Rbf
from scipy.interpolate import griddata
from scipy.interpolate import UnivariateSpline


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

# model_area = Dropdown(
    # options=mod_list,
    # description='Model:',
    # background_color='cyan',
    # border_color='black',
    # border_width=2)
# display(model_area)

with open('dir_list.txt', 'w') as f:
    for i in dir_list:
        f.write('{}\n'.format(i))


# In[ ]:

agelay = 3


# In[ ]:

# model = model_area.value
# model_ws = [item for item in dir_list if model in item][0]
# nam_file = '{}.nam'.format(model)
# print("working model is {}".format(model_ws))


# In[ ]:

for pth in dir_list:
    model = os.path.normpath(pth).split(os.sep)[2]
    model_ws = [item for item in dir_list if model in item][0]
    nam_file = '{}.nam'.format(model)
    print("working model is {}".format(model_ws))


    # In[ ]:

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
    # top = dis.gettop()
    # hk = upw.hk.get_value()

    hnoflo = bas.hnoflo
    ibound = np.asarray(bas.ibound.get_value())
    hdry = upw.hdry
    row_to_plot = np.int32(dis.nrow / 2)

    print ('   ... done') 


    # In[ ]:

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


    # In[ ]:

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


    # In[ ]:

    src_pth = os.path.dirname(model_ws)
    src = os.path.join(src_pth, 'top.tif')
    nf = gdal.Open(src)
    band = nf.GetRasterBand(1)
    land_surface = band.ReadAsArray()
    gt = nf.GetGeoTransform()
    proj = nf.GetProjection()
    nf = None


    # In[ ]:

    src = os.path.join(model_ws, fpmg.namefile)
    name_file_df = pd.read_table(src, header=None, comment='#', delim_whitespace=True, 
                  names=['package', 'unit', 'filename', 'type'])

    name_file_df['package'] = name_file_df.package.str.lower()
    name_file_df.set_index('unit', inplace=True)

    head_file_name = name_file_df.loc[oc.iuhead, 'filename']
    bud_file_name = name_file_df.loc[oc.get_budgetunit(), 'filename']


    # In[ ]:

    src = os.path.join(model_ws, head_file_name)
    hd_obj = fp.utils.HeadFile(src)
    head_df = pd.DataFrame(hd_obj.recordarray)

    heads = hd_obj.get_data(kstpkper=(0, 0))

    heads[heads == hnoflo] = np.nan
    heads[heads <= hdry] = np.nan
    heads[heads > 1E+29] = np.nan

    hin = np.argmax(np.isfinite(heads), axis=0)
    row, col = np.indices((hin.shape))
    water_table = heads[hin, row, col]


    # In[ ]:

    # Transform GWPs into fractional row, column coordinates for each model    
    # format the geotransformation list into an affine transformation matrix
    forward_transform = np.array(gt).reshape(2, -1)
    # add a row to get homogeneous coodinates (offsets are in the first column)
    forward_transform = np.vstack((forward_transform, [1, 0, 0]))

    # reverse transform cell-center coordinates to projected coordinates
    r, c = np.indices(water_table.shape)
    dum = np.column_stack((np.ones_like(c.ravel()), c.ravel() + 0.5, r.ravel() + 0.5)) 
    dat = forward_transform.dot(dum.T).T
    xdat = dat[:,0].reshape(water_table.shape)
    ydat = dat[:,1].reshape(water_table.shape)

    index = np.isfinite(water_table[row_to_plot, :])

    xplot = xdat[row_to_plot, index]


    # In[ ]:

    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 8,
            'sans-serif' : 'Arial'}

    plt.rc('font', **font)


    # In[ ]:

    fig, ax1 = plt.subplots(1, 1, figsize=(7.4,4))

    colors = ['green', 'red', 'gray']

    ax1.plot(xplot, land_surface[row_to_plot, index], label='land surface', color='black', lw=0.5)
    ax1.plot(xplot, water_table[row_to_plot, index], label='simulated\nwater table', color='blue', lw=0.75)
    ax1.fill_between(xplot, land_surface[row_to_plot, index], bot[0, row_to_plot, index], alpha=0.25, 
                     color='blue', lw=0.75)
    for lay in range(dis.nlay - 1):
        ax1.fill_between(xplot, bot[lay, row_to_plot, index], bot[lay+1, row_to_plot, index],  
                        color=colors[lay], alpha=0.250, lw=0.75)
    ax1.plot(xplot, bot[-2, row_to_plot, index], label='bedrock', color='red', linestyle='dotted', lw=1.5)
    ax1.plot(xplot, bot[-1, row_to_plot, index], color='black', linestyle='solid', lw=0.5)
    ax1.legend(loc=0, frameon=False, ncol=1)#, bbox_to_anchor=(1.0, 0.5))
    ax1.set_ylabel('Altitude in meters')
    ax1.set_xlabel('Albers Equal Area meters GRS80')
    fig.tight_layout()

    form_list = ['png', 'pdf', 'tif']
    for form in form_list:
        line = '{}_xs_cal.{}'.format(model, form)
        fig_name = os.path.join(fig_dir, line)
        plt.savefig(fig_name)
    plt.close()


    # In[ ]:

    # form the path to the endpoint file
    mpname = '{}_flux'.format(fpmg.name)
    endpoint_file = '{}.{}'.format(mpname, 'mpend')
    endpoint_file = os.path.join(model_ws, endpoint_file)


    # In[ ]:

    ep_data = fit_parametric_distributions.read_endpoints(endpoint_file, dis, time_dict)


    # In[ ]:

    n = 100 # number of points to interpolate to in a vertical column at cell center
    tthk = water_table[row_to_plot, index] - bot[-1, row_to_plot, index]
    incr = np.linspace(0, 1, n, endpoint=True).reshape(n, 1)
    pt = bot[-1, row_to_plot, index] + incr * tthk
    xx = xplot * np.ones(pt.shape)


    # In[ ]:

    xpoints = ep_data.loc[ep_data['Initial Row'] == row_to_plot, ['Initial Column', 'Initial Local X']].sum(axis=1)
    ypoints = ep_data.loc[ep_data['Initial Row'] == row_to_plot, ['Initial Row', 'Initial Local Y']].sum(axis=1)
    dum = np.column_stack((np.ones_like(xpoints), xpoints, ypoints)) 
    points = forward_transform.dot(dum.T).T
    points[:, 2] = ep_data.loc[ep_data['Initial Row'] == row_to_plot, 'Initial Global Z']


    # In[ ]:

    values = ep_data.loc[ep_data['Initial Row'] == row_to_plot, 'rt'].values
    xi = np.column_stack((xx.ravel(), pt.ravel()))
    tmp = griddata(points[:, 0::2], values, xi, method='linear')
    age = tmp.reshape(pt.shape)


    # In[ ]:

    fig, ax1 = plt.subplots(1, 1, figsize=(7.4, 4.6))

    colors = ['green', 'red', 'gray']
    colors_poly = plt.cm.rainbow(np.linspace(0, 1, nlay+1))
    colors_mark = plt.cm.spectral(np.linspace(0, 1, 6))
    alfa_poly = 0.50
    alfa_mark = 1.0

    im = ax1.contourf(xplot * np.ones((n, 1)), pt[:], age[:], colors=colors_poly, alpha=0.5, 
                      levels=[0, 10, 50, 100, 500, 10000])

    cbar = fig.colorbar(im, orientation='horizontal', shrink=0.5, pad=0.10, use_gridspec=True) 
    cbar.ax.set_xlabel('Particle travel time / porosity in years', rotation=0, y=1.5, ha='center')

    ax1.plot(xplot, land_surface[row_to_plot, index], label='land surface', color='black', lw=0.5)
    ax1.plot(xplot, water_table[row_to_plot, index], label='simulated\nwater table', color='blue', lw=0.75)
    ax1.plot(xplot, bot[-2, row_to_plot, index], label='bedrock', color='black', linestyle='dotted', lw=1.5)
    ax1.plot(xplot, bot[-1, row_to_plot, index], color='black', linestyle='solid', lw=0.5)
    ax1.legend(loc=0, frameon=False, ncol=1)
    ax1.set_ylabel('Altitude in meters')
    ax1.set_xlabel('Albers Equal Area meters GRS80')
    fig.text(0.01, 0.95, 'a', fontdict={'weight':'bold'})
    fig.tight_layout()

    form_list = ['png', 'pdf', 'tif']
    for form in form_list:
        line = 'Paper #2017WR021531-f03a.{}'.format(form)
        fig_name = os.path.join(fig_dir, line)
        plt.savefig(fig_name, dpi=300)
    plt.close()


    # In[ ]:

    # initiate a plot for individual model residuals (small areas)
    fig2, ax2 = plt.subplots(1, 1, figsize=(7.4, 4.6))
    im = ax2.contourf(xdat, ydat, water_table, cmap=plt.cm.spectral, interpolation='none', alpha=0.4)

    cbar = fig2.colorbar(im, orientation='horizontal', shrink=0.5, pad=0.10, use_gridspec=True) 
    cbar.ax.set_xlabel('Simulated water table altitude in meters', rotation=0, y=1.5, ha='center')

    src = os.path.dirname(model_ws)
    fname1 = 'domain_outline.shp'
    f = os.path.join(src, fname1)
    basin = gp.read_file(f)
    basin.plot(ax=ax2, color='none', linewidth=1.00, alpha=1.0, **{'edgecolor':'k'})    

    fname2 = 'clip_box.shp'
    f = os.path.join(src, fname2)
    clip = gp.read_file(f)
    clip.plot(ax=ax2, color='none', linewidth=1.0, alpha=0.5, **{'edgecolor':'k'})

    fname3 = 'NHD_clip.shp'
    f = os.path.join(src, fname3)
    streams = gp.read_file(f)
    streams.plot(ax=ax2, color='b', linewidth=1.0, alpha=0.5)

    ax2.plot(xdat[row_to_plot, index], ydat[row_to_plot, index], color='k', lw=2)

    #     to make small area plots compact, first set the axis limits to the data extent
    ymin = ydat.min()
    ymax = ydat.max()
    xmin = xdat.min()
    xmax = xdat.max()
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
        
        # make sure the scale for both axes are equal
    fig2.gca().set_aspect('equal', adjustable='datalim', anchor='SW')  
    ax2.set_xlabel('Albers Equal Area meters GRS80')
    fig2.text(0.01, 0.95, 'b', fontdict={'weight':'bold'})

    fig2.set_tight_layout(True)

    try:
        src = os.path.join(model_ws, 'WEL', 'well_gdf.shp')
        well = gp.read_file(src)
        well.plot(ax=ax2, color='black', markersize=6)
    except:
        pass

    form_list = ['png', 'pdf', 'tif']
    for form in form_list:
        line = '{}_head_map.{}'.format(model, form)
        fig_name = os.path.join(fig_dir, line)
        plt.savefig(fig_name)
    plt.close()


    # In[ ]:

    agelay = 3
    age2d = ep_data.loc[ep_data['Initial Layer'] == agelay, :]
    age2d = age2d.groupby('initial_node_num').median()

    ncells_in_layer = nrow * ncol
    begin = (agelay - 1) * ncells_in_layer
    end = begin + ncells_in_layer

    nindex = np.arange(begin, end, 1)

    age2d = age2d.reindex(nindex)

    ageL = age2d.rt.values.reshape(nrow, ncol)


    # In[ ]:

    # initiate a plot for individual model residuals (small areas)
    fig2, ax2 = plt.subplots(1, 1, figsize=(7.4, 4.6))

    im = ax2.contourf(xdat, ydat, ageL, colors=colors_poly, alpha=0.5, 
                      levels=[0, 10, 50, 100, 500, 10000])

    # cbar = fig2.colorbar(im, orientation='horizontal', shrink=0.5, pad=0.10, use_gridspec=True) 
    # cbar.ax.set_xlabel('Particle travel time / porosity in years', rotation=0, y=1.5, ha='center')

    src = os.path.dirname(model_ws)
    fname1 = 'domain_outline.shp'
    f = os.path.join(src, fname1)
    basin = gp.read_file(f)
    basin.plot(ax=ax2, color='none', linewidth=1.00, alpha=1.0, **{'edgecolor':'k'})    

    # fname2 = 'clip_box.shp'
    # f = os.path.join(src, fname2)
    # clip = gp.read_file(f)
    # clip.plot(ax=ax2, color='none', linewidth=1.0, alpha=0.5, **{'edgecolor':'k'})

    fname3 = 'NHD_clip.shp'
    f = os.path.join(src, fname3)
    streams = gp.read_file(f)
    streams.plot(ax=ax2, color='b', linewidth=0.5, alpha=1.0)

    ax2.plot(xdat[row_to_plot, index], ydat[row_to_plot, index], color='k', lw=1)

    # ax2.set_xlabel('Albers Equal Area meters GRS80')
    #     to make small area plots compact, first set the axis limits to the data extent
    # ymin = ydat.min()
    # ymax = ydat.max()
    # xmin = xdat.min()
    # xmax = xdat.max()
    # ax2.set_xlim(xmin, xmax)
    # ax2.set_ylim(ymin, ymax)
        
        # make sure the scale for both axes are equal
    ax2.set_aspect('equal', adjustable='box-forced', anchor='SW')  
    fig2.text(0.01, 0.95, 'b', fontdict={'weight':'bold'})
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)


    fig2.set_tight_layout(True)

    line = '{}_age_layer_{}.png'.format(model, agelay)
    fig_name = os.path.join(fig_dir, line)
    plt.savefig(fig_name)

    form_list = ['png', 'pdf', 'tif']
    for form in form_list:
        line = 'Paper #2017WR021531-f03b.{}'.format(form)
        fig_name = os.path.join(fig_dir, line)
        plt.savefig(fig_name, dpi=300)
    plt.close()


    # In[ ]:

    dst = '{}_lay{}.tif'.format(model, agelay)
    dst_file = os.path.join(fig_dir, dst)

    import gdal
    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(dst_file, ncol, nrow, 1, gdal.GDT_Float32)
    dst.SetGeoTransform(gt)
    dst.SetProjection(proj)
    band = dst.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    band.WriteArray(ageL)
    dst = None


    xs_x = xdat[row_to_plot, index]
    xs_y = ydat[row_to_plot, index]

    from shapely.geometry import Point, LineString
    line = LineString([Point(xs_x[0], xs_y[0]), Point(xs_x[-1], xs_y[-1])])

    xs_line = gp.GeoDataFrame(geometry = gp.GeoSeries(line), crs=streams.crs)

    dst = '{}_xsline.shp'.format(model)
    dst_file = os.path.join(fig_dir, dst)


    dst_file

    xs_line.to_file(dst_file)


    # In[ ]:

    fig = plt.figure(figsize=(7.4, 6))

    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.0, hspace=0.08, top=0.98, left=0.08, right=0.92)

    ax1 = plt.subplot(gs[0:3, 0:9])
    ax2 = plt.subplot(gs[3:-1, 0:9], sharex=ax1)
    ax3 = plt.subplot(gs[2:7, 9])

    colors_poly = plt.cm.rainbow(np.linspace(0, 1, nlay+1))
    alfa_poly = 0.60
    levels = [0, 10, 50, 100, 500, 10000]

    im = ax1.contourf(xplot * np.ones((n, 1)), pt[:], age[:], colors=colors_poly, alpha=alfa_poly, 
                      levels=levels, antialiased=True)

    ax1.plot(xplot, land_surface[row_to_plot, index], label='land surface', color='black', lw=0.5)
    ax1.plot(xplot, water_table[row_to_plot, index], label='simulated\nwater table', color='blue', lw=0.75)
    ax1.plot(xplot, bot[num_surf_layers-2, row_to_plot, index], label='bedrock',
             color='black', linestyle='solid', lw=0.5, alpha=alfa_poly)
    ax1.plot(xplot, bot[num_surf_layers-1, row_to_plot, index], label='bedrock', 
             color='black', linestyle='solid', lw=0.5, alpha=alfa_poly)
    ax1.plot(xplot, bot[-1, row_to_plot, index], color='black', linestyle='solid', lw=0.5)

    ax1.set_aspect('auto', adjustable='box-forced', anchor='NE')  
    ax1.set_ylabel('Altitude in meters')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(True)
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params('x', length=0)

    im = ax2.contourf(xdat, ydat, ageL, colors=colors_poly, alpha=alfa_poly, levels=levels,
                     antialiased=True)

    src = os.path.dirname(model_ws)
    fname1 = 'domain_outline.shp'
    f = os.path.join(src, fname1)
    basin = gp.read_file(f)
    basin.plot(ax=ax2, color='none', linewidth=1.00, alpha=1.0, **{'edgecolor':'k'})    

    # fname2 = 'clip_box.shp'
    # f = os.path.join(src, fname2)
    # clip = gp.read_file(f)
    # clip.plot(ax=ax2, color='none', linewidth=1.0, alpha=0.5, **{'edgecolor':'k'})

    fname3 = 'NHD_clip.shp'
    f = os.path.join(src, fname3)
    streams = gp.read_file(f)
    streams.plot(ax=ax2, color='b', linewidth=0.5, alpha=1.0)

    ax2.plot(xdat[row_to_plot, index], ydat[row_to_plot, index], color='k', lw=1)

    ax2.set_aspect('equal', adjustable='box-forced', anchor='NE')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params('both', length=5)

    cbar = fig.colorbar(im, cax=ax3, ax=ax3, orientation='vertical')#, shrink=0.5, pad=0.10, use_gridspec=True) 
    cbar.ax.set_ylabel('Particle travel time / porosity in years', rotation=90, x=0.1, y=0.5, ha='center')
    ax3.set_xmargin(0)
    ax3.set_ymargin(0)
    ax3.set_aspect(15)

    fig.text(0.01, 0.96, 'a', fontdict={'weight':'bold'})
    fig.text(0.01, 0.68, 'b', fontdict={'weight':'bold'})
    fig.text(0.90, 0.12, 'Albers Equal Area meters GRS80', ha='right')

    form_list = ['png', 'pdf', 'tif']
    for form in form_list:
        line = 'Paper #2017WR021531-f03_combined.{}'.format(form)
        fig_name = os.path.join(fig_dir, line)
        plt.savefig(fig_name, dpi=300)
    plt.close()


    # In[ ]:

    # experimental

    # from matplotlib.transforms import Affine2D
    # import mpl_toolkits.axisartist.floating_axes as floating_axes
    # import mpl_toolkits.axisartist.angle_helper as angle_helper
    # from matplotlib.projections import PolarAxes
    # from mpl_toolkits.axisartist.grid_finder import (FixedLocator, MaxNLocator,
    #                                                  DictFormatter)

    # x, y = clip.loc[0, 'geometry'].exterior.coords.xy
    # exts = (x[0], x[2], y[3], y[1])

    # src = os.path.join(os.path.dirname(model_ws), 'grid_spec.txt')

    # with open(src) as f:
    #     lines = f.readlines()

    # key = 'Rotation about upper left corner in radians and degrees from positive x axis\n'
    # lineno = [item for item in enumerate(lines) if key in item][0][0] + 1
    # angle = np.float32(lines[lineno].split()[1])

    # def setup_axes1(fig, rect):
    #     """
    #     A simple one.
    #     """
    #     tr = Affine2D().scale(2, 1).rotate_deg(angle)

    #     grid_helper = floating_axes.GridHelperCurveLinear(
    #         tr, extremes=exts)

    #     ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    #     fig.add_subplot(ax1)

    #     aux_ax = ax1.get_aux_axes(tr)

    #     grid_helper.grid_finder.grid_locator1._nbins = 4
    #     grid_helper.grid_finder.grid_locator2._nbins = 4

    #     return ax1, aux_ax

    # fig = plt.figure(1, figsize=(8, 4))
    # # fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)

    # ax1, aux_ax1 = setup_axes1(fig, 111)
    # # streams.plot(ax=aux_ax1)
    # # basin.plot(ax=aux_ax1)
    # clip.plot(ax=aux_ax1)
    # # fig.gca().set_aspect('equal', adjustable='datalim', anchor='SW')  



    # In[ ]:



