
# coding: utf-8

# In[ ]:

__author__ = 'Jeff Starn'
# get_ipython().magic('matplotlib notebook')
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('png', 'pdf')
# from IPython.display import Image
# from IPython.display import Math
import os
import sys
import shutil
import gdal
import pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import flopy as fp
import pandas as pd
import geopandas as gp
import scipy.stats as ss
import scipy.optimize as so
from scipy.interpolate import UnivariateSpline

# from ipywidgets import interact, Dropdown
# from IPython.display import display


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


# ## Read and process tracer input file from TracerLPM

# In[ ]:

# read input tracers
tracer_input_raw = pd.read_excel('../data/tracer input/Copy of TracerLPM_V_1_0B.xlsm',                            skiprows=3, sheetname='StoredTracerData', header=0)

col_list = ['Tracer', 'CFC-12', 'CFC-11', 'CFC-13', 'SF6', '3H', 'NO3-N']
tr_list = ['CFC-12', 'CFC-11', 'CFC-13', 'SF6', '3H', 'NO3-N']
tracer_input_df = tracer_input_raw.loc[:, col_list].copy()

# delete garbage header rows
tracer_input_df = tracer_input_df.iloc[3:, :]

# delete blank rows
tracer_input_df.dropna(axis=0, how='any', inplace=True)

# make sure all the tracer data is numeric
for col in col_list:
    tracer_input_df[col] = pd.to_numeric(tracer_input_df[col])

# reverse the date order so that oldest date is first
tracer_input_df = tracer_input_df.iloc[::-1]

# interpolate decimal years to a regular time series

# first change decimal years to equally spaced time series at approximately the same frequency (monthly)
# extract the year from the tracer decimal year
year = tracer_input_df.Tracer.astype(np.int32())
# convert year to a Datetime object
dto = pd.to_datetime(year, format='%Y')
# is it a leap year?
isleap = pd.DatetimeIndex(dto).is_leap_year
# find the number of days in the year
num_days_in_year = np.where(isleap, 366, 365)
# extract the fractional part of the year using modulus division (%)
fraction_of_year = tracer_input_df.Tracer % 1
# find the number of elapsed days within each year
num_days_by_year = fraction_of_year * num_days_in_year
# make the number of days a timedelta object
td = pd.to_timedelta(num_days_by_year, unit='D')
# sum the year (converted to a datetime object) and the timedelta
# make the datetime the index
tracer_input_df.set_index(dto + td, inplace=True)

# create a regular datetime series starting the middle of each month
# the frequency approximates the average month within this time span
freq = 30.436764
freq_str = '{}D'.format(freq)
dates = pd.date_range('1850-01-01', '2020-01-01', freq=freq_str) + pd.Timedelta(days=14)

# create a union of the original index and the desired index
t = tracer_input_df.index.union(dates)

# reindex the tracer df using the unioned index
tracer_input_df = tracer_input_df.reindex(t)

# fill in the gaps corresponding to the new dates using interpolation
tracer_input_df = tracer_input_df.interpolate('slinear')

# select only the rows with the new dates
tracer_input_df = tracer_input_df.loc[dates]

# delete blank rows
tracer_input_df.dropna(axis=0, how='any', inplace=True)

# create a spline function based on the last nonzero spl_per months (in this case 60 months)
# and use it to extrapolate the tracer concentrations to the end of the time series
spl_per = 60

for tr in tr_list:
    # index of last nonzero tracer value
    idx = tracer_input_df.loc[:, tr].nonzero()[0][-1]
    # dates of last spl_per monthly nonzero values
    x = tracer_input_df.iloc[(idx - spl_per):idx].index.to_julian_date()
    # concentrations of last spl_per monthly nonzero values
    y = tracer_input_df[tr].iloc[(idx - spl_per):idx]
    # create the spline function
    spl = UnivariateSpline(x, y)
    # dates of the ending zero tracer concentrations
    newx = tracer_input_df.iloc[idx:].index.to_julian_date()
    # use the spline to fill tracer concentrations to the end
    tracer_input_df.loc[idx:, tr] = spl(newx)   


# In[ ]:

class Tracer(object):
    def __init__(self, tracer_input_df, tr):
        self.tracer_input_df = tracer_input_df
        self.tr = tr
        self.tr_size = tracer_input_df.shape[0]
        self.input_date = tracer_input_df.index
        self.start_date = tracer_input_df.index.min()
        self.julian_time = self.input_date.to_julian_date()
        self.elapsed_time = self.julian_time - self.julian_time[0]
        
    def pad_tracer(self, rtd_size):
        self.tr_pad = np.zeros((self.tr_size + rtd_size * 2))

        self.lbackground = self.tracer_input_df.loc[self.tracer_input_df.index[0], self.tr]
        self.rbackground = self.tracer_input_df.loc[self.tracer_input_df.index[-1], self.tr]

        self.tr_pad[0 : rtd_size] = self.lbackground
        self.tr_pad[rtd_size : (self.tr_size + rtd_size)] = self.tracer_input_df.loc[:, self.tr]
        self.tr_pad[(self.tr_size + rtd_size) : ] = self.rbackground
        
class Sample_gdf(Tracer):
    def __init__(self, model_ws):
        super().__init__(tracer_input_df, tr)
        self.model_ws = model_ws
        self.src = os.path.join(self.model_ws, 'WEL', 'sample_gdf.shp')
        self.sample_gdf = gp.read_file(self.src)
        self.sample_gdf['STAID'] = self.sample_gdf.STAID.astype(np.int64())
        self.sample_gdf['DATES'] = pd.to_datetime(self.sample_gdf['DATES'])
        self.sample_gdf.index = self.sample_gdf['DATES']
        
    def extract_well(self, well, sa_in):
        self.well = well
        self.sa_in = sa_in
        self.well_data = self.sample_gdf.loc[self.sample_gdf.STAID == well, ['DATES', 'NetworkTyp', 'SuCode', self.sa_in]]
        self.well_data = self.well_data.dropna(axis=0, how='any', inplace=False)
        self.well_data['jdate'] = self.well_data.index.to_julian_date()
        self.well_data['well'] = well
        self.elapsed_sample_time = self.well_data['jdate'] - self.start_date.to_julian_date()
        
class Resepy(object):
    def __init__(self, por, freq):
        self.por = por
        self.freq = freq
        
    def get_fit(self, well_dict, method):
        # compute the pdf in log space
        self.px = np.linspace(0.1, 10, 50000, endpoint=True)
        self.pxp = np.exp(self.px) * self.por * 365.25
        self.p = well_dict['par'][method]
        self.py = self.explicit_pdf(self.px, *self.p)
        self.xi = np.arange(0, 1E+04 * self.freq, self.freq)
        self.yi = np.interp(self.xi, self.pxp, self.py)        
        self.pmf = self.yi / self.yi.sum()
        self.rtd_size = self.pmf.shape[0]
            
    def explicit_pdf(self, t, sh_1, lo_1, sc_1, sh_2, lo_2, sc_2, fy):
        _cdf_1 = dist.pdf(t, sh_1, lo_1, sc_1)
        _cdf_2 = dist.pdf(t, sh_2, lo_2, sc_2)
        return fy * _cdf_1 + (1 - fy) * _cdf_2

    def dk(self, thalf):
        self.pmfdk = self.pmf * np.exp(-self.xi * np.log(2) / thalf)


# In[ ]:

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 8,
        'sans-serif' : 'Arial'}

plt.rc('font', **font)

chem_list = [('3H', 'Trit', 'TU', 4499.88)]
dist = ss.weibull_min
form = 'add'
method = '{}_{}'.format(form, dist.name)

start = dt.datetime.strptime('1950-01-01', '%Y-%m-%d')
end = dt.datetime.strptime('2020-12-31', '%Y-%m-%d')

diff_step = 0.01
sigma = 3.
kwargs = {'ftol':1E-07, 'xtol':1E-07, 'max_nfev':200, 'loss':'soft_l1', 'diff_step':diff_step}
# kwargs = {'ftol':1E-07, 'xtol':1E-07, 'max_nfev':200, 'loss':'huber', 'diff_step':diff_step}

for model_ws in dir_list:
    sample_fit = pd.DataFrame()
    model = os.path.normpath(model_ws).split(os.sep)[2]
    nam_file = '{}.nam'.format(model)
    new_ws = os.path.join(model_ws, 'WEL')
    geo_ws = os.path.dirname(model_ws)

    print("working model is {}".format(model_ws))
    try:
        src = os.path.join(model_ws, 'fit_dict_wells_{}.pickle'.format(model))
        with open(src, 'rb') as f:
            fit_dict = pickle.load(f)

        s = Sample_gdf(model_ws)
        tr_in, sa_in, unit, dkr8 = chem_list[0]
        t = Tracer(tracer_input_df, tr_in)

        for well, d in fit_dict.items():
            s.extract_well(well, sa_in)
            if s.well_data.shape[0] > 0:
                fig, ax = plt.subplots(1, 1)
                def simeq(xi, por):
                    r = Resepy(por, freq)
                    r.get_fit(d, method)
                    r.dk(dkr8)
                    t.pad_tracer(r.rtd_size)
                    btc = np.convolve(r.pmfdk, t.tr_pad, mode='valid')
                    return np.interp(s.elapsed_sample_time, t.elapsed_time, btc[:t.tr_size])  

                bnds = (0.01, 0.40)
                porhat, cov = so.curve_fit(simeq, s.well_data['DATES'], s.well_data[sa_in], 
                                           bounds = bnds, method='trf', sigma=sigma, **kwargs)#or 'dogbox') 

                r = Resepy(porhat, freq)
                r.get_fit(d, method)
                r.dk(dkr8)
                t.pad_tracer(r.rtd_size)
                btc = np.convolve(r.pmfdk, t.tr_pad, mode='valid')
                yhat = np.interp(s.elapsed_sample_time, t.elapsed_time, btc[:t.tr_size])

                if np.isfinite(cov).all():
                    s.well_data['cov'] = np.diag(cov)[0]
                else:
                    dpor = porhat * (1 + diff_step)
                    neweq = simeq(s.well_data['DATES'], dpor)
                    X =  (yhat - neweq) / (diff_step * porhat)
                    scov = sigma / (X.T.dot(X))
                    s.well_data['cov'] = scov

                colu = 'calc_{}_'.format(tr_in)
                s.well_data[colu] = yhat
                s.well_data['por'] = r.por[0]

                sample_fit = sample_fit.append(s.well_data)    
                line = '{} {:0.3f}'.format(s.well, porhat[0])

                ax.plot(t.tracer_input_df.index, btc[:t.tr_size], color='k', label='_nolegend_', linestyle='solid');
                ax.plot(s.well_data['DATES'], yhat, linestyle='None', marker='x', color='k', label=line, ms=8);
                ax.plot(s.well_data['DATES'], s.well_data[sa_in], linestyle='None', marker='o', label='_nolegend_', color='k');

                ax.set_xlim(start, end)
                ax.set_ylim(0.1, 10000)
                ax.set_ylabel('{} in {}'.format('Tritium concentration', 'tritium units'))
                ax.set_yscale('log')
                ax.legend()

                fig.set_tight_layout(True)

                dst = '{} distribution for well {}.png'.format(tr_in, well)
                dst_pth = os.path.join(model_ws, dst)
                plt.savefig(dst_pth, dpi=600)
                plt.close()
        dst = os.path.join(model_ws, 'sample_dict_wells.csv')
        sample_fit.to_csv(dst)

    except (FileNotFoundError, ValueError):
        print('no node or sample files for {}'.format(model))

