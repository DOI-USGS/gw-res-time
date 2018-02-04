import numpy as np
import pandas as pd
import scipy.optimize as so
import scipy.stats as ss

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
        super().__init__(self, tracer_input_df, tr)
        self.tracer_input_df = tracer_input_df
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