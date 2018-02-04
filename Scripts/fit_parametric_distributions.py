def fit_dists(ly, lprt, dist_list):
    import numpy as np
    import fit_parametric_distributions
    import pandas as pd
    import scipy.stats as ss
    import scipy.optimize as so
    
    ## Initialize dictionaries and choose distributions

    param_dict = {}
    cdf_dict = {}
    error_dict = {}
    tt_dict = {}
    pars = ['shape_early', 'loc_early', 'scale_early', 'shape_late', 
            'loc_late', 'scale_late', 'pct_early', 'error']
    fit_param_df = pd.DataFrame(columns=pars)

    # dist_list = [ss.invgauss, ss.gamma, ss.weibull_min] 
    first = lprt.min()
    s = ly.shape[0]
    tt_dict['rt'] = lprt
    tt_dict['rt_cdf'] = ly


    ## Define parametric models

    def distfit(t, sh_e, sc_e):
        first = t.min()
        _cdf = dist.cdf(t, sh_e, first, sc_e)
        return  _cdf

    def explicit(t, sh_e, sc_e, sh_l, sc_l, fy):
        first = t.min()
        _cdf_e = dist.cdf(t, sh_e, first, sc_e)
        _cdf_l = dist.cdf(t, sh_l, first, sc_l)
        return fy * _cdf_e + (1 - fy) * _cdf_l

    def implicit(t, sh_e, sc_e, sh_l, sc_l):
        first = t.min()
        _cdf_e = dist.cdf(t, sh_e, first, sc_e)
        _cdf_l = dist.cdf(t, sh_l, first, sc_l)
        return  _cdf_e / (1 + _cdf_e - _cdf_l)

    ## Fit all distributions

    # Two methods are tried for each model (listed below). The final model is the one with the lowest RMSE.

    # * 'trf' : Trust Region Reflective algorithm, particularly suitable for large sparse problems with bounds. Generally robust method.
    # * 'dogbox' : dogleg algorithm with rectangular trust regions, typical use case is small problems with bounds. Not recommended for problems with rank-deficient Jacobian.


    print('starting parametric fits...')

    for dist in dist_list:
        # fit unimodal dists
        lab = 'uni_{}'.format(dist.name)
        print('fitting {}'.format(lab))

        bnds = (0, [+np.inf, +np.inf])

        try:
            up1, cov = so.curve_fit(distfit, lprt, ly, bounds = bnds, method='trf')
            e1_cdf = distfit(lprt, *up1)
            e1 = ly - e1_cdf
            sse1 = e1.T.dot(e1)
        except Exception as e: 
            print(lab, e)
            sse1 = np.inf

        try:
            up2, cov = so.curve_fit(distfit, lprt, ly, bounds = bnds, method='dogbox')
            e2_cdf = distfit(lprt, *up2)
            e2 = ly - e2_cdf
            sse2 = e2.T.dot(e2)
        except Exception as e: 
            print(lab, e)
            sse2 = np.inf

        if sse1 < sse2:
            up = up1
            e_cdf = e1_cdf
            e = np.sqrt(sse1 / s)
            meth = 'trf'
        elif sse1 > sse2:
            up = up2
            e_cdf = e2_cdf
            e = np.sqrt(sse2 / s)
            meth = 'dogbox'
        else:
            up = (0, 0)
            e_cdf = np.nan
            e = np.nan
            meth = 'none'      

        up = np.insert(up, 1, first)
        param_dict[lab] = up
        error_dict[lab] = e
        cdf_dict[lab] = e_cdf
        fit_param_df.loc[lab, pars[:3]] = up
        fit_param_df.loc[lab, 'meth'] = meth
        fit_param_df.loc[lab, 'error'] = error_dict[lab]

        #     fit bimodal dists with implicit mixing 
        lab = 'imp_{}'.format(dist.name)
        print('fitting {}'.format(lab))

        bnds = (0, [+np.inf, +np.inf, +np.inf, +np.inf])
        p0 = (up[0], up[2], up[0], up[2])

        try:
            ip1, cov = so.curve_fit(implicit, lprt, ly, bounds = bnds, method='trf')
            e1_cdf = implicit(lprt, *ip1)
            e1 = ly - e1_cdf
            sse1 = e1.T.dot(e1)
        except Exception as e: 
            print(lab, e)
            sse1 = np.inf

        try:
            ip2, cov = so.curve_fit(implicit, lprt, ly, bounds = bnds, method='dogbox')
            e2_cdf = implicit(lprt, *ip2)
            e2 = ly - e2_cdf
            sse2 = e2.T.dot(e2)
        except Exception as e: 
            print(lab, e)
            sse2 = np.inf

        if sse1 < sse2:
            ip = ip1
            e_cdf = e1_cdf
            e = np.sqrt(sse1 / s)
            meth = 'trf'
        elif sse1 > sse2:
            ip = ip2
            e_cdf = e2_cdf
            e = np.sqrt(sse2 / s)
            meth = 'dogbox'
        else:
            ip = (0, 0, 0, 0)
            e_cdf = np.nan
            e = np.nan
            meth = 'none'      

        error_dict[lab] = e
        cdf_dict[lab] = e_cdf

        ip = np.insert(ip, 1, first)
        ip = np.insert(ip, 4, first)
        param_dict[lab] = ip

        fit_param_df.loc[lab, pars[:6]] = ip
        fit_param_df.loc[lab, 'error'] = e
        fit_param_df.loc[lab, 'meth'] = meth

        # fit bimodal dists with explicit mixing 
        lab = 'add_{}'.format(dist.name)
        print('fitting {}'.format(lab))
        bnds = (0, [+np.inf, +np.inf, +np.inf, +np.inf, 1.0]) 
        p0 = (up[0], up[2], up[0], up[2], 1.0)

        try:
            ep1, cov = so.curve_fit(explicit, lprt, ly, bounds=bnds, method='trf')
            e1_cdf = explicit(lprt, *ep1)
            e1 = ly - e1_cdf
            sse1 = e1.T.dot(e1)        
        except Exception as e: 
            print(lab, e)
            sse1 = np.inf

        try:
            ep2, cov = so.curve_fit(explicit, lprt, ly, bounds=bnds, method='dogbox')
            e2_cdf = explicit(lprt, *ep2)
            e2 = ly - e2_cdf
            sse2 = e2.T.dot(e2)
        except Exception as e: 
            print(lab, e)
            sse2 = np.inf

        if sse1 < sse2:
            ep = ep1
            e_cdf = e1_cdf
            e = np.sqrt(sse1 / s)
            meth = 'trf'
        elif sse1 > sse2:
            ep = ep2
            e_cdf = e2_cdf
            e = np.sqrt(sse2 / s)
            meth = 'dogbox'
        else:
            ep = (0, 0, 0, 0, 0)
            e_cdf = np.nan
            e = np.nan
            meth = 'none'      

        ep = np.insert(ep, 1, first)
        ep = np.insert(ep, 4, first)
        param_dict[lab] = ep
        error_dict[lab] = e
        cdf_dict[lab] = e_cdf
        fit_param_df.loc[lab, pars[:7]] = ep
        fit_param_df.loc[lab, 'meth'] = meth
        fit_param_df.loc[lab, 'error'] = error_dict[lab]

    print('   ... done')

    return {'cdf' : cdf_dict, 'par' : param_dict, 'err' : error_dict, 'tt' : tt_dict}
    
def read_endpoints(endpoint_file, dis, time_dict):
    import pandas as pd
    # count the number of header lines
    i = 0
    with open(endpoint_file) as f:
        while True:
            line = f.readline()
            i += 1
            if 'END HEADER' in line:
                break
            elif not line:
                break

    # columns names from MP6 docs 
    cols = ['Particle ID', 'Particle Group', 'Status', 'Initial Time', 'Final Time', 'Initial Grid', 
            'Initial Layer', 'Initial Row', 'Initial Column', 'Initial Cell Face', 'Initial Zone', 
            'Initial Local X', 'Initial Local Y', 'Initial Local Z', 'Initial Global X', 'Initial Global Y', 
            'Initial Global Z', 'Final Grid', 'Final Layer', 'Final Row', 'Final Column', 'Final Cell Face', 
            'Final Zone', 'Final Local X', 'Final Local Y', 'Final Local Z', 'Final Global X', 
            'Final Global Y', 'Final Global Z', 'Label']            

    # read the endpoint data
    ep_data = pd.read_table(endpoint_file, names=cols, header=None, skiprows=i, delim_whitespace=True)

    # select only 'Normally Terminated' particles; status code = 2
    ep_data = ep_data.loc[ep_data.Status == 2, :]

    # calculate initial and final zero-based sequence numbers from [lay, row, col]
    # tmp = ep_data[['Initial Layer', 'Initial Row', 'Initial Column']].values.tolist()
    # ep_data['initial_node_num'] = np.array(dis.get_node(tmp)) - 1

    tmp = ep_data[['Initial Layer', 'Initial Row', 'Initial Column']] - 1
    ep_data['initial_node_num'] = dis.get_node(tmp.values.tolist())

    tmp = ep_data[['Final Layer', 'Final Row', 'Final Column']] - 1
    ep_data['final_node_num'] = dis.get_node(tmp.values.tolist())
    
    # calculate particle travel time in years
    ep_data['rt'] = (ep_data['Final Time'] - ep_data['Initial Time']) / time_dict[dis.itmuni] / 365.25
    ep_data.set_index('initial_node_num', drop=True, inplace=True)
    return ep_data

