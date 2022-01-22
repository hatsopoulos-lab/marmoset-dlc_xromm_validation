# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 11:05:27 2021

@author: Dalton
"""

import dill
import pandas as pd
import numpy as np
import os
import itertools
import h5py
from scipy.stats import mannwhitneyu, kruskal
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import ttest_ind
from scipy.stats import chi2

load_type = 'hdf' # can be 'hdf' or 'pickle'

project_base = r'Z:/dalton_moore/dryad_deposit'
results_base = os.path.join(project_base, 'results_presented_in_manuscript')    

class path:
    if load_type == 'pickle':
        data = os.path.join(results_base, 'pickle_versions')
        fileCode = '.pickle'
    elif load_type == 'hdf':
        data = os.path.join(results_base, 'hdf_versions')
        fileCode = '.h5'
    
    processed_results_path = os.path.join(data, 'trajData_posError_trackingQuality_pixelErrors_dlcBases%s' % fileCode)
    full_sweep_results = os.path.join(data, 'parameter_sweep%s' % fileCode)
    
def mannwhitney_uTest_by_segment():
    mannWhitney_pos = pd.DataFrame(np.empty((6, 7)),  
                                   columns=['group1', 'group2', 'n1', 'n2', 'U', 'p', 'r'])
    
    groups = [g[1].posErr.dropna() for g in allFramesError.groupby(['segment'])]
    groupNames = [g[0] for g in allFramesError.groupby(['segment'])]
    all_combos = list(itertools.combinations(groups, 2))
    all_comboNames = list(itertools.combinations(groupNames, 2))
    
    two_mwu = [mannwhitneyu(comb[0], comb[1], alternative='two-sided') for comb in all_combos]
    
    greater_f = [u[0] / (comb[0].shape[0] * comb[1].shape[0]) for u, comb in zip(two_mwu, all_combos)]
    greater_u = [1 - f for f in greater_f]
    
    for idx, (pair, comb, mwu, f, u) in enumerate(zip(all_comboNames, all_combos, two_mwu, greater_f, greater_u)):
        mannWhitney_pos.iloc[idx] = [pair[0], pair[1], comb[0].shape[0], comb[1].shape[0], mwu[0], mwu[1]*len(all_combos), f - u]    
    
    H_val_pos = kruskal(groups[0], groups[1], groups[2], groups[3])

    return mannWhitney_pos, H_val_pos

def mannwhitney_uTest_test_vs_train():
    mannWhitney_pos_by_set = pd.DataFrame(np.empty((1, 7)),  
                                          columns=['group1', 'group2', 'n1', 'n2', 'U', 'p', 'r'])
    
    groups = [g[1].posErr.dropna() for g in allFramesError.groupby(['newLabCategory'])]
    groupNames = [g[0] for g in allFramesError.groupby(['newLabCategory'])]
    all_combos = list(itertools.combinations(groups, 2))
    all_comboNames = list(itertools.combinations(groupNames, 2))
    
    two_mwu = [mannwhitneyu(comb[0], comb[1], alternative='two-sided') for comb in all_combos]
    
    greater_f = [u[0] / (comb[0].shape[0] * comb[1].shape[0]) for u, comb in zip(two_mwu, all_combos)]
    greater_u = [1 - f for f in greater_f]
    
    for idx, (pair, comb, mwu, f, u) in enumerate(zip(all_comboNames, all_combos, two_mwu, greater_f, greater_u)):
        mannWhitney_pos_by_set.iloc[idx] = [pair[0], pair[1], comb[0].shape[0], comb[1].shape[0], mwu[0], mwu[1]*len(all_combos), f - u]    
    
    groups = [g[1].velErr.dropna() for g in allFramesError.groupby(['newLabCategory'])]
    groupNames = [g[0] for g in allFramesError.groupby(['newLabCategory'])]
    all_combos = list(itertools.combinations(groups, 2))
    all_comboNames = list(itertools.combinations(groupNames, 2))
    
    two_mwu = [mannwhitneyu(comb[0], comb[1], alternative='two-sided') for comb in all_combos]
    
    greater_f = [u[0] / (comb[0].shape[0] * comb[1].shape[0]) for u, comb in zip(two_mwu, all_combos)]
    greater_u = [1 - f for f in greater_f]  

    return mannWhitney_pos_by_set

def mannwhitney_uTest_test_vs_train_by_segment():

    mannWhitney_pos_by_setAndSeg = pd.DataFrame(np.empty((4, 7)),  
                                              columns=['group1', 'group2', 'n1', 'n2', 'U', 'p', 'r'])
    
    groups = [g[1].posErr.dropna() for g in allFramesError.groupby(['segment', 'newLabCategory'])]
    groupNames = [g[0] for g in allFramesError.groupby(['segment', 'newLabCategory'])]
    all_combos = list(itertools.combinations(groups, 2))
    all_comboNames = list(itertools.combinations(groupNames, 2))
    
    all_combos = [comb for comb, names in zip(all_combos, all_comboNames) if names[0][0] == names[1][0]]
    all_comboNames = [names for names in all_comboNames if names[0][0] == names[1][0]]
    
    two_mwu = [mannwhitneyu(comb[0], comb[1], alternative='two-sided') for comb in all_combos]
    
    greater_f = [u[0] / (comb[0].shape[0] * comb[1].shape[0]) for u, comb in zip(two_mwu, all_combos)]
    greater_u = [1 - f for f in greater_f]
    
    for idx, (pair, comb, mwu, f, u) in enumerate(zip(all_comboNames, all_combos, two_mwu, greater_f, greater_u)):
        mannWhitney_pos_by_setAndSeg.iloc[idx] = [pair[0], pair[1], comb[0].shape[0], comb[1].shape[0], mwu[0], mwu[1], f - u]    
    
    H_val_pos_by_setAndSeg = kruskal(groups[0], groups[1], groups[2], groups[3],
                                     groups[4], groups[5], groups[6], groups[7])
    
    groups = [g[1].velErr.dropna() for g in allFramesError.groupby(['segment', 'newLabCategory'])]
    groupNames = [g[0] for g in allFramesError.groupby(['segment', 'newLabCategory'])]
    all_combos = list(itertools.combinations(groups, 2))
    all_comboNames = list(itertools.combinations(groupNames, 2))
    
    all_combos = [comb for comb, names in zip(all_combos, all_comboNames) if names[0][0] == names[1][0]]
    all_comboNames = [names for names in all_comboNames if names[0][0] == names[1][0]]
    
    two_mwu = [mannwhitneyu(comb[0], comb[1], alternative='two-sided') for comb in all_combos]
    
    greater_f = [u[0] / (comb[0].shape[0] * comb[1].shape[0]) for u, comb in zip(two_mwu, all_combos)]
    greater_u = [1 - f for f in greater_f]
    
    return mannWhitney_pos_by_setAndSeg, H_val_pos_by_setAndSeg

def compute_FVAF():

    first = True
    for dp, xp in zip(trajectoryData.dlc, trajectoryData.xromm):
        if first:
            col_dlc      = dp
            col_xromm    = xp            
            first = False
        else:
            col_dlc      = np.dstack((col_dlc,      dp))
            col_xromm    = np.dstack((col_xromm,    xp))
    
    mean_xromm    = np.nanmean(col_xromm, axis = -1)
    
    mean_xromm    = np.repeat(np.reshape(mean_xromm, (mean_xromm.shape[0], mean_xromm.shape[1], 1)), 
                              col_xromm.shape[-1], axis = 2)
    
    pre_sum_numer = (col_dlc - col_xromm)**2
    pre_sum_denom = (col_xromm - mean_xromm)**2
    
    perCut = 100
    pre_sum_denom[pre_sum_numer > np.nanpercentile(pre_sum_numer, perCut)] = np.nan
    pre_sum_numer[pre_sum_numer > np.nanpercentile(pre_sum_numer, perCut)] = np.nan
    
    pre_sum_numer = np.reshape(pre_sum_numer, (pre_sum_numer.shape[0], 3 * pre_sum_numer.shape[2]))
    pre_sum_denom = np.reshape(pre_sum_denom, (pre_sum_denom.shape[0], 3 * pre_sum_denom.shape[2]))
    
    pre_sum_numer = pre_sum_numer.flatten()
    pre_sum_denom = pre_sum_denom.flatten()
    
    total_fvaf_pos = 1 - np.nansum(pre_sum_numer) / np.nansum(pre_sum_denom) 

    return total_fvaf_pos

global compute_parameter_regression_with_logLikeRatio
def compute_parameter_regression_with_logLikeRatio(sweep_results, depVar = 'median_err'):
            
    dependent = sweep_results[depVar]
    param_names = [name for name in sweep_results.columns if name not in ['median_err', 'mean_err', 'trackPercent']]
    parameters = sweep_results.loc[:, param_names]
    parameters_and_interactions = parameters.copy()
    
    for p1, par1 in enumerate(parameters.columns):
        for p2, par2 in enumerate(parameters.columns):
            if p2 > p1:
                parameters_and_interactions[par1+'*'+par2] = parameters[par1] * parameters[par2]
                
    parameters = sm.add_constant(parameters)
    parameters_and_interactions = sm.add_constant(parameters_and_interactions)

    simple_full = sm.OLS(dependent, parameters).fit()
    interaction_full = sm.OLS(dependent, parameters_and_interactions).fit()
    
    logLikeRatio = []
    pvals = []
    for par in parameters.columns[1:]:    
        reduced_params = parameters.drop(par, axis=1)
        reduced_model = sm.OLS(dependent, reduced_params).fit()
        
        #calculate likelihood ratio Chi-Squared test statistic
        logLikeRatio.append(-2*(reduced_model.llf - simple_full.llf))
        pvals.append(chi2.sf(logLikeRatio[-1], 2))
        
    simple_model_stats = pd.DataFrame(zip(parameters.columns[1:], logLikeRatio, pvals),
                                      columns = ['parameter', 'LogLikelihoodRatio', 'pVal'])

    logLikeRatio = []
    pvals = []
    for par in parameters_and_interactions.columns[1:]:    
        reduced_params = parameters_and_interactions.drop(par, axis=1)
        reduced_model = sm.OLS(dependent, reduced_params).fit()
        
        #calculate likelihood ratio Chi-Squared test statistic
        logLikeRatio.append(-2*(reduced_model.llf - interaction_full.llf))
        pvals.append(chi2.sf(logLikeRatio[-1], 2))
        
    interaction_model_stats = pd.DataFrame(zip(parameters_and_interactions.columns[1:], logLikeRatio, pvals),
                                      columns = ['parameter', 'LogLikelihoodRatio', 'pVal'])
    
    simple_model_stats.sort_values('pVal', inplace=True, ignore_index=True)
    interaction_model_stats.sort_values('pVal', inplace=True, ignore_index=True)

    return interaction_model_stats, simple_model_stats

def find_minimal_model(sweep_results, depVar, simple_model_stats):
            
    dependent = sweep_results[depVar]
    
    simple_model_stats.index = simple_model_stats.parameter
    parameters = sweep_results.loc[:, simple_model_stats.parameter]
                
    parameters = sm.add_constant(parameters)

    simple_full = sm.OLS(dependent, parameters).fit()
    
    logLikeRatio = []
    pvals = []
    params_kept = []
    for idx in range(len(simple_model_stats.parameter)+1, 0, -1):
        params_to_drop = parameters.columns[idx:]
        reduced_params = parameters.drop(params_to_drop, axis=1)
        
        reduced_model = sm.OLS(dependent, reduced_params).fit()
        
        #calculate likelihood ratio Chi-Squared test statistic
        
        logLikeRatio.append(-2*(reduced_model.llf - simple_full.llf))
        pvals.append(chi2.sf(logLikeRatio[-1], 2))
        params_kept.append(tuple(reduced_params.columns))
        
    minimal_model = pd.DataFrame(zip(params_kept, logLikeRatio, pvals),
                                 columns = ['model_params', 'LogLikelihoodRatio', 'pVal'])
    minimal_model.sort_values('pVal', inplace=True, ignore_index=True)
    
    return minimal_model

def load_hdf_to_pandas_dataframe(f, key):
    col_names = list(f[key].keys())
    index = list(f[['%s/%s' % (key, col) for col in col_names if col == 'index'][0]])
    if type(index[0]) == np.bytes_:
        index = [string.decode('UTF-8') for string in index]
    col_names.remove('index')
    tmp_df_size = (len(index), len(f[key].keys()) - 1)
    tmp_df = pd.DataFrame(data = np.empty(tmp_df_size), 
                          columns = col_names,
                          index = index)
    for col in f[key].keys():
        if col != 'index':
            arr = np.array(f['%s/%s' % (key, col)])
            if type(arr[0]) == np.bytes_:
                arr = [string.decode('UTF-8') for string in arr]
            tmp_df[col] = arr
    return key.split('/')[-1], tmp_df 

def load_hdf_to_numpy_array(f, key):
    tmp_arr = np.array(f[key])        
    return key.split('/')[-1], tmp_arr

def load_hdf_to_float(f, key):
    tmp_arr = np.array(f[key])       
    return key.split('/')[-1], tmp_arr

def load_hdf_to_list(f, key):
    tmp_list = [np.array(f['%s/%s' % (key, sub_key)]) for sub_key in f[key].keys()]
    return key.split('/')[-1], tmp_list

def load_hdf_to_class(f, key):
    class tmp_class:
        tmp = []
    for sub_key in f[key].keys():
        full_key = '%s/%s' % (key, sub_key)
        if f[full_key].attrs['dtype'] == 'pandas.DataFrame':
            name, tmp = load_hdf_to_pandas_dataframe(f, full_key)
            exec('tmp_class.' + name + ' = tmp')
        elif f[full_key].attrs['dtype'] == 'numpy.ndarray':
            name, tmp = load_hdf_to_numpy_array(f, full_key)
            exec('tmp_class.' + name + ' = tmp')
        elif f[full_key].attrs['dtype'] == 'list':
            name, tmp = load_hdf_to_list(f, full_key)
            exec('tmp_class.' + name + ' = tmp')
        elif f[full_key].attrs['dtype'] == 'class':
            name, tmp = load_hdf_to_class(f, full_key)
            exec('tmp_class.' + name + ' = tmp')
        elif f[full_key].attrs['dtype'] == 'float':
            name, tmp = load_hdf_to_float(f, full_key)
            exec('tmp_class.' + name + ' = tmp')
    del tmp_class.tmp
    return key, tmp_class

def load_hdf_to_original(hdf_file):
    with h5py.File(hdf_file, 'r') as f:
        names_list = []
        data_list = []
        for key in f.keys():
            if f[key].attrs['dtype'] == 'pandas.DataFrame':
                name, tmp = load_hdf_to_pandas_dataframe(f, key)
            elif f[key].attrs['dtype'] == 'numpy.ndarray':
                name, tmp = load_hdf_to_numpy_array(f, key)
            elif f[key].attrs['dtype'] == 'list':
                name, tmp = load_hdf_to_list(f, key)
            elif f[key].attrs['dtype'] == 'class':
                name, tmp = load_hdf_to_class(f, key)
            names_list.append(name)
            data_list.append(tmp)
    return names_list, data_list

if __name__ == "__main__":
            
    if load_type == 'pickle':
        with open(path.processed_results_path, 'rb') as fp:
            trajectoryData, posErrResults, trackingQuality, allFramesError, dlc_bases = dill.load(fp) 
        with open(path.full_sweep_results, 'rb') as fp:
            sweep_results = dill.load(fp)            
    elif load_type == 'hdf':
        names_list, data_list = load_hdf_to_original(path.processed_results_path)
        for name, data in zip(names_list, data_list):
            exec(name + ' = data')            
        names_list, data_list = load_hdf_to_original(path.full_sweep_results)
        for name, data in zip(names_list, data_list):
            exec(name + ' = data')
            
    mannWhitney_pos, H_val_pos = mannwhitney_uTest_by_segment()
    mannWhitney_pos_by_set = mannwhitney_uTest_test_vs_train()
    mannWhitney_pos_by_setAndSeg, H_val_pos_by_setAndSeg = mannwhitney_uTest_test_vs_train_by_segment()
    pos_fvaf = compute_FVAF()
    
    interaction_errorModel_stats, simple_errorModel_stats     = compute_parameter_regression_with_logLikeRatio(sweep_results, 'median_err')
    interaction_percentModel_stats, simple_percentModel_stats = compute_parameter_regression_with_logLikeRatio(sweep_results, 'trackPercent')
    
    minimal_error_model   = find_minimal_model(sweep_results, 'median_err'  , simple_errorModel_stats)
    minimal_percent_model = find_minimal_model(sweep_results, 'trackPercent', simple_percentModel_stats)
    
    medianErrs = allFramesError.groupby(['segment']).median().posErr
    median_without_torso = allFramesError.loc[allFramesError['segment'] != 'torso', 'posErr'].median()
    
    number_of_test_timepoints = -278
    for d in trajectoryData.dlc:
        number_of_test_timepoints += np.sum(~np.isnan(np.nanmean(d, axis = 0)[0])) 