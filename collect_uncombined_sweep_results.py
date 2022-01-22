# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 00:00:01 2022

@author: Dalton
"""
import pickle
import dill
import pandas as pd
import numpy as np
import os
import glob

project_base = r'Z:/dalton_moore/dryad_deposit'
results_base = os.path.join(project_base, 'results_presented_in_manuscript/pickle_versions') 
anipose_base = os.path.join(project_base, 'anipose_files')
dlc_base     = os.path.join(project_base, 'deeplabcut_files/combined_dlc_xromm_validation-Dalton-2021-08-28')

class dpath:
    base = sorted(glob.glob(os.path.join(anipose_base, 'parameter_sweep/all_combos_results/*')))
    results_storage_path = os.path.join(results_base, 'final_results/parameter_sweep_with_postReprojError.pickle')
    sweep_extra_data_path = os.path.join(results_base, 'all_datasets_results_from_sweep')

with open(dpath.results_storage_path, 'rb') as fp:
    full_sweep_results = np.array(dill.load(fp))

dataPaths = sorted(glob.glob(os.path.join(dpath.sweep_extra_data_path, '*')))        
for datapath in dataPaths:
    print(datapath)
    with open(datapath, 'rb') as fp:
        posErrResults, trackingQuality = dill.load(fp)
    
    parameter_identifier = os.path.basename(datapath).split('param_vals_')[1][:-7]
    parameter_details = parameter_identifier.split('_')
    parameter_details[-2] = float(parameter_details[-2][-2:]) / 100
    parameter_details = [float(val) for val in parameter_details]
    tmp = np.hstack((parameter_details,
                     posErrResults.descriptiveStats.loc['all', 'pos_MeanErr'],
                     posErrResults.descriptiveStats.loc['all', 'pos_MedErr' ],
                     trackingQuality.total_percentTracked))
    
    skip = False
    for row in full_sweep_results:
        if np.sum(tmp[:7] == row[:7]) == 7:
            skip = True
            continue
    print(skip)
    
    if not skip:
        full_sweep_results = np.vstack((full_sweep_results, 
                                        np.reshape(tmp, (1, len(tmp)))
                                        ))

full_sweep_results = pd.DataFrame(full_sweep_results,
                                  columns = ['offset_threshold', 
                                             'n_back', 
                                             'scale_smooth', 
                                             'scale_length', 
                                             'reproj_error_threshold', 
                                             'score_threshold',
                                             'post_reproj_threshold',
                                             'mean_err', 
                                             'median_err', 
                                             'trackPercent'])
full_sweep_results = full_sweep_results.sort_values(by = ['offset_threshold', 
                                                          'n_back', 
                                                          'scale_smooth', 
                                                          'scale_length', 
                                                          'reproj_error_threshold', 
                                                          'score_threshold',
                                                          'post_reproj_threshold'], ignore_index=True)
with open(dpath.results_storage_path, 'wb') as fp:
    dill.dump(full_sweep_results, fp, recurse = True, protocol = pickle.HIGHEST_PROTOCOL) 


