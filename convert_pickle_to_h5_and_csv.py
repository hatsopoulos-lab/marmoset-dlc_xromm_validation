# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:28:13 2022

@author: Dalton
"""

import dill
import pandas as pd
import numpy as np
import os
import glob
import inspect
import h5py

project_base = r'Z:/dalton_moore/dryad_deposit'

results_base       = os.path.join(project_base, 'results_presented_in_manuscript')
csv_folder = os.path.join(results_base, 'csv_versions')
hdf_folder = os.path.join(results_base, 'hdf_versions')
pickle_folder = os.path.join(results_base, 'pickle_versions') 

pickle_to_hdf_variable_names = [['no_epipolar_lines_pixel_error', 'human_error', 'all_human_errors', 'dlc_pixel_error', 'all_dlc_pixel_errors'], 
                                ['sweep_results'], 
                                ['trainingset_results'], 
                                ['trainingset_results'], 
                                ['trajectoryData', 'posErrResults', 'trackingQuality', 'allFramesError', 'dlc_bases']]

def save_pandas_dataframe_hdf(f, variableName, data):
    group = f.create_group(variableName)
    group.attrs['dtype'] = 'pandas.DataFrame'
    if type(data.index[0]) == str:
        data_save = f.create_dataset('%s/%s' % (variableName, 'index'), data = np.array(data.index, dtype='S'))
    else:
        data_save = f.create_dataset('%s/%s' % (variableName, 'index'), data = data.index)
    for col in data.columns:
        if type(data[col][0]) == str:
            data_save = f.create_dataset('%s/%s' % (variableName, col), data = np.array(data[col], dtype='S'))
        else:
            data_save = f.create_dataset('%s/%s' % (variableName, col), data = data[col])
    return

def save_list_hdf(f, variableName, data):
    group = f.create_group(variableName)
    group.attrs['dtype'] = 'list'
    idxDigits = len(str(len(data)))
    for idx, arr in enumerate(data):
        data_save = f.create_dataset('%s/%s' % (variableName, str(idx).zfill(idxDigits)), data = arr)
    return

def save_numpy_array_hdf(f, variableName, data):
    data_save = f.create_dataset(variableName, data = data)
    data_save.attrs['dtype'] = 'numpy.ndarray'
    return    

def save_float_hdf(f, variableName, data):
    data_save = f.create_dataset(variableName, data = data)
    data_save.attrs['dtype'] = 'float'
    return   

def save_class_hdf(f, variableName, data):
    topGroup = f.create_group(variableName)
    topGroup.attrs['dtype'] = 'class'
    
    varKeys = data.__dict__.keys()      
    varKeys = [key for key in varKeys 
               if 'vel' not in key
               and 'speed' not in key
               and key != 'mae'
               and '__module__' not in key 
               and '__doc__' not in key 
               and '__dict__' not in key 
               and '__weakref__' not in key]
    for key in varKeys:                    
        data_subvariable = getattr(data, key)
        if type(data_subvariable) == pd.DataFrame:
            save_pandas_dataframe_hdf(f, '%s/%s' %(variableName, key), data_subvariable)
        elif type(data_subvariable) == np.ndarray:
            save_numpy_array_hdf(f, '%s/%s' %(variableName, key), data_subvariable)
        elif type(data_subvariable) == list:
            save_list_hdf(f, '%s/%s' %(variableName, key), data_subvariable)
        elif type(data_subvariable) in [float, np.float16, np.float32, np.float64]:
            save_float_hdf(f, '%s/%s' %(variableName, key), data_subvariable)
    return

def pickle_to_hdf():
    for pkl, varNames in zip(sorted(glob.glob(os.path.join(pickle_folder, '*'))), pickle_to_hdf_variable_names):
        varFileBase = os.path.basename(pkl).split('.pickle')[0]
        with open(pkl, 'rb') as f:
            pklData = dill.load(f)
        if len(varNames) == 1:
            pklData = [pklData]
        with h5py.File(os.path.join(hdf_folder, varFileBase + '.h5'), 'a', libver = 'earliest') as f:
            for name, data in zip(varNames, pklData):
                print((varFileBase, name))    
                if inspect.isclass(data):
                    save_class_hdf(f, name, data)
                elif type(data) == pd.DataFrame:
                    save_pandas_dataframe_hdf(f, name, data)
                elif type(data) == np.ndarray:
                    save_numpy_array_hdf(f, name, data)
                elif type(data) == list:
                    save_list_hdf(f, name, data)

pickle_to_hdf()

# pickle_to_csv_variable_names = [['no_epipolar_lines_pixel_error', 'human_error', 'all_human_errors' 'dlc_pixel_error', 'all_dlc_pixel_errors'], 
#                                 ['sweep_results'], 
#                                 ['trainingset_results'], 
#                                 ['trainingset_results'], 
#                                 ['posErrResults', 'trackingQuality', 'allFramesError', 'dlc_bases']]

# def save_pandas_dataframe_csv(variableName, data):
#     data.to_
#     return

# def save_list_csv(variableName, data):

#     idxDigits = len(str(len(data)))
#     for idx, arr in enumerate(data):
#         str(idx).zfill(idxDigits) #data = arr)
#     return

# def save_numpy_array_csv(variableName, data):

#     return    

# def save_class_csv(variableName, data):    
#     varKeys = data.__dict__.keys()      
#     varKeys = [key for key in varKeys 
#                if 'vel' not in key
#                and 'speed' not in key
#                and key != 'mae'
#                and '__module__' not in key 
#                and '__doc__' not in key 
#                and '__dict__' not in key 
#                and '__weakref__' not in key]
#     for key in varKeys:                    
#         data_subvariable = getattr(data, key)
#         if type(data_subvariable) == pd.DataFrame:
#             save_pandas_dataframe_csv('%s/%s' %(variableName, key), data_subvariable)
#         elif type(data_subvariable) == np.ndarray:
#             save_numpy_array_csv('%s/%s' %(variableName, key), data_subvariable)
#         elif type(data_subvariable) == list:
#             save_list_csv('%s/%s' %(variableName, key), data_subvariable)
#     return
# def pickle_to_csv():
#     for pkl, varNames in zip(sorted(glob.glob(os.path.join(pickle_folder, '*'))), pickle_to_csv_variable_names):
#         varFileBase = os.path.basename(pkl).split('.pickle')[0]
#         with open(pkl, 'rb') as f:
#             pklData = dill.load(f)
#         if len(varNames) == 1:
#             pklData = [pklData]
#         for name, data in zip(varNames, pklData):
#             print((varFileBase, name))    
#             if inspect.isclass(data):
#                 save_class_csv(name, data)
#             elif type(data) == pd.DataFrame:
#                 save_pandas_dataframe_csv(name, data)
#             elif type(data) == np.ndarray:
#                 save_numpy_array_csv(name, data)
#             elif type(data) == list:
#                 save_list_csv(name, data)
#             elif type(data) in [float, np.float16, np.float32, np.float64]:
#                 save_float_csv(name, data)
# pickle_to_csv()
