# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:06:47 2022

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
hdf_folder = os.path.join(results_base, 'hdf_versions')

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

for hdf_file in sorted(glob.glob(os.path.join(hdf_folder, '*'))):
    names_list, data_list = load_hdf_to_original(hdf_file)
    for name, data in zip(names_list, data_list):
        exec(name + ' = data')