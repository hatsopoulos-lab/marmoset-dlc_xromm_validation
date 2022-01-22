# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 14:21:33 2021

@author: Dalton
"""

import shutil
import subprocess
import os
import deeplabcut
import glob
import toml

iteration = 1
 
dates = ['2019_04_14', '2019_04_15']

param_category = ['filter', 'filter', 'triangulation', 'triangulation', 'triangulation', 'triangulation']
test_params     = {'offset_threshold'       : 20, 
                   'n_back'                 : 5, 
                   'scale_smooth'           : 6, 
                   'scale_length'           : 2, 
                   'reproj_error_threshold' : 8,
                   'score_threshold'        : 0.45} 

param_names  = list(test_params)
param_values = list(test_params.values())

projectpath = '/home/marms/Documents/dlc_local/combined_dlc_xromm_validation-Dalton-2021-08-28'
aniposepath = '/media/marms/DATA/anipose_dlc_validation'
anipose_folders = sorted(glob.glob(os.path.join(aniposepath, 'test_training_fraction_effect_iteration0', '*')))

anipose_codePath = '/home/marms/anaconda3/envs/dlc/lib/python3.7/site-packages/anipose/pose_videos.py'
with open(anipose_codePath, 'r') as file:
    text = file.read()

dlc_config = os.path.join(projectpath,'config.yaml')
dlc_cfg    = deeplabcut.auxiliaryfunctions.read_config(dlc_config)

original_iteration      = dlc_cfg['iteration']
original_snapshotindex  = dlc_cfg['snapshotindex']
original_train_fraction = dlc_cfg['TrainingFraction']

dlc_cfg['iteration'] = iteration

for anipose_folder in anipose_folders:

    os.chdir(anipose_folder)
    ani_config = os.path.join(anipose_folder, 'config.toml')
    ani_cfg = toml.load(ani_config)
    
    snapIdx = int(anipose_folder.split('snapshotindex_')[1])
    train_frac = float(anipose_folder.split('trainFrac_')[1][:2]) / 100
    shuffle = anipose_folder.split('shuffle_')[1][0]
    
    dlc_cfg['TrainingFraction'] = [train_frac]
    dlc_cfg['snapshotindex']    = snapIdx
    deeplabcut.auxiliaryfunctions.write_config(dlc_config, dlc_cfg)
    prev_shuffle = text.split('shuffle=')[1][0]    
    text = text.replace('shuffle=' + prev_shuffle, 'shuffle=' + shuffle)
    with open(anipose_codePath, 'w') as file:
        file.write(text)

    for cat, key, param in zip(param_category, param_names, param_values):
        ani_cfg[cat][key] = param
    
    ani_cfg['filter']['type'] = 'viterbi'
    
    with open(ani_config, 'w') as f:
        toml.dump(ani_cfg, f)

    subprocess.call(['anipose', 'analyze'])
    subprocess.call(['anipose', 'filter'])
    
    for date in dates:        
        shutil.copytree(os.path.join(anipose_folder, date, 'pose-2d'), 
                        os.path.join(anipose_folder, date, 'pose-2d-raw'))
        shutil.copytree(os.path.join(anipose_folder, date, 'pose-2d-filtered'), 
                        os.path.join(anipose_folder, date, 'pose-2d-viterbi-only'))
        
        shutil.rmtree(os.path.join(anipose_folder, date, 'pose-2d'))
        shutil.rmtree(os.path.join(anipose_folder, date, 'pose-2d-filtered'))
        
        shutil.copytree(os.path.join(anipose_folder, date, 'pose-2d-viterbi-only'),
                        os.path.join(anipose_folder, date, 'pose-2d'))
    
    ani_cfg['filter']['type'] = 'autoencoder'  
    with open(ani_config, 'w') as f:
        toml.dump(ani_cfg, f)
        
    subprocess.call(['anipose', 'train-autoencoder'])
    subprocess.call(['anipose', 'filter'])
    subprocess.call(['anipose', 'triangulate']) 

print("resetting snapshotindex and iteration")
dlc_cfg['iteration'] = original_iteration
dlc_cfg['snapshotindex'] = original_snapshotindex
dlc_cfg['TrainingFraction'] = original_train_fraction
deeplabcut.auxiliaryfunctions.write_config(dlc_config, dlc_cfg)

text = text.replace('shuffle=' + prev_shuffle, 'shuffle=1')
with open(anipose_codePath, 'w') as file:
    file.write(text)

                                