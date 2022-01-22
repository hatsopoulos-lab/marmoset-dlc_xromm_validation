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
test_params     = {'offset_threshold'       : [20, 25, 30], 
                   'n_back'                 : [1, 3, 5], 
                   'scale_smooth'           : [0, 2, 4, 6, 8, 10], 
                   'scale_length'           : [0, 2, 4, 6], 
                   'reproj_error_threshold' : [5, 8],
                   'score_threshold'        : [0.15, 0.3, 0.45, 0.6]} 

param_names  = list(test_params)
param_values = list(test_params.values())

projectpath = '/home/marms/Documents/dlc_local/combined_dlc_xromm_validation-Dalton-2021-08-28'
aniposepath = '/media/marms/DATA/anipose_dlc_validation/parameter_sweep'

dlc_config=os.path.join(projectpath,'config.yaml')
dlc_cfg=deeplabcut.auxiliaryfunctions.read_config(dlc_config)

original_iteration=dlc_cfg['iteration']
dlc_cfg['iteration'] = iteration
original_snapshotindex = dlc_cfg['snapshotindex']
original_train_fraction = dlc_cfg['TrainingFraction']

anipose_folder = glob.glob(os.path.join(aniposepath, 'iteration_%d_trainFrac_*' % iteration))[0]   
os.chdir(anipose_folder)
ani_config = os.path.join(anipose_folder, 'config.toml')
ani_cfg = toml.load(ani_config)

snapIdx = int(anipose_folder.split('snapshotindex_')[1])
train_frac = float(anipose_folder.split('trainFrac_')[1][:2]) / 100
dlc_cfg['snapshotindex'] = snapIdx    
dlc_cfg['TrainingFraction'] = [train_frac]

deeplabcut.auxiliaryfunctions.write_config(dlc_config, dlc_cfg)

for param0 in param_values[0]:
    for param1 in param_values[1]:
        for param2 in param_values[2]:
            for param3 in param_values[3]:
                for param4 in param_values[4]:
                    for param5 in param_values[5]:
                        param_list = [param0, param1, param2, param3, param4, param5]
                        print('\n\n\n\n')
                        print(param_list)
                        print('\n\n\n\n')
                        for cat, key, param in zip(param_category, param_names, param_list):
                            ani_cfg[cat][key] = param
                        
                        storage_folder = os.path.join(aniposepath, 
                                                      'all_combos_results',
                                                      'param_vals_%d_%d_%d_%d_%d_pt%d' % (param0, 
                                                                                          param1, 
                                                                                          param2,
                                                                                          param3,
                                                                                          param4,
                                                                                          param5*100))                        
                        
                        if os.path.isdir(storage_folder):
                            continue
                        
                        ani_cfg['filter']['type'] = 'viterbi'
                        
                        with open(ani_config, 'w') as f:
                            toml.dump(ani_cfg, f)
                            
                        subprocess.call(['anipose', 'filter'])
                        
                        for date in dates:
                            os.makedirs(os.path.join(storage_folder, date))
                            
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
                        
                        for date in dates:
                            shutil.copytree(os.path.join(anipose_folder, date, 'pose-3d'),
                                            os.path.join(storage_folder, date, 'pose-3d'))
                            shutil.copytree(os.path.join(anipose_folder, date, 'pose-2d-filtered'),
                                            os.path.join(storage_folder, date, 'pose-2d-filtered'))
                            shutil.copytree(os.path.join(anipose_folder, date, 'pose-2d-viterbi-only'),
                                            os.path.join(storage_folder, date, 'pose-2d-viterbi-only'))

                            shutil.rmtree(os.path.join(anipose_folder, date, 'pose-2d'))
                            shutil.rmtree(os.path.join(anipose_folder, date, 'pose-2d-filtered'))
                            shutil.rmtree(os.path.join(anipose_folder, date, 'pose-2d-viterbi-only'))
                            shutil.rmtree(os.path.join(anipose_folder, date, 'pose-3d'))
                            
                            shutil.copytree(os.path.join(anipose_folder, date, 'pose-2d-raw'), 
                                            os.path.join(anipose_folder, date, 'pose-2d'))
                            shutil.rmtree(os.path.join(anipose_folder, date, 'pose-2d-raw'))
                        
                        os.remove(os.path.join(anipose_folder, 'autoencoder.pickle'))

print("resetting snapshotindex and iteration")
dlc_cfg['iteration'] = original_iteration
dlc_cfg['snapshotindex'] = original_snapshotindex
dlc_cfg['TrainingFraction'] = original_train_fraction
deeplabcut.auxiliaryfunctions.write_config(dlc_config, dlc_cfg)

                                