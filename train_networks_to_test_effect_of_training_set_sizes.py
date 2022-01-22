# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 17:56:00 2021

@author: Dalton
"""

import shutil
import os
import glob
import pandas as pd
import deeplabcut

iteration = 1
shuffle_list = [1, 2, 3]
train_fractions = [0.95, 0.85, 0.7, 0.5, 0.3]

maxIters = 300000

projectpath = '/home/marms/Documents/dlc_local/combined_dlc_xromm_validation-Dalton-2021-08-28'
aniposepath = '/media/marms/DATA/anipose_dlc_validation'

config=os.path.join(projectpath,'config.yaml')
cfg=deeplabcut.auxiliaryfunctions.read_config(config)

original_iteration=cfg['iteration']
cfg['iteration'] = iteration
deeplabcut.auxiliaryfunctions.write_config(config,cfg)

original_snapshotindex = cfg['snapshotindex']
original_train_fraction = cfg['TrainingFraction']

for train_frac in train_fractions:
    cfg['TrainingFraction'] = [train_frac]
    deeplabcut.auxiliaryfunctions.write_config(config,cfg)
    
    for shuf in shuffle_list:
        
        test_path = os.path.join(aniposepath, 'test_training_fraction_effect_extra_training', 
                              'iteration_%d_trainFrac_%d_shuffle_%d_*' % (iteration, train_frac*100, shuf)) 
        if len(glob.glob(test_path)) == 1:
            continue
    
        deeplabcut.train_network(config, shuffle=shuf, max_snapshots_to_keep=None, maxiters=maxIters, gputouse=0, saveiters=10000)
    
        print("Evaluating...")
        cfg['snapshotindex'] = 'all'
        deeplabcut.auxiliaryfunctions.write_config(config,cfg)
        deeplabcut.evaluate_network(config, Shuffles=[shuf], plotting=False)
    
        print("Finding the snapshot to use for anipose analysis and saving to config file")
        evalResults = pd.read_csv(os.path.join(projectpath, 'evaluation-results/iteration-' + str(iteration), 'CombinedEvaluation-results.csv'))
        evalResults = evalResults.loc[(evalResults['Shuffle number']    == shuf) & 
                                      (evalResults['%Training dataset'] == train_frac*100)].reset_index(drop=True)        
        snapIdx = evalResults.loc[:, ' Test error(px)'].idxmin()
        snapshot = evalResults.loc[snapIdx, 'Training iterations:'] 
        cfg['snapshotindex'] = snapIdx
        deeplabcut.auxiliaryfunctions.write_config(config,cfg)
        
        print("Setting up anipose starter files")
        source = os.path.join(aniposepath, 'starter_files')
        target = os.path.join(aniposepath, 'test_training_fraction_effect_extra_training', 
                              'iteration_%d_trainFrac_%d_shuffle_%d_snapshot_%d_snapshotindex_%d' % (iteration, 
                                                                                                     train_frac*100, 
                                                                                                     shuf, 
                                                                                                     snapshot, 
                                                                                                     snapIdx))
        shutil.copytree(source, target)        

print("resetting snapshotindex and iteration")
cfg['iteration'] = original_iteration
cfg['snapshotindex'] = original_snapshotindex
cfg['TrainingFraction'] = original_train_fraction
deeplabcut.auxiliaryfunctions.write_config(config,cfg)

