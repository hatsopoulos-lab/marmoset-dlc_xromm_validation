# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:32:41 2022

@author: Dalton

This script converts anipose calibration to mayaCams format. The mayaCam format can
be used in XMALab if you wish to import DLC images and labels for manual corrections. 

"""

import toml
import numpy as np
import cv2
import re
import os

# it is expected that the calibration.toml file resides in the 'calibration' folder 
# within the nested anipose structure for a particular date/experiment
# Example: calib_file = '/path/to/experiment_or_date/calibration/calibration.toml'
calib_file = r'Z:/dalton_moore/Publications_and_grants/dlc_validation/dryad_deposit/anipose_files/starter_files/2019_04_15/calibration/calibration.toml'

# path to wherever you want to store the mayaCam files
mayaCam_dir = r'C:/Users/Dalton/Documents/mayaCam_files'

camera_set = 'apparatus'

# Main code
os.makedirs(mayaCam_dir, exist_ok=True)
date = os.path.basename(calib_file.split('/calibration/')[0])
anipose_calib = toml.load(calib_file)

for cam_key, calib in anipose_calib.items():
    if 'cam' in cam_key:
        # collect calibration information from anipose calibration
        image_size = calib['size']
        camera_matrix = np.array(calib['matrix'], dtype = np.float64)
        rotation_vector = np.array(calib['rotation'], dtype = np.float64)
        rotation, tmp = cv2.Rodrigues(rotation_vector)
        translation = np.expand_dims(np.array(calib['translation'], dtype = np.float64), axis=1)
        
        # write to mayaCam txt file
        mayaCam_text = 'image size\n' + str(image_size[0]) + ',' + str(image_size[1])
        mayaCam_text += '\n\ncamera matrix\n%s' % re.sub(' ', '', re.sub('[\[\]]', '', np.array2string(camera_matrix, separator=',', suppress_small=True)))        
        mayaCam_text += '\n\nrotation\n%s' % re.sub('[\[\]]', '', np.array2string(rotation, separator=','))
        mayaCam_text += '\n\ntranslation\n%s' % re.sub('[\[\]]', '', np.array2string(translation, separator=''))
        
        mayaCam_text = re.sub('0.,', '0,', mayaCam_text)
        mayaCam_text = re.sub('1.\n', '1\n', mayaCam_text)        
        
        with open(os.path.join(mayaCam_dir, '%s_%s_%s_mayaCam.txt' % (date, camera_set, cam_key)), 'w') as f:
            f.write(mayaCam_text)