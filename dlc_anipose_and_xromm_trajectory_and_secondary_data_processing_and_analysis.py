# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:54:29 2021

@author: Dalton
"""

import pickle
import dill
import pandas as pd
import numpy as np
import os
import glob
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.decomposition import PCA

project_base = r'Z:/dalton_moore/dryad_deposit'

modes = ['single', 'trainFrac', 'sweep']
mode = modes[0]
reprojIdx = 1 # Sets the base reproj_error_threshold from the list of options [10, 20] 

anipose_base       = os.path.join(project_base, 'anipose_files')
results_base       = os.path.join(project_base, 'results_presented_in_manuscript/pickle_versions')
dlc_base           = os.path.join(project_base, 'deeplabcut_files/combined_dlc_xromm_validation-Dalton-2021-08-28')
dlc_secondary_base = os.path.join(project_base, 'deeplabcut_files/dlc_files_for_secondary_analyses')

class dpath:
    if mode == 'sweep':
        base = sorted(glob.glob(os.path.join(anipose_base, 'parameter_sweep/all_combos_results/*')))
        results_storage_path  = os.path.join(results_base, 'parameter_sweep.pickle')
        sweep_extra_data_path = os.path.join(results_base, 'all_datasets_results_from_sweep')
    elif mode == 'trainFrac':
        iteration = 1
        base       = glob.glob(os.path.join(anipose_base, 'training_set_size_iteration%d/*' % iteration))
        results_storage_path = os.path.join(results_base, 'trainingset_set_size_iteration%d.pickle' % iteration)
    else:
        iteration = 1
        trainFrac = 85
        shuffle = 3
        base = glob.glob(os.path.join(anipose_base, 'anipose_on_iteration_1_trainFrac_85_shuffle_3_snapshotindex_23_param_vals_20_5_6_2_8_pt45'))
        results_storage_path = results_base
        human_error_base     = os.path.join(dlc_secondary_base, 'human_error_labels_')
        pre_eplines_labels   = os.path.join(dlc_secondary_base, 'labels_before_epipolar')
        post_eplines_labels  = os.path.join(dlc_secondary_base, 'labels_after_epipolar')
        projected_dlcanipose_error_base = base[0] 

    dates = ['2019_04_14', '2019_04_15']  
    labels = os.path.join(dlc_base, 'labeled_data')
    train_split_info = os.path.join(dlc_secondary_base, 'Documentation_data-combined_dlc_xromm_validation_85shuffle3.pickle')

class xpath:
    base = os.path.join(project_base, 'xromm_files/validation_trajectories')
    label_reorder = [12, 11, 10, 9, 8, 7, 6, 5, 3, 1, 0, 2, 4]

class params:
    nEvents = 17
    TY_events = [1, 2, 3, 7]
    PT_events  = list(set(range(nEvents)) - set(TY_events))
    acceptable_reprojectionError_threshold_day1                 = [10, 20]
    acceptable_reprojectionError_threshold_day2                 = acceptable_reprojectionError_threshold_day1
    acceptable_reprojectionError_threshold_day1_last_two_events = [25, 35]
    reprojNum = reprojIdx
    
    min_cams_threshold = 2
    min_chunk_length = 6
    max_gap_to_stitch = 30
    dlc_pixel_error_percentile_threshold = 99
    xromm_fps = 200
    dlc_fps   = 200 
    reach_idxs = [range(227, 558),
                  range(130, 341),
                  range(600, 1021),
                  range(1364, 2000),
                  range(906, 1551),
                  range(400, 840),
                  range(1387, 1700),
                  range(0, 550),
                  range(0, 725),
                  range(0, 610),
                  range(0, 975),
                  range(0, 1075),
                  range(300, 860),
                  range(0, 730),
                  range(0, 100),
                  range(0, 550),
                  range(0, 1140)]

class fig2_plot_params:
    dirColors = np.array([[27 , 158, 119],
                          [217, 95 , 2  ],
                          [117, 112, 179]])
    figSize = (5, 7)
    lw = 2

def load_dlc_data(data_dirs):
    data_files = []
    for data_dir in data_dirs:
        data_files.extend(glob.glob(os.path.join(data_dir, '*.csv')))
    
    dlc = []
    dlc_metadata = []    
    event_info = pd.DataFrame(np.empty((params.nEvents, 3)), columns = ['date', 'event', 'marm'])
    for fNum, f in enumerate(data_files):
        data = pd.read_csv(f)
        dataIdx = sorted(list(range(0, data.shape[1]-13, 6)) + 
                         list(range(1, data.shape[1]-13, 6)) + 
                         list(range(2, data.shape[1]-13, 6))) 
        metadataIdx = sorted(list(range(3, data.shape[1]-13, 6)) + 
                              list(range(4, data.shape[1]-13, 6)) + 
                              list(range(5, data.shape[1]-13, 6)))
        dlc_tmp = data.iloc[:, dataIdx].to_numpy(dtype=np.float64)
        dlc_metadata_tmp = data.iloc[:, metadataIdx]
        
        dlc_tmp = dlc_tmp.T
        dlc_tmp = np.reshape(dlc_tmp, (int(dlc_tmp.shape[0]/3), 3, dlc_tmp.shape[1]))
        
        dlc.append(dlc_tmp)
        dlc_metadata.append(dlc_metadata_tmp)
        
        event_name = os.path.basename(f)
        event_info.iloc[fNum] = [event_name[:10], int(event_name.split('event')[1][:3]), 'PT']
    
    event_info.iloc[params.TY_events, 2] = 'TY'
        
    return dlc, dlc_metadata, event_info

def load_xromm_data():
    data_files = glob.glob(os.path.join(xpath.base, '*event*.csv'))

    xromm = []
    for f in data_files:  
        data = pd.read_csv(f)
        xromm_tmp = data.to_numpy(dtype=np.float64).T
        if f == data_files[0]:
            nanVec = np.full_like(xromm_tmp[:3], np.nan)    
            xromm_tmp = np.insert(xromm_tmp, 0, nanVec, axis=0)
        xromm_tmp = np.reshape(xromm_tmp, (int(xromm_tmp.shape[0]/3), 3, xromm_tmp.shape[1]))
        xromm_tmp = xromm_tmp[xpath.label_reorder]
        xromm_tmp = xromm_tmp[:-2]
        
        xromm.append(xromm_tmp)
        
    return xromm     

def filter_dlc(dlc, dlc_metadata, rNum):
    dlc_filtered = []
    for eventNum, (pos, meta) in enumerate(zip(dlc, dlc_metadata)):
        marker_names = meta.columns[slice(0, len(meta.columns), 3)]
        marker_names = [name[:-6] for name in marker_names]
        pos_first_filter = pos.copy()
        pos_out      = np.full_like(pos, np.nan)
        
        if event_info.date[eventNum] == '2019_04_14' and event_info.event[eventNum] < 46:
            reproj_threshold = params.acceptable_reprojectionError_threshold_day1[rNum]
        elif event_info.date[eventNum] == '2019_04_14' and event_info.event[eventNum] >= 46:
            reproj_threshold = params.acceptable_reprojectionError_threshold_day1_last_two_events[rNum]
        else:
            reproj_threshold = params.acceptable_reprojectionError_threshold_day2[rNum] 
        
        # iterate over markers to filter out bad portions of trajctories
        for mNum, marker in enumerate(marker_names):
            
            # find frames where the reprojection error is worse than a specified threshold or 
            # the number of cams is below a threshold (min_cams_threshold=2 unless good reason to change it)
            # and set position in those frames to np.nan
            # fig, ax = plt.subplots()
            # ax.plot(meta.loc[:, marker+'_error'])
            # ax.plot(range(meta.shape[0]), np.repeat(reproj_threshold, meta.shape[0]), '-r')
            # plt.show()
            
            filterOut_idx = np.union1d(np.where(meta.loc[:, marker+'_error'] > reproj_threshold)[0],
                                       np.where(meta.loc[:, marker+'_ncams'] < params.min_cams_threshold)[0])
            pos_first_filter[mNum, :, filterOut_idx] = np.nan
            
            if np.sum(~np.isnan(pos_first_filter[mNum])) == 0:
                continue
            
            # Find beginning and end of each 'chunk' that remains after filtering
            gap_idxs = np.where(np.isnan(pos_first_filter[mNum, 0]))[0]
            if len(gap_idxs) == 0:
                chunk_starts = np.array([0])
                chunk_ends = np.array([pos_first_filter.shape[-1] - 1])
            else:
                chunk_starts = gap_idxs[np.hstack((np.diff(gap_idxs) > 1, False))] + 1
                chunk_ends   = gap_idxs[np.hstack((False, np.diff(gap_idxs) > 1))] - 1 
                if gap_idxs[0] != 0:
                    chunk_starts = np.hstack((0, chunk_starts))
                    chunk_ends   = np.hstack((gap_idxs[0] - 1, chunk_ends))
                if gap_idxs[-1] != pos_first_filter.shape[-1] - 1:
                    chunk_starts = np.hstack((chunk_starts, gap_idxs[-1] + 1))
                    chunk_ends   = np.hstack((chunk_ends  , pos_first_filter.shape[-1] - 1))
            
            left_gap_lengths = np.hstack((chunk_starts[0], chunk_starts[1:] - chunk_ends[:-1] - 1))
            
            # Write chunk and gap info into readable format. Note that to grab a particular chunk you would index 
            # with [chunk_info.start : chunk_info.end + 1]
            chunk_info = pd.DataFrame(data = zip(chunk_starts, 
                                                 chunk_ends, 
                                                 chunk_ends - chunk_starts + 1, 
                                                 left_gap_lengths), 
                                      columns=['start', 
                                               'end', 
                                               'chunk_length', 
                                               'prev_gap_length'])
            
            # remove chunks shorter than the minimum chunk_length
            while np.sum(chunk_info.chunk_length < params.min_chunk_length) > 0:
                idx = chunk_info.index[chunk_info.chunk_length < params.min_chunk_length][0]
                if idx < chunk_info.index.max():
                    chunk_info.loc[idx + 1, 'prev_gap_length'] += chunk_info.loc[idx, 'prev_gap_length'] + chunk_info.loc[idx, 'chunk_length']
                    chunk_info = chunk_info.drop(idx, axis = 0)     
                    chunk_info = chunk_info.reset_index(drop=True)
                else: 
                    chunk_info = chunk_info.drop(idx, axis = 0)
            
            # stitch together chunks with gaps shorter than the max_gap_to_stitch parameter
            while np.sum(chunk_info.prev_gap_length[1:] < params.max_gap_to_stitch) > 0:
                idx  = chunk_info.index[chunk_info.prev_gap_length < params.max_gap_to_stitch]
                idx_after_first_chunk = np.nonzero(idx)[0]
                if len(idx_after_first_chunk) != 0:
                    idx = idx[idx_after_first_chunk[0]]
                
                    chunk_info.loc[idx-1, 'end']          = chunk_info.loc[idx, 'end']
                    chunk_info.loc[idx-1, 'chunk_length'] = chunk_info.loc[idx-1, 'end'] - chunk_info.loc[idx-1, 'start'] + 1 
                    chunk_info = chunk_info.drop(idx, axis=0)
                    chunk_info = chunk_info.reset_index(drop=True)
            
            # produce a filtered position with only the remaining stitched-together chunks
            filtered_chunk_idxs = []
            for index, chunk in chunk_info.iterrows():
                filtered_chunk_idxs.extend(list(range(chunk.start, chunk.end+1)))
            pos_out[mNum, :, filtered_chunk_idxs] = pos[mNum, :, filtered_chunk_idxs]
        
        if eventNum == 3:# and mNum == 2:
            catchHere = []        
        
        dlc_filtered.append(pos_out)
    
    return dlc_filtered

def flip_axes(dlc, event_info):
    for eventNum, traj in enumerate(dlc):
        if event_info.date[eventNum] == '2019_04_14':
            for part in range(traj.shape[0]):
                traj[part] = traj[part, (0, 2, 1), :]
                traj[part, 1, :] = -1 * traj[part, 1, :]
        dlc[eventNum] = traj
    return dlc

def project_trajectories_on_common_axes(trajData):
    
    dlc       = trajData.dlc
    xromm     = trajData.xromm 
    xromm_raw = trajData.xromm_raw
    
    first = True
    stored_frame_counts = []
    for dTraj, xTraj in zip(dlc, xromm):
        stored_frame_counts.append(dTraj.shape[-1])
        if first:
            dlc_cat       = dTraj
            xromm_cat     = xTraj
            first = False
        else:
            dlc_cat       = np.dstack((dlc_cat,       dTraj))
            xromm_cat     = np.dstack((xromm_cat,     xTraj))
    
    day1_events = event_info.index[event_info.date == '2019_04_14']
    date_cutoff = np.sum(np.array(stored_frame_counts)[day1_events])    
    dlc_list   = [dlc_cat  [..., :date_cutoff], dlc_cat  [..., date_cutoff:]]
    xromm_list = [xromm_cat[..., :date_cutoff], xromm_cat[..., date_cutoff:]]        
    
    dlc_bases   = []
    xromm_bases = []
    for idx, (dlc_tmp, xromm_tmp) in enumerate(zip(dlc_list, xromm_list)):
                
        dlc_tmp = dlc_tmp.swapaxes(0, 1)
        dlc_tmp = np.reshape(dlc_tmp, (dlc_tmp.shape[0], dlc_tmp.shape[1]*dlc_tmp.shape[2])).T 
        dlc_tmp_idx = np.where(~np.isnan(dlc_tmp[:, 0]))[0]
        
        xromm_tmp = xromm_tmp.swapaxes(0, 1)
        xromm_tmp = np.reshape(xromm_tmp, (xromm_tmp.shape[0], xromm_tmp.shape[1]*xromm_tmp.shape[2])).T 
        xromm_tmp_idx = np.where(~np.isnan(xromm_tmp[:, 0]))[0]
        
        pca = PCA(n_components = 3)
        
        pca.fit_transform(dlc_tmp[dlc_tmp_idx])
        dlc_comps = pca.components_
        # dlc_variance = pca.explained_variance_ratio_
        pca.fit_transform(xromm_tmp[xromm_tmp_idx])
        xromm_comps = pca.components_
        # xromm_variance = pca.explained_variance_ratio_
        
        dlc_bases.append(dlc_comps)
        xromm_bases.append(xromm_comps)

    dlc_out       = []
    xromm_out     = []
    xromm_raw_out = []
    for eventNum, (dTraj, xTraj, xrawTraj) in  enumerate(zip(dlc, xromm, xromm_raw)):
               
        if eventNum in day1_events:
            dlc_basis = dlc_bases[0]
            xromm_basis = xromm_bases[0]
        else:
            dlc_basis = dlc_bases[1]
            xromm_basis = xromm_bases[1]
        
        dTraj = dTraj.swapaxes(1, 2)
        dTraj = dTraj @ dlc_basis.T
        dlc_out.append(dTraj.swapaxes(1, 2))

        xTraj = xTraj.swapaxes(1, 2)
        xTraj = xTraj @ xromm_basis.T
        xromm_out.append(xTraj.swapaxes(1, 2))
        
        xrawTraj = xrawTraj.swapaxes(1, 2)
        xrawTraj = xrawTraj @ xromm_basis.T
        xromm_raw_out.append(xrawTraj.swapaxes(1, 2))
        
    # compute temporary error after projection to check if any of the dimensions 
    # should be multiplied by -1 to match each other
    x_err = []
    y_err = []
    z_err = []
    for d, x in zip(dlc_out, xromm_out):
        tmp = np.nanmax(np.nanmedian(d-x, axis = 2), axis = 0)
        x_err.append(tmp[0])
        y_err.append(tmp[1])
        z_err.append(tmp[2])
    tmp_err = np.array([[np.mean(x_err[:10]), np.mean(y_err[:10]), np.mean(z_err[:10])],
                        [np.mean(x_err[10:]), np.mean(y_err[10:]), np.mean(z_err[10:])]])

    
    # adjust dimensions that need to be flipped
    for dim, dim_err in enumerate(tmp_err[0]):
        if dim_err > 0.2:
            for eventNum, dTraj in enumerate(dlc_out):
                if eventNum < 10:
                    for part in range(dTraj.shape[0]):
                        dTraj[part, dim, :] = -1 * dTraj[part, dim, :]
                    dlc_out[eventNum] = dTraj
                    
    for dim, dim_err in enumerate(tmp_err[1]):
        if dim_err > 0.2:    
            for eventNum, dTraj in enumerate(dlc_out):
                if eventNum >= 10:
                    for part in range(dTraj.shape[0]):
                        dTraj[part, dim, :] = -1 * dTraj[part, dim, :]
                    dlc_out[eventNum] = dTraj
    
    trajData.dlc       = dlc_out
    trajData.xromm     = xromm_out
    trajData.xromm_raw = xromm_raw_out
        
    return trajData, dlc_bases 

def align_and_mean_subtract_trajectories(xromm_in, dlc_filtered):
    bestShift = []
    meanSubtracted = []
    allDataPointsXromm = []
    dlc_shift_and_meanSubtracted = []
    xromm_shift_and_meanSubtracted = []
    for xromm, dlc_filt in zip(xromm_in, dlc_filtered):
        # identify file length differences
        diff = abs(np.shape(dlc_filt)[-1] - np.shape(xromm)[-1])
        if diff == 0:
            diff = 5
        
        # iterate over range of file length differences to find the correct shift
        lastError = 1000
        for delta in range(-diff, diff):
            dlc_copy = dlc_filt.copy()
            xromm_copy = xromm.copy()
            original_xromm = xromm.copy()
            
            # shift dlc relative to xromm and make frame numbers equal 
            if delta <= 0:
                start = np.zeros(np.shape(dlc_copy[:, :, 0:abs(delta)]))
                end = np.zeros(np.shape(dlc_copy[:, :, 0:diff - abs(delta)]))
                start[start == 0] = np.nan
                end[end == 0] = np.nan
                dlc_copy = np.dstack((start, dlc_copy, end))
            else:
                end = np.zeros(np.shape(dlc_copy[:, :, 0:2*delta]))
                end[end == 0] = np.nan 
                dlc_copy = np.dstack((dlc_copy, end))
                dlc_copy = np.delete(dlc_copy, np.s_[0:delta], axis = 2)
    
            tmpDiff = dlc_copy.shape[-1] - xromm_copy.shape[-1]
            adjust = np.zeros(np.shape(dlc_copy[:, :, 0: abs(tmpDiff) ]))
            adjust[adjust == 0] = np.nan
    
            if tmpDiff > 0:
                xromm_copy = np.dstack((xromm_copy, adjust))
                original_xromm = np.dstack((original_xromm, adjust))
            else:
                dlc_copy = np.dstack((dlc_copy, adjust))            
            
            # subtract mean position of each marker and compute mean absolute error for each shift
            error_tmp = np.full_like(dlc_copy[:, 0], np.nan)  
            meanSubtract = np.empty((12, 3))
            for part in range(np.size(dlc_copy, 0)):
                nonOverlap = np.where(np.logical_or(np.isnan(xromm_copy[part, 0, :]), np.isnan(dlc_copy[part, 0, :])))
                for dim in range(3):
                    dlc_copy[part, dim, nonOverlap] = np.nan
                    xromm_copy[part, dim, nonOverlap] = np.nan
                    
                    remSlice = np.where(~np.isnan(dlc_copy[part, dim, :]))
    
                    meanSubtract[part, dim] = np.mean(dlc_copy[part, dim, remSlice])
                    dlc_copy[part, dim, remSlice] = dlc_copy[part, dim, remSlice] - np.repeat(meanSubtract[part, dim], len(remSlice))
                    xromm_copy[part, dim, remSlice] = xromm_copy[part, dim, remSlice] - np.repeat(np.mean(xromm_copy[part, dim, remSlice]), len(remSlice))
                    if len(remSlice[0]) > 0:
                        original_xromm[part, dim, :] = original_xromm[part, dim, :] - np.repeat(np.mean(original_xromm[part, dim, remSlice]), len(original_xromm[0,0,:]))
                
                errorByDim = xromm_copy[part, :, :] - dlc_copy[part, :, :]
                if np.any(~np.isnan(errorByDim)):
                    error_tmp[part] = np.sqrt(np.square(errorByDim[0, :]) + np.square(errorByDim[1, :]) + np.square(errorByDim[2, :]))
            
            newError = np.nanmean(error_tmp.flatten())
            if newError < lastError:
                lastError = newError
                diff_tmp = delta      
                shifted_dlcTraj_tmp = dlc_copy
                shifted_xrommTraj_tmp = xromm_copy
                shifted_original_xromm = original_xromm
                bestShiftMean = meanSubtract
        
        if lastError == 1000:
            bestShift.append(0)
            diff = np.shape(dlc_filt)[-1] - np.shape(xromm)[-1]
            end = np.zeros(np.shape(dlc_copy[:, :, 0:abs(diff)]))
            end[end == 0] = np.nan
            if diff <= 0: 
                dlc_filt = np.dstack((dlc_filt, end))
            else:
                xromm = np.dstack((xromm, end))
            allDataPointsXromm.append(xromm)
        else:
            bestShift.append(diff_tmp)
            dlc_shift_and_meanSubtracted.append(shifted_dlcTraj_tmp)
            xromm_shift_and_meanSubtracted.append(shifted_xrommTraj_tmp)
            allDataPointsXromm.append(shifted_original_xromm)
            meanSubtracted.append(bestShiftMean)
            
    return dlc_shift_and_meanSubtracted, xromm_shift_and_meanSubtracted, allDataPointsXromm, bestShift, meanSubtracted

def subtract_mean_over_all_points(dlc, xromm, xromm_raw):
    
    first = True
    for dTraj, xTraj, xrawTraj in zip(dlc, xromm, xromm_raw):
        if first:
            dlc_cat       = dTraj
            xromm_cat     = xTraj
            xromm_raw_cat = xrawTraj
            first = False
        else:
            dlc_cat       = np.dstack((dlc_cat,       dTraj))
            xromm_cat     = np.dstack((xromm_cat,     xTraj))
            xromm_raw_cat = np.dstack((xromm_raw_cat, xrawTraj))
    
    avg_dlc   = np.expand_dims(np.nanmean(dlc_cat,   axis = -1), -1)
    avg_xromm = np.expand_dims(np.nanmean(xromm_cat, axis = -1), -1)
    
    for eventNum, (dTraj, xTraj, xrawTraj) in enumerate(zip(dlc, xromm, xromm_raw)):
        nFrames = dTraj.shape[-1]
        dlc[eventNum]       = dTraj    - np.repeat(avg_dlc  , repeats = nFrames, axis = -1)
        xromm[eventNum]     = xTraj    - np.repeat(avg_xromm, repeats = nFrames, axis = -1)
        xromm_raw[eventNum] = xrawTraj - np.repeat(avg_xromm, repeats = nFrames, axis = -1)
    
    return dlc, xromm, xromm_raw

global record_tracking_quality
def record_tracking_quality(trajData, frame_shift):
    
    global trackingQuality
    class trackingQuality:
        unadjusted_percentTracked = np.full((len(trajData.dlc), trajData.dlc[0].shape[0]), np.nan)
        percentTracked = np.full_like(unadjusted_percentTracked, np.nan)
        avgOverPart = []
        avgOverTraj = []

    reach_idxs = [np.array(idxs) - shift for idxs, shift in zip(params.reach_idxs, frame_shift)]
    
    dlcMatchingLength   = np.empty_like(trackingQuality.percentTracked)
    xrommTrackingLength = np.empty_like(trackingQuality.percentTracked)
    dlcTrajLength       = np.empty_like(trackingQuality.percentTracked)
    dlcReachLength      = np.empty_like(trackingQuality.percentTracked)
    xrommReachLength    = np.empty_like(trackingQuality.percentTracked)
    for eventNum, (dTraj, xrawTraj) in enumerate(zip(trajData.dlc, trajData.xromm_raw)):
        for part in range(dTraj.shape[0]):
            goodDLCPoints     = np.where(~np.isnan(dTraj   [part, 0]))[0]
            goodXrommPoints   = np.where(~np.isnan(xrawTraj[part, 0]))[0]
            dlcMatchingPoints = [idx for idx in goodDLCPoints if idx in goodXrommPoints]
            xrommReachPoints  = [idx for idx in goodXrommPoints if idx in reach_idxs[eventNum]]            
            dlcReachPoints    = [idx for idx in goodDLCPoints   if idx in xrommReachPoints]
            
            dlcReachLength     [eventNum, part] = len(dlcReachPoints)
            xrommReachLength   [eventNum, part] = len(xrommReachPoints)
            
            dlcMatchingLength  [eventNum, part] = len(dlcMatchingPoints)
            xrommTrackingLength[eventNum, part] = len(goodXrommPoints)
            dlcTrajLength      [eventNum, part] = len(goodDLCPoints)

            trackingQuality.unadjusted_percentTracked[eventNum, part] = dlcMatchingLength[eventNum, part] / xrommTrackingLength[eventNum, part]
            trackingQuality.percentTracked[eventNum, part] = dlcReachLength[eventNum, part] / xrommReachLength[eventNum, part]  
                
    trackingQuality.avgOverTraj = np.divide(np.nansum(np.multiply(trackingQuality.percentTracked, xrommReachLength), 1), 
                                            np.nansum(xrommReachLength, 1))
    trackingQuality.avgOverPart = np.divide(np.nansum(np.multiply(trackingQuality.percentTracked, xrommReachLength), 0), 
                                            np.nansum(xrommReachLength, 0))
    
    trackingQuality.total_percentTracked = np.nansum(dlcReachLength) / np.nansum(xrommReachLength)
    
    return trackingQuality, dlcTrajLength

def collect_data_from_hand_labeled_frames(frame_shift = None):
    
    label_folders = glob.glob(os.path.join(dpath.labels, '*'))
    
    cam1_label_paths = [os.path.join(p, 'CollectedData_Dalton.h5') for p in label_folders if 'cam1' in os.path.basename(p) and '_labeled' not in os.path.basename(p)]
    cam2_label_paths = [os.path.join(p, 'CollectedData_Dalton.h5') for p in label_folders if 'cam2' in os.path.basename(p) and '_labeled' not in os.path.basename(p)]

    with open(dpath.train_split_info, 'rb') as f:
        train_split_info = dill.load(f)
    
    trainset = [info['image'] for info in train_split_info[0]]
                                     
    class labels_info:
        eventNum        = []
        frames          = []
        labeled_parts   = []
        data            = []
        in_training_set = []
        

    for cam1_file, cam2_file in zip(cam1_label_paths, cam2_label_paths):
        
        event = int(cam1_file.split('event')[1][:3])
        date  = os.path.basename(os.path.dirname(cam1_file))[:10]
        eventNum = event_info.index[(event_info.date == date) & (event_info.event == event)][0]
        labels_info.eventNum.append(eventNum)  
        
        pdI = pd.IndexSlice
        cam1_labels = pd.read_hdf(cam1_file)
        cam2_labels = pd.read_hdf(cam2_file)
        
        frames_tmp        = []
        labeled_parts_tmp = []
        
        # check if refinement set wasn't merged for one camera and fill matching NaN data
        cam1_labels['frNum'] = [int(os.path.basename(f)[3:-4]) for f in cam1_labels.index]
        cam2_labels['frNum'] = [int(os.path.basename(f)[3:-4]) for f in cam2_labels.index]
        all_frames_tmp = np.union1d(cam1_labels['frNum'], cam2_labels['frNum'])
        
        if list(all_frames_tmp) == sorted(cam1_labels['frNum']) and list(all_frames_tmp) == sorted(cam2_labels['frNum']): 
            cam1_labels = cam1_labels.drop('frNum', axis = 1)
            cam2_labels = cam2_labels.drop('frNum', axis = 1)
        else:    
            print('\n\n\n\n\n\n event%d \n\n\n\n\n\n\n' % eventNum)
            base_img_path = os.path.split(cam1_labels.index[0])[0][:-1]
            cam1_index = []
            cam2_index = []
            for frNum in all_frames_tmp:
                cam1_index.append(base_img_path + '1/img' + str(frNum).zfill(4) + '.png')
                cam2_index.append(base_img_path + '2/img' + str(frNum).zfill(4) + '.png')                              
    
            cam1_labels_tmp = pd.DataFrame(np.full((all_frames_tmp.shape[0], cam1_labels.shape[1]-1), np.nan),
                                           columns = cam1_labels.columns[:-1],
                                           index = cam1_index)
            cam2_labels_tmp = pd.DataFrame(np.full((all_frames_tmp.shape[0], cam2_labels.shape[1]-1), np.nan),
                                           columns = cam2_labels.columns[:-1],
                                           index = cam2_index)
            
            for tmpIdx, frNum in enumerate(all_frames_tmp):
                for camNum, tmp, orig in zip(['1', '2'],
                                             [cam1_labels_tmp, cam2_labels_tmp], 
                                             [cam1_labels, cam2_labels]):
                    if frNum in list(orig.frNum):
                        tmp.iloc[tmpIdx, :] = orig.loc[orig.frNum == frNum, :].values[0][:-1]
            cam1_labels = cam1_labels_tmp
            cam2_labels = cam2_labels_tmp
        
        trimmed_cam1_labelPaths = [os.path.split(os.path.split(labpath)[0])[1] + '/' + os.path.split(labpath)[1] for labpath in cam1_labels.index]   
        trimmed_cam2_labelPaths = [os.path.split(os.path.split(labpath)[0])[1] + '/' + os.path.split(labpath)[1] for labpath in cam2_labels.index]   
        trimmed_trainset        = [os.path.split(os.path.split(labpath)[0])[1] + '/' + os.path.split(labpath)[1] for labpath in trainset]   
        
        cam1_x = cam1_labels.loc[:,  pdI[:, :, 'x']]     # alt formatting:  cam1_labels.loc[:, (slice(None), slice(None), 'x')]
        cam2_x = cam2_labels.loc[:,  pdI[:, :, 'x']]
        
        train_test = pd.DataFrame(np.empty((cam1_x.shape[0], 2)),
                                  columns = ['cam1', 'cam2'])
        for idx in range(len(cam1_labels)):
            frames_tmp.append(int(os.path.basename(cam1_labels.index[idx])[3:-4]))
            
            cam1_labeled = np.where(~np.isnan(cam1_x.iloc[idx, :]))[0]
            cam2_labeled = np.where(~np.isnan(cam2_x.iloc[idx, :]))[0]
            # labeled_parts_tmp.append(np.union1d(cam1_labeled, cam2_labeled)) # change to intersect1d if we want only parts labeled in both cameras
            labeled_parts_tmp.append(np.intersect1d(cam1_labeled, cam2_labeled)) # change to intersect1d if we want only parts labeled in both cameras
        
            train_test.iloc[idx, :] = [trimmed_cam1_labelPaths[idx] in trimmed_trainset,
                                       trimmed_cam2_labelPaths[idx] in trimmed_trainset]
        
        # adjust frame numbers with the frame shift calculated during alignment    
        if frame_shift is not None:
            frames_tmp = [fr - frame_shift[eventNum] for fr in frames_tmp]
        else: 
            labels_info.data.append([cam1_labels, cam2_labels])
            
        labels_info.frames.append(frames_tmp)
        labels_info.labeled_parts.append(labeled_parts_tmp)
        labels_info.in_training_set.append(train_test)
        
    return labels_info

global compare_pre_eplines_to_post
def compare_pre_eplines_to_post():

    pdI = pd.IndexSlice
    
    pre_paths_cam1  = glob.glob(os.path.join(dpath.pre_eplines_labels , '*cam1*'))
    post_paths_cam1 = glob.glob(os.path.join(dpath.post_eplines_labels, '*cam1*'))
    pre_paths_cam2  = glob.glob(os.path.join(dpath.pre_eplines_labels , '*cam2*'))
    post_paths_cam2 = glob.glob(os.path.join(dpath.post_eplines_labels, '*cam2*'))
    
    first = True
    for pre_f, post_f in zip(pre_paths_cam1, post_paths_cam1):
        pre  = pd.read_hdf(os.path.join(pre_f , 'CollectedData_Dalton.h5'))
        post = pd.read_hdf(os.path.join(post_f, 'CollectedData_Dalton.h5'))
        labels_before_refinement = [idx for idx, f in enumerate(post.index) if '/' not in f]
        post = post.iloc[labels_before_refinement, :]
        diff_tmp = np.square(post - pre)
        if first:
            pre_post_cam1 = np.sqrt(np.array(diff_tmp.loc[:, pdI[:, :, 'x']]) + 
                                    np.array(diff_tmp.loc[:, pdI[:, :, 'y']]))
            first = False
        else:
            pre_post_cam1 = np.vstack((pre_post_cam1, np.sqrt(np.array(diff_tmp.loc[:, pdI[:, :, 'x']]) + 
                                                              np.array(diff_tmp.loc[:, pdI[:, :, 'y']]))))
    
    first = True
    for pre_f, post_f in zip(pre_paths_cam2, post_paths_cam2):
        pre  = pd.read_hdf(os.path.join(pre_f , 'CollectedData_Dalton.h5'))
        post = pd.read_hdf(os.path.join(post_f, 'CollectedData_Dalton.h5'))
        labels_before_refinement = [idx for idx, f in enumerate(post.index) if '/' not in f]
        post = post.iloc[labels_before_refinement, :]
        diff_tmp = np.square(post - pre)
        if first:
            pre_post_cam2 = np.sqrt(np.array(diff_tmp.loc[:, pdI[:, :, 'x']]) + 
                                    np.array(diff_tmp.loc[:, pdI[:, :, 'y']]))
            first = False
        else:
            pre_post_cam2 = np.vstack((pre_post_cam2, np.sqrt(np.array(diff_tmp.loc[:, pdI[:, :, 'x']]) + 
                                                              np.array(diff_tmp.loc[:, pdI[:, :, 'y']]))))
    
    # pre_post_cam1 = np.nanmean(pre_post_cam1, 0)
    # pre_post_cam2 = np.nanmean(pre_post_cam2, 0)
    errors_without_epipolar_lines = pd.DataFrame(zip([np.nanmean(pre_post_cam1[:, :-2]), np.nanmean(pre_post_cam1[:, -2:])], 
                                                      [np.nanmean(pre_post_cam2[:, :-2]), np.nanmean(pre_post_cam2[:, -2:])]), 
                                                  index = ['arm/hand (9 labels)', 'torso (2 labels)'],
                                                  columns = ['cam1', 'cam2'])    
    return errors_without_epipolar_lines

global compute_human_and_dlc_pixel_labeling_error
def compute_human_and_dlc_pixel_labeling_error(dlc_filtered):
    pdI = pd.IndexSlice
    
    # compute dlc pixel labeling error with anipose projected pixel error
    labels_info = collect_data_from_hand_labeled_frames()
    data_dirs = []
    for date in dpath.dates:
        data_dirs.append(os.path.join(dpath.projected_dlcanipose_error_base, date, 'pose-2d-proj'))
    
    projected_cam1_files = []
    projected_cam2_files = []
    for d in data_dirs:
        projected_cam1_files.extend(glob.glob(os.path.join(d, '*cam1*')))
        projected_cam2_files.extend(glob.glob(os.path.join(d, '*cam2*')))    
    projected_cam1_files = [f for fNum, f in enumerate(projected_cam1_files) if fNum in labels_info.eventNum]
    projected_cam2_files = [f for fNum, f in enumerate(projected_cam2_files) if fNum in labels_info.eventNum]
    
    proj_vs_human_error = [[], []]
    good_calib_idxs = []
    eventPixel_error = np.empty((len(labels_info.frames), 2))
    for fNum, (f1, f2, labData, frames) in enumerate(zip(projected_cam1_files, 
                                                         projected_cam2_files, 
                                                         labels_info.data, 
                                                         labels_info.frames)):
        cam1_data = pd.read_hdf(f1)
        cam2_data = pd.read_hdf(f2)
        
        dataIdx = sorted(list(range(0, cam1_data.shape[1], 3)) + 
                         list(range(1, cam1_data.shape[1], 3))) 
        
        cam1_data = cam1_data.iloc[frames, dataIdx]
        cam2_data = cam2_data.iloc[frames, dataIdx]
        
        dlc_tmp = dlc_filtered[labels_info.eventNum[fNum]]
        for cNum, (cData_orig, lData_orig) in enumerate(zip([cam1_data, cam2_data], labData)):
            cData = cData_orig.copy()
            lData = lData_orig.copy()
            lData.index = [int(os.path.basename(idx)[3:-4]) for idx in lData.index]
            cData.columns = pd.MultiIndex.from_tuples([('Dalton', col[1], col[2]) for col in cData.columns])
            diff_tmp = np.square(cData - lData)
            for part in range(dlc_tmp.shape[0]):
                nanIdxs = np.where(np.isnan(dlc_tmp[part, 0]))[0]
                for frNum in diff_tmp.index:
                    if int(frNum) in nanIdxs:
                        frameIndex = np.where(diff_tmp.index == frNum)[0][0]
                        diff_tmp.iloc[frameIndex, part*2 : part*2+1] = np.nan
            tmp_array = np.sqrt(np.array(diff_tmp.loc[:, pdI[:, :, 'x']]) + 
                                np.array(diff_tmp.loc[:, pdI[:, :, 'y']]))
            eventPixel_error[fNum, cNum] = np.nanmedian(tmp_array)
            if fNum == 0:
                proj_vs_human_error[cNum] = tmp_array
            else:
                proj_vs_human_error[cNum] = np.vstack((proj_vs_human_error[cNum], 
                                                       tmp_array))
            if fNum not in [8, 9] and cNum == 0:
                good_calib_idxs.extend(range(proj_vs_human_error[cNum].shape[0] - tmp_array.shape[0], proj_vs_human_error[cNum].shape[0]))
    
    train_test_concat = labels_info.in_training_set[0]
    for inTrain in labels_info.in_training_set[1:]:
        train_test_concat = pd.concat([train_test_concat, inTrain], ignore_index=True)
    cam1_train_idx = np.where(train_test_concat.cam1)[0]
    cam1_test_idx  = np.where(train_test_concat.cam1 == False)[0]
    cam2_train_idx = np.where(train_test_concat.cam2)[0]
    cam2_test_idx  = np.where(train_test_concat.cam2 == False)[0]
    
    good_calib_idxs = np.array(good_calib_idxs)
    
    good_cam1_train = proj_vs_human_error[0][np.intersect1d(good_calib_idxs, cam1_train_idx)] 
    good_cam2_train = proj_vs_human_error[1][np.intersect1d(good_calib_idxs, cam2_train_idx)]
    good_cam1_test  = proj_vs_human_error[0][np.intersect1d(good_calib_idxs, cam1_test_idx)] 
    good_cam2_test  = proj_vs_human_error[1][np.intersect1d(good_calib_idxs, cam2_test_idx)]

    
    flat_train_errors = np.hstack((good_cam1_train.flatten(), good_cam2_train.flatten()))
    all_dlc_pixel_errors = pd.DataFrame(zip(flat_train_errors,
                                     np.full_like(flat_train_errors, np.nan),
                                     np.full_like(flat_train_errors, 1)),
                                 columns = ['train', 'test', 'cam'])
    all_dlc_pixel_errors.iloc[np.size(good_cam1_train):, -1] = 2
    all_dlc_pixel_errors['test'][: np.size(good_cam1_test)] = good_cam1_test.flatten()
    all_dlc_pixel_errors['test'][np.size(good_cam1_train) : np.size(good_cam1_train) + np.size(good_cam2_test)] = good_cam2_test.flatten()    
    
    percentile_thresh  = np.nanpercentile(all_dlc_pixel_errors, params.dlc_pixel_error_percentile_threshold)
    all_dlc_pixel_error_noOutliers = all_dlc_pixel_errors.copy()
    all_dlc_pixel_error_noOutliers.loc[all_dlc_pixel_error_noOutliers['train'] > percentile_thresh, 'train'] = np.nan
    all_dlc_pixel_error_noOutliers.loc[all_dlc_pixel_error_noOutliers['test' ] > percentile_thresh, 'test' ] = np.nan
    
    dlc_pixel_error = pd.DataFrame(zip([np.nanmean(all_dlc_pixel_errors['train']), 
                                        np.nanmean(all_dlc_pixel_errors['test']), 
                                        np.nanmean(all_dlc_pixel_errors.loc[:, ['train','test']])],
                                       [np.nanmean(all_dlc_pixel_error_noOutliers['train']), 
                                        np.nanmean(all_dlc_pixel_error_noOutliers['test']), 
                                        np.nanmean(all_dlc_pixel_error_noOutliers.loc[:, ['train','test']])],
                                       [np.nanmedian(all_dlc_pixel_errors['train']), 
                                        np.nanmedian(all_dlc_pixel_errors['test']), 
                                        np.nanmedian(all_dlc_pixel_errors.loc[:, ['train','test']])]),  
                                   columns = ['mean_error', 'mean_error_below_percentile', 'median_error'],
                                   index   = ['train', 'test', 'total'])  
    
    # compute human labeling error comparing labels by Dalton and Ariana    
    ariana_paths_cam1 = glob.glob(os.path.join(dpath.human_error_base + 'ariana', '*cam1*'))
    dalton_paths_cam1 = glob.glob(os.path.join(dpath.human_error_base + 'dalton', '*cam1*'))
    ariana_paths_cam2 = glob.glob(os.path.join(dpath.human_error_base + 'ariana', '*cam2*'))
    dalton_paths_cam2 = glob.glob(os.path.join(dpath.human_error_base + 'dalton', '*cam2*'))
    dalton_repeat_cam1 = glob.glob(os.path.join(dpath.human_error_base + 'repeatDalton', '*cam1*'))
    dalton_repeat_cam2 = glob.glob(os.path.join(dpath.human_error_base + 'repeatDalton', '*cam2*'))
    
    first = True
    for ar_f, da_f, da_rep_f in zip(ariana_paths_cam1, dalton_paths_cam1, dalton_repeat_cam1):
        dalton    = pd.read_hdf(os.path.join(da_f,     'CollectedData_Dalton.h5'))
        daltonRep = pd.read_hdf(os.path.join(da_rep_f, 'CollectedData_Dalton.h5'))
        ariana    = pd.read_hdf(os.path.join(ar_f,     'CollectedData_Dalton.h5'))
        labels_before_refinement = [idx for idx, f in enumerate(dalton.index) if '/' not in f]
        dalton = dalton.iloc[labels_before_refinement, :]
        diff_tmp    = np.square(dalton - ariana)
        diffRep_tmp = np.square(daltonRep - dalton)
        if first:
            error_cam1 = np.sqrt(np.array(diff_tmp.loc[:, pdI[:, :, 'x']]) + 
                                 np.array(diff_tmp.loc[:, pdI[:, :, 'y']]))
            errorRep_cam1 = np.sqrt(np.array(diffRep_tmp.loc[:, pdI[:, :, 'x']]) + 
                                    np.array(diffRep_tmp.loc[:, pdI[:, :, 'y']]))
            first = False
        else:
            error_cam1 = np.vstack((error_cam1, np.sqrt(np.array(diff_tmp.loc[:, pdI[:, :, 'x']]) + 
                                                        np.array(diff_tmp.loc[:, pdI[:, :, 'y']]))))
            errorRep_cam1 = np.vstack((errorRep_cam1, np.sqrt(np.array(diffRep_tmp.loc[:, pdI[:, :, 'x']]) + 
                                                              np.array(diffRep_tmp.loc[:, pdI[:, :, 'y']])))) 
            
    first = True
    for ar_f, da_f, da_rep_f in zip(ariana_paths_cam2, dalton_paths_cam2, dalton_repeat_cam2):
        dalton    = pd.read_hdf(os.path.join(da_f,     'CollectedData_Dalton.h5'))
        daltonRep = pd.read_hdf(os.path.join(da_rep_f, 'CollectedData_Dalton.h5'))
        ariana    = pd.read_hdf(os.path.join(ar_f,     'CollectedData_Dalton.h5'))
        labels_before_refinement = [idx for idx, f in enumerate(dalton.index) if '/' not in f]
        dalton = dalton.iloc[labels_before_refinement, :]
        diff_tmp = np.square(dalton - ariana)
        diffRep_tmp = np.square(daltonRep - dalton)
        if first:
            error_cam2 = np.sqrt(np.array(diff_tmp.loc[:, pdI[:, :, 'x']]) + 
                                 np.array(diff_tmp.loc[:, pdI[:, :, 'y']]))
            errorRep_cam2 = np.sqrt(np.array(diffRep_tmp.loc[:, pdI[:, :, 'x']]) + 
                                    np.array(diffRep_tmp.loc[:, pdI[:, :, 'y']]))
            first = False
        else:
            error_cam2 = np.vstack((error_cam2, np.sqrt(np.array(diff_tmp.loc[:, pdI[:, :, 'x']]) + 
                                                        np.array(diff_tmp.loc[:, pdI[:, :, 'y']]))))
            errorRep_cam2 = np.vstack((errorRep_cam2, np.sqrt(np.array(diffRep_tmp.loc[:, pdI[:, :, 'x']]) + 
                                                              np.array(diffRep_tmp.loc[:, pdI[:, :, 'y']])))) 
    errors_concat    = np.vstack((error_cam1   , error_cam2    ))
    errorsRep_concat = np.vstack((errorRep_cam1, errorRep_cam2))
    
    human_error_by_segment = pd.DataFrame(zip([np.nanmean(errors_concat[:,  :3]), 
                                               np.nanmean(errors_concat[:, 3:6]), 
                                               np.nanmean(errors_concat[:, 6:9]),
                                               np.nanmean(errors_concat[:, 9: ])],
                                              [np.nanmean(errorsRep_concat[:,  :3]), 
                                               np.nanmean(errorsRep_concat[:, 3:6]), 
                                               np.nanmean(errorsRep_concat[:, 6:9]),
                                               np.nanmean(errorsRep_concat[:, 9: ])]),
                                          columns= ['Ariana', 'Dalton'],
                                          index=['Hand', 'Forearm', 'UpperArm', 'Torso'])
    human_error_by_part = np.hstack((np.nanmean(errors_concat, axis = 0), np.nanmean(errorsRep_concat, axis = 0))) 
    # human_error = pd.DataFrame(zip([np.nanmedian(error_cam1), np.nanmedian(errorRep_cam1)], [np.nanmedian(error_cam2), np.nanmedian(errorRep_cam2)], [np.nanmedian(errors_concat), np.nanmedian(errorsRep_concat)]),  
    #                            index = ['ariana_median_error', 'dalton_median_error'],
    #                            columns = ['cam1', 'cam2', 'combined'])
    all_human_errors = pd.DataFrame(zip(errors_concat.flatten(), errorsRep_concat.flatten()), columns=['ariana_err', 'dalton_err'])         
    human_error = pd.DataFrame(zip([np.nanmean(error_cam1), np.nanmean(errorRep_cam1)], 
                                   [np.nanmean(error_cam2), np.nanmean(errorRep_cam2)], 
                                   [np.nanmean(errors_concat), np.nanmean(errorsRep_concat)]),  
                               index = ['ariana_mean_error', 'dalton_mean_error'],
                               columns = ['cam1', 'cam2', 'combined']) 
    return human_error, all_human_errors, human_error_by_segment, human_error_by_part, dlc_pixel_error, all_dlc_pixel_errors    

global compute_errors_and_tracking_quality
def compute_errors_and_tracking_quality(trajData, frame_shift):
    
    trackingQuality, dlcTrajLengths = record_tracking_quality(trajData, frame_shift)
    labels_info = collect_data_from_hand_labeled_frames(frame_shift)
    
    global posErrResults
    class posErrResults:
        mae = []
        mae_trajByPart = np.empty((len(trajData.dlc), trajData.dlc[0].shape[0]))
        mae_avgOverTraj = []
        mae_avgOverPart = []
        mae_medTrajByPart = np.empty_like(mae_trajByPart)
        mae_medOverTraj = np.empty((mae_trajByPart.shape[0], ))
        mae_medOverPart = np.empty((mae_trajByPart.shape[1], ))
        descriptiveStats = pd.DataFrame(np.empty((7, 5)), 
                                        index=['all', 'Pat', 'Tony', '04_14', '04_15', 'frames_byHand', 'frames_byDLC'], 
                                        columns=['pos_MeanErr', 'pos_std', 'pos_MedErr', 'normErr', 'medNormErr'])
    
    totalPoints = 0
    for traj in trajData.dlc:
        totalPoints += traj.shape[0] * traj.shape[-1]
    
    allFramesError = pd.DataFrame(np.empty((totalPoints, 6)), 
                                  columns=['posErr', 'eventNum', 'part', 
                                           'segment', 'labelingCategory', 'dummyLabel'])
    allFramesError.iloc[:, :] = np.nan
    allFramesError.loc[:, 'dummyLabel'] = 'All'
    segment = ['hand', 'forearm', 'upper arm', 'torso']
    
    rangeInfo = pd.DataFrame(np.empty((totalPoints, 1)), columns = ['posRange'])
    rangeInfo.iloc[:, :] = np.nan
    
    posRange = np.empty_like(posErrResults.mae_trajByPart)
    start = 0
    for eventNum, (dTraj, xTraj, xrawTraj) in enumerate(zip(trajData.dlc, 
                                                            trajData.xromm,  
                                                            trajData.xromm_raw)):  
        
        mae_tmp         = np.empty(np.shape(dTraj)[0])
        maeNorm_tmp     = np.empty_like(mae_tmp)
    
        for part in range(np.shape(dTraj)[0]):
            
            errorByDim = xTraj[part, :, :] - dTraj[part, :, :]                   
            error = np.sqrt(np.square(errorByDim[0, :]) + np.square(errorByDim[1, :]) + np.square(errorByDim[2, :]))
            
            if np.any(~np.isnan(error)):
                
                # compute mean absolute error for this part
                mae_tmp[part] = np.nanmean(error)
                
                # trim raw xromm traces to remove any points at the edges where tracking went off the rails
                xPos_raw_trimmed = xrawTraj[part, :, ~np.isnan(xrawTraj[part, 0])].T
                xPos_raw_trimmed = xPos_raw_trimmed[:, 5:-5]

                # compute the range of motion and max speed for this part
                tmp_posRange = np.nanmax(xPos_raw_trimmed, axis = 1) - np.nanmin(xPos_raw_trimmed, axis = 1)
                posRange[eventNum, part] = np.sqrt(tmp_posRange @ tmp_posRange)

                # compute normalized mean absolute error
                maeNorm_tmp[part] = np.divide(mae_tmp[part], posRange[eventNum, part])        
            else:
                mae_tmp[part]         = np.nan
                maeNorm_tmp[part]     = np.nan
                posRange[eventNum, part] = np.nan
                
            allFramesError.loc[start : start + len(error) - 1,  'posErr'  ] = error
            allFramesError.loc[start : start + len(error) - 1,  'eventNum'] = eventNum
            allFramesError.loc[start : start + len(error) - 1,  'part'    ] = part
            allFramesError.loc[start : start + len(error) - 1,  'segment' ] = segment[int(np.floor(part / 3))]
            
            rangeInfo.loc[start : start + len(error) - 1,  'posRange'  ] = posRange[eventNum, part]

            start += len(error)
                
        posErrResults.mae.append(mae_tmp)
        posErrResults.mae_trajByPart[eventNum, :] = mae_tmp
    
    # save maximum position range and maximum speed across all events for each part
    trajData.maxPosRange = np.nanmax(posRange, axis = 0)
    
    # extract points from labeled frames and record idxs for labeled train frames and for test frames
    labeled_frame_idxs = []
    for eventNum, frames, labParts in zip(labels_info.eventNum, labels_info.frames, labels_info.labeled_parts):
        eventErrors = allFramesError.loc[allFramesError.loc[:, 'eventNum'] == eventNum, ['posErr','part']]
        eventErrors['frameNum'] = np.tile(range(trajData.dlc[eventNum].shape[-1]), trajData.dlc[eventNum].shape[0])
        for fr in frames:
            labeled_frame_idxs.extend(list(eventErrors.index[eventErrors.frameNum == fr]))  
    allFramesError['newLabCategory'] = np.empty((np.shape(allFramesError)[0], 1))
    allFramesError.loc[:, 'newLabCategory'] = 'test'
    allFramesError.loc[labeled_frame_idxs, 'newLabCategory'] = 'train'
    test_frame_idxs = allFramesError.index[allFramesError.newLabCategory == 'test']
    
    # compute mean absolute errors average over events and over parts    
    posErrResults.mae_avgOverTraj = np.divide(np.nansum(np.multiply(posErrResults.mae_trajByPart, dlcTrajLengths), 1), 
                                              np.nansum(dlcTrajLengths, 1))
    posErrResults.mae_avgOverPart = np.divide(np.nansum(np.multiply(posErrResults.mae_trajByPart, dlcTrajLengths), 0), 
                                              np.nansum(dlcTrajLengths, 0))
    
    for eventNum in range(len(trajData.dlc)):
        errIdxs = [i for i, eNum in enumerate(allFramesError.loc[:, 'eventNum']) if eNum == eventNum]
        posErrResults.mae_medOverTraj[eventNum] = np.nanmedian(allFramesError.loc[errIdxs, 'posErr'])
        for part in range(trajData.dlc[0].shape[0]):
            errIdxs = [i for i, (eNum, pNum) in enumerate(zip(allFramesError.loc[:, 'eventNum'], allFramesError.loc[:, 'part'])) if eNum == eventNum and pNum == part]
            posErrResults.mae_medTrajByPart[eventNum, part] = np.nanmedian(allFramesError.loc[errIdxs, 'posErr'])
            
    for part in range(trajData.dlc[0].shape[0]):
        errIdxs = [i for i, pNum in enumerate(allFramesError.loc[:, 'part']) if pNum == part]
        posErrResults.mae_medOverPart[part] = np.nanmedian(allFramesError.loc[errIdxs, 'posErr'])
    
    for part in range(trajData.dlc[0].shape[0]):
        partIdxs = allFramesError.index[allFramesError.loc[:, 'part'] == part]
        rangeInfo.loc[partIdxs, 'posRange'] = rangeInfo.loc[partIdxs, 'posRange'].max() 
    
    allIdxs = range(len(dlc))
    params.patReaches  = event_info.index[event_info.marm == 'PT']
    params.tonyReaches = event_info.index[event_info.marm == 'TY']
    params.day1Reaches = event_info.index[event_info.date == '2019_04_14']
    params.day2Reaches = event_info.index[event_info.date == '2019_04_15']
    for storeIdx, eventIdxs in enumerate([allIdxs, 
                                         params.patReaches, params.tonyReaches,
                                         params.day1Reaches , params.day2Reaches]):
        posVec  = posErrResults.mae_trajByPart[eventIdxs, :].flatten()
        weights = dlcTrajLengths[eventIdxs, :].flatten()
        
        posStats = DescrStatsW(posVec[~np.isnan(posVec)], weights = weights[~np.isnan(posVec)])
    
        norm_posVec = np.divide(posVec, np.repeat(trajData.maxPosRange, len(eventIdxs)))
        norm_posStats = DescrStatsW(norm_posVec[~np.isnan(posVec)], weights = weights[~np.isnan(posVec)])
        
        errIdxs = [i for i, eNum in enumerate(allFramesError.loc[:, 'eventNum']) if eNum in eventIdxs]
        catPosErr = allFramesError.loc[errIdxs, 'posErr']
            
        posErrResults.descriptiveStats.iloc[storeIdx, :] = np.array([posStats.mean, posStats.std, 
                                                                     np.nanmedian(catPosErr), norm_posStats.mean,
                                                                     np.nanmedian(np.divide(catPosErr, rangeInfo.loc[errIdxs, 'posRange']))])

    # isolate error from training frames and compute stats
    trainError = allFramesError.loc[labeled_frame_idxs, :]
    posErrResults.descriptiveStats.loc['frames_byHand', :] = np.array([np.nanmean(trainError.posErr), 
                                                                       np.nanstd(trainError.posErr), 
                                                                       np.nanmedian(trainError.posErr), 
                                                                       np.nan, 
                                                                       np.nanmedian(np.divide(trainError.posErr, rangeInfo.loc[labeled_frame_idxs, 'posRange']))])
    del trainError
    
    # compute stats for test set error
    testError = allFramesError.loc[test_frame_idxs, :]
    posErrResults.descriptiveStats.loc['frames_byDLC', :] = np.array([np.nanmean(testError.posErr), 
                                                                      np.nanstd(testError.posErr), 
                                                                      np.nanmedian(testError.posErr), 
                                                                      np.nan, 
                                                                      np.nanmedian(np.divide(testError.posErr, rangeInfo.loc[test_frame_idxs, 'posRange']))])
    del testError
    
    return posErrResults, allFramesError, trackingQuality

def compute_mean_distances(meanSubtracted):
    
    dlc = trajectoryData.dlc
    
    meanDistances = pd.DataFrame(np.empty((4, 3)), 
                                 index=['hand', 'fore', 'upper', 'torso'], 
                                 columns=['dist 1-2', 'dist 1-3', 'dist 2-3'])
    
    handDist = np.zeros((3, len(dlc)))
    foreDist = np.zeros((3, len(dlc)))
    upperDist = np.zeros((3, len(dlc)))
    torsoDist = np.zeros((1, len(dlc)))
    totalFrames = np.zeros((4, 3))
    for fNum, (dlc, meanSub) in enumerate(zip(dlc, meanSubtracted)):
        for idx, (firstMark, secondMark) in enumerate(zip([0, 0, 1], [1, 2, 2])):
               
            m1 = dlc[firstMark, ...].squeeze()  + np.tile(np.reshape(meanSub[firstMark, :], (3,1)), (1, dlc.shape[-1]))
            m2 = dlc[secondMark, ...].squeeze() + np.tile(np.reshape(meanSub[secondMark, :], (3,1)), (1, dlc.shape[-1]))
            realFrames = np.intersect1d(np.where(~np.isnan(m1[0, :]))[0], np.where(~np.isnan(m2[0, :]))[0])
            handDist[idx, fNum] = np.mean(np.sqrt(np.square(m1[0, realFrames] - m2[0, realFrames]) + np.square(m1[1, realFrames] - m2[1, realFrames]) + np.square(m1[2, realFrames] - m2[2, realFrames]))) * len(realFrames)
            totalFrames[0, idx] += len(realFrames)
            
            m1 = dlc[firstMark  + 3, ...].squeeze()  + np.tile(np.reshape(meanSub[firstMark + 3, :], (3,1)), (1, dlc.shape[-1]))
            m2 = dlc[secondMark + 3, ...].squeeze() + np.tile(np.reshape(meanSub[secondMark + 3, :], (3,1)), (1, dlc.shape[-1]))
            realFrames = np.intersect1d(np.where(~np.isnan(m1[0, :]))[0], np.where(~np.isnan(m2[0, :]))[0])
            foreDist[idx, fNum] = np.mean(np.sqrt(np.square(m1[0, realFrames] - m2[0, realFrames]) + np.square(m1[1, realFrames] - m2[1, realFrames]) + np.square(m1[2, realFrames] - m2[2, realFrames]))) * len(realFrames)
            totalFrames[1, idx] += len(realFrames)
            
            m1 = dlc[firstMark  + 6, ...].squeeze()  + np.tile(np.reshape(meanSub[firstMark + 6, :], (3,1)), (1, dlc.shape[-1]))
            m2 = dlc[secondMark + 6, ...].squeeze() + np.tile(np.reshape(meanSub[secondMark + 6, :], (3,1)), (1, dlc.shape[-1]))
            realFrames = np.intersect1d(np.where(~np.isnan(m1[0, :]))[0], np.where(~np.isnan(m2[0, :]))[0])
            upperDist[idx, fNum] = np.mean(np.sqrt(np.square(m1[0, realFrames] - m2[0, realFrames]) + np.square(m1[1, realFrames] - m2[1, realFrames]) + np.square(m1[2, realFrames] - m2[2, realFrames]))) * len(realFrames)
            totalFrames[2, idx] += len(realFrames)
            
            if idx < 1:
                m1 = dlc[firstMark  + 9, ...].squeeze()  + np.tile(np.reshape(meanSub[firstMark  + 9, :], (3,1)), (1, dlc.shape[-1]))
                m2 = dlc[secondMark + 9, ...].squeeze()  + np.tile(np.reshape(meanSub[secondMark + 9, :], (3,1)), (1, dlc.shape[-1]))
                realFrames = np.intersect1d(np.where(~np.isnan(m1[0, :]))[0], np.where(~np.isnan(m2[0, :]))[0])
                torsoDist[idx, fNum] = np.mean(np.sqrt(np.square(m1[0, realFrames] - m2[0, realFrames]) + np.square(m1[1, realFrames] - m2[1, realFrames]) + np.square(m1[2, realFrames] - m2[2, realFrames]))) * len(realFrames)
                totalFrames[3, idx] += len(realFrames)
                            
    meanDistances.iloc[0, :] = np.nansum(handDist , axis=-1) / totalFrames[0, :]
    meanDistances.iloc[1, :] = np.nansum(foreDist , axis=-1) / totalFrames[1, :]
    meanDistances.iloc[2, :] = np.nansum(upperDist, axis=-1) / totalFrames[2, :]
    meanDistances.iloc[3, :] = np.nansum(torsoDist, axis=-1) / totalFrames[3, 0]
    
    return meanDistances

if __name__ == "__main__":
 
    first = True
    for base in dpath.base:
        print('\n\n\n' + base + '\n\n\n')
        if mode == 'sweep':
            os.makedirs(dpath.sweep_extra_data_path, exist_ok=True)
            parameter_identifier = os.path.basename(base)
            parameter_details = parameter_identifier.split('_')[2:]
            parameter_details[-1] = float(parameter_details[-1][-2:]) / 100
            parameter_details = [float(val) for val in parameter_details]
            parameter_details.append(params.acceptable_reprojectionError_threshold_day2[params.reprojNum])
            parameter_details = np.array(parameter_details)
            parameter_identifier = parameter_identifier + '_' + str(params.acceptable_reprojectionError_threshold_day2[params.reprojNum])
            dataset_save_path = os.path.join(dpath.sweep_extra_data_path, 
                                             'posError_and_trackingQuality_' + parameter_identifier + '.pickle')
            if os.path.isfile(dataset_save_path):
                continue
            
        data_dirs = []
        for date in dpath.dates:
            data_dirs.append(os.path.join(base, date, 'pose-3d'))
    
        dlc, dlc_metadata, event_info = load_dlc_data(data_dirs)
        xromm = load_xromm_data()
        dlc_filtered = filter_dlc(dlc, dlc_metadata, params.reprojNum)
        dlc_filtered = flip_axes(dlc_filtered, event_info)        
        dlc_aligned, xromm_aligned, xromm_raw_aligned, frame_shift, meanSubtracted = align_and_mean_subtract_trajectories(xromm, dlc_filtered)
        dlc_aligned, xromm_aligned, xromm_raw_aligned = subtract_mean_over_all_points(dlc_aligned, xromm_aligned, xromm_raw_aligned)
        
        global trajectoryData
        class trajectoryData:
            dlc           = dlc_aligned
            xromm         = xromm_aligned
            xromm_raw     = xromm_raw_aligned    
            
        trajectoryData, dlc_bases = project_trajectories_on_common_axes(trajectoryData)
    
        posErrResults, allFramesError, trackingQuality = compute_errors_and_tracking_quality(trajectoryData, frame_shift)
        
        if mode == 'sweep':
            with open(dataset_save_path, 'wb') as fp:
                dill.dump([posErrResults, trackingQuality], fp, recurse=True, protocol = pickle.HIGHEST_PROTOCOL)
            
            if first:
                tmp = np.hstack((parameter_details,
                                 posErrResults.descriptiveStats.loc['all', 'pos_MeanErr'],
                                 posErrResults.descriptiveStats.loc['all', 'pos_MedErr' ],
                                 trackingQuality.total_percentTracked))
                if os.path.isfile(dpath.results_storage_path):
                    with open(dpath.results_storage_path, 'rb') as fp:
                        full_sweep_results = np.array(dill.load(fp))
                    full_sweep_results = np.vstack((full_sweep_results, 
                                                    np.reshape(tmp, (1, len(tmp)))
                                                    ))
                else:
                    full_sweep_results = np.reshape(tmp, (1, len(tmp)))
                first = False
            else:
                tmp = np.hstack((parameter_details,
                                 posErrResults.descriptiveStats.loc['all', 'pos_MeanErr'],
                                 posErrResults.descriptiveStats.loc['all', 'pos_MedErr' ],
                                 trackingQuality.total_percentTracked))
                full_sweep_results = np.vstack((full_sweep_results, 
                                                np.reshape(tmp, (1, len(tmp)))
                                                ))            
        elif mode == 'trainFrac':
            trainingset_identifier = os.path.basename(base)
            trainingset_details = trainingset_identifier.split('_')[2:]
            trainingset_details = np.array([int(val) for idx, val in enumerate(trainingset_details) if idx % 2 == 1])
            print(trainingset_details)
            if first:
                tmp = np.hstack((trainingset_details,
                                 posErrResults.descriptiveStats.loc['all', 'pos_MeanErr'],
                                 posErrResults.descriptiveStats.loc['all', 'pos_MedErr' ],
                                 trackingQuality.total_percentTracked))
                trainingset_results = np.reshape(tmp, (1, len(tmp)))
                first = False
            else:
                tmp = np.hstack((trainingset_details,
                                 posErrResults.descriptiveStats.loc['all', 'pos_MeanErr'],
                                 posErrResults.descriptiveStats.loc['all', 'pos_MedErr' ],
                                 trackingQuality.total_percentTracked))
                trainingset_results = np.vstack((trainingset_results, 
                                                 np.reshape(tmp, (1, len(tmp)))
                                                 ))
    
    if mode == 'sweep':
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
    
    elif mode == 'trainFrac':
        trainingset_results = pd.DataFrame(trainingset_results,
                                           columns = ['trainFrac', 
                                                      'shuffle', 
                                                      'snapshot', 
                                                      'snapIdx', 
                                                      'mean_err', 
                                                      'median_err', 
                                                      'trackPercent'])
        with open(dpath.results_storage_path, 'wb') as fp:
            dill.dump(trainingset_results, fp, recurse = True, protocol = pickle.HIGHEST_PROTOCOL)                                                    
    
    elif mode == 'single':
        no_epipolar_lines_pixel_error = compare_pre_eplines_to_post() 
        human_error, all_human_errors, human_error_by_segment, human_error_by_part, dlc_pixel_error, all_dlc_pixel_errors = compute_human_and_dlc_pixel_labeling_error(dlc_filtered)
        meanDistances = compute_mean_distances(meanSubtracted)

        pixels_save_path = os.path.join(dpath.results_storage_path, 'human_vs_human_vs_DLC_error_and_preEpipolar_error.pickle')
        with open(pixels_save_path, 'wb') as fp:
            dill.dump([no_epipolar_lines_pixel_error, human_error, all_human_errors, dlc_pixel_error, all_dlc_pixel_errors], fp, protocol = pickle.HIGHEST_PROTOCOL)
                
        processed_results_save_path = os.path.join(dpath.results_storage_path, 'trajData_posError_trackingQuality_pixelErrors_dlcBases.pickle')
        with open(processed_results_save_path, 'wb') as fp:
            dill.dump([trajectoryData, posErrResults, trackingQuality, allFramesError, dlc_bases], fp, recurse=True) 
