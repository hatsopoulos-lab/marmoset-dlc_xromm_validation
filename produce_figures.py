# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 11:05:27 2021

@author: Dalton
"""

import matplotlib.pyplot as plt
import dill
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib
import cv2
import h5py

load_type = 'hdf' # can be 'hdf' or 'pickle'
iteration = 1     # can be 0 or 1

project_base = r'Z:/dalton_moore/dryad_deposit'
results_base = os.path.join(project_base, 'results_presented_in_manuscript')    

class path:
    figures_storage = os.path.join(results_base, 'figures')
    if load_type == 'pickle':
        data = os.path.join(results_base, 'pickle_versions')
        fileCode = '.pickle'
    elif load_type == 'hdf':
        data = os.path.join(results_base, 'hdf_versions')
        fileCode = '.h5'
    processed_results_path = os.path.join(data, 'trajData_posError_trackingQuality_pixelErrors_dlcBases%s' % fileCode)
    eval_results = os.path.join(project_base, 'deeplabcut_files/combined_dlc_xromm_validation-Dalton-2021-08-28/evaluation-results/iteration-1/CombinedEvaluation-results.csv')
    training_set_size_posErr = os.path.join(data, 'training_set_size_iteration%d%s' % (iteration, fileCode))
    full_sweep_results = os.path.join(data, 'parameter_sweep%s' % fileCode)
    labeling_error_results = os.path.join(data, 'human_vs_human_vs_DLC_error_and_preEpipolar_error%s' % fileCode)   
    
class params:
    axis_fontsize = 8
    dpi = 300 # set to 4000 for publication, reduced for testing speed
    anipose_param_names = ['Offset Threshold', 'N-Back', 'Smoothing Factor', 
                           'Spatial Constraint', 'Reprojection Error Threshold (Anipose)', 
                           'Score Threshold', 'Reprojection Error Threshold (post)']
    axis_linewidth = 1.25
    tick_length = 2
    tick_width = 1
    
class fig1_params:
    images_path = os.path.join(project_base, 'deeplabcut_files/dlc_files_for_secondary_analyses/frames_with_axes')
    dateIdx = 0
    axes_dotSize = 14
    line_thickness = 12
    originShift = [12, 0, 4]

class fig2_params:
    single_param_figsize = (1.5, 1.5)
    sorted_combinations_figSize = (4.5,3)
    sorted_combinations_lw = 2
    sorted_combinations_markerSize = 4
    sorted_combinations_markerEdgeWidth = 0.5
    sorted_err_ticks = [0.225, 0.23, 0.235, 0.24]
    sorted_per_ticks = [0.82, 0.85, 0.88, 0.91]
    
    err_range = [0.225, 0.24]#[.23, .245]
    percent_range = [0.8, 0.9]#[.87, .92]
    
    err_ticks = [0.225, 0.23, 0.235, 0.24]#[.235, .24, .245]
    per_ticks = [0.82, 0.85, 0.88]#[.88, .90, .92]

class fig3_params:
    dirColors = np.array([[27 , 158, 119],
                          [217, 95 , 2  ],
                          [117, 112, 179]])
    figSize   = (3.5, 2.75)
    figSize3d = (4.25, 4.25)
    angles3d = [18, -170]
    ylim = [-9, 12]
    thickMult = 0.012 
    textMult = 4
    lw = 1.25
    startMarkSize = 6
    unit = 'cm'
    yLen = 5
    panelB_scalebar_yStart = 6
    panelCD_scalebar_yStart = 2
    
    histFigSize = (6, 3)
    hist_lw = 2
    hist_median_lw = 2
    
    dlc_fps = 200
    xromm_fps = 200   

class fig2_notsig_params:
    single_param_figsize = (2, 0.67)
    err_range = [.235, .24]
    percent_range = [.8867, .9033]
    
    err_ticks = [.235, .24]
    per_ticks = [.89, .90]    

    
class figS1_params:
    iteration_figSize = (3.5, 2.5)
    figSize = (2.5, 2.5)

class figS2_params:
    heatmap_figSize = (1, 1)
    sorted_combinations_figSize = (5,3)
    
    
font = {'family'     : 'sans-serif',
        'sans-serif' : ['Arial'],
        'size'       : 8}

matplotlib.rc('font', **font)


def standardize_plot_appearance(ax, top=True, right=True, left=False, bottom=False):    
    ax.spines[['bottom', 'left', 'top', 'right']].set_linewidth(params.axis_linewidth)
    sns.despine(ax=ax, top=top, right=right, left=left, bottom=bottom)
    plt.tight_layout()
    
def plotTrajectories(trajNum, parts, colors, figSet, pos, vel, combined):
    
    if parts == 'all':
        parts = range(11)
    
    dlc           = trajectoryData.dlc
    xromm         = trajectoryData.xromm
    xromm_raw     = trajectoryData.xromm_raw
    
    colors = [tuple(col/255) for col in colors]
    
    jumpColors = np.array([[102,194,165],
                           [252,141,98],
                           [141,160,203]])
    jumpColors = [tuple(col/255) for col in jumpColors]
    
    eLabLoc = [-0.1, 0.5] # [0.5, 0.5] [-0.08, 0.5]
    tLabLoc = [-0.1, 0.5] # [0.5, 0.5]# [-0.08, 0.5]
    
    if pos:
        minFrames = []
        maxFrames = []
        for part in parts:
            tempFrames = np.where(~np.isnan(xromm_raw[trajNum][part, 0, :]))[0]
            if len(tempFrames) != 0:
                minFrames.append(np.nanmin(tempFrames))
                maxFrames.append(np.nanmax(tempFrames))    
        minFrame = np.min(minFrames)
        maxFrame = np.max(maxFrames)

    plt.style.use('seaborn-white')
    sns.set_style('ticks')
    for part in parts: 
        time = np.linspace(0, np.shape(dlc[trajNum])[2] / fig3_params.dlc_fps, num = np.shape(dlc[trajNum])[2])
        x_time = np.linspace(0, np.shape(xromm_raw[trajNum])[2] / fig3_params.xromm_fps, num = np.shape(xromm_raw[trajNum])[2])
        
        realFramesX = np.where(~np.isnan(xromm_raw[trajNum][part, 0, :]))[0]
        if trackingQuality.percentTracked[trajNum, part] == 0:
            continue
        elif len(realFramesX) == 0:
            continue

        frSliceX = slice(realFramesX[0], realFramesX[-1])
        tAdj = x_time[realFramesX[0]]
        
        errOff = 10
        if pos:
            if combined:
                fig, (axPos, axErrPos, axVel, axErrVel) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [4, 1, 4, 1]}, figsize=fig3_params.figSize)
            else:
                fig, (axPos, axErrPos) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=fig3_params.figSize)
   
            realFrames = np.where(~np.isnan(dlc[trajNum][part, 0, :]))
            frSlice = slice(realFrames[0][0], realFrames[0][-1])
            if len(realFrames[0]) == 0:
                firstSpace = 0
                secondSpace = 0
            else:
                firstSpace  = dlc[trajNum][part, 0, realFrames].squeeze().min() - dlc[trajNum][part, 1, realFrames].squeeze().max()
                secondSpace = dlc[trajNum][part, 1, realFrames].squeeze().min() - dlc[trajNum][part, 2, realFrames].squeeze().max()

            errorByDim = xromm_raw[trajNum][part, :, realFrames].squeeze() - dlc[trajNum][part, :, realFrames].squeeze() 
            error = np.sqrt(np.square(errorByDim[:, 0]) + np.square(errorByDim[:, 1]) + np.square(errorByDim[:, 2]))

            errorByDim = xromm_raw[trajNum][part, :, frSlice].squeeze() - dlc[trajNum][part, :, frSlice].squeeze() 
            error = np.sqrt(np.square(errorByDim[0]) + np.square(errorByDim[1]) + np.square(errorByDim[2]))

            axPos.plot(time[frSlice] - tAdj, dlc[trajNum][part, 0, frSlice].squeeze() - firstSpace  + 0.5, linestyle='-', color=colors[0], linewidth=fig3_params.lw)
            axPos.plot(time[frSlice] - tAdj, dlc[trajNum][part, 1, frSlice].squeeze()                    , linestyle='-', color=colors[1], linewidth=fig3_params.lw)
            axPos.plot(time[frSlice] - tAdj, dlc[trajNum][part, 2, frSlice].squeeze() + secondSpace - 0.5, linestyle='-', color=colors[2], linewidth=fig3_params.lw)
          
            axPos.plot(x_time[frSliceX][2*errOff:] - tAdj, xromm_raw[trajNum][part, 0, frSliceX][2*errOff:].squeeze() - firstSpace  + 0.5, linestyle='-.', color=colors[0], linewidth=fig3_params.lw*1.25)
            axPos.plot(x_time[frSliceX][2*errOff:] - tAdj, xromm_raw[trajNum][part, 1, frSliceX][2*errOff:].squeeze()                    , linestyle='-.', color=colors[1], linewidth=fig3_params.lw*1.25)
            axPos.plot(x_time[frSliceX][2*errOff:] - tAdj, xromm_raw[trajNum][part, 2, frSliceX][2*errOff:].squeeze() + secondSpace - 0.5, linestyle='-.', color=colors[2], linewidth=fig3_params.lw*1.25)    

            axErrPos.plot(time[realFramesX][errOff:] - tAdj + np.diff(time[realFramesX[[0, errOff]]]), np.repeat(np.nanmedian(error), len(time[realFramesX][errOff:])), linestyle='-', color=(.7, .7, .7), linewidth=fig3_params.lw*1.5) 
            axErrPos.plot(time[frSlice]  - tAdj, error, linestyle='-', color='k', linewidth=fig3_params.lw)
            
            y_yStart = xromm_raw[trajNum][part, 1, realFramesX].squeeze()[0]
            
            axPos.set_yticklabels([])
            axPos.set_xticklabels([])
            if combined:
                axErrPos.set_xticklabels([])
            else:
                axErrPos.set_xlabel('Time (s)', fontsize=params.axis_fontsize)
            
            if part == 0:
                axErrPos.set_xticks(np.arange(0, 4, 1))
            else:
                axErrPos.set_xticks(np.arange(x_time[realFramesX[0]] - tAdj, x_time[realFramesX[-1]] - tAdj, 1))  
            
            axErrPos.set_yticks([round(np.nanmedian(error), 1), 1.5])
            
            plt.tight_layout()
            plt.show()
        
            if figSet == 0:
                axErrPos.set_ylabel('Error (cm)', fontsize=params.axis_fontsize)
                axErrPos.yaxis.set_label_coords(eLabLoc[0], eLabLoc[1])
                axPos.set_ylabel('Position', fontsize=params.axis_fontsize)
                axPos.yaxis.set_label_coords(tLabLoc[0], tLabLoc[1])
            for item in [axPos.xaxis, axPos.yaxis]:
                item.label.set_fontsize(params.axis_fontsize)
            for label in (axErrPos.get_xticklabels() + axErrPos.get_yticklabels()):
                label.set_fontsize(params.axis_fontsize)
            
            
            axPos.spines['bottom'].set_linewidth(params.axis_linewidth)
            axErrPos.spines['bottom'].set_linewidth(params.axis_linewidth)
            axErrPos.spines['left'].set_linewidth(params.axis_linewidth)
            axErrPos.tick_params(bottom=True, left=True, length = params.tick_length, width = params.tick_width, direction='out')
            
            sns.despine(bottom=True, left=True, offset={'bottom': 0}, ax=axPos)
            sns.despine(bottom=False, left=False, ax=axErrPos)

            if trajNum == 16:
                y_yStart = fig3_params.panelB_scalebar_yStart 
            else:
                y_yStart = fig3_params.panelCD_scalebar_yStart
                
            y_xStart = time[minFrame] - tAdj  
            yThick = fig3_params.thickMult * (time[maxFrame] - time[minFrame])
            yScale = matplotlib.patches.Rectangle((y_xStart, y_yStart), yThick, fig3_params.yLen, angle=0.0, clip_on=False, facecolor=(0.1, 0.1, 0.1))
            axPos.add_patch(yScale)
            axPos.text(y_xStart - fig3_params.textMult*yThick, y_yStart + 0.25*fig3_params.yLen, str(fig3_params.yLen) + ' ' + fig3_params.unit, 
                       clip_on=False, rotation='vertical', fontsize=params.axis_fontsize)    
  
            axPos.set_xlim(   time[minFrame] - tAdj, time[maxFrame] - tAdj)        
            axErrPos.set_xlim(time[minFrame] - tAdj, time[maxFrame] - tAdj)         
            axPos.set_ylim(fig3_params.ylim[0], fig3_params.ylim[1])

        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(path.figures_storage, 'trajNum_' + str(trajNum) + '_part_' + str(part) + '_xyz.png'), bbox_inches='tight', dpi=params.dpi)
    
    part = parts[0]
    
    fig = plt.figure(figsize=fig3_params.figSize3d)
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.view_init(fig3_params.angles3d[0], fig3_params.angles3d[1])

    idx = [0, len(dlc[trajNum][part, 0, :])]
    
    blue = tuple(np.array([0, 0, 139])/255)
    lightblue = tuple(np.array([0,191,255])/255)
    
    firstReal = np.where(~np.isnan(dlc[trajNum][part, 0, :]))[0].min()
    
    ax3d.plot(dlc[trajNum][part, 0, idx[0]:idx[1]], dlc[trajNum][part, 1, idx[0]:idx[1]], dlc[trajNum][part, 2, idx[0]:idx[1]], color = blue, linestyle = '-', linewidth=fig3_params.lw)
    ax3d.plot(xromm[trajNum][part, 0, idx[0]:idx[1]], xromm[trajNum][part, 1, idx[0]:idx[1]], xromm[trajNum][part, 2, idx[0]:idx[1]], color = lightblue, linestyle = '-.', linewidth=fig3_params.lw*1.25)
    ax3d.plot(dlc[trajNum][part, 0, firstReal], dlc[trajNum][part, 1, firstReal], dlc[trajNum][part, 2, firstReal], color = blue, marker = 'o', markersize=fig3_params.startMarkSize)
    # ax3d.plot(dlc[trajNum][part, 0, firstReal:firstReal+1], dlc[trajNum][part, 1, firstReal:firstReal+1], dlc[trajNum][part, 2, firstReal:firstReal+1], color = blue, marker = 'o', markersize=fig3_params.startMarkSize)

    minLim = -2.96838
    maxLim =  3.20595
    
    ax3d.set_xlim(minLim, maxLim)
    ax3d.set_ylim(minLim, maxLim)
    ax3d.set_zlim(minLim, maxLim)
    ax3d.set_xlabel('x (cm)', fontsize=params.axis_fontsize)
    ax3d.set_ylabel('y (cm)', fontsize=params.axis_fontsize)
    ax3d.set_zlabel('z (cm)', fontsize=params.axis_fontsize)    
    for item in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        item.label.set_fontsize(params.axis_fontsize)
    for label in (ax3d.get_xticklabels() + ax3d.get_yticklabels() + ax3d.get_zticklabels()):
        label.set_fontsize(params.axis_fontsize)
    ax3d.legend(['DLC', 'XROMM'], loc='upper right', bbox_to_anchor=(0.8, 0.4), fontsize = params.axis_fontsize, shadow=False)
    ax3d.grid(False)
    for axis in [ax3d.w_xaxis, ax3d.w_yaxis, ax3d.w_zaxis]:
        axis.line.set_linewidth(params.axis_linewidth)   
        
    ax3d.tick_params(axis = 'x', length = params.tick_length, width = params.tick_width, direction='inout')
    ax3d.tick_params(axis = 'y', length = params.tick_length, width = params.tick_width, direction='inout')
    ax3d.tick_params(axis = 'z', length = params.tick_length, width = params.tick_width, direction='inout')
    plt.show()
    fig.savefig(os.path.join(path.figures_storage, 'trajNum_' + str(trajNum) + '_part_' + str(parts[0]) + '_3D.png'), dpi=params.dpi)

def labelExample_with_basisAxes(imgPath, savePath, axes, colors):
    img = cv2.imread(imgPath)
    
    colors = colors[:, (2, 1, 0)].astype(np.int64)  
    axes = axes.astype(np.int64)
    
    for endPts, color in zip(axes[1:], colors):
        color = tuple([int(x) for x in color])
        img = cv2.arrowedLine(img, tuple(axes[0][0]), tuple(endPts[0]), color=color, thickness=fig1_params.line_thickness)   
    img = cv2.circle(img, tuple(axes[0][0]), fig1_params.axes_dotSize, tuple([int(x) for x in [0,0,0]]), thickness=-1)        
    
    cv2.imwrite(savePath, img)

def fig1_project_basis_on_images(originShift = [0, 0, 0]):
    dlc1 = os.path.join(fig1_params.images_path, 'dlc_2019_04_14_event028_cam1_frame_150.png')
    dlc2 = os.path.join(fig1_params.images_path, 'dlc_2019_04_14_event028_cam2_frame_150.png')
    
    dlc1_with_axes = os.path.join(fig1_params.images_path, 'dlc_2019_04_14_event028_cam1_frame_150_with_PCA_axes.png')
    dlc2_with_axes = os.path.join(fig1_params.images_path, 'dlc_2019_04_14_event028_cam2_frame_150_with_PCA_axes.png')
    
    rvecs = np.array([[1.828952184318242, -0.9655738973550269, 0.5430500204720892],
                      [1.9549978901014753, 0.5182194712846203, -0.015591874583159708]], dtype = np.float32)
    tvecs = np.array([[1.5557070238429147, 9.251689275812153, 36.569331730559405],
                      [-5.67497848410687, 4.824943386986453, 35.710803979112946]], dtype = np.float32)
    camMats = np.array([[[2.37401990e+03,              0, 7.22472508e+02],
                         [             0, 2.36461960e+03, 6.08605089e+02],
                         [             0,              0,              1]],
                        [[1.59495385e+03,              0, 8.01129953e+02],
                         [             0, 1.59423315e+03, 5.80006439e+02],
                         [             0,              0,              1]]], dtype = np.float32)
    distCoeffs = np.array([[-4.73839306e-01,  5.21597237, 1.12650634e-03, 2.80757313e-03, -4.76103219e+01],
                           [    -0.31741825, -0.64961686,    -0.00421487,    -0.00304917,      3.03281784]], dtype = np.float32)
    
    origin_and_axes = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]], dtype = np.float32)   
    origin_and_axes = origin_and_axes @ dlc_bases[fig1_params.dateIdx].T
    origin_and_axes = origin_and_axes + np.tile(np.array(originShift, dtype=np.float32), (4, 1))    
    
    imagePoints_cam1, tmp = cv2.projectPoints(origin_and_axes, rvecs[0], tvecs[0], camMats[0], distCoeffs[0])
    imagePoints_cam2, tmp = cv2.projectPoints(origin_and_axes, rvecs[1], tvecs[1], camMats[1], distCoeffs[1])

    labelExample_with_basisAxes(dlc1, dlc1_with_axes, imagePoints_cam1, fig3_params.dirColors)
    labelExample_with_basisAxes(dlc2, dlc2_with_axes, imagePoints_cam2, fig3_params.dirColors)


def plot_learning_progress():
    eval_results = pd.read_csv(path.eval_results)
    eval_results = eval_results.loc[~np.isnan(eval_results['%Training dataset']), :]
    results_for_plot = pd.DataFrame(np.vstack((eval_results.loc[:, ['Training iterations:', '%Training dataset', 'Shuffle number', ' Train error(px)']],
                                               eval_results.loc[:, ['Training iterations:', '%Training dataset', 'Shuffle number', ' Test error(px)']])),
                                    columns = ['iteration', 'trainFrac', 'shuffle', 'error_px'])
    
    results_for_plot['trainFrac'] = results_for_plot['trainFrac'] / 100    

    results_for_plot['train/test'] = ['Train' for idx in range(eval_results.shape[0])] + ['Test' for idx in range(eval_results.shape[0])]
    fig, ax0 = plt.subplots(figsize=figS1_params.iteration_figSize)
    ax0 = sns.lineplot(data = results_for_plot, x = 'iteration', 
                       y = 'error_px', hue = 'trainFrac', style = 'train/test', 
                       palette = 'crest', err_style = 'bars', ci = 'sd')
    ax0.set_ylim(0, 25)
    ax0.set_xlabel('Iteration', fontsize=params.axis_fontsize)
    ax0.set_ylabel('Error (pixels)', fontsize=params.axis_fontsize)
    sns.despine(top=True, right=True, ax=ax0)
    ax0.spines[['bottom', 'left']].set_linewidth(params.axis_linewidth)
    ax0.tick_params(bottom=True, left=True, length = params.tick_length, width = params.tick_width, direction='out')

    plt.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left', borderaxespad=0)

    plt.show()
    fig.savefig(os.path.join(path.figures_storage, 'pixel_error_vs_iteration_for_all_trainFracs.png'), bbox_inches = 'tight', dpi=params.dpi)


def plot_training_size_effect(all_human_errors, trainingset_results):
    all_human_errors = pd.concat([all_human_errors, all_human_errors], ignore_index=True)   
    all_human_errors['x'] = np.hstack((np.repeat(np.nanmin(trainingset_results['trainFrac']/100), all_human_errors.shape[0]/2),
                                       np.repeat(np.nanmax(trainingset_results['trainFrac']/100), all_human_errors.shape[0]/2)))
    
    eval_results = pd.read_csv(path.eval_results)
    eval_results = eval_results.loc[~np.isnan(eval_results['%Training dataset']), :]
    eval_results = eval_results.sort_values(by = ['%Training dataset', 'Shuffle number', 'Training iterations:'])

    trainingset_results[['train_px_err', 'test_px_err']] = np.full((trainingset_results.shape[0], 2), np.nan)    
    for idx, vals in trainingset_results.iterrows():
        trainingset_results.iloc[idx, -2:] = eval_results.loc[(eval_results['%Training dataset']    == vals.trainFrac) &
                                                              (eval_results['Shuffle number']       == vals.shuffle  ) &
                                                              (eval_results['Training iterations:'] == vals.snapshot ),  
                                                              [' Train error(px)', ' Test error(px)']]#['Train error with p-cutoff', 'Test error with p-cutoff']]
   
    trainingset_results['trainFrac'] = trainingset_results['trainFrac'] / 100    
    trainSet_stacked = pd.concat([trainingset_results.iloc[:, :-2], 
                                  trainingset_results.iloc[:, :-2]])
    trainSet_stacked['px_error']   = np.hstack((trainingset_results['train_px_err'], trainingset_results['test_px_err']))
    trainSet_stacked['train/test'] = ['Train' for idx in range(trainingset_results.shape[0])] + ['Test' for idx in range(trainingset_results.shape[0])]
    fig, ax0 = plt.subplots(figsize=figS1_params.figSize)
    ax0 = sns.lineplot(data = all_human_errors, x = 'x', y = 'dalton_err',
                       ci=95, color='black')
    ax0 = sns.lineplot(data = all_human_errors, x = 'x', y = 'ariana_err',
                       ci=95, color='black', linestyle='-.')
    # ax0 = sns.lineplot(data = all_human_errors, x = 'x', y = 'ariana_err',
    #                    ci=95, color='black')
    ax0 = sns.lineplot(data = trainSet_stacked, x = 'trainFrac', 
                       y = 'px_error', hue = 'train/test', style='train/test', 
                       palette = 'Paired', ci = None, legend = 'full')
    ax0 = sns.scatterplot(data = trainSet_stacked, x = 'trainFrac', 
                          y = 'px_error', hue = 'train/test', 
                          palette = 'Paired', legend = 'full')
    ax0.set_xlabel('Training Set Fraction', fontsize=params.axis_fontsize)
    ax0.set_ylabel('Error (pixels)', fontsize=params.axis_fontsize)
    
    sns.despine(top=True, right=True, ax=ax0)
    
    ax0.spines[['bottom', 'left']].set_linewidth(params.axis_linewidth)
    ax0.tick_params(bottom=True, left=True, length = params.tick_length, width = params.tick_width, direction='out')
    ax0.set_xticks(np.unique(trainingset_results.trainFrac))
    ax0.set_yticks([np.mean(all_human_errors.dalton_err), np.mean(all_human_errors.ariana_err)])
    ax0.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

    plt.legend(bbox_to_anchor=(1.1, 1.1), loc='upper left', borderaxespad=0)

    plt.show()
    fig.savefig(os.path.join(path.figures_storage, 'pixel_error_vs_trainFracs.png'), bbox_inches = 'tight', dpi=params.dpi)
    
    colors = [tuple(col/255) for col in fig3_params.dirColors]

    fig, ax1 = plt.subplots(figsize=figS1_params.figSize)
    color = colors[0]
    ax1.set_xlabel('Training Set Fraction', fontsize=params.axis_fontsize)
    ax1.set_ylabel('Median Position Error (cm)', fontsize=params.axis_fontsize, color=color)
    ax1 = sns.lineplot(data = trainingset_results, x='trainFrac', y='median_err', 
                       linestyle = '-', err_style='band', ci=None, color = color)
    ax1 = sns.scatterplot(data = trainingset_results, x='trainFrac', y='median_err', 
                          color = color)
    # ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    color = colors[1]
    ax2.set_ylabel('Percent Frames Tracked', fontsize=params.axis_fontsize, color=color)
    ax2 = sns.lineplot(data = trainingset_results, x='trainFrac', y='trackPercent', 
                       linestyle = '-.', err_style='band', ci=None, color = color)    
    ax2 = sns.scatterplot(data = trainingset_results, x='trainFrac', y='trackPercent', 
                          color = color)
    # ax2.tick_params(axis='y', color=color)
    
    rangeMultiplier = 50
    ax1_range = trainingset_results.groupby('trainFrac').std().mean()['median_err'] * rangeMultiplier
    ax2_range = trainingset_results.groupby('trainFrac').std().mean()['trackPercent'] * rangeMultiplier
    
    start = 0.2
    end = 1
    ax1.set_ylim(start, start + ax1_range)
    ax2.set_ylim(end - ax2_range, end)

    sns.despine(top=True, right=False, ax=ax1)    
    sns.despine(top=True, right=False, ax=ax2)
    
    ax1.set_xticks(np.unique(trainingset_results.trainFrac))

    ax1.spines[['bottom', 'left']].set_linewidth(params.axis_linewidth)
    ax2.spines['right'].set_linewidth(params.axis_linewidth)

    ax1.tick_params(bottom=True, left=True, length = params.tick_length, width = params.tick_width, direction='out')
    ax2.tick_params(axis='y', length = params.tick_length, width = params.tick_width, direction='out')
    
    plt.show()    
    fig.savefig(os.path.join(path.figures_storage, 'posError_and_tracking_vs_trainFracs.png'), bbox_inches = 'tight', dpi=params.dpi)

    
def errors_histogram():

    allFramesError.loc[allFramesError['segment'] == 'hand', 'segment'] = 'Hand'    
    allFramesError.loc[allFramesError['segment'] == 'forearm', 'segment'] = 'Forearm'    
    allFramesError.loc[allFramesError['segment'] == 'upper arm', 'segment'] = 'Upper Arm'    
    allFramesError.loc[allFramesError['segment'] == 'torso', 'segment'] = 'Torso'    

    fig, ax0 = plt.subplots(figsize=fig3_params.histFigSize)
    sns.kdeplot(data=allFramesError, x="posErr", hue="segment", 
                 shade=False, linewidth = fig3_params.hist_lw, ax = ax0, palette='Set2', 
                 common_norm=False, legend = True)
    
    pal = sns.color_palette('Set2').as_hex()[:len(ax0.lines)]     
    medianErrs = allFramesError.groupby(['segment']).median().posErr
    medianErrs = medianErrs.iloc[[1, 0, 3, 2]]
    kdelines = ax0.lines
    for kdeline, med, col in zip(kdelines[::-1], medianErrs, pal):
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        ax0.vlines(med, 0, np.interp(med, xs, ys), colors=col, ls='-', lw=fig3_params.hist_lw)

    ax0.vlines(allFramesError['posErr'].median()+0.002, 0, np.max(kdelines[1].get_ydata()), 
               colors='black', lw = fig3_params.hist_median_lw, linestyle = '--')
    
    ax0.set_xlim(0, 1)
    ax0.set_xlabel('Position Error (cm)', fontsize=params.axis_fontsize)
    ax0.set_ylabel('Density', fontsize=params.axis_fontsize)
    ax0.set_yticklabels('')
    ax0.set_xticks([0, allFramesError['posErr'].median(), 0.5, 1])
    ax0.tick_params(bottom=True, left=False, length = params.tick_length, width = params.tick_width, direction='out')
    
    ax0.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
    
    ax0.spines[['bottom', 'left']].set_linewidth(params.axis_linewidth)
    sns.despine(ax=ax0)
    plt.tight_layout()
    plt.show()
    
    fig.savefig(os.path.join(path.figures_storage, 'positionDistributions.png'), bbox_inches = 'tight', dpi=params.dpi)
    
    return

def plot_single_param_effect_together(sweep_results, par, par_name, figsize, min_err, max_err, min_per, max_per):
    colors = [tuple(col/255) for col in fig3_params.dirColors]

    fig, ax0 = plt.subplots(figsize=figsize)
    color = colors[0]
    ax0.set_xlabel(par_name, fontsize=params.axis_fontsize)
    ax0 = sns.lineplot(data = sweep_results, x = par, 
                       y = 'median_err', err_style='bars',
                       color = color, ci = 95)

    ax0.set_xticks(np.unique(sweep_results[par]))
    ax0.set_yticks(fig2_params.err_ticks)

    ax0.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
    ax0.tick_params(axis='y', length = params.tick_length, width = params.tick_width, direction='in')
    ax0.tick_params(axis='x', length = params.tick_length, width = params.tick_width, direction='out')

    ax1 = ax0.twinx()
    color = colors[1]
    ax1 = sns.lineplot(data = sweep_results, x = par, 
                       y = 'trackPercent', err_style='bars',
                       color = color, ci = 95)
    
    ax1.set_yticks(fig2_params.per_ticks)
    ax1.tick_params(axis='y', length = params.tick_length, width = params.tick_width, direction='in')
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

    if par == 'score_threshold':
        ax0.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        ax0.set_ylabel('Median Error (cm)', fontsize=params.axis_fontsize, color=colors[0])
        ax1.set_ylabel('Percent frames tracked', fontsize=params.axis_fontsize, color=colors[1])
    elif par == 'n_back':
        ax0.set_ylabel('')
        ax1.set_ylabel('')
    else:
        ax0.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        ax0.set_ylabel('')
        ax1.set_ylabel('')
        ax0.axes.yaxis.set_ticklabels([])
        ax1.axes.yaxis.set_ticklabels([])
        
    ax0.set_ylim(min_err, max_err)    
    ax1.set_ylim(min_per, max_per)

    ax0.spines[['bottom', 'left']].set_linewidth(params.axis_linewidth)
    ax1.spines['right'].set_linewidth(params.axis_linewidth)
    
    sns.despine(top=True, right=False, ax=ax0)    
    sns.despine(top=True, right=False, ax=ax1)
    
    plt.show()    
    fig.savefig(os.path.join(path.figures_storage, 'percentTracked_and_medianError_' + par + '.png'), bbox_inches = 'tight', dpi=params.dpi)
    


def plot_parameter_effects_by_single_parameter(sweep_results):
    columns = [col for col in sweep_results.columns if col in ['scale_smooth', 'scale_length', 'score_threshold', 'n_back', 'reproj_error_threshold', 'offset_threshold', 'post_reproj_threshold']]
    for par, par_name in zip(columns, params.anipose_param_names):
        plot_single_param_effect_together(sweep_results, par, par_name, fig2_params.single_param_figsize,
                                          fig2_params.err_range[0], fig2_params.err_range[1], 
                                          fig2_params.percent_range[0], fig2_params.percent_range[1])
            
def interaction_error_heatmap(sweep_results, par0, par1, name0, name1, min_err, max_err, cmap): 
    sweep_tmp = sweep_results.pivot_table(index=par0, columns=par1, values='median_err', aggfunc=np.mean)
    fig, ax0 = plt.subplots(figsize=figS2_params.heatmap_figSize)
    ax0 = sns.heatmap(sweep_tmp.sort_index(ascending=False), vmin=min_err, 
                      vmax=max_err, square = True, cmap=cmap,
                      cbar_kws={'label' : 'Median Error (cm)',
                                'format': '%.2f',
                                'ticks' : [min_err, max_err, min_err + (max_err-min_err)/3, min_err + (max_err-min_err)*2/3]})
    ax0.set_xlabel(name1, fontsize=params.axis_fontsize)
    ax0.set_ylabel(name0, fontsize=params.axis_fontsize)
    
    ax0.tick_params(length = params.tick_length, width = params.tick_width, direction='out')
    if par0 == 'score_threshold':
        ax0.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        ax0.set_yticks(np.arange(0.5, sweep_tmp.shape[0], 1))
        ax0.set_yticklabels(np.unique(sweep_results[par0]).astype(np.float16)[::-1])
    else:
        ax0.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        ax0.set_yticks(np.arange(0.5, sweep_tmp.shape[0], 1))
        ax0.set_yticklabels(np.unique(sweep_results[par0]).astype(np.int16)[::-1])
    if par1 == 'score_threshold':
        ax0.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        ax0.set_xticks(np.arange(0.5, sweep_tmp.shape[1], 1))
        ax0.set_xticklabels(np.unique(sweep_results[par1]).astype(np.float16))
    else:
        ax0.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))        
        ax0.set_xticks(np.arange(0.5, sweep_tmp.shape[1], 1))
        ax0.set_xticklabels(np.unique(sweep_results[par1]).astype(np.int16)) 
    
    plt.show()     
    fig.savefig(os.path.join(path.figures_storage, 'interactions_medianError_vs_' + par0 + '_' + par1 + '.png'), bbox_inches = 'tight', dpi=params.dpi)
    return

def interaction_percent_heatmap(sweep_results, par0, par1, name0, name1, min_percent, max_percent, cmap): 
    sweep_tmp = sweep_results.pivot_table(index=par0, columns=par1, values='trackPercent', aggfunc=np.mean)
    fig, ax0 = plt.subplots(figsize=figS2_params.heatmap_figSize)   
    ax0 = sns.heatmap(sweep_tmp.sort_index(ascending=False), vmin=min_percent, 
                      vmax=max_percent, square = True, cmap=cmap,
                      cbar_kws={'label' : 'Percent Motion Tracked',
                                'format': '%.2f',
                                'ticks' : [min_percent, max_percent, min_percent + (max_percent-min_percent)/3, min_percent + (max_percent-min_percent)*2/3]})

    ax0.set_xlabel(name1, fontsize=params.axis_fontsize)
    ax0.set_ylabel(name0, fontsize=params.axis_fontsize)
    
    ax0.tick_params(length = params.tick_length, width = params.tick_width, direction='out')
    if par0 == 'score_threshold':
        ax0.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        ax0.set_yticks(np.arange(0.5, sweep_tmp.shape[0], 1))
        ax0.set_yticklabels(np.unique(sweep_results[par0]).astype(np.float16)[::-1])
    else:
        ax0.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
        ax0.set_yticks(np.arange(0.5, sweep_tmp.shape[0], 1))
        ax0.set_yticklabels(np.unique(sweep_results[par0]).astype(np.int16)[::-1])
    if par1 == 'score_threshold':
        ax0.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        ax0.set_xticks(np.arange(0.5, sweep_tmp.shape[1], 1))
        ax0.set_xticklabels(np.unique(sweep_results[par1]).astype(np.float16))
    else:
        ax0.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))        
        ax0.set_xticks(np.arange(0.5, sweep_tmp.shape[1], 1))
        ax0.set_xticklabels(np.unique(sweep_results[par1]).astype(np.int16))  
    
    plt.show()   
    fig.savefig(os.path.join(path.figures_storage, 'interactions_percentTracked_vs_' + par0 + '_' + par1 + '.png'), bbox_inches = 'tight', dpi=params.dpi)

    return

def plot_parameter_effects_by_heatmaps(sweep_results):    
    only_significant = True
    significant_error_interactions   = [('scale_smooth', 'scale_length'),
                                        ('scale_smooth', 'score_threshold'), 
                                        ('scale_length','post_reproj_threshold'),
                                        ('score_threshold', 'post_reproj_threshold'),
                                        ('scale_length', 'reproj_error_threshold'),
                                        ('scale_length', 'score_threshold')]
    significant_percent_interactions = [('scale_length','post_reproj_threshold'),
                                        ('score_threshold', 'post_reproj_threshold'),
                                        ('scale_smooth', 'post_reproj_threshold'),
                                        ('offset_threshold', 'n_back')]

    cmap = 'viridis'
    

    # min_percent = sweep_results['trackPercent'].min()
    # max_percent = sweep_results['trackPercent'].max()
    min_percent = 0.92
    max_percent = 0.8
    
    for p0, (par0, name0) in enumerate(zip(sweep_results.columns[:7], params.anipose_param_names)): 
        for p1, (par1, name1) in enumerate(zip(sweep_results.columns[:7], params.anipose_param_names)):
            if only_significant:
                if (par0, par1) in significant_error_interactions:
                    if (par0, par1) == ('scale_smooth', 'scale_length'):
                        min_err = 0.2275
                        max_err = 0.24
                    else:
                        min_err     = sweep_results['median_err'].min()
                        max_err     = sweep_results['median_err'].max()
                    interaction_error_heatmap(sweep_results, par0, par1, name0, name1, min_err, max_err, cmap)
            else:
                if p1 > p0:
                    interaction_error_heatmap(sweep_results, par0, par1, min_err, max_err, cmap)

    for p0, (par0, name0) in enumerate(zip(sweep_results.columns[:7], params.anipose_param_names)): 
        for p1, (par1, name1) in enumerate(zip(sweep_results.columns[:7], params.anipose_param_names)):
            if only_significant:
                if (par0, par1) in significant_percent_interactions:
                    interaction_percent_heatmap(sweep_results, par0, par1, name0, name1, min_percent, max_percent, cmap)
            else:
                if p1 > p0:
                    interaction_percent_heatmap(sweep_results, par0, par1, min_percent, max_percent, cmap)

def plot_parameter_selection_aids(scoreThresh_cut, sweep_results):
    if scoreThresh_cut == 0.45:
        figPath = os.path.join(path.figures_storage, 'percentTracked_medianError_sortBy_scoreThresh_lessOrEqual45_reprojPost_nback.png')
        figSize = fig2_params.sorted_combinations_figSize
    elif scoreThresh_cut == 0.6:
        figPath = os.path.join(path.figures_storage, 'percentTracked_medianError_sortBy_scoreThresh_lessOrEqual60_reprojPost_nback.png')
        figSize = figS2_params.sorted_combinations_figSize
    
    sweep_results = sweep_results.loc[(sweep_results.scale_length == 2) & 
                                      (sweep_results.scale_smooth == 6) & 
                                      (sweep_results.score_threshold <= scoreThresh_cut), :]
    
    sweep_results = sweep_results.sort_values(['score_threshold', 'post_reproj_threshold', 'n_back'], ignore_index=True)
    sweep_results['idx'] = sweep_results.index
    
    colors = [tuple(col/255) for col in fig3_params.dirColors]

    fig, ax0 = plt.subplots(figsize=figSize)
    color = colors[0]
    ax0.set_xlabel('Sorted Parameter Combination', fontsize=params.axis_fontsize)
    ax0.set_ylabel('Median Error (cm)', fontsize=params.axis_fontsize, color=color)
    ax0 = sns.lineplot(data = sweep_results, x='idx', y='median_err', lw = fig2_params.sorted_combinations_lw,
                        linestyle = '-', marker = 'o', markersize = fig2_params.sorted_combinations_markerSize, 
                        markeredgewidth = fig2_params.sorted_combinations_markerEdgeWidth, 
                        err_style='band', ci=None, color = color)
    ax1 = ax0.twinx()
    color = colors[1]
    ax1.set_ylabel('Percent frames tracked', fontsize=params.axis_fontsize, color=color)
    ax1 = sns.lineplot(data = sweep_results, x='idx', y='trackPercent', lw = fig2_params.sorted_combinations_lw, 
                        linestyle = '-', marker = 'o', markersize = fig2_params.sorted_combinations_markerSize, 
                        markeredgewidth = fig2_params.sorted_combinations_markerEdgeWidth,
                        err_style='band', ci=None, color = color)    
    
    ax0.set_ylim(np.min(sweep_results.median_err) - .0015, np.max(sweep_results.median_err) + .0015)    
    ax1.set_ylim(np.min(sweep_results.trackPercent) - .01, np.max(sweep_results.trackPercent) + .01)

    ax0.set_yticks(fig2_params.sorted_err_ticks)
    ax1.set_yticks(fig2_params.sorted_per_ticks)

    ax0.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%d'))
    ax0.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

    ax0.tick_params(axis='y', length = params.tick_length, width = params.tick_width, direction='in')
    ax0.tick_params(axis='x', length = params.tick_length, width = params.tick_width, direction='out')
    ax1.tick_params(axis='y', length = params.tick_length, width = params.tick_width, direction='in')

    sns.despine(top=True, right=False, ax=ax0)    
    sns.despine(top=True, right=False, ax=ax1)
    
    plt.show()    
    fig.savefig(figPath, bbox_inches = 'tight', dpi=params.dpi)
    
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
    
    os.makedirs(path.figures_storage, exist_ok=True)
    
    if load_type == 'pickle':
        with open(path.processed_results_path, 'rb') as fp:
            trajectoryData, posErrResults, trackingQuality, allFramesError, dlc_bases = dill.load(fp) 
        with open(path.training_set_size_posErr, 'rb') as fp:
            trainingset_results = dill.load(fp)
        with open(path.labeling_error_results, 'rb') as fp:
            no_epipolar_lines_pixel_error, human_error, all_human_errors, dlc_pixel_error, all_dlc_pixel_errors = dill.load(fp)
        with open(path.full_sweep_results, 'rb') as fp:
            sweep_results = dill.load(fp)
            
    elif load_type == 'hdf':
        names_list, data_list = load_hdf_to_original(path.processed_results_path)
        for name, data in zip(names_list, data_list):
            exec(name + ' = data')            
        names_list, data_list = load_hdf_to_original(path.training_set_size_posErr)
        for name, data in zip(names_list, data_list):
            exec(name + ' = data')
        names_list, data_list = load_hdf_to_original(path.labeling_error_results)
        for name, data in zip(names_list, data_list):
            exec(name + ' = data')
        names_list, data_list = load_hdf_to_original(path.full_sweep_results)
        for name, data in zip(names_list, data_list):
            exec(name + ' = data')

    # to recreate panels A-D in fig3, use these combos of (eventNum, part): good = (16, 0), bad = (5, 5) and (7, 1)
    for eventNum, part in zip([16, 5, 7], [0, 5, 1]):
        eventNums = [eventNum] 
        parts = [part] #[0] #'all' # can be a list of integers, or 'all'
        figSet = 0
        for eventNum in eventNums:
            plotTrajectories(eventNum, parts, fig3_params.dirColors, figSet, pos=True, vel=False, combined=False)
            
    plot_learning_progress()
    plot_training_size_effect(all_human_errors, trainingset_results)
    fig1_project_basis_on_images(originShift = fig1_params.originShift)
    plot_parameter_effects_by_single_parameter(sweep_results)
    plot_parameter_effects_by_heatmaps(sweep_results)
    plot_parameter_selection_aids(0.45, sweep_results)
    plot_parameter_selection_aids(0.6, sweep_results)
    errors_histogram()
