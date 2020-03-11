#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 10:55:49 2018

@author: truongp
"""

from .metrics_detector import compute_repeatability
from .metrics_descriptor import extract_descriptor_metrics
import numpy as np
from utils.metrics_descriptor import keep_shared_keypoints
import tqdm


def compute_metrics_results(img1, img2, method, data, real_H, kwarg):
    '''
    extracts both detector and descriptor metrics for a pair of images img1 and img2, 
    img1 being the reference image, for a particular detector/descriptor named in 'method'.
    arguments: 
                img1 - grayscale reference image
                img2 - grayscale transformed image
                data - dictionnary containing keypoints and descriptors of both images 
                method - string name of the detector/descriptor evaluated
                real_H - ground truth homography relating the pair of images
                kward - info given by user
    outputs:
                dictionnary with keys:
                    - repeatability - repeatability between kp1 and kp2
                    - curve_1_precision - list presenting the value of 1-precision 
                    for ratio threshold [0.4,0.45,..,0.95] used for matching descriptor
                    - curve_recall - list presenting the value of recall 
                    for ratio threshold [0.4,0.45,..,0.95] used for matching descriptor
                    - curve_tpr - list presenting true positive rate for ratio threshold 
                    [0.4,0.45,..,0.95] used for matching descriptor 
                    - curve_fpr - list presenting false positive rate for ratio threshold 
                    [0.4,0.45,..,0.95] used for matching descriptor 
                    - area - area under the curve of true positive rate versus false positive rate 
                    - M_score - value of the M.score calculated for a reatio threshold of 0.8
                    - homography - computed homography supposedly relating both images
                    - found_homography - 1 is a homography was found (enough matches), 0 otherwise
                    - acceptable_homography - 1 if the homography respected the criteria
                    for the registration to be acceptable
                    - inlier_ratio - inlier ratio of the homography found using RANSAC, equal
                    to nan if no homography was found
                    - RMSE - value of RMSE calculated based on the homography found at 
                    ratio threshold of 0.8
                    - MEE - value of MEE calculated based on the homography found at 
                    ratio threshold of 0.8
                    - coverage_ratio - coverage fraction of the true positive keypoint of image1
                    run_time: a dictionnary containing
                    - nbr_kp_per_pair - sum of kp1 and kp2 for this image pair
                
                run_time - dictionary containing
                     - run_time['matching'] - time in seconds of bf matching
                     - run_time['homography_estimation'] - time in seconds for homogrpahy estimation
                     using RANSAC
        '''

    kp1_shared=np.float32(data['kp1'])
    kp2_shared=np.float32(data['kp2'])

    repeatability=compute_repeatability(kp1_shared, kp2_shared, real_H, img1.shape, \
                                        distance_thresh=kwarg['distance_threshold'])
    
    homography, curve_1_precision, curve_recall, area, curve_tpr, curve_fpr, M_score, \
    RMSE, MEE, found_homography, acceptable_homography, coverage_ratio, inlier_ratio, run_time\
    =extract_descriptor_metrics(img1, img2, data, method, real_H,
    distance_threshold=kwarg['distance_threshold'], plot=kwarg['plot'],
    verbose=kwarg['verbose'])
    
    if kwarg['verbose']==True:
        print('Precision and recall for a ratio threshold of 0.8 are {} and {} and repeatability is \
{}'.format(float(1-curve_1_precision[8]), curve_recall[8], repeatability))

    return {'repeatability':repeatability, 'curve_1_precision': curve_1_precision,
            'curve_recall':curve_recall,  'area':area,
            'curve_tpr':curve_tpr, 'curve_fpr':curve_fpr, 'M_score':M_score,
            'homography':homography.tolist(), 'found_homography':found_homography,
            'acceptable_homography':acceptable_homography, 'inlier_ratio': inlier_ratio,
            'RMSE':RMSE, 'MEE':MEE,  'coverage_ratio':coverage_ratio, 'nbr_kp_per_pair':len(kp1_shared)+len(kp2_shared)}, run_time


def compute_metrics_results_null(kp1, kp2):
    '''
    For the case of insufficient number of keypoints for either img1 or img2 (<4), 
    computes all the metrics results. 
    arguments: 
                kp1 - keypoints of image1, shape N1 x2
                kp2 - keypoints of image2, shape N2 x2
    outputs:
                dictionnary with keys:
                    - repeatability - equal to 0
                    - curve_1_precision - list of 0
                    - curve_recall - list of 0
                    - curve_tpr - list of 0
                    - curve_fpr - list of 0
                    - area - equal to 0
                    - M_score - equal to 0
                    - homography - equal to 3x3 null matricx
                    - found_homography - equal to 0
                    - acceptable_homography - equal to 0
                    - inlier_ratio - equal to nan
                    - RMSE - equal to nan
                    - MEE - equal to nan
                    - coverage_ratio - equal to 0
                    - nbr_kp_per_pair - sum of kp1 and kp2 for this image pair
                
                run_time - dictionary containing
                     - run_time['matching'] - time in seconds of bf matching
                     - run_time['homography_estimation'] - time in seconds for homogrpahy estimation
                     using RANSAC
        '''
    run_time={'matching':float('nan'), 'homography_estimation':float('nan')}
    return {'repeatability':0, 'curve_1_precision': np.ones((12)).tolist(), 'curve_recall': np.zeros((12)).tolist(),
            'area':0, 'curve_tpr':np.zeros((12)).tolist(), 'curve_fpr':np.zeros((12)).tolist(),
            'M_score': 0, 'homography':np.zeros((3,3)).tolist(), 'found_homography':0, 'acceptable_homography':0,
            'inlier_ratio': float('nan'),'RMSE':float('nan'), 'MEE':float('nan'), 'coverage_ratio':0,
            'nbr_kp_per_pair':len(kp1)+len(kp2)}, run_time


def metrics_registration(MEE, RMSE, homography, acceptable_homography, inlier_ratio):
    '''
    computes the info on the registration quality
    :param MEE:
    :param RMSE:
    :param homography:
    :param acceptable_homography:
    :param inlier_ratio:
    :return:
    '''

    MEE_acceptable = []
    RMSE_acceptable = []
    MEE_inaccurate = []
    RMSE_inaccurate = []
    nbr_homography = np.sum(homography, dtype=np.float64)

    nbr_acceptable_homography = np.sum(acceptable_homography, dtype=np.float64)

    for i in range(len(MEE)):
        if (homography[i] == 1 and acceptable_homography[i] == 1):
            MEE_acceptable.append(MEE[i])
            RMSE_acceptable.append(RMSE[i])
        elif (homography[i] == 1 and acceptable_homography[i] == 0):
            MEE_inaccurate.append(MEE[i])
            RMSE_inaccurate.append(RMSE[i])

    assert (len(MEE_acceptable) == nbr_acceptable_homography)
    assert ((len(MEE_acceptable) + len(MEE_inaccurate)) == nbr_homography)

    mean_MEE_acceptable = np.mean(MEE_acceptable)
    std_MEE_acceptable = np.std(MEE_acceptable)
    mean_RMSE_acceptable = np.mean(RMSE_acceptable)
    std_RMSE_acceptale = np.std(RMSE_acceptable)

    mean_MEE_inaccurate = np.mean(MEE_inaccurate)
    std_MEE_inaccurate = np.std(MEE_inaccurate)
    mean_RMSE_inaccurate = np.mean(RMSE_inaccurate)
    std_RMSE_inaccurate = np.std(RMSE_inaccurate)

    mean_MEE_global = np.nansum(MEE, dtype=np.float64) / nbr_homography if nbr_homography != 0 else 0
    std_MEE_global = np.nanstd(MEE)
    mean_RMSE_global = np.nansum(RMSE, dtype=np.float64) / nbr_homography if nbr_homography != 0 else 0
    std_RMSE_global = np.nanstd(RMSE)
    mean_inlier_ratio = np.nansum(inlier_ratio, dtype=np.float64) / nbr_homography if nbr_homography != 0 else 0

    registration = {'mean_MEE_acceptable': mean_MEE_acceptable, 'std_MEE_acceptable': std_MEE_acceptable,
                    'mean_RMSE_acceptable': mean_RMSE_acceptable, 'std_RMSE_acceptable': std_RMSE_acceptale,
                    'mean_MEE_inaccurate': mean_MEE_inaccurate, 'std_MEE_inaccurate': std_MEE_inaccurate,
                    'mean_RMSE_inaccurate': mean_RMSE_inaccurate, 'std_RMSE_inaccurate': std_RMSE_inaccurate,
                    'mean_MEE_global': mean_MEE_global, 'std_MEE_global': std_MEE_global,
                    'mean_RMSE_global': mean_RMSE_global, 'std_RMSE_global': std_RMSE_global,
                    'nbr_homography': nbr_homography, 'nbr_acceptable_homography': nbr_acceptable_homography,
                    'mean_inlier_ratio': mean_inlier_ratio}

    return registration


def evaluate(test_dataloader, model,  args, name_method='GLAMpointsInference'):
    '''
    Computes the average metrics on all pair of images
    arguments:
        test_dataloader is the dataset
        model is the trained model
        name_method
    output:
        dictionnary containing the average metrics with keys the name of the descriptors
        average_results={'SIFT'=output, 'SURF'=output ....}
        output={'mean_area', 'variance_area', 'mean_coverage_fraction', 'variance_coverage_fraction'
                'mean_M_score', 'variance_M_score','mean_repeatability','variance_repeatability', \
                'mean_curve_recall', 'variance_curve_recall', 'mean_curve_1_precision',
                'variance_curve_1_precision','mean_precision_at_thresh_0_8',
                'mean_recall_at_thresh_0_8','variance_precision_at_thresh_0_8',\
                'variance_recall_at_thresh_0_8','mean_curve_tpr', 'mean_curve_fpr', \
                'registration', 'MEE', 'RMSE', 'homography',\
                'acceptable_homography', 'nbr_images', \
                'inlier_ratio', 'mean_nbr_kp'}
    '''
    # will calculate all the averages
    average_results = {}
    area = []
    repeatability = []
    curve_1_precision = []
    curve_recall = []
    curve_tpr = []
    curve_fpr = []
    M_score = []
    MEE = []
    RMSE = []
    homography = []
    acceptable_homography = []
    coverage_fraction = []
    inlier_ratio = []
    nbr_kp = []
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i_batch, mini_batch in pbar:
        # calculate metrics for each image pair
        img1 = mini_batch['image1']
        img2 = mini_batch['image2']
        real_H = mini_batch['H_gt_1_to_2']

        kp1, des1 = model.find_and_describe_keypoints(img1)
        kp2, des2 = model.find_and_describe_keypoints(img2)

        if kp1.shape[0] < 3 or kp2.shape[0] < 3:
            metrics_results, run_time=compute_metrics_results_null(kp1, kp2)
            # no keypoints are detected
        else:
            # we are working with kp that are visible in the shared region only
            kp1_shared, des1_shared = keep_shared_keypoints(kp1, des1, real_H, img1.shape, verbose=args['verbose'])
            kp2_shared, des2_shared = keep_shared_keypoints(kp2, des2, np.linalg.inv(real_H), img1.shape, verbose=args['verbose'])
            print('there are {} keypoints in the shared view for {}'.format(kp1_shared.shape[0], name_method))
            print('there are {} keypoints in the shared view for {}'.format(kp2_shared.shape[0], name_method))
            if kp1_shared.shape[0]<3 or kp2_shared.shape[0]<3:
                metrics_results, run_time=compute_metrics_results_null(kp1_shared, kp2_shared)
            else:
                keypoints={'kp1':kp1_shared, 'des1':des1_shared,
                           'kp2':kp2_shared, 'des2':des2_shared}
                metrics_results, run_time = compute_metrics_results(img1, img2, name_method, keypoints, real_H)
        '''metrics_results = {'repeatability', 'curve_1_precision', 'curve_recall', 'homography', 'area', \
                           'curve_tpr', 'curve_fpr', 'M_score', \
                           'RMSE', 'MEE', 'found_homography', 'acceptable_homography', \
                           'coverage_ratio', 'inlier_ratio', 'nbr_kp_per_pair'}'''

        area.append( metrics_results['{}'.format(name_method)]['metrics_results']['area'])
        repeatability.append(metrics_results['{}'.format(name_method)]['metrics_results']['repeatability'])
        curve_1_precision.append( metrics_results['{}'.format(name_method)]['metrics_results']['curve_1_precision'] )
        curve_recall.append(metrics_results['{}'.format(name_method)]['metrics_results']['curve_recall'] )
        curve_tpr.append(metrics_results['{}'.format(name_method)]['metrics_results']['curve_tpr'] )
        curve_fpr.append(metrics_results['{}'.format(name_method)]['metrics_results']['curve_fpr'] )
        M_score.append(metrics_results['{}'.format(name_method)]['metrics_results']['M_score'])
        homography.append(metrics_results['{}'.format(name_method)]['metrics_results']['found_homography'])
        acceptable_homography.append(metrics_results['{}'.format(name_method)]['metrics_results']['acceptable_homography'])
        MEE.append(metrics_results['{}'.format(name_method)]['metrics_results']['MEE'])
        RMSE.append(metrics_results['{}'.format(name_method)]['metrics_results']['RMSE'])
        coverage_fraction.append(metrics_results['{}'.format(name_method)]['metrics_results']['coverage_ratio'])
        inlier_ratio.append(metrics_results['{}'.format(name_method)]['metrics_results']['inlier_ratio'])
        nbr_kp.append(metrics_results['{}'.format(name_method)]['metrics_results']['nbr_kp_per_pair']/2)

    # calculates the average over the whole dataset
    mean_repeatability = np.mean(repeatability)
    variance_repeatability = np.var(repeatability)
    mean_coverage_fraction = np.mean(coverage_fraction)
    variance_coverage_fraction = np.var(coverage_fraction)
    mean_nbr_kp = np.mean(nbr_kp)

    mean_area = np.mean(area)
    variance_area = np.var(area)
    mean_M_score = np.mean(M_score)
    variance_M_score = np.var(M_score)

    curve_recall = np.float32(curve_recall)
    curve_1_precision = np.float32(curve_1_precision)
    curve_tpr = np.float32(curve_tpr)
    curve_fpr = np.float32(curve_fpr)

    mean_curve_recall = np.mean(curve_recall, axis=0, dtype=np.float64)
    variance_curve_recall = np.var(curve_recall, axis=0, dtype=np.float64)
    mean_curve_1_precision = np.mean(curve_1_precision, axis=0, dtype=np.float64)
    variance_curve_1_precision = np.var(curve_1_precision, axis=0, dtype=np.float64)
    mean_curve_tpr = np.mean(curve_tpr, axis=0, dtype=np.float64)
    mean_curve_fpr = np.mean(curve_fpr, axis=0, dtype=np.float64)

    mean_precision_at_thresh_0_8 = (1 - mean_curve_1_precision[8])
    variance_precision_at_thresh_0_8 = variance_curve_1_precision[8]
    variance_recall_at_thresh_0_8 = variance_curve_recall[8]
    mean_recall_at_thresh_0_8 = mean_curve_recall[8]

    registration = metrics_registration(MEE, RMSE, homography, acceptable_homography, inlier_ratio)

    output = {'mean_area': mean_area, 'variance_area': variance_area, \
              'mean_coverage_fraction': mean_coverage_fraction, 'variance_coverage_fraction': \
                  variance_coverage_fraction,
              'mean_M_score': mean_M_score, 'variance_M_score': variance_M_score, \
              'mean_repeatability': mean_repeatability, 'variance_repeatability': variance_repeatability, \
              'mean_curve_recall': mean_curve_recall.tolist(), \
              'variance_curve_recall': variance_curve_recall.tolist(), 'mean_curve_1_precision': \
                  mean_curve_1_precision.tolist(),
              'variance_curve_1_precision': variance_curve_1_precision.tolist(), \
              'mean_precision_at_thresh_0_8': mean_precision_at_thresh_0_8,
              'mean_recall_at_thresh_0_8': mean_recall_at_thresh_0_8, \
              'variance_precision_at_thresh_0_8': variance_precision_at_thresh_0_8, \
              'variance_recall_at_thresh_0_8': variance_recall_at_thresh_0_8, \
              'mean_curve_tpr': mean_curve_tpr.tolist(), 'mean_curve_fpr': mean_curve_fpr.tolist(), \
              'registration': registration, 'nbr_images': len(area),  'mean_nbr_kp': mean_nbr_kp}
    average_results['{}'.format(name_method)] = output
    return average_results
