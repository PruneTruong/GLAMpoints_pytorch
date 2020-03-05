#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 18:32:07 2018

@author: truongp
"""
import cv2
import numpy as np
from utils.plot import draw_matches, draw_keypoints
from matplotlib import pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from math import sqrt
import time


def keep_shared_keypoints(kp, des, real_H, shape, verbose=False):
    """
    Compute a list of keypoints and their corresponding detectors by keeping 
    only the points that once mapped by H are still inside the shape of the image.
    arguments:  kp: array of keypoint locations of shape Nx2. the first coordinate 
                the location in the horizontal direction.
                des: array of corresponding descriptors
                real_H: ground truth homography between two images. 3x3
                shape: height and width of image
    
    output:
                kp_shared: list of keypoint locations in shared viewpoints.
                Nx2 first coordinate is the horizontal direction
                des_shared: corresponding descriptors
    """ 
    def warp_keypoints(kp, real_H):
        '''
        calculates the transformed coordonates of an array of points according to a 
        homography transformation.
        arguments:
                    kp: list of keypoint locations Nx2, first coordinate is horizontal.
                    real_H: ground thruth homography
        outputs: warped_points: locations of warped kp, Nx2, first coordinate is also horizontal.
        '''
        
        num_points = kp.shape[0]
        homogeneous_points = np.concatenate([kp, np.ones((num_points, 1))],axis=1)
        warped_points = np.dot(real_H, np.transpose(homogeneous_points))
        warped_points=np.transpose(warped_points) #shape number_of_keypointsx3
        warped_points=warped_points[:, :2] / warped_points[:, 2:]
        #it was divided so that third coordinate is 1. shape number_keypointsx2
        return warped_points
    
    warped_points = warp_keypoints(kp, real_H) #first coordinate horizontal
    kp_shared=[]
    des_shared=[]
    for i in range(warped_points.shape[0]):
        if ( (warped_points[i, 0] >= 0) & (warped_points[i, 0] < shape[1]) &\
               (warped_points[i, 1] >= 0) & (warped_points[i, 1] < shape[0])  ):
            kp_shared.append(kp[i])
            des_shared.append(des[i])
    
    kp_shared=np.array(kp_shared)
    des_shared=np.array(des_shared)

    if verbose:
        print('there are {} keypoints in the shared view'.format(kp_shared.shape[0]))
    return kp_shared, des_shared


def compute_gt_matches(kp1, kp2, real_H, distance_thresh=None):
    """
    Compute the gt matches.
    arguments:
                kp1 N1x2
                kp2 N2x2
                real_H linking the 2 images
                distance_thresh: radius to consider a match as correct   
    outputs:
                nbr_possible_correct_matches, nbr of kp1 that warped are within a radius
                equal to distance_thresh of a kp2. 
                nbr_possible_incorrect_matches from image1 to image2
    """
                
    def warp_keypoints(kp, real_H):
        '''
        calculates the transformed coordonates of an array of points according to a 
        homography transformation.
        arguments:
                    kp: list of keypoint locations Nx2, first coordinate is horizontal.
                    real_H: ground thruth homography
        outputs: warped_points: locations of warped kp, Nx2, first coordinate is also horizontal.
        '''
        
        num_points = kp.shape[0]
        homogeneous_points = np.concatenate([kp, np.ones((num_points, 1))],axis=1)
        warped_points = np.dot(real_H, np.transpose(homogeneous_points))
        warped_points=np.transpose(warped_points) #shape number_of_keypointsx3
        warped_points=warped_points[:, :2] / warped_points[:, 2:]
        return warped_points


    if distance_thresh is None:
        distance_thresh=3

    true_warped_keypoints = warp_keypoints(np.float32(kp1), real_H)
    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1) #N1x1x2
    warped_keypoints = np.expand_dims(np.float32(kp2), 0) #1xN2x2
    # shapes are broadcasted to N1 x N2 x 2:
    norm = np.linalg.norm(true_warped_keypoints - warped_keypoints, ord=None, axis=2) #N1xN2
    min1 = np.min(norm, axis=1)

    count1 = np.sum(min1 <= distance_thresh)

    nbr_possible_correct_matches = count1

    nbr_possible_incorrect_matches = kp1.shape[0]-nbr_possible_correct_matches

    return nbr_possible_correct_matches, nbr_possible_incorrect_matches


def compute_homography(img1, img2, data, method, ratio_threshold, min_nb_pts=None, verbose=False):  
    '''
    goal: A brute force matcher is used that finds for each descriptor of the querry image the two closest neighbors 
    descriptors of the training image. A ratio is than applied to the matches to remove falses matches. 
    The homography matrix between the two images is found using RANSAC if there are enough corresponding points between 
    the two images.
    inputs: img1- first image (query image)
            img2- second image (source image)
            data - dict containing:
                kp1- list of extracted keypoints from image 1.
                des1- corresponding descriptor for image 1
                kp2- list of extractd keypoints from image 2
                des2- corresponding descriptors for image 2
                threshold_ratio- value of the threshold to apply to the ratio of the descriptor distance of the two closest neihbors
            min_nb_pts- minimum number of matches between to images to extract the homography matrice using RANSAC
            
    
    outputs: output: dictionnary containing
                - output['matches'] - Dstructure of remaining matches after RANSAC, giving indexes
                of corresponding keypoints on both images. can be accesses through
                kp1[matches.querryIdx] and kp2[matches.trainIdx]
                - output['homography'] - extracted homography after RANSAC, if not enough
                matches are found, it is equal to a 3x3 nul matrix
                - output['inlier_ratio'] - inlier ratio of RANSAC, if no homography is found, 
                it is equal to nan. 
             run_time: a dictionnary containing
                 - run_time['matching'] time in seconds of bf matching
                 - run_time['homography_estimation'] time in seconds for homogrpahy estimation
                 using RANSAC
             h - extracted homography matrice
             inlier_matches - 
    '''

    if min_nb_pts is None:
        min_nb_pts=4
    kp1=data['kp1']
    kp2=data['kp2']
    des1=data['des1']
    des2=data['des2']

    
    def unilateral_matching(des1, des2, method):
        if method == 'ORB' or method == 'FREAK' or method == 'BRISK':
            distance_metrics=cv2.NORM_HAMMING
        else:
            distance_metrics=cv2.NORM_L2
            
        bf = cv2.BFMatcher(distance_metrics)
        matches = bf.knnMatch(des1, des2, k=2)   
        good = [m for m,n in matches if (m.distance < ratio_threshold*n.distance)]
        return good
    
    def bilateral_matching(des1, des2, method):
        if method == 'ORB' or method == 'FREAK' or method == 'BRISK':
            distance_metrics=cv2.NORM_HAMMING
        else:
            distance_metrics=cv2.NORM_L2
        good=[]
        bf= cv2.BFMatcher(distance_metrics)
        matches_1 = bf.knnMatch(des1, des2, k=2)   
        good_1 = [m for m,n in matches_1 if (m.distance < ratio_threshold*n.distance)]
        
        matches_2 = bf.knnMatch(des2, des1, k=2)   
        good_2 = [m for m,n in matches_2 if (m.distance < ratio_threshold*n.distance)]
        for i in good_1:
            for j in good_2:
                if i.queryIdx==j.trainIdx and i.trainIdx==j.queryIdx:
                    good.append(i)
        return good
    
    t1=time.time()
    good=unilateral_matching(des1, des2, method)
    t2=time.time()
    #good=bilateral_matching(des1,des2, method)

    output={}
    if len(good)>= min_nb_pts:
        src_pts = np.float32([ kp1[m.queryIdx] for m in good ]).reshape(-1,1,2)
        # shape is number_good_matchesx1x2 (two coordinates of corresponding keypoint)
        dst_pts = np.float32([ kp2[m.trainIdx] for m in good ]).reshape(-1,1,2)       
        
        t3=time.time()
        H, inliers = cv2.findHomography(src_pts, dst_pts,cv2.RANSAC)
        t4=time.time()
        if H is None:
            H=np.zeros((3,3))
        inliers = inliers.ravel().tolist()
        '''
        this is a list specifying if each match is an inlier (1) or an outlier (0). len=number_matches
        inlier_matches=[good[i] for i in range(len(inliers)) if inliers[i] ]
        kp1_inliers = np.float32([ kp1[m.queryIdx] for m in inlier_matches]).reshape(-1,2)
        assert (kp1_inliers.shape[0]==np.sum(inliers))
        '''
        inlier_ratio=np.sum(inliers)/len(inliers)

        if verbose==True:
            print("with a ratio threshold of {}, there are {} matches found between \
the two images with {}".format(ratio_threshold, len(good), method))
            #print('after applying ransac, there are {} remaining matches '.format(len(inlier_matches)))
            
        output['matches']= good
        output['homography']= H
        output['inlier_ratio']=inlier_ratio
        run_time={'matching':t2-t1, 'homography_estimation':t4-t3}

    else:
        if verbose==True:
            print("with a ratio threshold of {}, there are {} matches found between \
the two images with {}".format(ratio_threshold, len(good), method))
            print('not enough matches to compute homography with {}'.format(method))

        output['matches']=good
        output['homography']=np.zeros((3,3))
        output['inlier_ratio']=float('nan')
        run_time={'matching':t2-t1, 'homography_estimation':float('nan')}

    return output, run_time
        #will have to change this


def true_positive_matches(kp1, kp2, matches, real_H, distance_threshold=None, verbose=False):
    ''' calculates number of true positive matches between 2 images given the 
    ground truth homography.
    arguments:
                kp1 array of keypoint locations of image1 (that are shared between the two images)
                kp2 array of keypoint locations of image2 (that are shared between the two images)
                matches Dstructure giving the index of corresponding keypoints in both images.
                matches[].queryIdx for image1 and matches[].trainIdx for image2.
                distance_threshold: distance between warped keypoints and true warped keypoint
                for which a match is considered to be true.
    output:
                tp: number of true positive matches 
                fp: number of false positive matches
                true_positive_matches: Dstructure giving the index of true corresponding keypoints.
                kp1_true_positive_matches: keypoitn location corresponding to the 
                true positive matches of image 1. shape tp x 2, horizontal coordonate first
    '''
    if distance_threshold is None:
        distance_threshold=3
        
    def warp_keypoint(kp, real_H):

        #kp must be horizontal, then vertical like usual 
        num_points = kp.shape[0]
        homogeneous_points = np.concatenate([kp, np.ones((num_points, 1))],axis=1)
        true_warped_points = np.dot(real_H, np.transpose(homogeneous_points))
        true_warped_points=np.transpose(true_warped_points)
        true_warped_points=true_warped_points[:, :2] / true_warped_points[:, 2:]
        return true_warped_points
    
    
    true_warped_keypoints = warp_keypoint(np.float32([ kp1[m.queryIdx] for m in matches ]).reshape(-1,2), real_H)
    # shape is number_good_matchesx2 (two coordinates of corresponding keypoint)
    warped_keypoints = np.float32([ kp2[m.trainIdx] for m in matches]).reshape(-1,2)
    norm=( np.linalg.norm(true_warped_keypoints-warped_keypoints, axis=1) <= distance_threshold )
    
    tp=np.sum(norm)
    fp=len(matches)-tp
    true_positive_matches=[matches[i] for i in range(norm.shape[0]) if (norm[i]==1)]
    kp1_true_positive_matches=np.float32([ kp1[m.queryIdx] for m in true_positive_matches]).reshape(-1,2)
    
    if verbose==True:
        print('the number of true positive matches is {}'.format(tp))
    return tp, fp, true_positive_matches, kp1_true_positive_matches


def descriptor_precision(tp, fp):
    return tp/(tp+fp) if tp != 0 else 0


def descriptor_recall(tp, nbr_possible_correct_matches):
    '''compute the recall
    arguments: 
                tp : number of true positive matches for a certain threshold after
                comparing to ground truth
                nbr_possible_correct_matches: number of same keypoints found in both images.
    output:     recall
    '''
    
    recall=tp/nbr_possible_correct_matches if nbr_possible_correct_matches!=0 else 0
    return recall


def compute_M_score(tp, nbr_kp1_shared):
    '''computes the M.score
    arguments:
                tp : number of true positive matches for a certain threshold after
                comparing to ground truth
                nbr_kp1_shared: number of keypoints extracted from image1
    output:     M.score
    '''
    if tp==0 or nbr_kp1_shared==0:
        return 0
    else:
        return tp/nbr_kp1_shared


def calculate_area(tpr, fpr):
    area=np.trapz(tpr, fpr)
    return area


def compute_registration_error(real_H, H, shape):
    '''
    computes RMSE, median error and maximum error between transformed points according
    to the ground truth and the computed homography. 
    inputs: real_H - ground truth homography
            H - computed homography
            shape - image shape
    outputs: RMSE root mean squared error
             MEE median error
             MAE max error
    '''
    if np.all(H==0):
        return float('nan'), float('nan'), float('nan')
    
    corners=np.array([[50, 50, 1], [50, shape[0] - 50, 1],
                      [shape[1] - 50, 50, 1], [shape[1] - 50, shape[0] - 50, 1],
                      [int(shape[1]/2), int(shape[0]/4),1], 
                      [int(shape[1]/2), int(3*shape[0]/4),1]])
    gt_warped_points = np.transpose(np.dot(real_H, np.transpose(corners)))
    gt_warped_points=gt_warped_points[:, :2] / gt_warped_points[:, 2:]
    
    warped_points = np.transpose(np.dot(H, np.transpose(corners)))
    warped_points=warped_points[:, :2] / warped_points[:, 2:]
    
    try:
        RMSE = sqrt(mean_squared_error(warped_points, gt_warped_points))
    except:
        RMSE=float('nan')

    # median error
    MEE = np.median( np.linalg.norm(gt_warped_points - warped_points, axis=1) )

    # max error
    MAE = np.max( np.linalg.norm(gt_warped_points - warped_points, axis=1) )

    return RMSE, MEE, MAE


def homography_is_accepted(H):
    '''criteria to decide if a homography is correct (not too squewed..)
    arguments: H - homography to consider
    output: True - homography is correct
            False - otherwise
    '''
    det=H[0,0]*H[1,1]-H[0,1]*H[1,0]
    if (det<0):
        return False
    N1=math.sqrt(H[0,0]*H[0,0]+H[1,0]*H[1,0])
    N2=math.sqrt(H[0,1]*H[0,1]+H[1,1]*H[1,1])
    if ((N1 > 4) or (N1 < 0.1)):
        return False
    if ((N2 > 4) or (N2 < 0.1)):
        return False
    return True


def class_homography(MEE, MAE):
    '''
    returns 1 if it is an acceptable homography, 0 otherwise. acceptable homography
    means MEE and MAE respect several criteria. 
    inputs: MEE - median error between transformed points according to the ground 
    truth and the computed homography. 
            MAE - maximum error
    outputs: found_homography - if a homography was found equal to 1, 0 otherwise
             acceptable_homography - if a homography was found, it can be acceptable or not. 
             equal to 1 if acceptable, 0 otherwise. 
    '''
    def is_acceptable_homography(MEE, MAE):
        if (MEE<10 and MAE<30):
            return 1
        else:
            return 0
        
        
    if math.isnan(MEE) is True:
        found_homography=0
        acceptable_homography=0
        #didn t find a homography
    else:
        found_homography=1
        acceptable_homography=is_acceptable_homography(MEE, MAE)
    return found_homography, acceptable_homography


def mask_coverage(img, kp):
    ''' calculates the coverage fraction by true positive key points of an image. each keypoint
    accounts for a circle of radius 25 on a mask of the image. 
    inputs:
        img 
        kp - keypoints to consider for coverage fraction, shape Nx2
    outpus:
        fraction - coverage fraction
    '''
    if (kp.shape[0]==0):
        return 0
    else:
        im=np.zeros(img.shape)
        for i in range(kp.shape[0]):
            image_coverage=cv2.circle(im, (np.uint(kp[i,0]),np.uint(kp[i,1])), 25, (255,255,255),thickness=-1)
            #white points
        image_coverage/=255
        image_coverage= np.uint8(image_coverage)
        covered_area=np.sum(image_coverage)
    
        fraction=covered_area/(img.shape[0]*img.shape[1])
    
        return fraction


def extract_descriptor_metrics(img1, img2, data, method, real_H, index_1, distance_threshold=None, plot=False, verbose=False):
    '''
    extracts all descriptor metrics for a pair of img1 and img2, img1 being the reference one. 
    arguments:  img1 - grayscale reference image
                img2 - grayscale transformed image
                data - dictionnary containing keypoints and descriptors of both images 
                method - string name of the detector/descriptor evaluated
                real_H - ground truth homography relating the pair of images
                index_1 - image number (only used to give information if plot or verbose is True)
                distance_threshold - optional, value for the radius used to evaluate 
                if an match is correct 
                plot - optional boolean to plot the results 
                verbose - optional boolean to write the results 
    outputs:
                homography - computed homography supposedly relating both images
                found_homography - 1 is a homography was found (enough matches), 0 otherwise
                acceptable_homography - 1 if the homography respected the criteria 
                for the registration to be acceptable
                output['inlier_ratio'] - inlier ratio of the homography found using RANSAC, equal
                to nan if no homography was found
                op_precision_curve - list presenting the value of 1-precision 
                for ratio threshold [0.4,0.45,..,0.95] used for matching descriptor
                recall_curve - list presenting the value of recall 
                for ratio threshold [0.4,0.45,..,0.95] used for matching descriptor
                tpr_curve - list presenting true positive rate for ratio threshold 
                [0.4,0.45,..,0.95] used for matching descriptor 
                fpr_curve - list presenting false positive rate for ratio threshold 
                [0.4,0.45,..,0.95] used for matching descriptor 
                area - area under the curve of true positive rate versus false positive rate 
                M_score - value of the M.score calculated for a reatio threshold of 0.8
                RMSE_thresh_0_8 - value of RMSE calculated based on the homography found at 
                ratio threshold of 0.8
                MEE_thresh_0_8 - value of MEE calculated based on the homography found at 
                ratio threshold of 0.8
                coverage_ratio - coverage fraction of the true positive keypoint of image1
                run_time: a dictionnary containing
                 - run_time['matching'] time in seconds of bf matching
                 - run_time['homography_estimation'] time in seconds for homogrpahy estimation
                 using RANSAC
    '''
                
    
    kp1_shared=np.float32(data['kp1'])
    kp2_shared=np.float32(data['kp2'])
    op_precision_curve=[]
    recall_curve=[]
    tpr_curve=[]
    fpr_curve=[]
    ratio_thresholds=np.round(np.arange(0.4,1.0,0.05),2)
    nbr_possible_correct_matches, nbr_incorrect_matches=compute_gt_matches(kp1_shared,
                                                                           kp2_shared, real_H,
                                                                           distance_thresh=distance_threshold)
    
    if (plot==True):
        img1_detected = draw_keypoints(img1, kp1_shared)
        img2_detected = draw_keypoints(img2, kp2_shared) 
        fig, ((axis1, axis2))=plt.subplots(1,2, figsize=(10,10))
        axis1.imshow(img1_detected, cmap='gray')
        axis1.set_title('image1 (index {}) detected keypoints with {}\n{} keypoints'.\
                        format(index_1, method, kp1_shared.shape[0]))
        axis2.imshow(img2_detected, cmap='gray')
        axis2.set_title('image2 (index {}) detected keypoints with {}\n{} keypoints'.\
                        format(index_1+1, method, kp2_shared.shape[0]))
        fig, m_axs = plt.subplots(1,ratio_thresholds.shape[0], figsize = (20, 20))
    
        for ratio_threshold, axis1 in zip(ratio_thresholds, m_axs.flatten()):
            output, run_time=compute_homography(img1, img2, data, method, ratio_threshold, \
                                      min_nb_pts=None, verbose=verbose)
            #output={'matches', 'homography'}
            tp, fp, true_positive_match, kp1_true_positive_matches=true_positive_matches(kp1_shared, kp2_shared, \
                                                          output['matches'], real_H, \
                                                          distance_threshold=distance_threshold, verbose=verbose)
            precision=descriptor_precision(tp, fp)
            recall=descriptor_recall(tp, nbr_possible_correct_matches)
            op_precision_curve.append(1-precision)
            recall_curve.append(recall)
            tpr=float(tp/nbr_possible_correct_matches) if nbr_possible_correct_matches!=0 else 0
            fpr=float(fp/nbr_incorrect_matches) if nbr_incorrect_matches!=0 else 0
            tpr_curve.append( tpr )
            fpr_curve.append( fpr )

            if ratio_threshold==0.8:
                coverage_ratio=mask_coverage(img1, kp1_true_positive_matches)
                homography=output['homography']
                M_score=compute_M_score(tp, kp1_shared.shape[0])
                RMSE_thresh_0_8, MEE_thresh_0_8, MAE_thresh_0_8=compute_registration_error(real_H, homography, img1.shape)
                found_homography, acceptable_homography=class_homography(MEE_thresh_0_8, MAE_thresh_0_8)
                is_acceptable=homography_is_accepted(homography)
            
            img_matches=draw_matches(img1_detected,img2_detected,kp1_shared, kp2_shared, output['matches'], true_positive_match)
            axis1.imshow(img_matches)
            axis1.set_title('{}\nthresh= {}\n{} matches\ntp={}\n{} pos true m'.format(method, ratio_threshold\
                            ,len(output['matches']), tp, nbr_possible_correct_matches), fontsize='large')
        
        transformed_img1 = cv2.warpPerspective(img1, homography, (img2.shape[1], img2.shape[0]))
        fig, axis7=plt.subplots(1,1, figsize=(10,10))
        axis7.imshow(transformed_img1, cmap='gray')
        axis7.set_title('Transformed image 1 with {}\n MEE={}\n MAE={}\n RMSE={}\n acceptable={}\ncoverage fraction= {}\nacceptable={}'\
                        .format(method, \
                        MEE_thresh_0_8, MAE_thresh_0_8, RMSE_thresh_0_8, acceptable_homography, coverage_ratio, is_acceptable))

    else:
        for ratio_threshold in ratio_thresholds:
            # makes the different curves

            tp, fp, true_positive_match, kp1_true_positive_matches=true_positive_matches(kp1_shared, kp2_shared, output['matches'], \
                                                          real_H, distance_threshold=distance_threshold, verbose=verbose)
            precision=descriptor_precision(tp, fp)
            recall=descriptor_recall(tp, nbr_possible_correct_matches)
            op_precision_curve.append(1-precision)
            recall_curve.append(recall)
            tpr=float(tp/nbr_possible_correct_matches) if nbr_possible_correct_matches!=0 else 0
            fpr=float(fp/nbr_incorrect_matches) if nbr_incorrect_matches!=0 else 0
            tpr_curve.append( tpr )
            fpr_curve.append( fpr )

            if ratio_threshold==0.8:
                M_score=compute_M_score(tp, kp1_shared.shape[0])
                # computes homography for this ratio threshold on descriptor matching
                output, run_time = compute_homography(img1, img2, data, method,
                                                      ratio_threshold,
                                                      min_nb_pts=None, verbose=verbose)
                coverage_ratio = mask_coverage(img1, kp1_true_positive_matches)
                homography=output['homography']
                RMSE_thresh_0_8, MEE_thresh_0_8, MAE_thresh_0_8 = compute_registration_error(real_H, homography, img1.shape)
                found_homography, acceptable_homography = class_homography(MEE_thresh_0_8, MAE_thresh_0_8)
    
    area = calculate_area(tpr_curve, fpr_curve)

    return homography, op_precision_curve, recall_curve, area, tpr_curve, \
            fpr_curve, M_score, \
            RMSE_thresh_0_8, MEE_thresh_0_8,\
            found_homography, acceptable_homography, coverage_ratio, output['inlier_ratio'], run_time



