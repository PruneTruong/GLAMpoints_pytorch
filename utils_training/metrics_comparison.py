#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# GLAMPOINTS LICENSE CONDITIONS

# Copyright (2019), RetinAI Medical AG.

# This software for the training and application of Greedily Learned Matching keypoints is being made available for individual research use only. For any commercial use contact RetinAI Medical AG.

# For further details on obtaining a commercial license, contact RetinAI Medical AG Office (sales@retinai.com). 

# RETINAI MEDICAL AG MAKES NO REPRESENTATIONS OR
# WARRANTIES OF ANY KIND CONCERNING THIS SOFTWARE.

# This license file must be retained with all copies of the software,
# including any modified or derivative versions.

import cv2
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from math import sqrt


def compute_homography(kp1, kp2, des1, des2, method, ratio_threshold):
    def unilateral_matching(des1, des2, method):
        if method == 'ORB' or method == 'FREAK' or method == 'BRISK':
            distance_metrics = cv2.NORM_HAMMING
        else:
            distance_metrics = cv2.NORM_L2

        bf = cv2.BFMatcher(distance_metrics)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if (m.distance < ratio_threshold * n.distance)]
        return good

    if des1 is not None and des2 is not None:
        if des1.shape[0] > 2 and des2.shape[0] > 2:
            good = unilateral_matching(des1, des2, method)
            if len(good) >= 4:
                src_pts = np.float32([kp1[m.queryIdx] for m in good]).reshape(-1, 1, 2)
                # shape is number_good_matchesx1x2 (two coordinates of corresponding keypoint)
                dst_pts = np.float32([kp2[m.trainIdx] for m in good]).reshape(-1, 1, 2)

                H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
                if H is None:
                    # H=np.identity(3)
                    H = np.zeros((3, 3))
                    ratio_inliers = 0.0
                else:
                    inliers = inliers.ravel().tolist()
                    ratio_inliers = np.sum(inliers) / len(inliers) if len(inliers) != 0 else 0.0

                homography = H

            else:
                # homography=np.identity(3)
                homography = np.zeros((3, 3))
                ratio_inliers = 0.0
        else:
            # homography=np.identity(3)
            homography = np.zeros((3, 3))
            ratio_inliers = 0.0
    else:
        # homography=np.identity(3)
        homography = np.zeros((3, 3))
        ratio_inliers = 0.0
    return homography, ratio_inliers


def homography_is_accepted(H):
    accepted = True
    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
    '''we compute the determinant of the 2x2 submatrix of homography matrix. 
    This [2x2] matrix called R contains rotation component of the estimated transformation.
    Correct rotation matrix has it's determinant value equals to 1. 
    In our case R matrix may contain scale component, so it's determinant can have 
    other values, but in general for correct rotation and scale values it's always greater than zero.
    '''
    if (det < 0.2):
        accepted = False
    N1 = math.sqrt(H[0, 0] * H[0, 0] + H[1, 0] * H[1, 0])
    N2 = math.sqrt(H[0, 1] * H[0, 1] + H[1, 1] * H[1, 1])
    N3 = math.sqrt(H[2, 0] * H[2, 0] + H[2, 1] * H[
        2, 1])  # projective transforms, must be low because almost affine transformation

    if ((N1 > 4) or (N1 < 0.1)):
        accepted = False
    if ((N2 > 4) or (N2 < 0.1)):
        accepted = False
    if (math.fabs(N1 - N2) > 0.5):
        accepted = False
        # scale in x and y directions must be more or less the same
    if (N3 > 0.002):
        accepted = False

    return accepted


def compute_registration_error(real_H, H, shape):
    if np.all(H == 0):
        return float('nan'), float('nan'), float('nan')

    corners = np.array([[50, 50, 1], [50, shape[0] - 50, 1],
                        [shape[1] - 50, 50, 1], [shape[1] - 50, shape[0] - 50, 1],
                        [int(shape[1] / 2), int(shape[0] / 4), 1],
                        [int(shape[1] / 2), int(3 * shape[0] / 4), 1]])
    gt_warped_points = np.transpose(np.dot(real_H, np.transpose(corners)))
    gt_warped_points = gt_warped_points[:, :2] / gt_warped_points[:, 2:]

    warped_points = np.transpose(np.dot(H, np.transpose(corners)))
    warped_points = warped_points[:, :2] / warped_points[:, 2:]

    if (np.isnan(np.sum(warped_points)) == False) and (np.isnan(np.sum(gt_warped_points)) == False):
        try:
            RMSE = sqrt(mean_squared_error(warped_points, gt_warped_points))
            MEE = np.median(np.linalg.norm(gt_warped_points - warped_points, axis=1))
            # median error
            MAE = np.max(np.linalg.norm(gt_warped_points - warped_points, axis=1))
            return RMSE, MEE, MAE
        except:
            return float('nan'), float('nan'), float('nan')
    else:
        return float('nan'), float('nan'), float('nan')
    # max error


def class_homography(MEE, MAE):
    '''
    returns 1 if it is a correct homography, 0 otherwise.
    add later criteria on skewness this kind og things
    '''

    def is_acceptable_homography(MEE, MAE):
        if (MEE < 10 and MAE < 30):
            return 1
        else:
            return 0

    if math.isnan(MEE) is True:
        found_homography = 0
        acceptable_homography = 0
        # didn t find a homography
    else:
        found_homography = 1
        acceptable_homography = is_acceptable_homography(MEE, MAE)
    return found_homography, acceptable_homography


def compute_repeatability(kp1, kp2, real_H, shape, distance_thresh=None):
    """
    Compute the repeatability.
    arguments:
                kp1
                kp2
                real_H linking the 2 images
                shape of image              """

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
        homogeneous_points = np.concatenate([kp, np.ones((num_points, 1))], axis=1)
        warped_points = np.dot(real_H, np.transpose(homogeneous_points))
        warped_points = np.transpose(warped_points)  # shape number_of_keypointsx3
        warped_points = warped_points[:, :2] / warped_points[:, 2:]
        return warped_points

    def keep_shared_keypoints(kp, real_H, shape):
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
        """
        warped_points = warp_keypoints(kp, real_H)  # first coordinate horizontal
        kp_shared = []

        for i in range(warped_points.shape[0]):
            if ((warped_points[i, 0] >= 0) & (warped_points[i, 0] < shape[1]) & \
                    (warped_points[i, 1] >= 0) & (warped_points[i, 1] < shape[0])):
                kp_shared.append(kp[i])
        kp_shared = np.array(kp_shared)
        return kp_shared

    #    repeatability = []
    if distance_thresh is None:
        distance_thresh = 3

    warped_keypoints = keep_shared_keypoints(kp2, np.linalg.inv(real_H), shape)  # kp2
    # coordonate of the keypoints in image 2 that are commun to both images
    keypoints = keep_shared_keypoints(kp1, real_H, shape)  # kp1

    # Compute the repeatability
    N1 = keypoints.shape[0]
    N2 = warped_keypoints.shape[0]
    if N2 != 0 and N1 != 0:
        true_warped_keypoints = warp_keypoints(keypoints, real_H)
        true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)  # N1x1x2
        warped_keypoints = np.expand_dims(warped_keypoints, 0)  # 1xN2x2
        # shapes are broadcasted to N1 x N2 x 2:
        norm = np.linalg.norm(true_warped_keypoints - warped_keypoints, ord=None, axis=2)  # N1xN2
        count1 = 0
        count2 = 0

        min1 = np.min(norm, axis=1)  # so that each point of image2 has only one match possible on image1
        count1 = np.sum(min1 <= distance_thresh)

        min2 = np.min(norm, axis=0)
        count2 = np.sum(min2 <= distance_thresh)
        if N1 + N2 > 0:
            repeatability = (count1 + count2) / (N1 + N2)
    else:
        repeatability = 0

    return repeatability


def get_repeatability(kp1, kp2, real_H, size):
    if kp1.shape[0]==0 or kp2.shape[0]==0:
        repeatability=0
    else:
        repeatability=compute_repeatability(kp1, kp2, real_H, size )

    return repeatability


def detector_SIFT(image):
    #this is ROOT_SIFT
    eps=1e-7
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=600, contrastThreshold=0.0275, edgeThreshold=12, sigma=1)
    kp, des= sift.detectAndCompute(image, None)

    if des is not None:
        des /= (des.sum(axis=1, keepdims=True) + eps )
        des = np.sqrt(des)
    kp=np.array([m.pt for m in kp], dtype=np.int32)
    return kp, des


def info_sift(image1, image2, real_H):
    kp1, des1=detector_SIFT(np.uint8(image1) )
    kp2, des2=detector_SIFT(np.uint8(image2) )
    sift_repeatability=get_repeatability(kp1, kp2, real_H, image1.shape)
    sift_homography, ratio_inliers=compute_homography(kp1, kp2, des1, des2, 'SIFT', 0.8)
    
    return sift_repeatability, sift_homography, ratio_inliers





