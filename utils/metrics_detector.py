#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:30:11 2018

@author: truongp
"""
import numpy as np

def compute_repeatability(kp1, kp2, real_H, shape, distance_thresh=None):
    """
    Compute the repeatability. 
    arguments:
                kp1
                kp2
                real_H - linking the 2 images
                shape of image              
                distance_thresh - optional, value of radius used to evaluated 
                if two keypoints on img1 and img2 are the same
    output: 
        repeatability 
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

    def keep_shared_keypoints(kp,real_H, shape):
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
        warped_points = warp_keypoints(kp, real_H) #first coordinate horizontal
        kp_shared=[]

        for i in range(warped_points.shape[0]):
            if ( (warped_points[i, 0] >= 0) & (warped_points[i, 0] < shape[1]) &\
               (warped_points[i, 1] >= 0) & (warped_points[i, 1] < shape[0])  ):
                kp_shared.append(kp[i])
        kp_shared=np.array(kp_shared)
        return kp_shared

#    repeatability = []
    if distance_thresh is None:
        distance_thresh=3

    warped_keypoints = keep_shared_keypoints(kp2, np.linalg.inv(real_H), shape)

    #coordonate of the keypoints in image 2 that are commun to both images
    keypoints=keep_shared_keypoints(kp1, real_H, shape)
    true_warped_keypoints = warp_keypoints(keypoints, real_H)



    # Compute the repeatability
    N1 = true_warped_keypoints.shape[0]
    N2 = warped_keypoints.shape[0]
    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1) #N1x1x2
    warped_keypoints = np.expand_dims(warped_keypoints, 0) #1xN2x2
    # shapes are broadcasted to N1 x N2 x 2:
    norm = np.linalg.norm(true_warped_keypoints - warped_keypoints, ord=None, axis=2) #N1xN2
    count1 = 0
    count2 = 0
    if N2 != 0:
        min1 = np.min(norm, axis=1) #so that each point of image2 has only one match possible on image1
        count1 = np.sum(min1 <= distance_thresh)
    if N1 != 0:
        min2 = np.min(norm, axis=0)
        count2 = np.sum(min2 <= distance_thresh)
    if N1 + N2 > 0:
        repeatability=(count1 + count2) / (N1 + N2)
    return repeatability
