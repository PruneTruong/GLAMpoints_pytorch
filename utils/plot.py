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


def horizontal_combine_images(img1, img2):

    ratio=img1.shape[0]/img2.shape[0]
    imgs_comb=np.hstack((img1, cv2.resize(img2,None,fx=ratio, fy=ratio)))
    return imgs_comb


def draw_matches(img1, img2, kp1, kp2, matches, matches_true):
    '''
    arguments: gray images
                kp1 is shape Nx2, N number of feature points, first point in horizontal direction
                matches is Dmatch object of length the number of matches
                '''
    h,w=img1.shape[:2]
    img=horizontal_combine_images(img1, img2)

    src_pts = np.float32([ kp1[m.queryIdx] for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx] for m in matches]).reshape(-1,1,2)
    
    src_pts_true = np.float32([ kp1[m.queryIdx] for m in matches_true]).reshape(-1,1,2)
    dst_pts_true = np.float32([ kp2[m.trainIdx] for m in matches_true]).reshape(-1,1,2)
    #shape Mx1x2 M number of matches 
    dst_pts[:,:,0]=dst_pts[:, :,0]+w
    dst_pts_true[:,:,0]=dst_pts_true[:, :,0]+w

    for i in range(src_pts.shape[0]):
        img=cv2.line(img,(src_pts[i,0,0],src_pts[i,0,1]),(dst_pts[i,0,0], dst_pts[i,0,1]),(255,0,0),5)
        
    for j in range(src_pts_true.shape[0]):
        img=cv2.line(img,(src_pts_true[j,0,0],src_pts_true[j,0,1]),(dst_pts_true[j,0,0], dst_pts_true[j,0,1]),(0,0,255),6)
    return img
    

def draw_keypoints(img, kp):
    '''
        arguments: gray images
                kp1 is shape Nx2, N number of feature points, first point in horizontal direction
                '''
    image_copy=np.copy(img)
    nbr_points=kp.shape[0]
    for i in range(nbr_points):
        image=cv2.circle(image_copy, (np.int32(kp[i,0]),np.int32(kp[i,1])), 5, (0,255,0),thickness=2)
    return image


