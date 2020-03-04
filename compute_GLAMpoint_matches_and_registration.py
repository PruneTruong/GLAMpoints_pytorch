#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 14:42:15 2018

@author: truongp
"""
import numpy as np
from matplotlib import pyplot as plt

import cv2
import argparse
import imageio
import os
from models.glampoints import GLAMpoints, SIFT_noorientation


def horizontal_combine_images(img1, img2):

    ratio=img1.shape[0]/img2.shape[0]
    imgs_comb=np.hstack((img1, cv2.resize(img2,None,fx=ratio, fy=ratio)))
    return imgs_comb


def draw_keypoints(img, kp):
    '''
        arguments: gray images
                kp1 is shape Nx2, N number of feature points, first point in horizontal direction
                '''
    image_copy=np.copy(img)
    nbr_points=kp.shape[0]
    for i in range(nbr_points):
        image=cv2.circle(image_copy, (np.int32(kp[i,0]),np.int32(kp[i,1])), 3, (0,0,0),thickness=-1)
    return image


def compute_homography(kp1, kp2, des1, des2, method, ratio_threshold):

    def unilateral_matching(des1, des2, method, ratio_threshold):
        if method == 'ORB' or method == 'FREAK' or method == 'BRISK':
            distance_metrics=cv2.NORM_HAMMING
        else:
            distance_metrics=cv2.NORM_L2
            
        bf = cv2.BFMatcher(distance_metrics)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m,n in matches if (m.distance < ratio_threshold*n.distance)]
        return good

    # for cases when no homography is found
    good = None
    homography = np.zeros((3, 3))
    ratio_inliers = 0.0

    if des1 is not None and des2 is not None:
        if des1.shape[0]>2 and des2.shape[0]>2:
            good = unilateral_matching(des1, des2, method, ratio_threshold)
            if len(good) >= 4:
                src_pts = np.float32([ kp1[m.queryIdx] for m in good ]).reshape(-1,1,2)
                # shape is number_good_matchesx1x2 (two coordinates of corresponding keypoint)
                dst_pts = np.float32([ kp2[m.trainIdx] for m in good ]).reshape(-1,1,2)       
                
                H, inliers = cv2.findHomography(src_pts, dst_pts,cv2.RANSAC)
                if H is None:
                    H = np.zeros((3,3))
                    ratio_inliers = 0.0
                else:
                    inliers = inliers.ravel().tolist()
                    ratio_inliers = np.sum(inliers)/len(inliers) if len(inliers)!=0 else 0.0

                homography = H

    return homography, ratio_inliers, good


def draw_matches(img1, img2, kp1, kp2, matches):
    '''
    arguments: gray images
                kp1 is shape Nx2, N number of feature points, first point in horizontal direction
                matches is Dmatch object of length the number of matches
                '''
    h,w=img1.shape[:2]
    img=horizontal_combine_images(img1, img2)

    src_pts = np.float32([ kp1[m.queryIdx] for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx] for m in matches]).reshape(-1,1,2)
    
    #shape Mx1x2 M number of matches 
    dst_pts[:,:,0]=dst_pts[:, :,0]+w

    for i in range(src_pts.shape[0]):
        img=cv2.line(img,(src_pts[i,0,0],src_pts[i,0,1]),(dst_pts[i,0,0], dst_pts[i,0,1]),(0,0,255),1)

    return img


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Computing matches and registration using GLAMpoints')
    parser.add_argument('--path_image1', type=str, default='path_to_image1',
    help='Path to the first image.')
    parser.add_argument('--path_image2', type=str, default='path_to_image1',
    help='Path to the second image.')
    parser.add_argument('--write_dir', type=str, default='path_to_result_dir',
    help='Directory where to write output figure.')
    parser.add_argument('--path_GLAMpoints_weights', type=str, default='weights/model-34',
    help='Path to pretrained weights file of GLAMpoint model (default: weights/model-34).')
    parser.add_argument('--SIFT', type=bool, default=True,
    help='Compute matches and registration with SIFT detector as a comparison? (default:True)')
    parser.add_argument('--NMS', type=int, default=15,
    help='Value of the NMS window applied on the score map output of GLAMpoint (default:15)')
    parser.add_argument('--min_prob', type=float, default=0.0,
    help='Minimum probability of a keypoint for GLAMpoint (default:0)')
    parser.add_argument('--green_channel', type=bool, default=True,
    help='Use the green channel (default:True)')
    opt = parser.parse_args()
    kwarg=vars(parser.parse_args(args=None, namespace=None))
    kwarg_SIFT={'nfeatures':600, 'contrastThreshold':0.0275, 'edgeThreshold':12.0, 'sigma':1.0}
    

    try:
        image1 = imageio.imread(opt.path_image1)
        image2 = imageio.imread(opt.path_image2)
    except ValueError:
        print('cannot read your images')

    if opt.green_channel:
        image1_gray = image1[:,:,1]
        image2_gray = image2[:,:,1]
    else:
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    glampoints = GLAMpoints(**kwarg)

    # gets kp and descriptor from both images using glampoint
    kp1, des1 = glampoints.find_and_describe_keypoints(image1_gray)
    kp2, des2 = glampoints.find_and_describe_keypoints(image2_gray)
    im1 = np.uint8(draw_keypoints(cv2.cvtColor(image1, cv2.COLOR_RGB2BGR), kp1))
    im2 = np.uint8(draw_keypoints(cv2.cvtColor(image2, cv2.COLOR_RGB2BGR), kp2))

    # compute matches and homography
    homography, inlier_ratio, matches = compute_homography(kp1, kp2, des1, des2, method='GLAMpoint', ratio_threshold=0.8)

    fig, ((axis1, axis2, axis3), (axis4, axis5, axis6))=plt.subplots(2,3, figsize=(30,30))
    axis1.imshow(im1)
    axis1.set_title('detected kp on image 1, NMS={}'.format(opt.NMS))
    axis2.imshow(im2)
    axis2.set_title('detected kp on image 2, NMS={}'.format(opt.NMS))
    if matches is not None:
        img_matches = draw_matches(im1, im2, kp1, kp2, matches)
        axis3.imshow(img_matches)
        axis3.set_title('matches found')
    axis4.imshow(image1)
    axis4.set_title('image 1')
    axis5.imshow(image2)
    axis5.set_title('image 2')
    axis6.imshow(cv2.warpPerspective(image1, homography, (image1.shape[1], image1.shape[0])))
    axis6.set_title('image 1 transformed after \n estimating homography with RANSAC')
    fig.savefig(os.path.join(opt.write_dir, 'registration_GLAMpoints_NMS_{}_minprob_{}.png'.\
                format(opt.NMS, opt.min_prob)))
    plt.close(fig)

    if opt.SIFT:
        # compare to SIFT by extracting SIFT kp and descriptors
        sift = SIFT_noorientation(**kwarg_SIFT) # here for fair comparison, SIFT does not have orientation
        # can also be SIFT with orientation sift=SIFT(**kwarg_SIFT)
        kp1, des1=sift.find_and_describe_keypoints(image1_gray)
        kp2, des2=sift.find_and_describe_keypoints(image2_gray)
        im1=np.uint8(draw_keypoints(cv2.cvtColor(image1, cv2.COLOR_RGB2BGR), kp1))
        im2=np.uint8(draw_keypoints(cv2.cvtColor(image2, cv2.COLOR_RGB2BGR), kp2))

        homography, inlier_ratio, matches=compute_homography(kp1, kp2, des1, des2, 'RetiNet', 0.8)
        fig, ((axis1, axis2, axis3), (axis4, axis5, axis6))=plt.subplots(2,3, figsize=(30,30))
        axis1.imshow(im1)
        axis1.set_title('detected kp on image 1')
        axis2.imshow(im2)
        axis2.set_title('detected kp on image 2')
        if matches is not None:
            img_matches = draw_matches(im1, im2, kp1, kp2, matches)
            axis3.imshow(img_matches)
            axis3.set_title('matches found')
        axis4.imshow(image1)
        axis4.set_title('image 1')
        axis5.imshow(image2)
        axis5.set_title('image 2')
        axis6.imshow(cv2.warpPerspective(image1, homography, (image1.shape[1], image1.shape[0])))
        axis6.set_title('image 1 transformed after \n estimating homography with RANSAC')
        fig.savefig(os.path.join(opt.write_dir, 'registration_SIFT.png'))
        plt.close(fig)