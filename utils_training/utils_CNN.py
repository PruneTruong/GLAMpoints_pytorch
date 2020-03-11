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

import numpy as np
import math
import os
import cv2
pi=math.pi
from scipy import sqrt, arctan2
from utils_training.metrics_comparison import info_sift, homography_is_accepted, \
    class_homography, compute_registration_error
from matplotlib import pyplot as plt
import torch
import shutil

def save_checkpoint(state, is_best, save_path, filename):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth'))


def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    best_val=-1
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #scheduler.load_state_dict(checkpoint['scheduler'])
        try:
            best_val=checkpoint['best_loss']
        except:
            best_val=-1
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, best_val


def plot_training(image1, image2, kp_map1, kp_map2, computed_reward1, loss, mask_batch1, metrics_per_image,
                  epoch, save_path, name_to_save):

    fig, ((axis1, axis2), (axis3, axis4), (axis5, axis6), (axis7, axis8)) = \
        plt.subplots(4, 2, figsize=(20, 20))
    nbr=1
    size_image = image1[nbr, :, :].shape
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.0002, hspace=0.4)
    # plot pair of images, red kp are tp points
    axis1.imshow(image1[nbr, :, :], cmap='gray', vmin=0, vmax=255)
    axis1.scatter(metrics_per_image['{}'.format(nbr)]['to_plot']['tp_kp1'][1],
                  metrics_per_image['{}'.format(nbr)]['to_plot']['tp_kp1'][0], s=4, color='red')
    axis1.set_title('epoch{}, img1:original_image, \n true positive points in red'.format(epoch), fontsize='small')

    im = axis2.imshow(image2[nbr, :, :], cmap='gray', vmin=0, vmax=255)
    axis2.scatter(metrics_per_image['{}'.format(nbr)]['to_plot']['warped_tp_kp1'][1],
                  metrics_per_image['{}'.format(nbr)]['to_plot']['warped_tp_kp1'][0], s=2,
                  color='red')
    axis2.set_title('img2:warped_image', fontsize='small')
    fig.colorbar(im, ax=axis2)

    # plot kp map corresponding to both images
    axis3.imshow(kp_map1[nbr, :, :], vmin=0.0, vmax=1.0, interpolation='nearest')
    axis3.set_title('kp map of image1, max {}, min {}'.format(np.max(kp_map1[nbr]),
                                                              np.min(kp_map1[nbr])),
                    fontsize='small')
    im4 = axis4.imshow(kp_map2[nbr, :, :], vmin=0.0, vmax=1.0, interpolation='nearest')
    axis4.set_title('kp map of image2', fontsize='small')
    fig.colorbar(im4, ax=axis4)

    # scatter the points afte NMS and in red the tp points
    axis5.imshow(image1[nbr, :, :], cmap='gray', origin='upper', vmin=0.0, vmax=255.0)
    axis5.scatter(metrics_per_image['{}'.format(nbr)]['to_plot']['keypoints_map1'][1],
                  metrics_per_image['{}'.format(nbr)]['to_plot']['keypoints_map1'][0], s=2,
                  color='green')
    axis5.scatter(metrics_per_image['{}'.format(nbr)]['to_plot']['tp_kp1'][1],
                  metrics_per_image['{}'.format(nbr)]['to_plot']['tp_kp1'][0], s=4, color='red')
    axis5.set_title('kp_map after NMS of image1 in green, nbr_kp {}, \n nbr tp keypoints found {} (in red)'.format(
        metrics_per_image['{}'.format(nbr)]['nbr_kp1'],
        metrics_per_image['{}'.format(nbr)]['total_nbr_kp_reward1']), fontsize='medium')

    im6 = axis6.imshow(image2[nbr, :, :], cmap='gray', origin='upper', vmin=0.0, vmax=255.0)
    axis6.scatter(metrics_per_image['{}'.format(nbr)]['to_plot']['keypoints_map2'][1],
                  metrics_per_image['{}'.format(nbr)]['to_plot']['keypoints_map2'][0], s=2,
                  color='green')
    axis6.scatter(metrics_per_image['{}'.format(nbr)]['to_plot']['warped_tp_kp1'][1],
                  metrics_per_image['{}'.format(nbr)]['to_plot']['warped_tp_kp1'][0], s=4,
                  color='red')
    axis6.set_title('kp_map after NMS of image2 in green, nbr_kp {} \n true positive points in red'.format(
        metrics_per_image['{}'.format(nbr)]['nbr_kp2']), fontsize='small')
    fig.colorbar(im6, ax=axis6)

    # plot the reward and the loss function
    axis7.imshow(computed_reward1[nbr, :, :], vmin=0, vmax=1, interpolation='nearest')
    axis7.set_title('computed reward:,\nrepeatability={}, total_nbr_tp_kp={} \n binary mask for backpropagation sum={}'.format(
        metrics_per_image['{}'.format(nbr)]['repeatability'],
        metrics_per_image['{}'.format(nbr)]['total_nbr_kp_reward1'],
    np.sum(mask_batch1[nbr, :, :])), fontsize='small')


    # plot the transformed image according to estimated homography
    im8 = axis8.imshow(
        cv2.warpPerspective(image1[nbr, :, :], metrics_per_image['{}'.format(nbr)]
        ['computed_H'], (size_image[1], size_image[0])), cmap='gray', vmin=0, vmax=255)
    axis8.set_title('warped image 1 according to computed homography,\n inlier_ratio={}, '
                     'true_homo={}, class_acceptable={}'.format(
        metrics_per_image['{}'.format(nbr)]['inlier_ratio'],
        metrics_per_image['{}'.format(nbr)]['homography_correct'],
        metrics_per_image['{}'.format(nbr)]['class_acceptable']), fontsize='small')
    fig.colorbar(im8, ax=axis8)
    fig.savefig(os.path.join(save_path, name_to_save + '.jpg'), bbox_inches='tight')
    plt.close(fig)


def find_true_positive_matches(kp1, kp2, matches, real_H, distance_threshold=3, verbose=False):
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
                find_true_positive_matches: Dstructure giving the index of true corresponding keypoints.
                kp1_true_positive_matches : kp1 location that correspond to find_true_positive_matches
                kp1_false_positive_matches: kp1 locations that correspond to the false positive matches

    '''

    def warp_keypoint(kp, real_H):
        # kp must be horizontal, then vertical like usual
        num_points = kp.shape[0]
        homogeneous_points = np.concatenate([kp, np.ones((num_points, 1))], axis=1)
        true_warped_points = np.dot(real_H, np.transpose(homogeneous_points))
        true_warped_points = np.transpose(true_warped_points)
        true_warped_points = true_warped_points[:, :2] / true_warped_points[:, 2:]
        return true_warped_points

    # warp the source points accordint to gt homography
    true_warped_keypoints = warp_keypoint(np.float32([kp1[m.queryIdx] for m in matches]).reshape(-1, 2), real_H)
    warped_keypoints = np.float32([kp2[m.trainIdx] for m in matches]).reshape(-1, 2)
    norm = (np.linalg.norm(true_warped_keypoints - warped_keypoints, axis=1) <= distance_threshold)

    tp = np.sum(norm)
    fp = len(matches) - tp
    true_positive_matches = [matches[i] for i in range(norm.shape[0]) if (norm[i] == 1)]
    false_positive_matches = [matches[i] for i in range(norm.shape[0]) if (norm[i] == 0)]

    kp1_true_positive_matches = np.int32([kp1[m.queryIdx] for m in true_positive_matches]).reshape(-1, 2)
    kp1_false_positive_matches = np.int32([kp1[m.queryIdx] for m in false_positive_matches]).reshape(-1, 2)

    if verbose:
        print('the number of true positive matches is {}'.format(tp))
    return tp, fp, true_positive_matches, kp1_true_positive_matches, kp1_false_positive_matches


def sift(SIFT, image, kp_before):
    kp, des = SIFT.compute(image, kp_before)
    if des is not None:
        eps = 1e-7
        des /= (des.sum(axis=1, keepdims=True) + eps)
        des = np.sqrt(des)
    return kp, des


def get_sift(image1, image2, homography_batch):
    sift_H_batch = np.zeros((image1.shape[0], 3, 3), dtype=np.float32)
    sift_ratio_inliers = np.zeros((image1.shape[0]), dtype=np.float32)
    repeatability_SIFT = []
    accepted_SIFT = []
    SIFT_class_acceptable = []

    for i in range(image1.shape[0]):
        homography = homography_batch[i, :, :]
        s_repeatability, sift_homography, s_ratio_inliers = info_sift(np.uint8(image1[i, :, :, 0]),
                                                                      np.uint8(image2[i, :, :, 0]), homography)
        repeatability_SIFT.append(s_repeatability)
        s_accepted = homography_is_accepted(sift_homography)
        accepted_SIFT.append(s_accepted)
        sift_H_batch[i] = sift_homography
        sift_ratio_inliers[i] = s_ratio_inliers
        RMSE, MEE, MAE = compute_registration_error(homography, sift_homography, image1[i, :, :, 0].shape)
        found_homography, acceptable_homography = class_homography(MEE, MAE)
        SIFT_class_acceptable.append(acceptable_homography)
    return accepted_SIFT, repeatability_SIFT, SIFT_class_acceptable


def warp_kp(kp, homography, shape):
    '''
    warp the keypoints kp according to the given homography and only keeps the ones 
    that are within the shape. 
    input:
        kp 2xN, kp[0] contains array coordinates of the keypoints in vertical direction !,
                kp[1] the coordinates in horizontal direction (corresponds to x direction)
    of the array (results of np.where)
        homography 3x3 matrix
        shape 2d
    output:
        kp resulting warped keypoints in shape 2xN, kps[0] contains the x coordinates.
    '''
    # convert source kp in homogeneous coordinate: Nx3 (x, y, 1)
    homogeneous_points = np.concatenate([kp.T[:, [1,0]], np.ones((len(kp[0]), 1))], axis=1)

    # warp the source kp
    warped_points = np.dot(homography, np.transpose(homogeneous_points))
    warped_points = np.transpose(warped_points)
    warped_points = warped_points[:, :2] / warped_points[:, 2:]
    warped_points = np.int32(np.round(warped_points))
    # warped_points first coordinate is horizontal

    kp = []

    # check that warp kp are still within the image
    for i in range(warped_points.shape[0]):
        if ((warped_points[i, 0] >= 0) & (warped_points[i, 0] < shape[1]) &
               (warped_points[i, 1] >= 0) & (warped_points[i, 1] < shape[0])):
            kp.append(warped_points[i])
    kp = np.array(kp) # horizontal coordinate first, Nx2

    # return to 2xN, [0] contains coordinate in vertical direction, [1] in horizontal direction (x direction)
    return kp.T[[1,0], :]

        
def resize_keeping_ratio(image1, image2, size, homography):
    '''
    resize both image1 and image 2 so that their smallest dimension is equal to size, and modifies 
    the homography relating the two accordinly. 
    inputs:
        image1
        image2
        size size of the desired square image output
        homogrpahy relating image1 to image2
    outputs:
        image1 - resized image1
        image2 - resized image2
        modified_homo - modified homography
    '''
    if image1.shape[0]<image1.shape[1]:
        factor=size/image1.shape[0]
        image1=cv2.resize(image1,None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
        #image1=image[:,x:x+256]
        
    else:
        factor=size/image1.shape[1]
        image1=cv2.resize(image1,None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
        #x=random.randint(0,image.shape[0]-256)
        #image1=image[x:x+256, :]
    modified_homo=np.copy(homography)
    modified_homo[0,2]*=factor
    modified_homo[1,2]*=factor
    modified_homo[2,0]/=factor
    modified_homo[2,1]/=factor
    return image1, image2, modified_homo


def resize_to_patch(image1, image2, homography, size):
    '''
    resize both image1 and image 2 to a square of size size, and modifies 
    the homography relating the two accordinly. 
    inputs:
        image1
        image2
        size size of the desired square image output
        homogrpahy relating image1 to image2
    outputs:
        image1 - resized image1
        image2 - resized image2
        modified_homo - modified homography
    '''
    factorx=size/image1.shape[1]
    factory=size/image1.shape[0]
    image1=cv2.resize(image1,None, fx=factorx, fy=factory, interpolation = cv2.INTER_CUBIC)
    image2=cv2.resize(image2,None, fx=factorx, fy=factory, interpolation = cv2.INTER_CUBIC)
    
    modified_homography=np.copy(homography)
    modified_homography[0,1]/=factory
    modified_homography[1,0]/=factorx
    modified_homography[0,2]*=factorx
    modified_homography[1,2]*=factory
    modified_homography[2,0]/=factorx
    modified_homography[2,1]/=factory
    return image1, image2, modified_homography





