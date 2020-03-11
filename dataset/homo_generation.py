#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 09:35:26 2019

@author: truongp
"""
import numpy as np
from dataset.data_augmentation import random_brightness, random_contrast, additive_speckle_noise, \
    additive_gaussian_noise, add_shade, motion_blur
import cv2
import random


def homography_sampling(shape, parameters, seed=None):
    """Sample a random valid homography, as a composition of translation, rotation,
    scaling, shearing and perspective transforms. 

    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        parameters: dictionnary containing all infor on the transformations to apply. 
        ex:
        parameters={}
        scaling={'use_scaling':True, 'min_scaling_x':0.7, 'max_scaling_x':2.0, \
                 'min_scaling_y':0.7, 'max_scaling_y':2.0}
        perspective={'use_perspective':False, 'min_perspective_x':0.000001, 'max_perspective_x':0.0009, \
                  'min_perspective_y':0.000001, 'max_perspective_y':0.0009}
        translation={'use_translation':True, 'max_horizontal_dis':100, 'max_vertical_dis':100}
        shearing={'use_shearing':True, 'min_shearing_x':-0.3, 'max_shearing_x':0.3, \
                  'min_shearing_y':-0.3, 'max_shearing_y':0.3}
        rotation={'use_rotation':True, 'max_angle':90}
        parameters['scaling']=scaling
        parameters['perspective']=perspective
        parameters['translation']=translation
        parameters['shearing']=shearing
        parameters['rotation']=rotation
    Returns:
        A 3x3 matrix corresponding to the homography transform.
    """
    if seed is not None:
        random.seed(seed)
    if parameters['rotation']['use_rotation']:
        (h, w) = shape
        center = (w // 2, h // 2)
        y= random.randint(-parameters['rotation']['max_angle'], \
                          parameters['rotation']['max_angle'])
        # perform the rotation
        M = cv2.getRotationMatrix2D(center, y, 1.0)
        homography_rotation=np.concatenate([M, np.array([[0,0,1]])], axis=0)
    else:
        homography_rotation=np.eye(3)
    
    if  parameters['translation']['use_translation']:
        tx=random.randint(-parameters['translation']['max_horizontal_dis'],\
                          parameters['translation']['max_horizontal_dis'])
        ty=random.randint(-parameters['translation']['max_vertical_dis'],\
                          parameters['translation']['max_vertical_dis'])
        homography_translation=np.eye(3)
        homography_translation[0,2]=tx
        homography_translation[1,2]=ty
    else:
        homography_translation=np.eye(3)
    
    if parameters['scaling']['use_scaling']:
        scaling_x=random.choice(np.arange(parameters['scaling']['min_scaling_x'],\
                                          parameters['scaling']['max_scaling_x'], 0.1))
        scaling_y=random.choice(np.arange(parameters['scaling']['min_scaling_y'],\
                                          parameters['scaling']['max_scaling_y'], 0.1))
        homography_scaling=np.eye(3)
        homography_scaling[0,0]=scaling_x
        homography_scaling[1,1]=scaling_y
    else:
        homography_scaling=np.eye(3)
        
    if parameters['shearing']['use_shearing']:
        shearing_x=random.choice(np.arange(parameters['shearing']['min_shearing_x'],\
                                          parameters['shearing']['max_shearing_x'], 0.0001))
        shearing_y=random.choice(np.arange(parameters['shearing']['min_shearing_y'],\
                                          parameters['shearing']['max_shearing_y'], 0.0001))
        homography_shearing=np.eye(3)
        homography_shearing[0,1]=shearing_y
        homography_shearing[1,0]=shearing_x
    else:
        homography_shearing=np.eye(3)
    
    if parameters['perspective']['use_perspective']:
        perspective_x=random.choice(np.arange(parameters['perspective']['min_perspective_x'],\
                                          parameters['perspective']['max_perspective_x'], 0.00001))
        perspective_y=random.choice(np.arange(parameters['perspective']['min_perspective_y'],\
                                          parameters['perspective']['max_perspective_y'], 0.00001))
        homography_perspective=np.eye(3)
        homography_perspective[2,0]=perspective_x
        homography_perspective[2,1]=perspective_y
    else:
        homography_perspective=np.eye(3)
    
    homography=np.matmul(np.matmul(np.matmul(np.matmul(homography_rotation, homography_translation),\
                                             homography_shearing),homography_scaling),\
                                             homography_perspective)
    return homography


def apply_augmentations(image, augmentations, seed=None):
    def apply_augmentation(image, aug, augmentations, seed=None):
        '''
        arguments: image - gray image with intensity scale 0-255
                    aug - name of the augmentation to apply
                    seed
        output:
                    image - gray image after augmentation with intensity scale 0-255
        '''
        if aug == 'additive_gaussian_noise':
            image, kp = additive_gaussian_noise(image, [], seed=seed,
                                                std=(augmentations['additive_gaussian_noise']['std_min'],
                                                     augmentations['additive_gaussian_noise']['std_max']))
        if aug == 'additive_speckle_noise':
            image, kp = additive_speckle_noise(image, [], intensity=augmentations['additive_speckle_noise']['intensity'])
        if aug == 'random_brightness':
            image, kp = random_brightness(image, [], seed=seed)
        if aug == 'random_contrast':
            image, kp = random_contrast(image, [], seed=seed)
        if aug == 'add_shade':
            image, kp = add_shade(image, [], seed=seed)
        if aug == 'motion_blur':
            image, kp = motion_blur(image, [], max_ksize=augmentations['motion_blur']['max_ksize'])
        if aug == 'gamma_correction':
            # must be applied on image with intensity scale 0-1
            maximum = np.max(image)
            image_preprocessed = image / maximum if maximum > 0 else 0
            random_gamma = random.uniform(augmentations['gamma_correction']['min_gamma'], \
                                          augmentations['gamma_correction']['max_gamma'])
            image_preprocessed = image_preprocessed ** random_gamma
            image = image_preprocessed * maximum
        if aug == 'opposite':
            # must be applied on image with intensity scale 0-1
            maximum = np.max(image)
            image_preprocessed = image / maximum if maximum > 0 else 0
            image_preprocessed = 1 - image_preprocessed
            image = image_preprocessed * maximum
        if aug == 'no_aug':
            pass
        return image

    random.seed(seed)
    list_of_augmentations = augmentations['augmentation_list']
    index = random.sample(range(len(list_of_augmentations)), 3)
    for i in index:
        aug = list_of_augmentations[i]
        image = apply_augmentation(image, aug, augmentations, seed)
    image_preprocessed = image / (np.max(image) + 0.000001)
    return image, image_preprocessed


def homography_sampling_2(image, sample_homography, seed=None):
    param = 0.0009
    if seed is None:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(seed)
        random.seed(seed)

    homography_perspective = np.array([[1 - param + 2 * param * random_state.rand(),
                                        -param + 2 * param * random_state.rand(),
                                        -param + 2 * param * random_state.rand()],
                                       [-param + 2 * param * random_state.rand(),
                                        1 - param + 2 * param * random_state.rand(),
                                        -param + 2 * param * random_state.rand()],
                                       [-param + 2 * param * random_state.rand(),
                                        -param + 2 * param * random_state.rand(),
                                        1 - param + 2 * param * random_state.rand()]])

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    y = random.randint(-sample_homography['max_rotation'], sample_homography['max_rotation'])
    # perform the rotation
    M = cv2.getRotationMatrix2D(center, y, 1.0)
    homography_rotation = np.concatenate([M, np.array([[0, 0, 1]])], axis=0)

    tx = random.randint(-sample_homography['max_horizontal_dis'], \
                        sample_homography['max_horizontal_dis'])
    ty = random.randint(-sample_homography['max_vertical_dis'], \
                        sample_homography['max_vertical_dis'])
    homography_translation = np.eye(3)
    homography_translation[0, 2] = tx
    homography_translation[1, 2] = ty

    final_homography = np.matmul(homography_perspective, np.matmul(homography_rotation, homography_translation))
    image_transformed = cv2.warpPerspective(np.uint8(image), final_homography, (image.shape[1], image.shape[0]))
    return image_transformed, final_homography




