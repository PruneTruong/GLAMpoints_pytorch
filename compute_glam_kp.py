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


"""
Created on Thu Dec  6 14:42:15 2018

@author: truongp
"""
import numpy as np
import argparse
from models.glampoints import GLAMpointsInference
import cv2
import os
from utils.plot import draw_keypoints


def get_kp_glampoints(image_color, glampoints, save_path, name, green_channel=False):
    if green_channel:
        image = image_color[:, :, 1]
    else:
        image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    kp, des = glampoints.find_and_describe_keypoints(image)

    image_kp = np.uint8(draw_keypoints(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB), kp))
    cv2.imwrite('{}/{}_glampoints.png'.format(save_path, name), image_kp)
    print('saved {}'.format(name))
    return kp


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Computing metrics')
    parser.add_argument('--path_images', type=str,
                        help='Directory where to find the images.')
    parser.add_argument('--write_dir', type=str,
                        help='Directory where to write output text containing the metrics')
    parser.add_argument('--path_glam_weights', type=str, default='weights/Unet4_retina_images_converted_tf_weights.pth',
                        help='Path to pretrained weights file of GLAMpointsInference model (default: weights/Unet4_retina_images_converted_tf_weights.pth)')
    parser.add_argument('--NMS', type=int, default=10,
                        help='Value of the NMS window applied on the score map output of GLAMpointsInference (default:10)')
    parser.add_argument('--min_prob', type=float, default=0.0,
                        help='Minimum probability of a keypoint for GLAMpointsInference (default:0)')
    parser.add_argument('--save_text', type=bool, default=False,
                        help='Save matrix of extracted kp as a directory in a text file (default:False)')
    parser.add_argument('--green_channel', type=bool, default=True,
                        help='Use the green channel (default:True)')

    '''
    parser.add_argument('--preprocessing', type=bool, default=False,
    help='Applying preprocessing on the images ? (default: False).')
    parser.add_argument('--preprocessing_equalization', type=int, default=8,
    help='Size of the histogram equalisation window applied on images during preprocessing (default: 8).')
    parser.add_argument('--preprocessing_blurring', type=int, default=2,
    help='Size of the bilinear blurring kernel applied on images during preprocessing (default: 2).')
    '''

    opt = parser.parse_args()
    if not os.path.isdir(opt.write_dir):
        os.makedirs(opt.write_dir)
    glampoints = GLAMpointsInference(path_weights=opt.path_glam_weights, nms=opt.NMS, min_prob=opt.min_prob)
    kp_dict = {}

    if os.path.isdir(opt.path_images):
        path_to_images = [os.path.join(opt.path_images, f) for f in sorted(os.listdir(opt.path_images)) if
                                     f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.ppm'))]
    else:
        # it is just the path to a single image
        path_to_images = []
        path_to_images.append(opt.path_images)

    for i, path_file in enumerate(path_to_images):
        print(path_file)
        try:
            image = cv2.imread(path_file)
            if image is None:
                continue
        except:
            continue

        kp = get_kp_glampoints(image, glampoints, opt.write_dir, '{}_{}'.format(os.path.basename(os.path.normpath(opt.path_images)), i),
                               green_channel=opt.green_channel)
        kp_dict[i] = kp.tolist()
        with open('{}/kp_image{}.txt'.format(opt.write_dir, i), 'w') as outfile:
            outfile.write('{} {}\n'.format(len(kp), len(kp)))
            for m in range(len(kp)):
                outfile.write('{} {}\n'.format(kp[m, 0], kp[m, 1]))
