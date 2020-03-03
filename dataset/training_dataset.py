from os import path as osp

import cv2
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from .homo_generation import homography_sampling, apply_augmentations


def center_crop(img, size):
    """
    Get the center crop of the input image
    Args:
        img: input image [HxWx3]
        size: size of the center crop (tuple)
    Output:
        img_pad: center crop
        x, y: coordinates of the crop
    """

    if not isinstance(size, tuple):
        size = (size, size)

    img = img.copy()
    h, w = img.shape[:2]

    pad_w = 0
    pad_h = 0
    if w < size[1]:
        pad_w = np.uint16((size[1] - w) / 2)
    if h < size[0]:
        pad_h = np.uint16((size[0] - h) / 2)
    img_pad = cv2.copyMakeBorder(img,
                                 pad_h,
                                 pad_h,
                                 pad_w,
                                 pad_w,
                                 cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
    x1 = w // 2 - size[1] // 2
    y1 = h // 2 - size[0] // 2

    img_pad = img_pad[y1:y1 + size[0], x1:x1 + size[1], :]

    return img_pad


class TrainingDataset(Dataset):
    def __init__(self, path, size, cfg, mask, seed=None
                 ):
        '''
         path - path to directory containing images
                size - (H, W)
                mask - boolean if a mask of the foreground of the umages needs to be computed
                augmentation - dictionnary from config giving info on augmentation
    outputs:
                batch - image N x H x W x 1  intensity scale 0-255
                if mask is True
                also mask N x H x W x 1

        '''

        self.path_directory_original_images = path
        self.list = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.ppm'))]

        # do size something
        self.size = size
        self.mask = mask
        self.cfg = cfg
        self.seed=None
        if seed is not None:
            self.seed=seed

    def __len__(self):
        return len(self.list)


    def __getitem__(self, idx):

        name_image = self.list[idx]
        path_original_image = os.path.join(self.path_directory_original_images, name_image)
        image_full = cv2.imread(path_original_image)
        image_full = center_crop(image_full, self.size)

        if self.cfg['use_green_channel']:
            image = image_full[:, :, 1]
        else:
            image = cv2.cvtColor(image_full, cv2.COLOR_BGR2GRAY)

        if self.mask:
            mask = (image < 230) & (image > 25)


        h1 = homography_sampling(image.shape, self.cfg['sample_homography'], seed=self.seed)
        image1 = cv2.warpPerspective(np.uint8(image), h1, (image.shape[1], image.shape[0]))
        image1, image1_preprocessed = apply_augmentations(image1, self.cfg['augmentation'], seed=self.seed)


        # image2
        h2 = homography_sampling(image.shape, self.cfg['sample_homography'], seed=self.seed)
        image2 = cv2.warpPerspective(np.uint8(image), h2, (image.shape[1], image.shape[0]))
        image2, image2_preprocessed = apply_augmentations(image2, self.cfg['augmentation'], seed=self.seed)

        # homography relatig image1 to image2
        H = np.matmul(h2, np.linalg.inv(h1))

        output = {'image1': image1,
                'image2': image2,
                'image1_preprocessed': image1_preprocessed,
                'image2_preprocessed': image2_preprocessed,
                'H1_to_2': H}

        if mask:
            output['mask'] = mask
        return output