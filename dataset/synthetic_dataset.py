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


class SyntheticDataset(Dataset):
    def __init__(self, cfg, train=True):
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

        self.cfg = cfg
        self.seed = cfg['training']['seed']
        self.training = train
        if train:
            self.root_dir = os.path.join(cfg['training']['TRAIN_DIR'])
            if cfg['training']['train_list'] != ' ':
                self.list_original_images = []
                self.list_original_images = open(cfg['training']['train_list']).read().splitlines()
            else:
                self.list_original_images = [f for f in os.listdir(self.root_dir) if
                                             f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.ppm'))]
        else:
            # we are evaluating
            self.root_dir = os.path.join(cfg['validation']['VAL_DIR'])
            if cfg['validation']['val_list'] != ' ':
                self.list_original_images = []
                self.list_original_images = open(cfg['validation']['val_list']).read().splitlines()
            else:
                self.list_original_images = [f for f in os.listdir(self.root_dir) if
                                             f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.ppm'))]
        #self.mask = mask

    def __len__(self):
        return len(self.list_original_images)

    def __getitem__(self, idx):

        name_image = self.list_original_images[idx]
        path_original_image = os.path.join(self.root_dir, name_image)

        # reads the image
        image_full = cv2.imread(path_original_image)

        # crops it to the desired size
        if self.training:
            image = center_crop(image_full, (self.cfg['training']['image_size_h'], self.cfg['training']['image_size_w']))
        else:
            image = center_crop(image_full,
                                (self.cfg['validation']['image_size_h'], self.cfg['validation']['image_size_w']))

        # apply correct preprocessing
        if self.cfg['augmentation']['use_green_channel']:
            image = image[:, :, 1]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        '''
        if self.mask:
            mask = (image < 230) & (image > 25)
        '''

        # sample homography and creates image1
        h1 = homography_sampling(image.shape, self.cfg['sample_homography'], seed=self.seed * (idx + 1))
        image1 = cv2.warpPerspective(np.uint8(image), h1, (image.shape[1], image.shape[0]))
        # applies appearance augmentations to image1, the seed is fixed so that results are reproducible
        image1, image1_preprocessed = apply_augmentations(image1, self.cfg['augmentation'], seed=self.seed * (idx + 1))

        # sample homography and creates image2
        h2 = homography_sampling(image.shape, self.cfg['sample_homography'], seed=self.seed * (idx + 2))
        image2 = cv2.warpPerspective(np.uint8(image), h2, (image.shape[1], image.shape[0]))
        # applies appearance augmentations to image1
        image2, image2_preprocessed = apply_augmentations(image2, self.cfg['augmentation'], seed=self.seed * (idx + 2))

        # homography relatig image1 to image2
        H = np.matmul(h2, np.linalg.inv(h1))

        output = {'image1': torch.Tensor(image1.astype(np.int32)).unsqueeze(0), # put the images (gray) so that batch will be Bx1xHxW
                  'image2': torch.Tensor(image2.astype(np.int32)).unsqueeze(0),
                  'image1_normed': torch.Tensor(image1_preprocessed.astype(np.float32)).unsqueeze(0),
                  'image2_normed': torch.Tensor(image2_preprocessed.astype(np.float32)).unsqueeze(0),
                  'H1_to_2': torch.Tensor(H.astype(np.float32))}

        '''
        if mask:
            output['mask'] = torch.Tensor(mask.astype(np.uint8))
        '''
        return output