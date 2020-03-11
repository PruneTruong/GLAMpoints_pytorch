import torch
import cv2
from utils_training.metrics_comparison import get_repeatability, compute_homography, homography_is_accepted, \
    class_homography, compute_registration_error
from models.glampoints import non_max_suppression
from utils_training.utils_CNN import warp_kp, find_true_positive_matches, sift
import numpy as np
import random


def compute_reward(image1, image2, kp_map1, kp_map2, homographies, nms, distance_threshold, device, compute_metrics=False):
    '''
    parameters:
        - image1: torch Tensor of size Bx1xHxW
        - image2: torch Tensor of size Bx1xHxW
        - kp_map1: torch Tensor of size Bx1xHxW
        - kp_map2: torch Tensor of size Bx1xHxW
        - homographies: torch Tensor of size Bx3x3
        - nms:
        - distance_threshold:
        - compute_metrics:
    :return:
        - reward_batch1: torch Tensor of size Bx1xHxW
        - mask_batch1: torch Tensor of size Bx1xHxW
        - optional metrics_per_image: dict containing the resuls
    '''

    # convert torch Tensor into numpy arrays, and put it in BxHxWxC
    kp_map1, kp_map2 = kp_map1.cpu().numpy().squeeze(), kp_map2.cpu().numpy().squeeze() # now BxHxW, before Bx1xHxW
    image1, image2 = image1.cpu().numpy().squeeze(), image2.cpu().numpy().squeeze() # now BxHxW, before Bx1xHxW because images are grayscale
    homographies = homographies.cpu().numpy()

    # created the reward and mask matrix
    reward_batch1 = np.zeros((image1.shape), np.float32)
    mask_batch1 = np.zeros((image1.shape), np.float32)

    # list that will store the metrics
    rep_glampoints = []
    homo_glampoints = []
    acceptable_homo_glampoints = []
    kp_glampoints = []
    tp_glampoints = []
    metrics_per_image = {}

    SIFT = cv2.xfeatures2d.SIFT_create()

    # computes the reward and mask for each element of the batch
    for i in range(kp_map1.shape[0]):
        shape = kp_map1[i, :, :].shape
        # for storing information
        plot = {}
        metrics = {}

        # reward an homography of current image pair
        reward1 = reward_batch1[i, :, :]
        homography = homographies[i, :, :]

        # apply NMS to the score map to get the final kp
        kp_map1_nonmax = non_max_suppression(kp_map1[i, :, :], nms, 0)
        kp_map2_nonmax = non_max_suppression(kp_map2[i, :, :], nms, 0)
        keypoints_map1 = np.where(kp_map1_nonmax > 0)
        keypoints_map2 = np.where(kp_map2_nonmax > 0)

        # transform numpy point to cv2 points and compute the corresponding descriptors
        kp1_array = np.array([keypoints_map1[1], keypoints_map1[0]]).T
        kp2_array = np.array([keypoints_map2[1], keypoints_map2[0]]).T
        kp1_cv2 = [cv2.KeyPoint(kp1_array[i, 0], kp1_array[i, 1], 10) for i in range(len(kp1_array))]
        kp2_cv2 = [cv2.KeyPoint(kp2_array[i, 0], kp2_array[i, 1], 10) for i in range(len(kp2_array))]

        kp1, des1 = sift(SIFT, np.uint8(image1[i, :, :]), kp1_cv2)
        kp2, des2 = sift(SIFT, np.uint8(image2[i, :, :]), kp2_cv2)

        # reconverts the cv2 kp into numpy, because descriptor might have removed points
        kp1 = np.array([m.pt for m in kp1], dtype=np.int32)
        kp2 = np.array([m.pt for m in kp2], dtype=np.int32)

        # compute the reward and the mask
        if des1 is not None and des2 is not None:
            if des1.shape[0] > 2 and des2.shape[0] > 2:
                # match descriptors
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                matches1 = bf.match(des1, des2)
                tp, fp, true_positive_matches1, kp1_true_positive_matches, kp1_false_positive_matches = \
                    find_true_positive_matches(kp1, kp2, matches1, homography, distance_threshold=distance_threshold)

                # reward and mask equal to 1 at the position of the TP keypoints
                reward1[kp1_true_positive_matches[:, 1].tolist(), kp1_true_positive_matches[:, 0].tolist()] = 1
                mask_batch1[i, kp1_true_positive_matches[:, 1].tolist(), kp1_true_positive_matches[:, 0].tolist()] = 1
                if tp >= fp:
                    # if there are more tp than fp, backpropagate through all matches
                    mask_batch1[i, kp1_false_positive_matches[:, 1].tolist(), kp1_false_positive_matches[:, 0].tolist()] = 1.0
                else:
                    # otherwise, find a subset of the fp matches of the same size than tp
                    index = random.sample(range(len(kp1_false_positive_matches)), tp)
                    mask_batch1[i, kp1_false_positive_matches[index, 1].tolist(),
                                kp1_false_positive_matches[index, 0].tolist()] = 1.0

        if compute_metrics:
            # compute metrics as an indication and plot the different steps

            # metrics about estimated homography
            computed_H, ratio_inliers = compute_homography(kp1, kp2, des1, des2, 'SIFT', 0.80)
            tf_accepted = homography_is_accepted(computed_H)
            RMSE, MEE, MAE = compute_registration_error(homography, computed_H, shape)
            found_homography, acceptable_homography = class_homography(MEE, MAE)
            metrics['computed_H'] = computed_H
            metrics['homography_correct'] = tf_accepted
            metrics['inlier_ratio'] = ratio_inliers
            metrics['class_acceptable'] = acceptable_homography

            # computes and stores repeatability
            if (kp1.shape[0] != 0) and (kp2.shape[0] != 0):
                repeatability = get_repeatability(kp1, kp2, homography, shape)
            else:
                repeatability = 0
            metrics['repeatability'] = repeatability

            # original kp after NMS
            # results same shape than np.where,
            # Nx2, [0] contains coordinate in vertical direction, [1] in horizontal direction (corresponds to x)
            plot['keypoints_map1'] = keypoints_map1
            plot['keypoints_map2'] = keypoints_map2
            metrics['nbr_kp1'] = len(keypoints_map1[0])
            metrics['nbr_kp2'] = len(keypoints_map2[0])

            # true positive kp: results same shape than np.where,
            # Nx2, [0] contains coordinate in vertical direction, [1] in horizontal direction (corresponds to x)
            tp_kp1 = kp1_true_positive_matches.T[[1,0], :]

            # warped tp kp: results same shape than np.where,
            # Nx2, [0] contains coordinate in vertical direction, [1] in horizontal direction (corresponds to x)
            if len(tp_kp1[1]) != 0:
                where_warped_tp_kp1 = warp_kp(tp_kp1, homography, (kp_map1.shape[1], kp_map1.shape[2]))
            else:
                where_warped_tp_kp1 = np.zeros((2, 1))

            plot['tp_kp1'] = tp_kp1
            plot['warped_tp_kp1'] = where_warped_tp_kp1
            metrics['total_nbr_kp_reward1'] = np.sum(reward1)

            metrics['to_plot'] = plot
            metrics_per_image['{}'.format(i)] = metrics

            rep_glampoints.append(repeatability)
            homo_glampoints.append(tf_accepted)
            acceptable_homo_glampoints.append(acceptable_homography)
            kp_glampoints.append(metrics['nbr_kp1'])
            tp_glampoints.append(metrics['total_nbr_kp_reward1'])

    # from numpy array, reconvert to torch Tensor, put them in cuda and the right dimention Bx1xHxW
    reward_batch1 = torch.from_numpy(reward_batch1).unsqueeze(1).to(device)
    mask_batch1 = torch.from_numpy(mask_batch1).unsqueeze(1).to(device)
    del SIFT

    if compute_metrics:
        metrics_per_image['sum_rep'] = np.sum(rep_glampoints)
        metrics_per_image['nbr_homo_correct'] = np.sum(homo_glampoints)
        metrics_per_image['nbr_homo_acceptable'] = np.sum(acceptable_homography)
        metrics_per_image['nbr_kp'] = np.sum(kp_glampoints)
        metrics_per_image['nbr_tp'] = np.sum(tp_glampoints)
        metrics_per_image['nbr_images'] = len(tp_glampoints)
        return reward_batch1, mask_batch1, metrics_per_image
    else:
        return reward_batch1, mask_batch1


def compute_loss(reward, kpmap, mask):
    '''

    :param reward: torch Tensor of shape Bx1xHxW
    :param kpmap: torch Tensor of shape Bx1xHxW
    :param mask: torch Tensor of shape Bx1xHxW, float32
    :return:
    '''

    assert reward.size() == kpmap.size()
    assert reward.size() == mask.size()

    loss_matrix = (reward - kpmap)**2 * mask
    loss = loss_matrix / (torch.sum(mask, axis=[1,2,3]).unsqueeze(1).unsqueeze(1).unsqueeze(1)+1e-6)
    # attention here, the loss is still a matrix, divided by the number of mask element per image.
    return loss.sum()