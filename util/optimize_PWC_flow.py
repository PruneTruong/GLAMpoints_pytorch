import numpy as np
import cv2

from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils.evaluate import epe
from utils.pixel_wise_mapping import remap_using_flow_fields, remap_using_correspondence_map
from utils_training.multiscale_loss import multiscaleEPE, realEPE
from model_CNN import non_max_suppression
from matplotlib import pyplot as plt
import time
from util.utils_CNN import warp_kp, batch, plot_training, find_true_positive_matches, sift, get_sift
from util.metrics_comparison import get_repeatability, compute_homography, homography_is_accepted, \
    class_homography, compute_registration_error


def compute_reward(image1, image2, kp_map1, kp_map2, homographies, nms, distance_threshold=5, compute_metrics=False):
    reward_batch1 = np.zeros((image1.shape), np.float32)
    mask_batch1 = np.zeros((image1.shape), np.float32)

    metrics_per_image = {}
    SIFT = cv2.xfeatures2d.SIFT_create()

    # computes the reward and mask for each element of the batch
    for i in range(kp_map1.shape[0]):
        # for storing information
        plot = {}
        metrics = {}

        # reward an homography of current image pair
        reward1 = reward_batch1[i, :, :, 0]
        homography = homographies[i, :, :]

        # apply NMS to the score map to get the final kp
        kp_map1_nonmax = non_max_suppression(kp_map1[i, :, :, 0], nms, 0)
        kp_map2_nonmax = non_max_suppression(kp_map2[i, :, :, 0], nms, 0)
        keypoints_map1 = np.where(kp_map1_nonmax > 0)
        keypoints_map2 = np.where(kp_map2_nonmax > 0)

        # transform numpy point to cv2 points and compute the corresponding descriptors
        kp1_array = np.array([keypoints_map1[1], keypoints_map1[0]]).T
        kp2_array = np.array([keypoints_map2[1], keypoints_map2[0]]).T
        kp1_cv2 = [cv2.KeyPoint(kp1_array[i, 0], kp1_array[i, 1], 10) for i in range(len(kp1_array))]
        kp2_cv2 = [cv2.KeyPoint(kp2_array[i, 0], kp2_array[i, 1], 10) for i in range(len(kp2_array))]

        kp1, des1 = sift(SIFT, np.uint8(image1[i, :, :, 0]), kp1_cv2)
        kp2, des2 = sift(SIFT, np.uint8(image2[i, :, :, 0]), kp2_cv2)

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
                mask_batch1[
                    i, kp1_true_positive_matches[:, 1].tolist(), kp1_true_positive_matches[:, 0].tolist(), 0] = 1
                if tp >= fp:
                    # if there are more tp than fp, backpropagate through all matches
                    mask_batch1[i, kp1_false_positive_matches[:, 1].tolist(),
                                kp1_false_positive_matches[:, 0].tolist(), 0] = 1
                else:
                    # otherwise, find a subset of the fp matches of the same size than tp
                    index = random.sample(range(len(kp1_false_positive_matches)), tp)
                    mask_batch1[i, kp1_false_positive_matches[index, 1].tolist(),
                                kp1_false_positive_matches[index, 0].tolist(), 0] = 1

        if compute_metrics:
            # compute metrics as an indication and plot the different steps
            # metrics about estimated homography
            computed_H, ratio_inliers = compute_homography(kp1, kp2, des1, des2, 'SIFT', 0.80)
            tf_accepted = homography_is_accepted(computed_H)
            RMSE, MEE, MAE = compute_registration_error(homography, computed_H, kp_map1[i, :, :, 0].shape)
            found_homography, acceptable_homography = class_homography(MEE, MAE)
            metrics['computed_H'] = computed_H
            metrics['homography_correct'] = tf_accepted
            metrics['inlier_ratio'] = ratio_inliers
            metrics['class_acceptable'] = acceptable_homography

            # repeatability
            if (kp1.shape[0] != 0) and (kp2.shape[0] != 0):
                repeatability = get_repeatability(kp1, kp2, homography, kp_map1[i, :, :, 0].shape)
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

    return reward_batch1, mask_batch1, metrics_per_image


def loss(reward, kpmap, mask):
    loss_matrix = torch.square(reward - kpmap) * mask
    loss = torch.div_no_nan(loss_matrix,
                            torch.sum(mask, axis=[1, 2, 3]), name='division')
    return loss
def train_epoch(net,
                optimizer,
                train_loader,
                device,
                epoch,
                train_writer,

                save_path=None,
                ):
    """
    Training epoch script
    Args:
        net: model architecture
        optimizer: optimizer to be used for traninig `net`
        train_loader: dataloader
        device: `cpu` or `gpu`
        criterion_grid: criterion for esimation pixel correspondence (L1Masked)
        criterion_matchability: criterion for mask optimization
        loss_grid_weights: weight coefficients for each grid estimates tensor
            for each level of the feature pyramid
        L_coeff: weight coefficient to balance `criterion_grid` and
            `criterion_matchability`
    Output:
        running_total_loss: total training loss

        here outptu at every level is flow inteprolated but not scaled. we only use the groudn truth flow as higest
        resolution and downsample it without scaling alos
    """
    n_iter = epoch*len(train_loader)
    # everywhere when they say flow it is actuallt mapping
    net.train()
    running_total_loss = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, mini_batch in pbar:
        image1 = mini_batch['image1'].to(device)
        image2 = mini_batch['image2'].to(device)
        image1_normed = mini_batch['image1_normed'].to(device)
        image2_normed = mini_batch['image2_normed'].to(device)
        homographies = mini_batch['H1_to_2']

        optimizer.zero_grad()

        kp_map2 = net(image2_normed)
        kp_map1 = net(image1_normed)

        # backpropagation will not go through this
        computed_reward1, mask_batch1, metrics_per_image = compute_reward(image1, image2, kp_map1.clone(), kp_map2.clone(),
                                                                          homographies, nms,
                                                                          distance_threshold=distance_threshold,
                                                                          compute_metrics=True)

        # loss calculation
        Loss = loss(reward=computed_reward1, kpmap=kp_map1, mask=mask_batch1)
        Loss.backward()
        optimizer.step()

        if plot and b % cfg["training"]["plot_every_x_batches"] == 0:
            plot_training(image1, image2, kp_map1, kp_map2, computed_reward1, tr_loss, mask_batch1,
                          metrics_per_image, nbr, epoch, save_path,
                          name_to_save='epoch{}_batch{}.jpg'.format(epoch, b))

        running_total_loss += Loss.item()
        train_writer.add_scalar('train_loss_per_iter', Loss.item(), n_iter)
        n_iter += 1
        pbar.set_description(
                'training: R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1),
                                             Loss.item()))
    running_total_loss /= len(train_loader)
    return running_total_loss


def validate_epoch(net,
                   val_loader,
                   device,
                   epoch,
                   save_path,
                   div_flow=1,
                   loss_grid_weights=None,
                   apply_mask=False):
    """
    Validation epoch script
    Args:
        net: model architecture
        val_loader: dataloader
        device: `cpu` or `gpu`
        criterion_grid: criterion for esimation pixel correspondence (L1Masked)
        criterion_matchability: criterion for mask optimization
        loss_grid_weights: weight coefficients for each grid estimates tensor
            for each level of the feature pyramid
        L_coeff: weight coefficient to balance `criterion_grid` and
            `criterion_matchability`
    Output:
        running_total_loss: total validation loss
    """

    net.eval()

    running_total_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        aepe_array=[]
        for i, mini_batch in pbar:




            running_total_loss += Loss.item()
            pbar.set_description(
                ' validation R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1),
                                             Loss.item()))
        mean_epe = np.mean(aepe_array)

    return running_total_loss / len(val_loader), mean_epe
