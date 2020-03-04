import numpy as np
import cv2
from utils_training.loss import compute_reward, compute_loss
from tqdm import tqdm
import torch
from utils_training.utils_CNN import plot_training


def train_epoch(net,
                optimizer,
                train_loader,
                train_writer,
                cfg,
                device,
                epoch,
                nms,
                compute_metrics=False,
                save_path=None):
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

    if compute_metrics:
        nbr_images = 0.0
        nbr_homo_correct = 0.0
        nbr_homo_accept = 0.0
        nbr_tp = 0.0
        nbr_kp = 0.0
        add_repeatability = 0.0

        '''
        # puts them to zero at the beginning of the epoch
        train_writer.add_scalar('cumulative_ratio_correct_homographies_per_iter', nbr_homo_correct, n_iter)
        train_writer.add_scalar('cumulative_ratio_acceptable_homographies_per_iter', nbr_homo_accept, n_iter)
        train_writer.add_scalar('cumulative_average_number_of_kp_per_iter', nbr_kp, n_iter)
        train_writer.add_scalar('cumulative_average_number_of_tp_per_iter', nbr_tp, n_iter)
        train_writer.add_scalar('cumulative_repeatability_per_iter', add_repeatability, n_iter)
        '''

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, mini_batch in pbar:
        # all the images are dimneison Bx1xHxW
        image1 = mini_batch['image1'].to(device)
        image2 = mini_batch['image2'].to(device)
        image1_normed = mini_batch['image1_normed'].to(device)
        image2_normed = mini_batch['image2_normed'].to(device)
        homographies = mini_batch['H1_to_2']

        optimizer.zero_grad()

        kp_map1 = net(image1_normed)
        kp_map2 = net(image2_normed)

        # backpropagation will not go through this
        if compute_metrics:
            computed_reward1, mask_batch1, metrics_per_image = compute_reward(image1, image2, kp_map1.clone().detach(),
                                                                              kp_map2.clone().detach(),
                                                                              homographies, nms,
                                                                              distance_threshold=cfg['training']['distance_threshold'],
                                                                              device=device,
                                                                              compute_metrics=compute_metrics)

            nbr_images += metrics_per_image['nbr_images']
            nbr_homo_correct += metrics_per_image['nbr_homo_correct']
            nbr_homo_accept += metrics_per_image['nbr_homo_acceptable']
            nbr_tp += metrics_per_image['nbr_tp']
            nbr_kp += metrics_per_image['nbr_kp']
            add_repeatability += metrics_per_image['sum_rep']

            train_writer.add_scalar('cumulative_ratio_correct_homographies_per_iter', float(nbr_homo_correct) / nbr_images,
                                  n_iter)
            train_writer.add_scalar('cumulative_ratio_acceptable_homographies_per_iter', float(nbr_homo_accept) / nbr_images,
                                  n_iter)
            train_writer.add_scalar('cumulative_average_number_of_kp_per_iter', float(nbr_kp) / nbr_images, n_iter)
            train_writer.add_scalar('cumulative_average_number_of_tp_per_iter', float(nbr_tp) / nbr_images, n_iter)
            train_writer.add_scalar('cumulative_repeatability_per_iter', float(add_repeatability) / nbr_images, n_iter)
        else:
            computed_reward1, mask_batch1 = compute_reward(image1, image2, kp_map1.clone().detach(), kp_map2.clone().detach(),
                                                           homographies, nms,
                                                           distance_threshold=cfg['training']['distance_threshold'],
                                                           device=device, compute_metrics=compute_metrics)

        # loss calculation
        Loss = compute_loss(reward=computed_reward1, kpmap=kp_map1, mask=mask_batch1)
        Loss.backward()
        optimizer.step()

        if i % cfg["training"]["plot_every_x_batches"] == 0 and compute_metrics:
            plot_training(image1.cpu().numpy().squeeze(), image2.cpu().numpy().squeeze(),
                          kp_map1.detach().cpu().numpy().squeeze(), kp_map2.detach().cpu().numpy().squeeze(),
                          computed_reward1.cpu().numpy().squeeze(), Loss.item(),
                          mask_batch1.cpu().numpy().squeeze(), metrics_per_image, epoch, save_path,
                          name_to_save='epoch{}_batch{}.jpg'.format(epoch, i))

        running_total_loss += Loss.item()
        train_writer.add_scalar('train_loss_per_iter', Loss.item(), n_iter)
        n_iter += 1
        pbar.set_description(
                'training: R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1),
                                             Loss.item()))
    running_total_loss /= len(train_loader)
    return running_total_loss


def validate_epoch(net,
                optimizer,
                val_loader,
                val_writer,
                cfg,
                device,
                epoch,
                nms,
                compute_metrics=False,
                save_path=None):
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

    n_iter = epoch * len(val_loader)
    net.eval()

    running_total_loss = 0

    if compute_metrics:
        nbr_images = 0.0
        nbr_homo_correct = 0.0
        nbr_homo_accept = 0.0
        nbr_tp = 0.0
        nbr_kp = 0.0
        add_repeatability = 0.0

        '''
        # puts them to zero at the beginning of the epoch
        val_writer.add_scalar('cumulative_ratio_correct_homographies_per_iter', nbr_homo_correct, n_iter)
        val_writer.add_scalar('cumulative_ratio_acceptable_homographies_per_iter', nbr_homo_accept, n_iter)
        val_writer.add_scalar('cumulative_average_number_of_kp_per_iter', nbr_kp, n_iter)
        val_writer.add_scalar('cumulative_average_number_of_tp_per_iter', nbr_tp, n_iter)
        val_writer.add_scalar('cumulative_repeatability_per_iter', add_repeatability, n_iter)
        '''

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        for i, mini_batch in pbar:
            image1 = mini_batch['image1'].to(device)
            image2 = mini_batch['image2'].to(device)
            image1_normed = mini_batch['image1_normed'].to(device)
            image2_normed = mini_batch['image2_normed'].to(device)
            homographies = mini_batch['H1_to_2']

            kp_map2 = net(image2_normed)
            kp_map1 = net(image1_normed)

            # backpropagation will not go through this
            if compute_metrics:
                computed_reward1, mask_batch1, metrics_per_image = compute_reward(image1, image2,
                                                                                   kp_map1.clone().detach(),
                                                                                   kp_map2.clone().detach(),
                                                                                   homographies, nms,
                                                                                   distance_threshold=cfg['training']['distance_threshold'],
                                                                                   device=device,
                                                                                   compute_metrics=compute_metrics)

                nbr_images += metrics_per_image['nbr_images']
                nbr_homo_correct += metrics_per_image['nbr_homo_correct']
                nbr_homo_accept += metrics_per_image['nbr_homo_acceptable']
                nbr_tp += metrics_per_image['nbr_tp']
                nbr_kp += metrics_per_image['nbr_kp']
                add_repeatability += metrics_per_image['sum_rep']

                val_writer.add_scalar('cumulative_ratio_correct_homographies_per_iter', float(nbr_homo_correct)/nbr_images, n_iter)
                val_writer.add_scalar('cumulative_ratio_acceptable_homographies_per_iter', float(nbr_homo_accept)/nbr_images, n_iter)
                val_writer.add_scalar('cumulative_average_number_of_kp_per_iter', float(nbr_kp)/nbr_images, n_iter)
                val_writer.add_scalar('cumulative_average_number_of_tp_per_iter', float(nbr_tp)/nbr_images, n_iter)
                val_writer.add_scalar('cumulative_repeatability_per_iter', float(add_repeatability)/nbr_images, n_iter)

            else:
                computed_reward1, mask_batch1 = compute_reward(image1, image2, kp_map1.clone(), kp_map2.clone(),
                                                               homographies, nms,
                                                               distance_threshold=cfg['training']['distance_threshold'],
                                                               device=device,
                                                               compute_metrics=compute_metrics)

            # loss calculation
            Loss = compute_loss(reward=computed_reward1, kpmap=kp_map1, mask=mask_batch1)

            if i < 2 and compute_metrics:
                plot_training(image1.cpu().numpy().squeeze(), image2.cpu().numpy().squeeze(),
                              kp_map1.detach().cpu().numpy().squeeze(), kp_map2.detach().cpu().numpy().squeeze(),
                              computed_reward1.cpu().numpy().squeeze(), Loss.item(),
                              mask_batch1.cpu().numpy().squeeze(), metrics_per_image, epoch, save_path,
                              name_to_save='epoch{}_batch{}.jpg'.format(epoch, i))

            running_total_loss += Loss.item()
            val_writer.add_scalar('val_loss_per_iter', Loss.item(), n_iter)
            n_iter += 1
            pbar.set_description(
                'validation: R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1),
                                                       Loss.item()))
        running_total_loss /= len(val_loader)
        if compute_metrics:
            val_writer.add_scalar('ratio_correct_homographies_per_epoch',
                                  float(nbr_homo_correct) / nbr_images, epoch)
            val_writer.add_scalar('ratio_acceptable_homographies_per_epoch',
                                  float(nbr_homo_accept) / nbr_images, epoch)
            val_writer.add_scalar('average_number_of_kp_per_epoch', float(nbr_kp) / nbr_images, epoch)
            val_writer.add_scalar('average_number_of_tp_per_epoch', float(nbr_tp) / nbr_images, epoch)
            val_writer.add_scalar('repeatability_per_epoch', float(add_repeatability) / nbr_images, epoch)

    return running_total_loss / len(val_loader)
