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
import os.path as osp
import tensorflow as tf
import cv2
import os
import random
import shutil
import argparse
import yaml
from sklearn.utils import shuffle
from dataset.training_dataset import TrainingDataset
from model_CNN import non_max_suppression, Unet_model_4
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler

from util.optimize_PWC_flow import train_epoch, validate_epoch
from util.utils_CNN import load_checkpoint, save_checkpoint
from tensorboardX import SummaryWriter


def training_no_test_set(cfg, plot):
    save_path = os.path.join(cfg['write_dir'], cfg['name_exp'])
    # creates the directory to save the images and checkpoints
    os.makedirs(save_path, exist_ok=True)
    shutil.copy2(os.path.basename(__file__), save_path)

    # read information from the yaml file
    nbr = 0
    epochs = cfg['training']['nbr_epochs']
    batch_size = cfg['training']['batch_size']
    distance_threshold = cfg['training']['distance_threshold']
    loss_training = []

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean_vector = np.array([0.485, 0.456, 0.406])
    std_vector = np.array([0.229, 0.224, 0.225])
    normTransform = transforms.Normalize(mean_vector, std_vector)
    dataset_transforms = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(240),
                                             transforms.ToTensor(),
                                             normTransform])


    train_dataset = TrainingDataset(path, size, cfg, mask)

    val_dataset = TrainingDataset(path, size, cfg, mask)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=args.n_threads)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=20,
                                shuffle=False,
                                num_workers=args.n_threads)

    model =


    # Optimizer
    optimizer = \
        optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=args.lr,
                   weight_decay=args.weight_decay)
    # Scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[2, 15, 30, 45, 60],
                                         gamma=0.1)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            # it is pointing directly to the pre trained model file
            path_to_pretrained_model = args.pretrained
        else:
            # it is only a directory, need to find the proper file now
            epochfiles = [f for f in os.listdir(args.pretrained) if f.startswith('epoch_')]
            epoch_sorted = sorted(epochfiles, key=lambda x: int(x.split("_")[1].split(".")[0]))
            path_to_pretrained_model = os.path.join(args.pretrained, epoch_sorted[-1])

        # reload from pre_trained_model
        print('path to pretrained model is {}'.format(path_to_pretrained_model))
        model, optimizer, scheduler, start_epoch, best_val = load_checkpoint(model, optimizer, scheduler,
                                                                             filename=path_to_pretrained_model)
        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        cur_snapshot = os.path.basename(os.path.dirname(args.pretrained))

    else:
        if not os.path.isdir(args.snapshots):
            os.mkdir(args.snapshots)

        cur_snapshot = time.strftime('%Y_%m_%d_%H_%M')

        if not osp.isdir(osp.join(args.snapshots, cur_snapshot)):
            os.mkdir(osp.join(args.snapshots, cur_snapshot))

        with open(osp.join(args.snapshots, cur_snapshot, 'args.pkl'), 'wb') as f:
            pickle.dump(args, f)

        best_val = -1
        start_epoch = 0
        train_losses = []

    # create summary writer
    save_path = osp.join(args.snapshots, cur_snapshot)
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path, 'test', str(i))))

    model = nn.DataParallel(model)
    model = model.to(device)

    # Criterions


    for epoch in range(start_epoch, args.n_epoch):
        scheduler.step()
        print('starting epoch {}: info scheduler last_epoch is {}, learning rate is {}'.format(epoch,
                                                                                               scheduler.last_epoch,
                                                                                               scheduler.get_lr()))
        # Training one epoch
        train_loss = train_epoch(model,
                                 optimizer,
                                 train_dataloader,
                                 device,
                                 save_path=os.path.join(save_path, 'train'),
                                 epoch=epoch,
                                 criterion_grid=criterion_grid,
                                 criterion_matchability=criterion_match,
                                 loss_grid_weights=weights_loss_coeffs)
        train_losses.append(train_loss)
        train_writer.add_scalar('train loss', train_loss, epoch)
        train_writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        print(colored('==> ', 'green') + 'Train average loss:', train_loss)



    # as an indication
    rep_glampoints = []
    homo_glampoints = []
    acceptable_homo_glampoints = []



        # compute this as an indication for the training
        for j in range(len(image1)):
            rep_glampoints.append(metrics_per_image['{}'.format(j)]['repeatability'])
            homo_glampoints.append(metrics_per_image['{}'.format(j)]['homography_correct'])
            acceptable_homo_glampoints.append(metrics_per_image['{}'.format(j)]['class_acceptable'])
        loss_training_epoch.append(np.sum(tr_loss))

    loss_training.append(np.sum(loss_training_epoch))
    # as an indication
    file_metrics_training = open(os.path.join(save_path_test, 'metrics_training.txt'), 'a')
    file_metrics_training.write('epoch {} \n'.format(epoch))
    file_metrics_training.write('training: TF CNN, {} homography found on {}, {} are of \
    acceptable class, rep {}, loss normalised by the number of matches {}\n'.format(np.sum(homo_glampoints),
                                                                                    len(homo_glampoints),
                                                                                    np.sum(acceptable_homo_glampoints),
                                                                                    np.mean(rep_glampoints),
                                                                                    np.sum(loss_training_epoch)))


    # TESTING
    if cfg['testing']['testing'] and epoch % cfg["testing"]["testing_every"] == 0:
        # to gather data
        test_rep_tf = []
        test_homo_tf = []
        test_acceptable_homo_tf = []
        loss_testing_epoch = []

        # Validation
        val_loss_grid, val_mean_epe = validate_epoch(model,
                                                     val_dataloader,
                                                     device,
                                                     output_writers=output_writers,
                                                     save_path=os.path.join(save_path, 'test'),
                                                     epoch=epoch,
                                                     criterion_grid=criterion_grid,
                                                     criterion_matchability=criterion_match,
                                                     loss_grid_weights=weights_loss_coeffs)
        print(colored('==> ', 'blue') + 'Val average grid loss :',
              val_loss_grid)
        print(colored('==> ', 'blue') + 'epoch :', epoch + 1)
        test_writer.add_scalar('mean EPE', val_mean_epe, epoch)
        test_writer.add_scalar('val loss', val_loss_grid, epoch)
        val_losses.append(val_loss_grid)
        val_epe.append(val_mean_epe)



            '''
            if plot and b_t == 0:
                plot_training(test_image1, test_image2, test_kp_map1, test_kp_map2, test_computed_reward1, test_loss,
                              test_mask_batch1, test_metrics_per_image, nbr, epoch, save_path_test,
                              name_to_save='epoch{}_test_batch{}.jpg'.format(epoch, b_t))
            '''

            for j in range(len(test_image1)):
                test_rep_tf.append(test_metrics_per_image['{}'.format(j)]['repeatability'])
                test_homo_tf.append(test_metrics_per_image['{}'.format(j)]['homography_correct'])
                test_acceptable_homo_tf.append(test_metrics_per_image['{}'.format(j)]['class_acceptable'])
            loss_testing_epoch.append(np.sum(test_loss))

        # give the results of the loss + write metrics in file
        loss_testing.append(np.sum(loss_testing_epoch))
        file_metrics_training.write('test: TF CNN, {} homography found on {}, {} are of \
        acceptable class, rep {}, loss normalised by the number of matches {}\n\n '.format(np.sum(test_homo_tf),
        len(test_homo_tf), np.sum(test_acceptable_homo_tf), np.mean(test_rep_tf), np.sum(loss_testing_epoch)))

        if epoch == 0:
            # compute testing for SIFT only once
            test_accepted_SIFT, test_repeatability_SIFT, test_SIFT_class_acceptable=\
                get_sift(test_image1, test_image2, test_homographies)
            test_rep_SIFT.extend(test_repeatability_SIFT)
            test_homo_SIFT.extend(test_accepted_SIFT)
            test_acceptable_homo_SIFT.extend(test_SIFT_class_acceptable)
            file_metrics_training.write('test: SIFT, {} homography found on {}, {} are of acceptable class, rep {}\n'.format(
                np.sum(test_homo_SIFT), len(test_homo_SIFT), np.sum(test_acceptable_homo_SIFT), np.mean(test_rep_SIFT)))

    file_metrics_training.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training the detector')
    parser.add_argument('--path_ymlfile', type=str,
                        help='Path to yaml file.')
    parser.add_argument('--plot', type=bool, default=False,
                        help='Plot during training? (default: False).')

    opt = parser.parse_args()

    with open(opt.path_ymlfile, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    tf.set_random_seed(11)
    np.random.seed(0)
    random.seed(0)

    training_no_test_set(cfg, opt.plot)
