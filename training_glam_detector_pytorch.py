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



#https://discuss.pytorch.org/t/why-cant-i-reimplement-my-tensorflow-model-with-pytorch/44347/4
import time
import pickle
import numpy as np
import os.path as osp
import tensorflow as tf
import os
import random
import shutil
import argparse
import yaml
from termcolor import colored
from dataset.training_dataset import TrainingDataset
from models.Unet_model import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from utils_training.optimize import train_epoch, validate_epoch
from utils_training.utils_CNN import load_checkpoint, save_checkpoint
from tensorboardX import SummaryWriter


def training_no_test_set(cfg, compute_metrics):

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

    # read information from the yaml file
    nbr = 0
    epochs = cfg['training']['nbr_epochs']
    batch_size = cfg['training']['batch_size']
    distance_threshold = cfg['training']['distance_threshold']

    train_dataset = TrainingDataset(path= opt['path_images_training'], size=(cfg['training']['image_size_h'],
                                                                             cfg['training']['image_size_w']),
                                    cfg=cfg, mask=False)

    val_dataset = TrainingDataset(path= opt['path_images_testing'], size=(cfg['testing']['image_size_h'],
                                                                          cfg['testing']['image_size_w']),
                                  cfg=cfg, mask=False)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=args.n_threads)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=20,
                                shuffle=False,
                                num_workers=args.n_threads)

    model = UNet()
    # attention, expect inputs to have channel first ! 

    # Optimizer
    optimizer = \
        optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=cfg['training']['learning_rate'],
                   weight_decay=0) #torch with weight decay is equivelent to tensorflow tfa.optimizers.AdamW
    # we used tf.train.AdamOptimizer, which does not have weight decay. however, i dont know what is eauivelent to
    # the regularisation function in layers of tensorflow


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
    save_path = os.path.join(cfg['write_dir'], cfg['name_exp'])
    # creates the directory to save the images and checkpoints
    os.makedirs(save_path, exist_ok=True)
    shutil.copy2(os.path.basename(__file__), save_path)
    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    val_writer = SummaryWriter(os.path.join(save_path, 'val'))

    model = nn.DataParallel(model)
    model = model.to(device)

    # Criterions
    for epoch in range(start_epoch, epochs):
        scheduler.step()
        print('starting epoch {}: info scheduler last_epoch is {}, learning rate is {}'.format(epoch,
                                                                                               scheduler.last_epoch,
                                                                                               scheduler.get_lr()))
        if epoch == 0:
            nms = cfg['training']['NMS_epoch0']
        else:
            nms = cfg['training']['NMS_others']

        # Training one epoch
        train_loss = train_epoch(model,
                                 optimizer,
                                 train_dataloader,
                                 train_writer,
                                 cfg,
                                 device,
                                 epoch,
                                 nms,
                                 distance_threshold=distance_threshold,
                                 compute_metrics=compute_metrics,
                                 save_path=os.path.join(save_path, 'train'))
        train_writer.add_scalar('train loss', train_loss, epoch)
        train_writer.add_scalar('learning_rate', scheduler.get_lr()[0], epoch)
        print(colored('==> ', 'green') + 'Train average loss:', train_loss)


        # TESTING
        if cfg['testing']['testing'] and epoch % cfg["testing"]["testing_every"] == 0:
            # Validation
            val_loss = validate_epoch(model,
                                      optimizer,
                                      val_dataloader,
                                      val_writer,
                                      cfg,
                                      device,
                                      epoch,
                                      nms,
                                      distance_threshold=distance_threshold,
                                      compute_metrics=compute_metrics,
                                      save_path=os.path.join(save_path, 'val'))

            print(colored('==> ', 'blue') + 'Val average grid loss :',
                  val_loss)
            print(colored('==> ', 'blue') + 'epoch :', epoch + 1)
            val_writer.add_scalar('val loss', val_loss, epoch)

            if best_val < 0:
                best_val = val_loss

            is_best = val_loss < best_val
            best_val = min(val_loss, best_val)
        else:
            is_best = False
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'best_loss': best_val},
                        is_best, save_path, 'epoch_{}.pth'.format(epoch + 1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training the detector')
    parser.add_argument('--path_ymlfile', type=str,
                        help='Path to yaml file.')
    parser.add_argument('--compute_metrics', type=bool, default=False,
                        help='Plot during training? (default: False).')

    opt = parser.parse_args()

    with open(opt.path_ymlfile, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    tf.set_random_seed(11)
    np.random.seed(0)
    random.seed(0)

    training_no_test_set(cfg, opt.compute_metrics)
