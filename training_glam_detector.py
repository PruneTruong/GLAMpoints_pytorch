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


import pickle
import numpy as np
import os.path as osp
import os
import random
import shutil
import argparse
import yaml
from termcolor import colored
from dataset.synthetic_dataset import SyntheticDataset
from models.Unet_model import UNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils_training.optimize import train_epoch, validate_epoch
from utils_training.utils_CNN import load_checkpoint, save_checkpoint
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Training the detector')
parser.add_argument('--path_ymlfile', type=str, default='config_training.yaml',
                    help='Path to yaml file.')
parser.add_argument('--compute_metrics', type=bool, default=True,
                    help='Compute metrics and plot during training? (default: True).')

args = parser.parse_args()

with open(args.path_ymlfile, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
compute_metrics = args.compute_metrics

random.seed(cfg['training']['seed'])
np.random.seed(cfg['training']['seed'])
torch.manual_seed(cfg['training']['seed'])
torch.cuda.manual_seed(cfg['training']['seed'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = SyntheticDataset(cfg=cfg, train=True)
val_dataset = SyntheticDataset(cfg=cfg, train=False)
train_dataloader = DataLoader(train_dataset,
                              batch_size=cfg['training']['batch_size'],
                              shuffle=True,
                              num_workers=cfg['training']['n_threads'])

val_dataloader = DataLoader(val_dataset,
                            batch_size=cfg['validation']['batch_size'],
                            shuffle=False,
                            num_workers=cfg['training']['n_threads'])

# defines the model
model = UNet()


# Optimizer
optimizer = \
    optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
               lr=cfg['training']['learning_rate'],
               weight_decay=0)
'''
torch with weight decay is equivelent to tensorflow tfa.optimizers.AdamW
we used tf.train.AdamOptimizer, which does not have weight decay. however, i dont know what is eauivelent to
the regularisation function in layers of tensorflow
'''

if cfg['training']['load_pre_training']:
    path_to_pretrained_model = cfg['training']['pre_trained_weights']
    if not os.path.isfile(cfg['training']['pre_trained_weights']):
        raise ValueError('The path to the model pre-trained weights you indicated does not exist !')

    # reload from pre_trained_model
    print('path to pretrained model is {}'.format(path_to_pretrained_model))
    model, optimizer, start_epoch, best_val = load_checkpoint(model, optimizer, filename=path_to_pretrained_model)
    # now individually transfer the optimizer parts...
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    save_path = os.path.join(cfg['write_dir'], cfg['name_exp'])

else:
    save_path = os.path.join(cfg['write_dir'], cfg['name_exp'])
    # creates the directory to save the images and checkpoints
    if not osp.isdir(save_path):
        os.mkdir(save_path)

    with open(osp.join(save_path, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    best_val = -1
    start_epoch = 0

# create summary writer
shutil.copy2(os.path.basename(__file__), save_path)
train_writer = SummaryWriter(os.path.join(save_path, 'train'))
val_writer = SummaryWriter(os.path.join(save_path, 'val'))

model = nn.DataParallel(model)
model = model.to(device)

# Criterions
for epoch in range(start_epoch, cfg['training']['nbr_epochs']):
    print('starting epoch {}'.format(epoch))
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
                             compute_metrics=compute_metrics,
                             save_path=os.path.join(save_path, 'train'))
    train_writer.add_scalar('train loss', train_loss, epoch)
    print(colored('==> ', 'green') + 'Train average loss:', train_loss)


    # TESTING
    if cfg['validation']['validation'] and epoch % cfg["validation"]["validation_every"] == 0:
        # Validation
        val_loss = validate_epoch(model,
                                  val_dataloader,
                                  val_writer,
                                  cfg,
                                  device,
                                  epoch,
                                  nms,
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
                     'best_loss': best_val},
                    is_best, save_path, 'epoch_{}.pth'.format(epoch + 1))
