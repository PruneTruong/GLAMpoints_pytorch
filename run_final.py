# coding: utf-8

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
from models.glampoints import GLAMpointsInference
import json
import argparse
import torch
import os
import yaml
import cv2
from utils.metrics import evaluate



parser = argparse.ArgumentParser(description='Computing metrics for a dataset')
# Paths
parser.add_argument('--data_dir', metavar='DIR', type=str,
                    default='/scratch_net/biwidl214/truongp/dataset/synthetic_dataset/CityScape/bonn',
                    help='path to folder containing images and flows')
parser.add_argument('--pre_trained_weights',
                    help='path to pre_trained_weights')
parser.add_argument('--NMS', type=int, default=10,
                    help='Value of the NMS window applied on the score map output of GLAMpointsInference (default:10)')
parser.add_argument('--min_prob', type=float, default=0.0,
                    help='Minimum probability of a keypoint for GLAMpointsInference (default:0)')
parser.add_argument('--batch-size', type=int, default=1,
                    help='evaluation batch size')
parser.add_argument('--save_dir', type=str, default='evaluation/',
                    help='path to directory to save the text files and results')
parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')

args = parser.parse_args()

torch.cuda.empty_cache()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # either gpu or cpu

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
with torch.no_grad():
    model = GLAMpointsInference(path_weights=args.pre_trained_weights, nms=args.NMS, min_pron=args.min_prob)
    test_set = TestDataset(args.data_dir, first_image_transform=input_transform,
                                                           second_image_transform=input_transform,
                                                           target_transform=target_transform,
                                                           co_transform=co_transform, split=0)  # only test

    test_dataloader = DataLoader(test_set,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=8)

    average_results = evaluate(test_dataloader, model, name_method='GLAMpointsInference')

with open('{}/{}.txt'.format(args.save_dir, 'average_results'), 'w') as outfile:
    json.dump(average_results, outfile, ensure_ascii=False, separators=(',', ':'))
    print('written to file ')

