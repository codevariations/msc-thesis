import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt

import random
import shutil
import time
import warnings

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from custom_loss import PoincareXEntropyLoss
from poincare_model import PoincareDistance
import pdb


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
args = parser.parse_args()
#load poincare embedding
poinc_emb = torch.load(
            '/home/hermanni/thesis/msc-thesis/code/model/nouns_id.pth')

# Data loading code
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val_white')

train_dataset = datasets.ImageFolder(
              traindir)

imgnet_classes = train_dataset.classes

#create poincare embedding that only contains imagenet synsets
imgnet2poinc_idx = [poinc_emb['objects'].index(i) for i in imgnet_classes]
imgnet_poinc_wgt = poinc_emb['model']['lt.weight'][[imgnet2poinc_idx]]
imgnet_poinc_labels = [poinc_emb['objects'][i] for i in imgnet2poinc_idx]

import numpy as np
from sklearn.manifold import TSNE
import matplotlib

#X_embedded = TSNE(n_components=2).fit_transform(imgnet_poinc_wgt)
XX = TSNE(n_components=2, n_iter=5000, perplexity=100).fit_transform(poinc_emb['model']['lt.weight'])

plt.scatter(XX[:,0], XX[:,1])












pdb.set_trace()
