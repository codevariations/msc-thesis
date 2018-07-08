import torch
import torchvision
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision import transforms, datasets
from torch.autograd import Function
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
from HEInception3 import *
import pdb
from poincare_model import PoincareDistance

#define data loader
data_transforms = {
                'train': transforms.Compose([
                           transforms.RandomResizedCrop(299),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()]),
                'val': transforms.Compose([
                         transforms.Resize(300),
                         transforms.CenterCrop(299),
                         transforms.ToTensor()])}

data_dir = "/home/hege/Documents/Thesis/msc-thesis/small_data/"
img_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                                        for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(img_datasets[x],
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=4) for x
                                              in ['train', 'val']}
dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}
class_names = img_datasets['train'].classes

emb_dir = '/home/hege/Documents/Thesis/msc-thesis/code/poincare-embeddings-fbresearch/mammals.pth'
poinc_emb = torch.load(emb_dir)
emb_matrix = poinc_emb['model']['lt.weight']
emb_labels = poinc_emb['objects']

format_lables = [emb_labels[x].split('.', 1)[0].replace('_',' ') for x in
        range(len(emb_labels))]

inp, _ = next(iter(dataloaders['val']))

model = he_inception_v3(pretrained=True)
out, out2 = model(inp)

#manual implementation of poincare distance

#class PoincareDist(Function):
#    boundary = 1-eps
#
#    def forward(self, u, v):
#        self.squnorm = th.clamp(th.sum(u*u, dim=-1), 0, self.boundary)
#        self.sqvnorm = th.clamp(th.sum(v*v, dim=-1), 0, self.boundary)
#        self.sqdist = th.sum(th.pow(u-v, 2), dim=-1)
#        x = self.sqdist / ((1-self.squnorm)*(1-self.sqvnorm)) * 2 + 1
#        z = th.sqrt(th.pow(x, 2) - 1)
#        return th.log(x+z)


pdb.set_trace()








