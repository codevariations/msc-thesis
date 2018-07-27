import os
import argparse
os.environ["CUDA_DEVICE_ORDER"]
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

import random
import shutil
import time
import warnings

import pandas as pd






import nltk
from nltk.corpus import wordnet as wn

import pdb

zsl_dir = '/fast-data/datasets/imagenet/fa2011'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
zsl_data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(zsl_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,])), batch_size=256, shuffle=False, num_workers=20,
            pin_memory=True)

val_1k_dir = '/fast-data/datasets/ILSVRC/2012/clsloc/val_white'
val_1k_loader = torch.utils.data.DataLoader(datasets.ImageFolder(val_1k_dir))

#all class names
hop_df = pd.read_csv('21k_wnids.csv')
all_wnids = hop_df.iloc[:,1].tolist()

all_21k_wnids = zsl_data_loader.dataset.classes
all_1k_wnids = val_1k_loader.dataset.classes
pure_2h_wnids = all_wnids[1000:2549]
pure_3h_wnids = all_wnids[2549:8860]

#converting to alternate form
all_21k_wnids_ = [str(s.split('n')[1] + 'n') for s in all_21k_wnids]
all_1k_wnids_= [str(s.split('n')[1]+'n') for s in all_1k_wnids]
pure_2h_wnids_ = [str(s.split('n')[1]+'n') for s in pure_2h_wnids]
pure_3h_wnids_ = [str(s.split('n')[1]+'n') for s in pure_3h_wnids]

#converting to synsets
all_21k_syns = [wn.of2ss(s) for s in all_21k_wnids_]
all_1k_syns = [wn.of2ss(s) for s in all_1k_wnids_]
pure_2h_syns = [wn.of2ss(s) for s in pure_2h_wnids_]
pure_3h_syns = [wn.of2ss(s) for s in pure_3h_wnids_]

dists = []
for classes in pure_3h_syns[:30]:
    mindist = min([classes.shortest_path_distance(j) for j in all_1k_syns])
    print(mindist)

