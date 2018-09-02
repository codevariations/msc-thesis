import os
import torch
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
import pdb
import pickle as pkl

all_ids = pd.read_csv('21k_wnids.csv')
all_ids = all_ids.iloc[:, 1].tolist()
all_ids = [i.split('n')[1]+'-n' for i in all_ids]
all_ss = [wn.of2ss(i) for i in all_ids]

hyper = lambda s: s.hypernyms()

all_hypers = []
all_dists = []
for i in all_ss:
  holder = ['n'+wn.ss2of(i).split('-')[0]]
  cur_hypers = list(i.closure(hyper))
  dists = [1.]
  dists.extend([(i.shortest_path_distance(j) + 1.) for j in cur_hypers])
  dists = [1. / i for i in dists]
  all_dists.append(dists)
  cur_hypers = ['n' + wn.ss2of(i).split('-')[0] for i in cur_hypers]
  holder.extend(cur_hypers)
  all_hypers.append(holder)

hyper_dict = {}
hyper_dict['ss'] = ['n'+wn.ss2of(i).split('-')[0] for i in all_ss]
hyper_dict['hypers'] = all_hypers
hyper_dict['dists'] = all_dists

with open('robust_labels.pickle', 'wb') as handle:
    pkl.dump(hyper_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)


