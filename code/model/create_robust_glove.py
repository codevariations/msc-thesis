import os
import torch
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
import pdb
import pickle as pkl

def rem_dups(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

#load glove labels
with open('w2v_emb.pkl', 'rb') as f:
    glove_emb = pkl.load(f, encoding='latin')
glove_idx = glove_emb['objects']

all_hops = pd.read_csv('21k_wnids.csv')

wnids_21k = all_hops.iloc[:, 1].tolist()
wnids_20k = wnids_21k[1000:]
wnids_1k = wnids_21k[:999]
wnids_2hop = wnids_21k[1000:2549]
wnids_3hop = wnids_21k[1000:8860]
wnids_2h_1k = wnids_21k[:2549]
wnids_3h_1k = wnids_21k[:8860]

#select which hop data-set to use
chosen_hop_data = wnids_20k

all_ids = chosen_hop_data
all_ids = [i.split('n')[1]+'-n' for i in all_ids]
all_ss = [wn.of2ss(i) for i in all_ids]

hyper = lambda s: s.hypernyms()

all_hypers = []
all_dists = []
for i in all_ss:
  holder = ['n'+wn.ss2of(i).split('-')[0]]
  cur_hypers = list(i.closure(hyper))
  cur_hypers = ['n' + wn.ss2of(i).split('-')[0] for i in cur_hypers]
  cur_hypers = [i for i in cur_hypers if i in glove_idx]
  holder.extend(cur_hypers)
  all_hypers.append(holder)
exc_ids = [i[0] for i in all_hypers if len(i)<5]
all_hypers = [i for i in all_hypers if i[0] not in exc_ids]
all_hypers_ids = [i for j in all_hypers for i in j]
all_hypers_ids = rem_dups(all_hypers_ids)

#constrain everything to 5 edges upwards max
all_hypers = [i[:5] for i in all_hypers]

hyper_dict = {}
hyper_dict['exc_ids'] = exc_ids
hyper_dict['all_hyper_ids'] = all_hypers_ids
hyper_dict['hyper_labels'] = all_hypers

with open('glove_robust_labels_20k.pickle', 'wb') as handle:
    pkl.dump(hyper_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

#old
#hyper_dict['ss'] = ['n'+wn.ss2of(i).split('-')[0] for i in all_ss]

