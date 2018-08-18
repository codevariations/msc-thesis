import torch
from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np
import pdb
import time
import pickle

labels_hop_order = pd.read_csv('21k_wnids.csv')

#load all zero-shot class ids sorted on hops
labels_hop_order = pd.read_csv('21k_wnids.csv')
wnids_21k = labels_hop_order.iloc[:, 1].tolist()
wnids_21k.remove('n11196627')
wnids_21k.remove('n11318824')
wnids_21k.remove('n10994097')
wnids_21k.remove('n09450163')
wnids_21k.remove('n04399382')

wnids_20k = wnids_21k[1000:]
wnids_1k = wnids_21k[:999]
wnids_2hop = wnids_21k[1000:2549]
wnids_3hop = wnids_21k[1000:8860]
wnids_2h_1k = wnids_21k[:2549]
wnids_3h_1k = wnids_21k[:8860]

IT = 0

def find_k_hierarchy(ss, k, ref_synsets):
    global IT
    IT += 1
    hcorrectSet = []
    hcorrectSet.extend([ss])
    real_new = hcorrectSet
    while len(hcorrectSet) < k:
        hypers = [i.hypernyms() for i in real_new]
        hypers = [item for l in hypers for item in l]
        hypos = [i.hyponyms() for i in real_new]
        hypos = [item for l in hypos for item in l]
        suggest_new = set(hypos + hypers)
        real_new = suggest_new - set(hcorrectSet)
        real_new = [i for i in real_new]
        valid_real_new = [i for i in real_new if i in ref_synsets]
        hcorrectSet.extend(valid_real_new)
    print(IT)
    return hcorrectSet

def create_hp_data(k, chosen_offset_data):
    _offsets = [i.split('n')[1] + '-n' for i in chosen_offset_data]
    _synsets = [wn.of2ss(s) for s in _offsets]

    return {'n'+i.split('-')[0]:['n'+ wn.ss2of(j).split('-')[0]
            for j in find_k_hierarchy(wn.of2ss(i), k, _synsets)]
            for i in _offsets}

def save_dict(obj, name):
    with open('dicts/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('dicts/'+name+'.pkl', 'rb') as f:
        return pickle.load(f)

#k=20
#data = create_hp_data(k, wnids_3hop)
#save_dict(data, 'hptarget_3hop_20')

#print(np.mean([len(data[i]) for i in data.keys()]))
#test = [j  for i in _offsets[1000:2549] for j in list(d1[i])]
#def find_k_hierarchy2(ss, k, reference):
#    dists = [ss.shortest_path_distance(wn.of2ss(i)) for i in reference]
#    hcorrectSet = []
#    i = 0
#    while  len(hcorrectSet) < k:
#        npdist = np.array(dists)
#        idx = np.where(npdist==i)[0]
#        wids = np.array(reference)[idx]
#        wids = [i for i in wids]
#        hcorrectSet.extend(wids)
#        i += 1
#    return hcorrectSet


