import torch
import pdb
import os

smax_func = torch.nn.Softmax(dim=0)
poinc_emb = torch.load('/home/hermanni/thesis/msc-thesis/code/model/nouns_id.pth')

poinc_wgts = poinc_emb['model']['lt.weight']
poinc_wgts = torch.tensor(poinc_wgts, dtype=torch.double).cuda()

def p2k_coords(emb_mat):
    #num = 2*emb_mat
    #den = torch.sum(emb_mat.pow(2), dim=2, keepdim=True)+1
    return 2*emb_mat.div(torch.sum(emb_mat.pow(2), dim=2, keepdim=True)+1)

def k2p_coords(emb_mat):
    sqnorm = emb_mat.pow(2).sum(dim=1, keepdim=True)
    den = torch.sqrt(torch.add(-sqnorm, 1)).add(1)
    return emb_mat.div(den)

def calc_lorenz_factors(emb_mat):
    return 1. / (torch.sqrt(1 - torch.sum(emb_mat.pow(2), dim=2)))

def einstein_midpoint(emb_mat, lfactors, weights):
    batch_size = emb_mat.size(0)
    T = emb_mat.size(1)
    n_emb_dims = emb_mat.size(2)
    wgts = torch.mul(weights, lfactors).div(torch.sum(torch.mul(weights,
       lfactors), dim=1, keepdim=True))
    return torch.mul(wgts.view(batch_size, T, 1), emb_mat).sum(dim=1)

def klein_dist(U, V):
    num = 1 - torch.sum(torch.mul(U, V), dim=1)
    lhs = torch.sqrt(1 - torch.sum(U.pow(2), dim=1))
    rhs = torch.sqrt(1 - torch.sum(V.pow(2), dim=1))
    den = lhs*rhs
    x = num.div(den)
    return torch.clamp(torch.log(x + torch.sqrt(torch.clamp(
        torch.mul(x,x), min=1.0) - 1.0)), min=0.0)

