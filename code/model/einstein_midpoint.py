import torch
import pdb
import os

smax_func = torch.nn.Softmax(dim=0)
poinc_emb = torch.load('/home/hermanni/thesis/msc-thesis/code/model/nouns_id.pth')

poinc_wgts = poinc_emb['model']['lt.weight']
poinc_wgts = torch.tensor(poinc_wgts, dtype=torch.double).cuda()

def p2k_coords(emb_mat):
    num = 2*emb_mat
    den = torch.sum(emb_mat.pow(2), dim=1, keepdim=True)+1
    return num.div(den)

def calc_lorenz_factors(emb_mat):
    return 1. / torch.sqrt(1 - torch.sum(emb_mat.pow(2), dim=1))

def einstein_midpoint(emb_mat, lfactors, weights):
    wgts = torch.mul(weights, lfactors).div(torch.sum(torch.mul(weights,
       lfactors))).view(-1, 1)
    return torch.sum(wgts*emb_mat, 0)

def klein_dist(U, V):
    num = 1 - torch.sum(torch.mul(U, V), dim=1)
    lhs = torch.sqrt(1 - torch.sum(U.pow(2), dim=1))
    rhs = torch.sqrt(1 - torch.sum(V.pow(2), dim=1))
    den = lhs*rhs
    x = num.div(den)
    return torch.clamp(torch.log(x + torch.sqrt(torch.clamp(
        torch.mul(x,x), min=1.0) - 1.0)), min=0.0)
