import torch
from torch import nn
from poincare_model import PoincareDistance2, EuclideanDistance
import pdb
from torch.autograd import Function



class PoincareEmbHingeLoss(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.bsize = batch_size
        self.poincdist = PoincareDistance2()

    def forward(self, X, Y):
        dist = self.poincdist(X, Y)
        sum_dist = torch.sum(dist)
        return torch.div(sum_dist, self.bsize)

