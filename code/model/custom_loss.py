import torch
from torch import nn
import torch.backends.cudnn as cudnn
from poincare_model import PoincareDistance, PoincareDistance2
import pdb
from torch.autograd import Function

cudnn.fastest =True

pdist = torch.nn.DataParallel(PoincareDistance)

def poincare_emb_hinge_loss(pred_embs, target_embs, all_embs, margin):
    batch_size = pred_embs.size(0)
    n_emb_dims = pred_embs.size(1)
    n_classes = all_embs.size(0)
    #calculate distance of a predicted embedding to all
    #the embeddings
    dist2wrong = PoincareDistance.apply(pred_embs.repeat(1,
                                        n_classes).view(-1, n_emb_dims),
                                        all_embs.repeat(batch_size,
                                        1).cuda(non_blocking=True))
    #calculate distance to correct embeddings
    dist2correct = PoincareDistance.apply(pred_embs.repeat(1,
                                          n_classes).view(-1, n_emb_dims),
                                          target_embs.repeat(1,
                                          n_classes).view(-1, n_emb_dims))
    #ranking hinge loss to move such that closes to correct
    #embedding
    hinge_loss = torch.sum(torch.clamp(dist2correct.sub(
                           dist2wrong).add(margin), min=0.0))
    #below is done because when predicting correct label separation
    #is zero so margin will still be added. This is a hack to really just
    #calculate loss over incorrect labels
    hinge_l_adj_margin = hinge_loss.add(-margin*batch_size)
    #average over observations in a batch
    return  torch.div(hinge_l_adj_margin, batch_size)

class PoincareEmbHingeLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.marg = margin

    def forward(self, pred_embs, target_embs, all_embs):
        _assert_no_grad(target_embs)
        return poincare_emb_hinge_loss(pred_embs, target_embs,
                                       all_embs, self.marg)

def poincare_dist_to_label_emb(pred_embs, all_embs):
    batch_size = pred_embs.size(0)
    n_emb_dims = pred_embs.size(1)
    n_classes = all_embs.size(0)
    pdist = torch.nn.DataParallel(PoincareDistance2()).cuda()
    #calculate distance of a predicted embedding to all possible true
    #embedding
    return pdist(pred_embs.repeat(1,
                                  n_classes).view(-1, n_emb_dims),
                                  all_embs.repeat(batch_size,
                                  1).cuda(non_blocking=True)).view(
                                  batch_size, -1)

def _assert_no_grad(tensor):
    assert not tensor.requires_grad



class PoincareXEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xeloss = nn.CrossEntropyLoss().cuda()

    def forward(self, pred_embs, target_idx, all_embs):
        _assert_no_grad(target_idx)
        _assert_no_grad(all_embs)
        scores = poincare_dist_to_label_emb(pred_embs, all_embs)
        #since smaller distance is good need to inver the exponent in
        #softmax
        neg_scores = -1 * scores
        return self.xeloss(neg_scores, target_idx)



