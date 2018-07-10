import torch
from poincare_model import PoincareDistance
import pdb
from torch.autograd import Function

def poincare_emb_hinge_loss(pred_embs, target_embs, all_embs,
                            num_classes, emb_size, margin, batch_size):
    poincdist = PoincareDistance()
    #calculate distance of a predicted embedding to all
    #the embeddings
    dist2wrong = poincdist(pred_embs.repeat(1,
                           num_classes).view(-1, emb_size),
                           all_embs.repeat(batch_size,
                               2).cuda(non_blocking=True))
    #calculate distance to correct embeddings
    dist2correct = poincdist(pred_embs.repeat(1,
                             num_classes).view(-1, emb_size),
                             target_embs.repeat(1,
                             num_classes).view(-1, emb_size))
    #ranking hinge loss to move such that closes to correct
    #embedding
    hinge_loss = torch.sum(torch.clamp(dist2correct.sub(
                           dist2wrong).add(margin), min=0.0))
    #below is done because when predicting correct label separation
    #is zero so margin will still be added. This is a hack to really just
    #calculate loss over incorrect labels
    hinge_l_adj_margin = hinge_loss.add(-margin*batch_size)
    #average by over observations in a batch
    return  torch.div(hinge_l_adj_margin, batch_size)


def _assert_no_grad(tensor):
    assert not tensor.requires_grad


class PoincareEmbHingeLoss(Function):
    def __init__(self, n_classes, emb_size, margin, batch_size):
        super().__init__()
        self.nclasses = n_classes
        self.nemb = emb_size
        self.marg = margin
        self.bsize = batch_size

    def forward(self, predicted_embeddings,
                target_embeddings, all_embeddings):

        _assert_no_grad(target_embeddings)
        return poincare_emb_hinge_loss(predicted_embeddings, target_embeddings,
                                       all_embeddings, self.nclasses,
                                       self.nemb, self.marg, self.bsize)






