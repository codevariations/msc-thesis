import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

import random
import shutil
import time
import warnings
import pickle

import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn.modules.distance import PairwiseDistance


from cust_loader import CustImageFolder
from custom_loss import EmbXEntropyLoss

import pdb


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

best_prec1 = 0


def main():
    global args, best_prec1, glove_emb, imgnet_glove_wgt, all_hyper_ids
    global glove_emb_hop_wgt, imgnet_glove_labels
    global target2tree_idx, numeric_robust_hyper_labels, expand_all_embs
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        orig_vgg = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        orig_vgg = models.__dict__[args.arch]()

    #Change model to project into poincare space
    model = EmbVGG(orig_vgg, n_emb_dims=50)

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()


    # define loss function (criterion) and optimizer
    criterion = EmbXEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                       args.lr,
                                       momentum=args.momentum,
                                       weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    ## Data loading code

    #regular 1k imagenet training data
    origdir = '/fast-data/datasets/ILSVRC/2012/clsloc/train/'
    orig_data = datasets.ImageFolder(origdir)

    #load the different hop ids so that load correct imgs

    all_hops = pd.read_csv('21k_wnids.csv')

    wnids_21k = all_hops.iloc[:, 1].tolist()
    wnids_20k = wnids_21k[1000:]
    wnids_1k = wnids_21k[:999]
    wnids_2hop = wnids_21k[1000:2549]
    wnids_3hop = wnids_21k[1000:8860]
    wnids_2h_1k = wnids_21k[:2549]
    wnids_3h_1k = wnids_21k[:8860]

    chosen_hop_data = wnids_3hop

    #load labels for current robust prediction
    with open('dicts/glove_robust_labels_3hop.pickle', 'rb') as f:
        robust_labels = pickle.load(f)
    all_hyper_ids = robust_labels['all_hyper_ids']
    robust_hyper_labels = robust_labels['hyper_labels']
    numeric_robust_hyper_labels = torch.tensor([[all_hyper_ids.index(i)
                                                 for i in j] for j in
                                                 robust_hyper_labels]).cuda()
    exc_ids = robust_labels['exc_ids']
    chosen_hop_data = [i for i in chosen_hop_data if i not in exc_ids]

    #ZSL data loading code
    valdir = args.data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_data = CustImageFolder(valdir, chosen_hop_data, transforms.Compose([
                               transforms.Resize(256), transforms.CenterCrop(224),
                               transforms.ToTensor(), normalize,]))
    val_loader = torch.utils.data.DataLoader(img_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    val_classes = val_loader.dataset.classes

    #load poincare embedding data (include embs for only curren hops)
    with open('w2v_emb.pkl', 'rb') as f:
        glove_emb = pickle.load(f, encoding='latin')

    glove_emb_wgt = torch.tensor(glove_emb['model'], dtype=torch.float)
    glove_emb_orig_idx = [glove_emb['objects'].index(i)
                          for i in orig_data.classes]
    glove_emb_orig_wgt = glove_emb_wgt[[glove_emb_orig_idx]]

    glove_emb_hop_idx = [glove_emb['objects'].index(i)
                         for i in all_hyper_ids]
    glove_emb_hop_wgt = glove_emb_wgt[[glove_emb_hop_idx]]
    glove_emb_hop_labels = [glove_emb['objects'][i] for i in
                            glove_emb_hop_idx]

    #locate target idx in tree idx
    target2tree_idx = [chosen_hop_data.index(i) for i in val_classes]

    #this is needed in prediction
    expand_all_embs = glove_emb_hop_wgt.repeat(args.batch_size,
                    1).cuda(args.gpu, non_blocking=True)

    #finally, run evaluation
    validate(val_loader, model)
    return

def validate(val_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top3 = AverageMeter()
    top4 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    #needed for converting class idx to IDs
    class2idx = val_loader.dataset.class_to_idx
    idx2class = {v: k for k, v in class2idx.items()}

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            tree_target_idx = [target2tree_idx[i] for i in target]
            tree_targets = numeric_robust_hyper_labels[[tree_target_idx]]

            # compute output
            output = model(input)

            # measure accuracy and record loss
            prec1, prec2, prec3, prec4, prec5  = accuracy(output,
                                               glove_emb_hop_wgt,
                                               tree_targets, topk=(1, 2,
                                               3, 4, 5))
            top1.update(prec1.item(), input.size(0))
            top2.update(prec2.item(), input.size(0))
            top3.update(prec3.item(), input.size(0))
            top4.update(prec4.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@2 {top2.val:.3f} ({top2.avg:.3f})\t'
                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\t'
                      'Prec@4 {top4.val:.3f} ({top4.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time,
                       top1=top1, top2=top2, top3=top3,
                       top4=top4, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}\t'
              'Prec@3 {top3.avg:.3f} Prec@4 {top4.avg:.3f}\t'
              'Prec@5 {top5.avg:.3f}'.format(top1=top1, top2=top2,
              top3=top3, top4=top4, top5=top5))

    return top1.avg, top2.avg, top3.avg, top4.avg, top5.avg

class EmbVGG(nn.Module):

    def __init__(self, vgg_model, n_emb_dims):
        super(EmbVGG, self).__init__()
        self.features = vgg_model.features
        self.fc = nn.Sequential(*list(
                                vgg_model.classifier.children())[:-1])
        self.classifier = nn.Sequential(nn.Linear(4096, n_emb_dims))
        self.eps = 1e-9

        #freeze weights except classifier layer 
        self.unfreeze_features(False)
        self.unfreeze_fc(True)

    def unfreeze_features(self, unfreeze):
        for p in self.features.parameters():
            p.requires_grad = unfreeze

    def unfreeze_fc(self, unfreeze):
        for p in self.fc.parameters():
            p.requires_grad = unfreeze

    def forward(self, x):
        f = self.features(x)
        if hasattr(self, 'fc'):
            f = f.view(f.size(0), -1)
            f = self.fc(f)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 1
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*0.1

def prediction(output, all_embs, knn=1):
    """Predicts the nearest class based on poincare distance"""
    with torch.no_grad():
        eucdist = torch.nn.DataParallel(PairwiseDistance()).cuda()
        batch_size = output.size(0)
        n_emb_dims = output.size(1)
        n_classes = all_embs.size(0)
        dists_to_all = eucdist(output.repeat(1, n_classes).view(-1, n_emb_dims),
                               expand_all_embs)
        #pred_norms = expand_output.pow(2).sum(dim=1).sqrt()
        #label_norms = expand_all_embs.pow(2).sum(dim=1).sqrt()
        #norm_wgt = -(1 + (label_norms.cuda(args.gpu, non_blocking=True)
        #                  - pred_norms)*1000)
        #tree_scores = torch.mul(dists_to_all.view(batch_size, -1),
        #                        norm_wgt.view(batch_size, -1))
        topk_per_batch = torch.topk(dists_to_all.view(batch_size, -1),
                                    k=knn, dim=1,
                                    largest=False)[1]
        if knn==1:
            return topk_per_batch.view(-1)
        return topk_per_batch

def accuracy(output, all_embs, targets, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        batch_size = output.size(0)
        maxk = max(topk)
        preds = prediction(output, all_embs, knn=maxk)
        batch_size = output.size(0)
        correct_tmp = preds.eq(targets)
        acc_wgts = np.array([[1, 1/2, 1/3, 1/4, 1/5]])
        res = []
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        predcpu = preds.cpu().numpy()
        targetcpu = targets.cpu().numpy()
        for k in topk:
            preds_tmp = predcpu[:, :k]
            matches = np.any((targetcpu.reshape(targetcpu.shape
                    + (1,)).reshape(targetcpu.shape[0], 1,
                    targetcpu.shape[1])
                    - preds_tmp.reshape(preds_tmp.shape+(1,)))==0, axis=1)
            res.append(np.mean(np.sum(
                       matches*acc_wgts, axis=1) / np.sum(acc_wgts)))
        return res





def find_name(wnid):
    ss = wnid.split('n')[1] + '-n'
    name = wn.of2ss(ss)
    return name

if __name__ == '__main__':
    main()
