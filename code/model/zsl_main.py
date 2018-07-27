import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import random
import shutil
import time
import warnings

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from custom_loss import PoincareXEntropyLoss
from poincare_model import PoincareDistance
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
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
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
    global args, best_prec1, poinc_emb, imgnet_poinc_wgt, imgnet_poinc_labels
    args = parser.parse_args()

    #load poincare embedding
    poinc_emb = torch.load(
            '/home/hermanni/thesis/msc-thesis/code/model/nouns_id.pth')

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
    model = PoincareVGG(orig_vgg, n_emb_dims=10)

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

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    valdir = args.data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
                      train_dataset)
    else:
        train_sampler = None

    val_loader = torch.utils.data.DataLoader(
               datasets.ImageFolder(valdir, transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     normalize,])),
               batch_size=args.batch_size, shuffle=True,
               num_workers=args.workers, pin_memory=True)
    val_classes = val_loader.dataset.classes

    #create poincare embedding with all synsets
    imgnet2poinc_idx = [poinc_emb['objects'].index(i) for i in val_classes]
    imgnet_poinc_wgt = poinc_emb['model']['lt.weight'][[imgnet2poinc_idx]]
    imgnet_poinc_labels = [poinc_emb['objects'][i] for i in imgnet2poinc_idx]

    #finally, run evaluation 
    validate(val_loader, model)
    return


### Functions 

def validate(val_loader, model):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    top5 = AverageMeter()
    top10 = AverageMeter()
    top20 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            # measure accuracy and record loss
            prec1, prec2, prec5, prec10, prec20  = accuracy(output,
                                                            imgnet_poinc_wgt,
                                                            target, topk=(1, 2,
                                                             5, 10, 20))
            top1.update(prec1.item(), input.size(0))
            top2.update(prec2.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            top10.update(prec10.item(), input.size(0))
            top20.update(prec20.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@2 {top2.val:.3f} ({top2.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Prec@10 {top10.val:.3f} ({top10.avg:.3f})\t'
                      'Prec@20 {top20.val:.3f} ({top20.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time,
                       top1=top1, top2=top2, top5=top5, top10=top10,
                       top20=top20))

        print(' * Prec@1 {top1.avg:.3f} Prec@2 {top2.avg:.3f}\t'
                'Prec@5 {top5.avg:.3f} Prec@10 {top10.avg:.3f}\t'
                'Prec@20 {top20.avg:.3f}'.format(top1=top1, top2=top2,
                                                 top5=top5, top10=top10,
                                                 top20=top20))

    return top1.avg, top2.avg, top5.avg, top10.avg, top20.avg



class PoincareVGG(nn.Module):

    def __init__(self, vgg_model, n_emb_dims):
        super(PoincareVGG, self).__init__()
        self.features = vgg_model.features
        self.fc = nn.Sequential(*list(
                                vgg_model.classifier.children())[:-1])
        self.classifier = nn.Sequential(nn.Linear(4096, n_emb_dims))
        self.eps = 1e-5

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
        y_norm = y.pow(2).sum(dim=1, keepdim=True).pow(0.5)
        y_normsq = y_norm.pow(2)
        y_normsqplus = torch.add(y_normsq, 1)
        y_unitnorm = torch.div(y, y_norm)
        return torch.add(torch.div(torch.mul(y_unitnorm,
                         y_normsq), y_normsqplus), -self.eps)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def prediction(output, all_embs, knn=1):
    """Predicts the nearest class based on poincare distance"""
    with torch.no_grad():
        batch_size = output.size(0)
        n_emb_dims = output.size(1)
        n_classes = all_embs.size(0)
        dists_to_all = PoincareDistance.apply(output.repeat(1,
                                              n_classes).view(-1, n_emb_dims),
                                              all_embs.repeat(batch_size,
                                              1).cuda(args.gpu,
                                                      non_blocking=True))
        topk_per_batch = torch.topk(dists_to_all.view(batch_size, -1),
                                     k=knn, dim=1,
                                     largest=False)[1]
        if knn==1:
            return topk_per_batch.view(-1)
        return topk_per_batch


def accuracy(output, all_embs, targets, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        preds = prediction(output, all_embs, knn=maxk)
        batch_size = output.size(0)
        res = []
        for k in topk:
            i = k
            preds_tmp = preds[:, :i]
            correct_tmp = preds_tmp.eq(targets.view(batch_size, -1).repeat(1, i))
            res.append(torch.sum(correct_tmp).float() / batch_size)
        return res

if __name__ == '__main__':
    main()
