import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os


#load pre-trained model insead
inception3 =  models.inception_v3(pretrained=True)


#define data loader
data_transforms = {
                'train': transforms.Compose([
                           transforms.RandomResizedCrop(229),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()]),
                'val': transforms.Compose([
                         transforms.Resize(229),
                         transforms.ToTensor()])
                }

data_dir = "/home/hege/Documents/Thesis/msc-thesis/small_data/"
img_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x]) for 
                                        x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(img_datasets[x], 
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=4)
                  for x in ['train', 'val']}

dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes











