import torch
import torchvision
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import os
import copy
import numpy as np

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
                                        data_transforms[x]) 
                                        for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(img_datasets[x], 
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=4)                                                            for x in ['train', 'val']}
dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}
class_names = img_datasets['train'].classes

#input example image and plot it
img_in, imb_label = next(iter(dataloaders['train']))

#plot it
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

out = torchvision.utils.make_grid(img_in)
imshow(out)
plt.show()

#define function to train a model
def train_model(model, criterion, optimizer, scheduler, n_epochs=25):
    since = time.time()

    best_model_ws = copy.deepcopy(model.state_dict())
    best_top_acc = 0.0 
    best_five_acc = 0.0 

    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch, n_epochs-1))
        print('-'*10)

        #training and validation per epoch
        for stage in ['train', 'val']:
            if stage == 'train':
                scheduler.step()
                model.train() #activate training mode
            else:
                model.eval() #activate validation mode
            
            run_loss = 0.0
            run_correct = 0

            #iterate over data
            for inputs, labels in dataloaders[stage]:
                optimizer.zero_grad()

                #forward pass
                with torch.set_grad_enabled(stage == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    #backwards pass
                    if stage == 'train':
                        loss.backward()
                        optimizer.step()
                 
                run_loss += loss.item()*inputs.size(0)
                run_correct += torch.sum(preds==labels.data)

            epoch_loss = run_loss / dataset_sizes[stage]
            epoch_acc = run_correct.double() / dataset_sizes[stage]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                stage, epoch_loss, epoch_acc))

            #deep copy the model
            if stage == 'val' and epoch_acc > best_top_acc:
                best_acc = epoch_acc
                best_model_ws = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since 
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_ws)
    return model

#visualize model prediction
def visualize_pred(model, n_imgs=6):
    was_training = model.training
    model.eval()
    imgs_so_far = 0 
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumarate(dataloaders['val']):
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                imgs_so_far += 1
                ax = plt.subplot(n_imgs//2, 2, imgs_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if imgs_so_far == n_imgs:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


#load pre-trained model 
model =  models.inception_v3(pretrained=True)
n_feats = model.fc.in_features
model.fc = nn.Linear(n_feats, 2)
criterion = nn.CrossEntropyLoss()

optimizer_m = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_m, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer_m, exp_lr_scheduler, 
       n_epochs=25)



















