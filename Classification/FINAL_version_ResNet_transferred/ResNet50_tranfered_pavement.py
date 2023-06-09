#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# License: BSD
# Author: Sasank Chilamkurthy


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode

from random import *
from tqdm.notebook import tqdm, trange
from time import sleep
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from warnings import filterwarnings
from sklearn.metrics import accuracy_score
filterwarnings('ignore')


device = torch.device("cpu")
device



##  codes for data augmentation
train_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.FiveCrop(224),
    transforms.Lambda(lambda crops: crops[0]), 
    transforms.RandomHorizontalFlip(p=0.5), ## tamim: image will move left and right
    transforms.RandomVerticalFlip(p=0.5), ## tamim: image will come to eye vertically
    transforms.RandomRotation(degrees=(.5, 5)), ## very small rotation of the cracks
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


test_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), ## tamim: image will move left and right
    transforms.RandomVerticalFlip(p=0.5), ## tamim: image will come to eye vertically
    transforms.RandomRotation(degrees=(.5, 5)), ## very small rotation of the cracks
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


## Load data
from torchvision.datasets import ImageFolder
data = ImageFolder("../Data/Data_Structure(Annotated)", transform=train_trans , )
test_folder= ImageFolder("../Data/Data_Structure(Raw images)", transform=test_trans, )


## hyperparameters

batch_size = 32
num_classes = 3
learning_rate = 0.001

## for epoch and optimizer in HPC

import sys

# read the epoch parameter from the command line / shellscript
num_epochs = int(sys.argv[1])

# read the optimizers from the command line / shellscript
optimizers = str(sys.argv[2])

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig(f"labels_{num_epochs}_{optimizers}.JPG")

# your code for training the model with the given epoch

print("Follwing classes are there : \n",data.classes)


print("data length:", len(data))

classes = ('Alligator Cracks', 'Longitudinal Cracks', 'Transverse Cracks')


##Splitting Data and Prepare Batches: 
## Source: https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48 
val_size = 40 ##  Tamim:30% data for validation ## 
train_size = len(data) - val_size 

## To randomly split the images into training and testing, PyTorch provides random_split()
train_data, val_data = random_split(data,[train_size,val_size]) 
print(f"Length of Train Data : {len(train_data)}")     ## changed the folder names
print(f"Length of Validation Data : {len(val_data)}")

# Splitting train and validation data on batches

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size) ## defined train data & val data
val_loader = torch.utils.data.DataLoader(val_data, shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_folder, shuffle=False, batch_size=batch_size)

# visualize images of a single batch

dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


model = models.resnet50(pretrained=True).to(device)

model.eval().to(device)
# model.fc = nn.Linear(2048, 4) # original model has outputs for 1000 classes. Tamim changed it to 4

print(model)


# https://www.kaggle.com/code/pmigdal/transfer-learning-with-resnet-50-in-pytorch 
# 


# Freeze all layers except the last fc layer
for param in model.parameters():
    param.requires_grad = False   
    
model.fc = nn.Sequential(
               nn.Linear(2048, 512),  # reduce hidden units to 512
               nn.ReLU(inplace=True),
               nn.Linear(512, 3), 
               nn.Softmax(dim=1),
               nn.Dropout(p=0.5))


# Print the modified ResNet50 architecture
print(model)


## Defining model optimizer and loss function
loss_fn = nn.CrossEntropyLoss()

if optimizers == 'Adam':
    opt = optim.Adam(model.parameters(), lr=1e-3)
elif optimizers == 'SGD':
    opt = optim.SGD(model.parameters(), lr=1e-3)
else:
    raise ValueError('Invalid optimizer name: {}'.format(optimizer_name))


def mean(l: list):
    return sum(l) / len(l)


def plot_losses_and_acc(train_losses, train_accuracies, valid_losses, valid_accuracies): 
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].plot(train_losses, label='train_losses')
    axes[0].plot(valid_losses, label='valid_losses')
    axes[0].set_title('Losses')
    axes[0].legend()
    axes[1].plot(train_accuracies, label='train_accuracies')
    axes[1].plot(valid_accuracies, label='valid_accuracies')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    plt.savefig(f"learning_curve_{num_epochs}_{optimizers}.JPG")


def plot_losses(train_losses, valid_losses): 
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(train_losses, label='train_losses')
    ax.plot(valid_losses, label='valid_losses')
    ax.set_title('Losses')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.savefig(f"learning_curve_{num_epochs}_{optimizers}.jpg")

def validate(model, valid_data, loss_fn):
    valid_losses, valid_accuracies = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in tqdm(valid_data, leave=False):
            X_batch, y_batch = X_batch.float(), y_batch.long()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            valid_losses.append(loss.item())
            preds = torch.argmax(logits, axis=1)          
            valid_accuracies.append(((preds == y_batch).sum() / len(preds)).item())
    return mean(valid_losses), mean(valid_accuracies)
    



def train(model, train_data, valid_data, loss_fn, opt, epoches):
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []
    
    for epoch in tqdm(range(epoches)):
        train_loss = []
        train_acc = []
        
        model.train()
        
        for X_batch, y_batch in tqdm(train_data, leave=False):
            opt.zero_grad()

            X_batch, y_batch = X_batch.float(), y_batch.long()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch,)
            train_loss.append(loss.item())

            pred = torch.argmax(logits, dim=1)
            train_acc.append(((pred == y_batch).sum() / len(pred)).item())
            loss.backward()
            opt.step()

        valid_loss, valid_accuracy = validate(model, valid_data, loss_fn)

        train_accuracies.append(mean(train_acc))
        train_losses.append(mean(train_loss))
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        
        print(f'epoch: {epoch}: train_loss: {mean(train_losses)}, train_acc: {mean(train_acc)}, val_loss: {valid_loss}, val_acc: {valid_accuracy}')
    plot_losses(train_losses, valid_losses)
    return model, train_losses, train_accuracies, valid_losses, valid_accuracies



from tqdm import tqdm



model, train_losses, train_accuracies, valid_losses, valid_accuracies = train(model, train_loader, val_loader, loss_fn, opt, epoches=num_epochs)

# resource: #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

## loss calculation

valid_loss, valid_acc = validate(model, test_loader, loss_fn)

print(valid_loss, valid_acc)



##Testing

dataiter = iter(test_loader)
images, labels = next(dataiter)

# resource: #https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


# Save model
PATH = f'./New_{num_epochs}_{optimizers}.pth'
torch.save(model.state_dict(), PATH)


# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


##Let us look at how the network performs on the whole dataset.

correct = 0
total = 0

# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')



# prepare to count predictions for each class on iphone image data
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed because we already trained
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

