#!/usr/bin/env python
# coding: utf-8

# ## inspired code: https://www.kaggle.com/code/gokulkarthik/image-segmentation-with-unet-pytorch/notebook. I took UNet model from but,
# their data was integratred, so they splitted. But I didn't do that. I defined my dataset to load my data for this model. 

# In[ ]:


import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm.notebook import tqdm


# In[ ]:


torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


train_images_dir = "./CRACKTREE206/train/images"
train_masks_dir = "./CRACKTREE206/train/masks"


val_images_dir = "./CRACKTREE206/test/images"
val_masks_dir= "./CRACKTREE206/test/masks"



import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = img_name.split('.')[0] + '.jpg'
        mask_path = os.path.join(self.mask_dir, mask_name)
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            # remove the first dimension of the mask tensor
            mask = mask.long()  # convert the mask tensor to Long data type
        return image, mask



# define transformations to apply to the data
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])


# create dataset and dataloader for training data
train_dataset = ImageMaskDataset(train_images_dir, train_masks_dir, transform=transform)
print(len(train_dataset))


## use data lodaer to create train dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# create dataset and dataloader for validation data
val_dataset = ImageMaskDataset(val_images_dir, val_masks_dir, transform=transform)
print(len(val_dataset))

val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


# get a batch of data from the train_loader
images, masks = next(iter(train_loader))
print(images.shape, masks.shape)

# process the batch of data in your model
# ...

# get a batch of data from the val_loader
val_images, val_masks = next(iter(val_loader))
print(val_images.shape, val_masks.shape)

# process the batch of data in your model
# ...


class UNet(nn.Module):
    
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=512, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block
    
    def forward(self, X):
        contracting_11_out = self.contracting_11(X) # [-1, 64, 256, 256]
        contracting_12_out = self.contracting_12(contracting_11_out) # [-1, 64, 128, 128]
        contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, 128, 128]
        contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, 64, 64]
        contracting_31_out = self.contracting_31(contracting_22_out) # [-1, 256, 64, 64]
        contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, 32, 32]
        contracting_41_out = self.contracting_41(contracting_32_out) # [-1, 512, 32, 32]
        contracting_42_out = self.contracting_42(contracting_41_out) # [-1, 512, 16, 16]
        middle_out = self.middle(contracting_42_out) # [-1, 1024, 16, 16]
        expansive_11_out = self.expansive_11(middle_out) # [-1, 512, 32, 32]
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, 64, 64]
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, 128, 128]
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        expansive_41_out = self.expansive_41(expansive_32_out) # [-1, 64, 256, 256]
        expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        output_out = self.output(expansive_42_out) # [-1, num_classes, 256, 256]
        return output_out


num_classes = 2
model = UNet(num_classes=num_classes)

import sys
epochs = int(sys.argv[1])
lr = 0.001


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


train_losses = []
val_losses = []
best_val_loss = float('inf')

for epoch in tqdm(range(epochs)):
    epoch_train_loss = 0
    epoch_val_loss = 0
    
    # Train loop
    model.train()
    for X, Y in tqdm(train_loader, total=len(train_loader), leave=False):
        X, Y = X.to(device), Y.to(device)
        Y = torch.argmax(Y, dim=1)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    train_losses.append(epoch_train_loss/len(train_loader))
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        for X_val, Y_val in tqdm(val_loader, total=len(val_loader), leave=False):
            X_val, Y_val = X_val.to(device), Y_val.to(device)
            Y_val = torch.argmax(Y_val, dim=1)
            Y_val_pred = model(X_val)
            val_loss = criterion(Y_val_pred, Y_val)
            epoch_val_loss += val_loss.item()
        val_losses.append(epoch_val_loss/len(val_loader))
    
    # Save best model based on validation loss
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        torch.save(model.state_dict(), f"best_model_{epochs}.pt")
    
    # Print and update progress bar
    tqdm.write(f"Epoch {epoch+1}/{epochs} - Train loss: {train_losses[-1]:.4f} - Val loss: {val_losses[-1]:.4f}")


import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(f"learning_curve_{epochs}.png")


model_name = f'U-Net_{epochs}.pth'
torch.save(model.state_dict(), model_name)

model_path = f"U-Net_{epochs}.pth"
model_ = UNet(num_classes=num_classes).to(device)
model_.load_state_dict(torch.load(model_path))


import os


# create a folder to save the images
os.makedirs(f'results_{epochs}', exist_ok=True)

with torch.no_grad():
    dice_score = 0
    for X_val, Y_val in tqdm(val_loader, total=len(val_loader), leave=False):
        X_val = X_val.to(device)
        Y_pred_val = model(X_val)
        Y_pred_val = torch.sigmoid(Y_pred_val)
        predicted_masks = Y_pred_val.cpu().numpy()
        
        # Threshold predicted masks to obtain binary masks
        threshold = 0.66
        predicted_masks = np.where(predicted_masks > threshold, 1, 0)
        
        # Compute the Dice score between the predicted and target masks
        intersection = np.sum(predicted_masks * Y_val.cpu().numpy())
        union = np.sum(predicted_masks) + np.sum(Y_val.cpu().numpy())
        dice = (2. * intersection) / (union + 1e-7) # add epsilon to avoid divide by zero error
        dice_score += dice
        print(f"Dice score: {dice:.4f}")
        
        # Plot the original image, target, and predicted segmentation mask
        for i in range(len(predicted_masks)):
            fig, axs = plt.subplots(1, 3, figsize=(10,10))
            axs[0].imshow(X_val[i].cpu().numpy().transpose(1,2,0), cmap=None)
            axs[0].set_title("Original Image")
            axs[1].imshow(Y_val[i][0].cpu().numpy(), cmap='gray')
            axs[1].set_title("Target Mask")
            axs[2].imshow(predicted_masks[i][0], cmap='gray')
            axs[2].set_title("Predicted Mask")
            
            # save the figure as an image
            plt.savefig(f'results_{epochs}/image_{i}.jpg')
            
            # close the figure to free up memory
            plt.close(fig)

    mean_dice_score = dice_score / len(val_loader)
    print(f"Mean Dice score: {mean_dice_score:.4f}")



