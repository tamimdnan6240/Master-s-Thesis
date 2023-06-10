#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import cv2 
from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms, models


# In[4]:


## Load the images
# This function will read the image using its path with opencv
def Load_Image(Path):
    img = cv2.imread(Path)[:,:,::-1] # opencv read the images in BGR format 
                                    # so we use [:,:,::-1] to convert from BGR to RGB
    return img


# In[5]:


## display the image and corresponding masks/annotations/labelings: Tamim

def display_images(image, mask): 
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title("Mask")
    plt.show()


# In[6]:


## Rotate the image and corresponding masks/annotations/labelings : Tamim

def rotate(image, mask, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle * (-1) , 1.0) ## 1 is the scaling factor to remain the size same 
    image = cv2.warpAffine(image, M, (w, h))
    mask = cv2.warpAffine(mask, M, (w, h))
    return image, mask


# In[7]:


## Flip the image and corresponding mask to left_right and up_down. This function is taken from https://github.com/qinnzou/DeepCrack/blob/master/codes/data/augmentation.py
## DeepCrack source
def t_random(min=0, max=1):
    return min + (max - min) * np.random.rand()

def t_randint(min, max):
    return np.random.randint(low=min, high=max)

def RandomFlip(img, mask, FLIP_LEFT_RIGHT=True, FLIP_TOP_BOTTOM=True):

    if FLIP_LEFT_RIGHT and t_random() < 0.5:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    if FLIP_TOP_BOTTOM and t_random() < 0.5:
        img = cv2.flip(img, 0)
        mask = cv2.flip(mask, 0)
    return img, mask


# In[8]:


## CenterCrop : Tamim

import numpy as np
import torchvision.transforms as transforms

def centerCrop(img, msk, size):
    # Convert to PIL images
    img = transforms.ToPILImage()(img)
    msk = transforms.ToPILImage()(msk)

    # Center crop images
    img_cropped = transforms.CenterCrop(size)(img)
    msk_cropped = transforms.CenterCrop(size)(msk)

    # Convert back to numpy arrays
    img_cropped = np.array(img_cropped)
    msk_cropped = np.array(msk_cropped)

    return img_cropped, msk_cropped

## DeepCrack source : Tamim

def RandomBlur(img, mask):

    r = 5

    if t_random() < 0.2:
        return cv2.GaussianBlur(img,(r,r),0), mask

    if t_random() < 0.15:
        return cv2.blur(img,(r,r)), mask

    if t_random() < 0.1:
        return cv2.medianBlur(img,r), mask


    return img, mask

## DeepCrack source : Tamim 
def RandomColorJitter(img, mask, brightness=32, contrast=0.5, saturation=0.5, hue=0.1,
                        prob=0.5):
    if brightness != 0 and t_random() > prob:
        img = _Brightness(img, delta=brightness)
    if contrast != 0 and t_random() > prob:
        img = _Contrast(img, var=contrast)
    if saturation != 0 and t_random() > prob:
        img = _Saturation(img, var=saturation)
    if hue != 0 and t_random() > prob:
        img = _Hue(img, var=hue)

    return img, mask

## DeepCrack source : Tamim 

def _Brightness(img, delta=32):
    img = img.astype(np.float32) + t_random(-delta, delta)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def _Contrast(img, var=0.3):
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
    alpha = 1.0 + t_random(-var, var)
    img = alpha * img.astype(np.float32) + (1 - alpha) * gs
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def _Hue(img, var=0.05):
    var = t_random(-var, var)
    to_HSV, from_HSV = [
        (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB),
        (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR)][t_randint(0, 2)]
    hsv = cv2.cvtColor(img, to_HSV).astype(np.float32)

    hue = hsv[:, :, 0] / 179. + var
    hue = hue - np.floor(hue)
    hsv[:, :, 0] = hue * 179.

    img = cv2.cvtColor(hsv.astype('uint8'), from_HSV)
    return img


def _Saturation(img, var=0.3):
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gs = np.expand_dims(gs, axis=2)
    alpha = 1.0 + t_random(-var, var)
    img = alpha * img.astype(np.float32) + (1 - alpha) * gs.astype(np.float32)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

