# Master-s-Thesis

# For Classification: 

## Dataset derived from the EdmCrack600. Total number of 142 images (we uploaded all folders in github, if needed download from the following link)
#### Link: https://drive.google.com/drive/folders/11i6iLq_KSbmxs91MaSCgtIhFTH9N2uJl?usp=sharing 

### The model was trained on 8 different epochs and 2 optimizers conditions, pretrained models in each conditions has been saved in this directory. Also,
#### All relevant outputs can be found here. Link: https://drive.google.com/drive/folders/1Kqtw9D_9VP7FDdtj7eFtWldTFkdyd8dy?usp=sharing 

### We also reduced some transverse cracks to make the dataset less imabalanced for the transferred-ResNet50 model.. Links: https://drive.google.com/drive/folders/1WDGmXSi1wGe4ewxHwLRdK0VvylIIDhD9?usp=sharing 


## For Segmentation

#### We fine-tuned a U-Net model from Kaggle: https://www.kaggle.com/code/gokulkarthik/image-segmentation-with-unet-pytorch

#### We trained and tested this U-Net model with two segmentation - datasets : 1)EdmCrack600 and 2)CRACKTREE260 

### EdmCrack600: 
#### The raw images had noises, so we cropped the raw images and corresponding masks to 512 x 512 with Python programming. Then some raw images missed the cracks, therefore, we removed the missed images, actually, we have deleted them, but kept some of them to show why we excluded them. Finally, 470 images were selected from the 600 images for this research. The dataset is uploaded in github, but if needed download from the given link of google drive. All output files are here: 


### CRACKTREE260: 
#### This dataset has 260 images but 206 images are the same sizes of 800 x 600. Therefore, our study just used 206 images for training and testing on the U-Net model to see the performances. No preprocessing was made here.  The dataset is uploaded in github, but if needed download from the given link of google drive. All output files are here: 
