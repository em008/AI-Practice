"""
Segmentation
"""

# Train a neural network model to cut out silhouette image
# Use Segmentation Full Body MADS Dataset from Kaggle
dataset_path = 'segmentation_full_body_mads_dataset'

import os
import matplotlib.pyplot as plt

img_path = os.path.join(dataset_path,'images')
mask_path = os.path.join(dataset_path,'masks')

fnames = os.listdir(img_path)

def load_image(img_name):
    img = plt.imread(os.path.join(img_path,img_name))
    mask = plt.imread(os.path.join(mask_path,img_name))
    return img,mask

img, mask = load_image(fnames[5])

fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(img)
ax[1].imshow(mask)
ax[0].axis('off')
ax[1].axis('off')
