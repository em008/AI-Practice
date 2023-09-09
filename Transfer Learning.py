import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

# Download this dataset and extract it into data directory
if not os.path.exists('data/kagglecatsanddogs_5340.zip'):
    !wget -P data https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip

import zipfile
if not os.path.exists('data/PetImages'):
    with zipfile.ZipFile('data/kagglecatsanddogs_5340.zip', 'r') as zip_ref:
        zip_ref.extractall('data')

# Loading the Dataset
data_dir = 'data/PetImages'
batch_size = 64
ds_train = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = 'training',
    seed = 13,
    image_size = (224,224),
    batch_size = batch_size
)
ds_test = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = 'validation',
    seed = 13,
    image_size = (224,224),
    batch_size = batch_size
)

# Dataset automatically picks up class names from directories
ds_train.class_names

# Datasets obtained can be directly passed to fit function to train the model
for x,y in ds_train:
    print(f"Training batch shape: features={x.shape}, labels={y.shape}")
    x_sample, y_sample = x,y
    break

display_dataset(x_sample.numpy().astype(np.int),np.expand_dims(y_sample,1),classes=ds_train.class_names)

# Pre-trained models
# VGG-16 model
vgg = keras.applications.VGG16()
inp = keras.applications.vgg16.preprocess_input(x_sample[:1])

res = vgg(inp)
print(f"Most probable class = {tf.argmax(res,1)}")

keras.applications.vgg16.decode_predictions(res.numpy())

# Architecture of the VGG-16 network
vgg.summary()

# GPU computations
# Check if Tensorflow is able to use GPU
tf.config.list_physical_devices('GPU')

# Extracting VGG features
# Instantiate VGG-16 model without top layers
vgg = keras.applications.VGG16(include_top=False)

inp = keras.applications.vgg16.preprocess_input(x_sample[:1])
res = vgg(inp)
print(f"Shape after applying VGG-16: {res[0].shape}")
plt.figure(figsize=(15,3))
plt.imshow(res[0].numpy().reshape(-1,512))

# Manually take some portion of images and pre-compute their feature vectors.
num = batch_size*50
ds_features_train = ds_train.take(50).map(lambda x,y : (vgg(x),y))
ds_features_test = ds_test.take(10).map(lambda x,y : (vgg(x),y))

for x,y in ds_features_train:
    print(x.shape,y.shape)
    break

# Train a simple dense classifier to distinguish between cats and dogs
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(7,7,512)),
    keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
hist = model.fit(ds_features_train, validation_data=ds_features_test)
# Can distinguish between a cat and a dog with almost 95% probability

# Transfer learning using one VGG network
# Construct a network with dense classifier on top of it, and then train the whole network using back propagation
model = keras.models.Sequential()
model.add(keras.applications.VGG16(include_top=False,input_shape=(224,224,3)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1,activation='sigmoid'))

model.layers[0].trainable = False

model.summary()

# Train network
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
hist = model.fit(ds_train, validation_data=ds_test)

# Saving and Loading the Model
# Once the model is trained the model architecture and trained weights can be saved to a file for future use
model.save('data/cats_dogs.tf')

# Can load the model from file at any time
model = keras.models.load_model('data/cats_dogs.tf')

# Fine-tuning transfer learning
# Training convolutional layers and unfreeze the convolutional filter parameters
model.layers[0].summary()

# Unfreeze a few final layers of convolutions because they contain higher level patterns that are relevant for the images
# Freeze all layers except the last 4
for i in range(len(model.layers[0].layers)-4):
    model.layers[0].layers[i].trainable = False
model.summary()

hist = model.fit(ds_train, validation_data=ds_test)

# Other computer vision models - ResNet-50 model
resnet = keras.applications.ResNet50()
resnet.summary()

# This network contains yet another type of layer: Batch Normalization.
# The idea of batch normalization is to bring values that flow through the neural network to right interval.
# Batch normalization layer computes average and standard deviation for all values of the current minibatch, and uses them to normalize the signal before passing it through a neural network layer. This significantly improves the stability of deep networks.
