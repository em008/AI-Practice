"""
Convolutional neural networks with MNIST dataset
"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# loading the dataset using Keras built-in functions
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

"""
Covolutional layers
"""
# convolution, flatten 24x24x9 tensor into one vector of size 5184, and then add linear layer, to produce 10 classes
# use relu activation function in between layers
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=9, kernel_size=(5,5), input_shape=(28,28,1),activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
])

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['acc'])

model.summary()

# reshape data before starting training
x_train_c = np.expand_dims(x_train,3)
x_test_c = np.expand_dims(x_test,3)
hist = model.fit(x_train_c,y_train,validation_data=(x_test_c,y_test),epochs=5)

"""
Visualizing Convolutional Layers
"""
# visualize the weights of trained convolutional layers to try and make some more sense of what is going on
fig,ax = plt.subplots(1,9)
l = model.layers[0].weights[0]
for i in range(9):
    ax[i].imshow(l[...,0,i])
    ax[i].axis('off')

"""
Multi-layered CNNs and pooling layers
"""
# add several convolutional layers with pooling layers in between them to decrease dimensions of the image
# increase the number of filters to help look for more combinations
# this architecture is also called pyramid architecture because of decreasing spatial dimensions and increasing feature/filters dimensions
model = keras.models.Sequential([
    keras.layers.Conv2D(filters=10, kernel_size=(5,5), input_shape=(28,28,1),activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(filters=20, kernel_size=(5,5), activation='relu'),
    keras.layers.MaxPooling2D(),    
    keras.layers.Flatten(),
    keras.layers.Dense(10)
])

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['acc'])

model.summary()
