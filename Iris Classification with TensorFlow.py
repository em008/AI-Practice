"""
Iris Classification with TensorFlow
"""

"""
Getting the Dataset
"""
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
features = iris['data']
labels = iris['target']
class_names = iris['target_names']
feature_names = iris['feature_names']

print(f"Features: {feature_names}, Classes: {class_names}")

"""
Visualize the Data
"""
import seaborn as sns
import pandas as pd

df = pd.DataFrame(features,columns=feature_names).join(pd.DataFrame(labels,columns=['Label']))

df

sns.pairplot(df,hue='Label')

"""
Normalize and Encode the Data
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# Initialize the scaler
scaler = MinMaxScaler()

# Normalize the data
normalized_data = scaler.fit_transform(features)

# Create a new dataframe with normalized data
normalized_df = pd.DataFrame(normalized_data, columns=feature_names)

# Create an instance of the OneHotEncoder class
encoder = OneHotEncoder()

# Fit the encoder to the data and transform the data
encoded_labels = encoder.fit_transform(labels.reshape(-1, 1)).toarray()

"""
Split the Data into Train and Test
"""
from sklearn.model_selection import train_test_split

# Split the dataset into train and test sets
train_x, test_x = train_test_split(normalized_df, test_size=0.2, random_state=42)
train_labels, test_labels = train_test_split(encoded_labels, test_size=0.2, random_state=42)

# Print the shapes of the train and test sets
print("Train X data shape:", train_x.shape)
print("Test X data shape:", test_x.shape)

print("Train lables data shape:", train_labels.shape)
print("Test labels data shape:", test_labels.shape)

"""
Define and Train Neural Network
"""
# Define the neural network architecture
model = keras.models.Sequential()
model.add(keras.layers.Dense(1,input_shape=(2,),activation='sigmoid'))

# Compile the model
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.2),loss='binary_crossentropy',metrics=['accuracy'])

# Train the model
model.fit(x=train_x,y=train_labels,validation_data=(test_x,test_labels),epochs=10,batch_size=1)

# Visualize train/validation accuracy graph
hist = model.fit(x=train_x,y=train_labels,validation_data=(test_x,test_labels),epochs=10,batch_size=1)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
