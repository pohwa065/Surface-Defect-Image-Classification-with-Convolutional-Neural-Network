import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from pathlib import Path
from scipy.io import loadmat
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import cv2
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.constraints import max_norm
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers


#Load data

X_train = np.load('X_train_1.npy')
X_train_lowSNR= np.load('X_train_1_lowSNR.npy')
#Y_train = np.load('Y_train_ex6.npy')
X_test = np.load('X_test_1.npy')
#Y_test = np.load('Y_test_ex1.npy')

y_test_orig = np.load('y_test_orig_1.npy')
y_train_orig = np.load('y_train_orig_1.npy')

print('X_train shape:' +str(X_train.shape))
#print('Y_train shape:' +str(Y_train.shape))
print('X_test shape:' + str(X_test.shape))
# #print('Y_test shape:' +str(Y_test.shape))
print('y_train_orig shape:' + str(y_train_orig.shape))
print('y_test_orig shape:' + str(y_test_orig.shape))

# Model configuration
img_width, img_height = 128, 128
input_shape = (img_width, img_height, 1)
batch_size = 150
no_epochs = 50
validation_split = 0.2
verbosity = 1
max_norm_value = 2.0
noise_factor = 0.55
number_of_visualizations = 6

# Load MNIST dataset
input_train = X_train
target_train =y_train_orig
input_test =X_test
target_test = y_test_orig

# Reshape data based on channels first / channels last strategy.
# This is dependent on whether you use TF, Theano or CNTK as backend.
# Source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Add noise
pure = input_train
pure_test = input_test
noise = np.random.normal(0, 0.1, pure.shape)
noise_test = np.random.normal(0, 0.1,pure_test.shape)
noisy_input = X_train_lowSNR+ noise_factor * noise
#noisy_input = pure + noise_factor * noise
noisy_input_test = pure_test + 0.55* noise_test

idx =2
imgs = [pure, pure_test,noisy_input,noisy_input_test]

for img in imgs:
  show = img[idx,:,:,0]
  plt.imshow(show)
  plt.colorbar()
  plt.show()



# Create the model
model = tf.keras.Sequential([layers.Conv2D(16, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape,padding="same"),
layers.Conv2D(8, kernel_size=(1, 1), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform',padding="same"),
layers.Conv2DTranspose(8, kernel_size=(1,1), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform',padding="same"),
layers.Conv2DTranspose(16, kernel_size=(1,1), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform',padding="same"),
layers.Conv2D(1, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='sigmoid', padding='same')])




# Compile and fit data
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()
model.fit(noisy_input, pure,
                epochs=no_epochs,
                batch_size=batch_size,
                validation_split=validation_split)


# Generate denoised images
samples = noisy_input_test[:number_of_visualizations]
targets = target_test[:number_of_visualizations]
denoised_images = model.predict(samples)
model.save('AF__1')

#Extract feature

extractor = tf.keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])


features = extractor(pure_test)



#layer0
#shape = (m, 128,128,1)
c = 0   #channel
fig = plt.figure()
for i in range(1,5):
  img = pure_test[i,:,:,c]
  fig.add_subplot(2,2, i)
  plt.imshow(img)

plt.title('Layer0')
#plt.show()



#layer1
#features[0]    #shape = (m, 128,128,64)
c = 1   #channel
fig = plt.figure()
for i in range(1,5):
  L1 = tf.keras.backend.eval(features[0])
  img = L1[i,:,:,c]
  fig.add_subplot(2,2, i)
  plt.imshow(img)

plt.title('Layer1')
#plt.show()


#layer2
#features[1]    #shape = (m, 128,128,32)
c = 1   #channel
fig = plt.figure()
for i in range(1,5):
  L2 = tf.keras.backend.eval(features[1])
  img = L2[i,:,:,c]
  fig.add_subplot(2,2, i)
  plt.imshow(img)

plt.title('Layer2')
#plt.show()


#layer3
#features[2]    #shape = (m, 128,128,32)
c = 1   #channel
fig = plt.figure()
for i in range(1,5):
  L3 = tf.keras.backend.eval(features[2])
  img = L3[i,:,:,c]
  fig.add_subplot(2,2, i)
  plt.imshow(img)

plt.title('Layer3')
#plt.show()


#layer4
#features[3]    #shape = (m, 128,128,64)
c = 1   #channel
fig = plt.figure()
for i in range(1,5):
  L4 = tf.keras.backend.eval(features[3])
  img = L4[i,:,:,c]
  fig.add_subplot(2,2, i)
  plt.imshow(img)

plt.title('Layer4')
#plt.show()


#layer5 (output)
#features[4]    #shape = (m, 128,128,1)
c = 0   #channel
fig = plt.figure()
for i in range(1,5):
  L5 = tf.keras.backend.eval(features[4])
  img = L5[i,:,:,0]
  fig.add_subplot(2,2, i)
  plt.imshow(img)

plt.title('Layer5')
#plt.show()



# Plot denoised images
for i in range(0, number_of_visualizations):
  # Get the sample and the reconstruction
  noisy_image = noisy_input_test[i][:, :, 0]
  pure_image  = pure_test[i][:, :, 0]
  denoised_image = denoised_images[i][:, :, 0]
  for i in range(len(denoised_image)):
    for j in range(len(denoised_image)):
      if denoised_image[i,j] < 0.94*np.max(denoised_image):
        denoised_image[i,j] =0

  #            image_resize3[i,j] = image_resize3[i,j] + 0.7*np.abs((np.max(image_resize3)-image_resize3[i,j]))

  # Matplotlib preparations
  fig, axes = plt.subplots(1, 3)
  fig.set_size_inches(8, 3.5)
  # Plot sample and reconstruciton
  axes[0].imshow(noisy_image)
  axes[0].set_title('Noisy image')
  axes[1].imshow(pure_image)
  axes[1].set_title('Pure image')
  axes[2].imshow(denoised_image)
  axes[2].set_title('Denoised image')
  plt.show()

