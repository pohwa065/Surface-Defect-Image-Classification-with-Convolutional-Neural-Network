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
import random

#Load data


X_test = np.load('X_C6_e11.npy')
y_test_orig = np.load('Y_C6_e11.npy')

print('X_test shape:' + str(X_test.shape))
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
number_of_visualizations = len(X_test)

# Load MNIST dataset
input_test =X_test
target_test = y_test_orig

# Reshape data based on channels first / channels last strategy.
# This is dependent on whether you use TF, Theano or CNTK as backend.
# Source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

pure_test = input_test
#noisy_input = X_train_lowSNR+ noise_factor * noise
#noisy_input = pure + noise_factor * noise
noisy_input_test = pure_test + 0* noise_test

model = tf.keras.models.load_model("AF__6")
model.summary()
print('model loaded')

# Generate denoised images
samples = noisy_input_test[:number_of_visualizations]
targets = target_test[:number_of_visualizations]
denoised_images = model.predict(samples)


#Extract feature
extractor = tf.keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])

features = extractor(pure_test)


myfourdarray = []

# Plot denoised images
for k in range(0, number_of_visualizations):
  # Get the sample and the reconstruction
  noisy_image = noisy_input_test[k][:, :, 0]
  pure_image  = pure_test[k][:, :, 0]
  denoised_image = denoised_images[k][:, :, 0]
  for i in range(len(denoised_image)):
    for j in range(len(denoised_image)):
      if denoised_image[i,j] < 0.98*np.max(denoised_image):
        #denoised_image[i,j] = 0
        denoised_image[i,j] = denoised_image[i,j] - 25*np.abs((np.max(denoised_image)-denoised_image[i,j]))
      if i > 110:
        denoised_image[i, j] = denoised_image[i-3, j]
      if j >110:
        denoised_image[i, j] = denoised_image[i, j-3]
      if i < 20:
        denoised_image[i, j] = denoised_image[i+3, j]
      if j <20:
        denoised_image[i, j] = denoised_image[i, j+3]
  denoised_image = 1/(1+np.exp(-denoised_image))  # image to 0-1
  denoised_image = denoised_image/np.max(denoised_image)

  myfourdarray.append(denoised_image)

  # Matplotlib preparations
  #fig, axes = plt.subplots(1, 3)
  #fig.set_size_inches(8, 3.5)
  # Plot sample and reconstruciton
  #axes[0].imshow(noisy_image)
  #axes[0].set_title('Noisy image')
  #axes[1].imshow(pure_image)
  #axes[1].set_title('Pure image')
  #axes[2].imshow(denoised_image)
  #axes[2].set_title('Denoised image')
  #plt.show()

X_ = np.stack(myfourdarray, axis=0)
np.save('X_C6_e11_1', X_)
print(X_.shape)


list = [random.randint(1, 3000) for iter in range(50)]

for i in list:
    plt.imshow(X_[i])
    plt.colorbar()
    plt.show()
    print(y_test_orig[i])
