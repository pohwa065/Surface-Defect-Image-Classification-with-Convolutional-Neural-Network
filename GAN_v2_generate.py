from __future__ import print_function, division
from skimage.transform import resize
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import model_from_json
import matplotlib.pyplot as plt

import numpy as np


json_file = open('generator_C6_9800.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("generator_weight_C6_9800.hdf5")
print("Loaded model from disk")

loaded_model.summary()


myfourdarray = []
generate = 10000
y_gen = {}
label = 6
#dir ="C:\Po-Hsiang\cs230\GAN\C6_128x128_Gen\%d.png"

for i in range(generate):
    r, c = 1, 1
    noise = np.random.normal(0, 1, (r * c, 100))
    #sampled_labels = np.array([num for _ in range(r) for num in range(c)])
    sampled_labels = np.zeros(generate)
    valid = np.ones((1, 1))
    gen_imgs = loaded_model.predict([noise, sampled_labels])
    gen_imgs = gen_imgs[0,:,:,:]
    #Rescale images 0 - 1
    gen_imgs= 0.5 * gen_imgs + 0.5
    #gen_imgs_up = resize(gen_imgs, (128, 128))
    #print(gen_imgs.shape)
    #print(gen_imgs_up[60,60,0])
    #plt.imshow(gen_imgs_up[:,:,0])
    #plt.colorbar()
    #plt.savefig(dir %i)
    #plt.show()
    myfourdarray.append(gen_imgs)

X_gen = np.stack(myfourdarray, axis=0)
y_gen = np.full((generate,1),label)


print(X_gen.shape)     # (m, 128, 128, 1)
print(y_gen.shape)     # (m,1)
print(y_gen)

np.save('X_gen_C6_128128_3' , X_gen)
np.save('y_gen_C6_128128_3.npy' , y_gen)