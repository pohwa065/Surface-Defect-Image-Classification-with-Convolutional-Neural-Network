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
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical


#Load data

X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')

y_test_orig = np.load('y_test_orig.npy')
y_train_orig = np.load('y_train_orig.npy')

print('X_train shape:' +str(X_train.shape))
print('Y_train shape:' +str(Y_train.shape))
print('X_test shape:' + str(X_test.shape))
print('Y_test shape:' +str(Y_test.shape))
print('y_train_orig shape:' + str(y_train_orig.shape))
print('y_test_orig shape:' + str(y_test_orig.shape))


batch_size = 32
epochs = 200
inChannel = 1
x, y = 128, 128
input_shape = (x, y, inChannel)
num_classes = 7

def model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    # kernel_regularizer=regularizers.l2(0.01)
    X_input = Input(input_shape)

    # CONV1 -> BN -> RELU Block applied to X
    X = Conv2D(8, (4, 4), strides = (1, 1), name = 'conv0',kernel_regularizer=regularizers.l2(0.001),padding="same")(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)
    #X = Dropout(0.5)(X)

    # MAXPOOL1
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # CONV2 -> BN -> RELU Block applied to X
    X = Conv2D(16, (2, 2), strides=(1, 1), name='conv1',kernel_regularizer=regularizers.l2(0.001),padding="same")(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)
    #X = Dropout(0.5)(X)

    # MAXPOOL2
    X = MaxPooling2D((2, 2), name='max_pool2')(X)

    # CONV3 -> BN -> RELU Block applied to X
    X = Conv2D(32, (1, 1), strides=(1, 1), name='conv2',kernel_regularizer=regularizers.l2(0.001),padding="same")(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)
    #X = Dropout(0.5)(X)

    # MAXPOOL3
    X = MaxPooling2D((2, 2), name='max_pool3')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(7, activation='softmax', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='CNN')

    return model

CNN_model = model(input_shape)
CNN_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
CNN_model.summary()
Train = CNN_model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_test, Y_test))

plt.plot(Train.history['accuracy'])
plt.plot(Train.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

test_eval = CNN_model.evaluate(X_test,Y_test)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


predicted_classes = CNN_model.predict(X_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)





correct = [i for i,item in enumerate(predicted_classes) if item == y_test_orig[i]]
wrong = [i for i,item in enumerate(predicted_classes) if item != y_test_orig[i]]

print(predicted_classes)
print(y_test_orig)
print(correct)
print(wrong)



accuracy={}
for i in range(7):
    all = np.sum(y_train_orig == i)
    correct = np.array([predicted_classes == y_train_orig]) & np.array([predicted_classes == i])
    correct_count = np.sum(correct)
    accuracy[i] = correct_count/all
    print(all)
    print(correct_count)


accuracy={}
for i in range(7):
    all = np.sum(y_test_orig == i)
    correct = np.array([predicted_classes == y_test_orig]) & np.array([predicted_classes == i])
    correct_count = np.sum(correct)
    accuracy[i] = correct_count/all
    print(all)
    print(correct_count)

print('C0 accuracy = '+ str(accuracy[0]))
print('C1 accuracy = '+ str(accuracy[1]))
print('C2 accuracy = '+ str(accuracy[2]))
print('C3 accuracy = '+ str(accuracy[3]))
print('C4 accuracy = '+ str(accuracy[4]))
print('C5 accuracy = '+ str(accuracy[5]))
print('C6 accuracy = '+ str(accuracy[6]))

#img = correct[1]
#plt.imshow(X_test[img][:,:,0])
#plt.show()

for i in range(len(wrong)):
    print(Y_test[wrong[i]], 'ground truth:' +str(y_test_orig[wrong[i]]), 'predict:' +str(predicted_classes[wrong[i]]))
    plt.imshow(X_test[wrong[i]][:,:,0])
    plt.colorbar()
    plt.show()



#print(Y_test[img],y_test_orig[img], prediction[img])