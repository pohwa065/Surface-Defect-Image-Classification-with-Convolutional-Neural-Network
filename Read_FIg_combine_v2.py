from pathlib import Path
from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import cv2
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops


def one_hot_matrix(labels, C):

    ### START CODE HERE ###

    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name='C')

    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)

    # Create the session (approx. 1 line)
    sess = tf.Session()

    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)

    # Close the session (approx. 1 line). See method 1 above.
    sess.close()

    ### END CODE HERE ###

    return one_hot

def read_and_label (Source,label):
    X_train_orig_dict = {}
    file_name = []
    for file in Path(Source).iterdir():
        name = file.stem
        dir= Source +'\\'+ name + '.fig'
        file_name.append(dir)
        matlab_fig = loadmat(dir)
        image = matlab_fig['hgS_070000'][0][0][3][0][0][3][0][0][2][0][0][0]
        image_resize = resize(image, (128, 128))
        X_train_orig_dict[dir]= image_resize

    plt.imshow(X_train_orig_dict[file_name[1]])
    #plt.show()


    myfourdarray = []
    file_name = []

    for file in Path(Source).iterdir():
        name = file.stem
        dir = Source + '\\' + name + '.fig'
        file_name.append(dir)
        matlab_fig = loadmat(dir)
        image = matlab_fig['hgS_070000'][0][0][3][0][0][3][0][0][2][0][0][0]
        image_resize = resize(image, (128, 128))
        image_resize2 = image_resize / np.max(image_resize)  # image to 0-1
        img2 = cv2.merge((image_resize2, image_resize, image_resize))  # resize very image to same dimension
        myfourdarray.append(img2)

    X_ = np.stack(myfourdarray, axis=0)
    #print(X_.shape)

    #plt.imshow(X_[0][:,:,0])
    #plt.show()


    y_ = {}
    file_name = []

    for file in Path(Source).iterdir():
        name = file.stem
        dir= Source +'\\'+ name + '.fig'
        file_name.append(dir)
        y_[dir]= label

    y_ = np.array([y_[key] for key in file_name]).T
    #print(y_.shape)

    return X_,y_


X_0, y_0 = read_and_label('C:\Po-Hsiang\cs230\MS2_Test_Train\_labeled data\C0',0)
X_0 = X_0[:,:,:,0].reshape(X_0.shape[0],X_0.shape[1],X_0.shape[2],1)
print(X_0.shape,y_0.shape)
X_1, y_1 = read_and_label('C:\Po-Hsiang\cs230\MS2_Test_Train\_labeled data\C1',1)
X_1 = X_1[:,:,:,0].reshape(X_1.shape[0],X_1.shape[1],X_1.shape[2],1)
print(X_1.shape,y_1.shape)
X_2, y_2 = read_and_label('C:\Po-Hsiang\cs230\MS2_Test_Train\_labeled data\C2',2)
X_2 = X_2[:,:,:,0].reshape(X_2.shape[0],X_2.shape[1],X_2.shape[2],1)
print(X_2.shape,y_2.shape)
X_3, y_3 = read_and_label('C:\Po-Hsiang\cs230\MS2_Test_Train\_labeled data\C3',3)
X_3 = X_3[:,:,:,0].reshape(X_3.shape[0],X_3.shape[1],X_3.shape[2],1)
print(X_3.shape,y_3.shape)
X_4, y_4 = read_and_label('C:\Po-Hsiang\cs230\MS2_Test_Train\_labeled data\C4',4)
X_4 = X_4[:,:,:,0].reshape(X_4.shape[0],X_4.shape[1],X_4.shape[2],1)
print(X_4.shape,y_4.shape)
X_5, y_5 = read_and_label('C:\Po-Hsiang\cs230\MS2_Test_Train\_labeled data\C5',5)
X_5 = X_5[:,:,:,0].reshape(X_5.shape[0],X_5.shape[1],X_5.shape[2],1)
print(X_5.shape,y_5.shape)
X_6, y_6 = read_and_label('C:\Po-Hsiang\cs230\MS2_Test_Train\_labeled data\C6',6)
X_6 = X_6[:,:,:,0].reshape(X_6.shape[0],X_6.shape[1],X_6.shape[2],1)
print(X_6.shape,y_6.shape)


# Concatenate 

args = (X_0,X_1,X_2,X_3,X_4,X_5,X_6)
arg2 = (y_0,y_1,y_2,y_3,y_4,y_5,y_6)
t = np.concatenate(args,axis=0)
t2 = np.concatenate(arg2,axis=0)
# Train_test split
print(t.shape)
print(t2.shape)


X_train, X_test, y_train, y_test = train_test_split(t,t2, test_size=0.99, random_state=42)

Y_train = one_hot_matrix(y_train, 7).T
Y_test = one_hot_matrix(y_test, 7).T


np.save('X_train', X_train)
np.save('X_test', X_test)
np.save('Y_train', Y_train)
np.save('Y_test', Y_test)

np.save('y_test_orig', y_test)
np.save('y_train_orig', y_train)


print('X_train shape:' +str(X_train.shape))
print('Y_train shape:' +str(Y_train.shape))
print('X_test shape:' + str(X_test.shape))
print('Y_test shape:' +str(Y_test.shape))
print('y_train_orig shape:' + str(y_train.shape))
print('y_test_orig shape:' + str(y_test.shape))



list = [1,4,5,7,4,564,432,526,2,14,567]

for i in list:
    plt.imshow(X_train[i][:,:,0])
    plt.colorbar()
    plt.show()
    print(y_train[i], Y_train[i])
