# Surface-Defect-Image-Classification-with-Convolutional-Neural-Network
## Intorduction
The proposed idea is to use convolutional neural network (CNN) to classify defects on the semiconductor wafer substrate. There are total 7 classes to be classified. Data augmentation by Generative Adversarial Network (GAN) is applied on 2 of the classes to improve classification accuracy. 

![Picture4](https://user-images.githubusercontent.com/65942005/100525604-f61e1280-3176-11eb-82cf-4179bc905247.png)
(a) Sample class distribution. C5 and C6 have GAN generated synthetic images added; noise are removed for C1 and C6 (b) Proposed data preprocessing flow for training a CNN model (c) AC-GAN model (d) Autoencoder (AE) model



## Model Structure
![Picture2](https://user-images.githubusercontent.com/65942005/100526142-f91b0200-317a-11eb-8e2f-c76940327ca0.png)

a) Generator consists of several convolution layers (CONV) follow by activation, batch normalization (BN) and up-sampling. (b) Discriminator also contains series of CONV layers follow by activation and dropout. At the end, it has a fully-connected layer with sigmoid function for binary classification.  (c) Stacked autoencoder with convolution layers (encoder) and transposed convolution layers (decoder). 

## Dataset
About 12k of images (resize to 128x128x1) are labeled. Most of them have well-defined defect shapes. Apart from labeled data, 210 and 3000 synthetic images of C5 and C6 were added to the dataset respectively. Also,  in some of the experiments, C1 and C6 are replaced with denoised version of the images. 

## Result
### GAN generated synthetic images 
![Picture1](https://user-images.githubusercontent.com/65942005/100526144-fa4c2f00-317a-11eb-9d0e-c09fa6c5b962.png) <br>
(Upper) Original and synthetic images of C6. 28x28, 64x64 generated images are resized to 128x128 <br>
(Lower) Original and synthetic images of C5. The effect of different kernel size in the convolution layer is shown <br>

### AE denoised images 
![Picture3](https://user-images.githubusercontent.com/65942005/100525609-fcac8a00-3176-11eb-9668-ac2a60c9c6e9.png) <br>
Original and AE denoised images. The intensity of the feature are boosted to different levels


### CNN Classifier performance with data augmentation and denoised images
The synthetic images from GAN are similar to the original images, confirmed not only visually but also through a baseline classifier. With the help of these GAN generated images in the training set, the classification accuracy of the class having small labeled data improves by 4.5%. AE successfully removed the background noise in the images from the training set. However, the model failed to recognize key features of the raw images in the test set.  

Table1: Sample distribution in training and testing set <br>
![Picture8](https://user-images.githubusercontent.com/65942005/100525874-a93b3b80-3178-11eb-962c-bf7851644fd5.png)<br>
Table2: Accuracy of each class for each testcase <br>
![Picture9](https://user-images.githubusercontent.com/65942005/100525876-aa6c6880-3178-11eb-8fb3-57c26d47f4b1.png)<br>

### Error analysis 

![Picture10](https://user-images.githubusercontent.com/65942005/100526215-a5f57f00-317b-11eb-9bf8-4eede5fd6216.png)
