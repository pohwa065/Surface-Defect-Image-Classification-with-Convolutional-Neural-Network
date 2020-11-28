# Surface-Defect-Image-Classification-with-Convolutional-Neural-Network
## Intorduction
The proposed idea is to use convolutional neural network (CNN) to classify defects on the semiconductor wafer substrate. There are total 7 classes to be classified. Data augmentation by Generative Adversarial Network (GAN) is applied on 2 of the classes to improve classification accuracy. 

![Picture4](https://user-images.githubusercontent.com/65942005/100525604-f61e1280-3176-11eb-82cf-4179bc905247.png)
(a) Sample class distribution. C5 and C6 have GAN generated synthetic images added; noise are removed for C1 and C6 (b) Proposed data preprocessing flow for training a CNN model (c) AC-GAN model (d) Autoencoder (AE) model



## Model Structure
![Picture5](https://user-images.githubusercontent.com/65942005/100525877-aa6c6880-3178-11eb-90a8-6b1d598e491c.png)
![Picture6](https://user-images.githubusercontent.com/65942005/100525878-aa6c6880-3178-11eb-8bc7-2a0aa8ebb5ed.png)

a) Generator consists of several convolution layers (CONV) follow by activation, batch normalization (BN) and up-sampling. (b) Discriminator also contains series of CONV layers follow by activation and dropout. At the end, it has a fully-connected layer with sigmoid function for binary classification.  

## Dataset
About 12k of images (resize to 128x128x1) are labeled. Most of them have well-defined defect shapes. Apart from labeled data, 210 and 3000 synthetic images of C5 and C6 were added to the dataset respectively. Also,  in some of the experiments, C1 and C6 are replaced with denoised version of the images. 

## Result
### GAN generated synthetic images 
![Picture1](https://user-images.githubusercontent.com/65942005/100525607-f9190300-3176-11eb-9937-6debe36097b2.png)
Original and synthetic images of C6. 28x28, 64x64 generated images are resized to 128x128 

![Picture2](https://user-images.githubusercontent.com/65942005/100525608-fb7b5d00-3176-11eb-9fe7-e0f9670a2e12.png)
Original and synthetic images of C5. The effect of different kernel size in the convolution layer is shown

### AE denoised images 
![Picture3](https://user-images.githubusercontent.com/65942005/100525609-fcac8a00-3176-11eb-9668-ac2a60c9c6e9.png)
Original and AE denoised images. The intensity of the feature are boosted to different levels




![Picture8](https://user-images.githubusercontent.com/65942005/100525874-a93b3b80-3178-11eb-962c-bf7851644fd5.png)
![Picture9](https://user-images.githubusercontent.com/65942005/100525876-aa6c6880-3178-11eb-8fb3-57c26d47f4b1.png)

