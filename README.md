# Surface-Defect-Image-Classification-with-Convolutional-Neural-Network
The proposed idea is to use convolutional neural network (CNN) to classify defects on the semiconductor wafer substrate. There are total 7 classes to be classified. Data augmentation by Generative Adversarial Network (GAN) is applied on 2 of the classes to improve classification accuracy. 

![Picture4](https://user-images.githubusercontent.com/65942005/100525604-f61e1280-3176-11eb-82cf-4179bc905247.png)

### Original and synthetic images of C6. 28x28, 64x64 generated images are resized to 128x128 
![Picture1](https://user-images.githubusercontent.com/65942005/100525607-f9190300-3176-11eb-9937-6debe36097b2.png)

### Original and synthetic images of C5. The effect of different kernel size in the convolution layer is shown
![Picture2](https://user-images.githubusercontent.com/65942005/100525608-fb7b5d00-3176-11eb-9fe7-e0f9670a2e12.png)

### Original and AE denoised images. The intensity of the feature are boosted to different levels
![Picture3](https://user-images.githubusercontent.com/65942005/100525609-fcac8a00-3176-11eb-9668-ac2a60c9c6e9.png)
