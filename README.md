# Enhance-fuzzy-license-plates-image
Restore information in images and improve its quality.

## Problem definition 

When we resize an image to small dimension and resize it back to its original size we realize leak of information in the image, so we want to restore this information back to enhance quality of the image.

## Approaches
### U-net architecture with CNN (Convolution neural network) layers.

for more information about U-net https://en.wikipedia.org/wiki/U-Net , https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47

## Dataset 
https://github.com/5amessi/license_plates.git
https://github.com/SeyedHamidreza/car_plate_dataset.git

## GAN architecture with CNN (Convolution neural network) layers.
### Network Details:
* 16 Residual blocks used.
* Loss Function: Perceptual loss. It comprises of Content(Reconstruction) loss and Adversarial loss.
### How it Works:
* We process the HQ(High Quality) images to get down-sampled LQ(Low Quality) images. 
  Now we have both HQ and LQ images for training data set.
* We pass LQ images through Generator which gives HQ(High Quality) images.
* We use a discriminator to distinguish the HQ images and back-propagate the GAN loss to train the discriminator
  and the generator.
* As a result of this, the generator learns to produce more and more realistic images(High Quality images) as 
  it trains.
  
## ResNet.
# Network Details:
* 16 Residual blocks used.

# Examples

![alt text](https://github.com/5amessi/Enhance-fuzzy-license-plates-images/blob/master/images/plat2/1.jpg) 
![alt text](https://github.com/5amessi/Enhance-fuzzy-license-plates-images/blob/master/images/plat2/2.jpg)
![alt text](https://github.com/5amessi/Enhance-fuzzy-license-plates-images/blob/master/images/plat2/4.jpg)
![alt text](https://github.com/5amessi/Enhance-fuzzy-license-plates-images/blob/master/images/plat2/5.jpg)
ــــــــــ
![alt text](https://github.com/5amessi/Enhance-fuzzy-license-plates-images/blob/master/images/plat3/1.jpg) 
![alt text](https://github.com/5amessi/Enhance-fuzzy-license-plates-images/blob/master/images/plat3/2.jpg)
![alt text](https://github.com/5amessi/Enhance-fuzzy-license-plates-images/blob/master/images/plat3/3.jpg)
![alt text](https://github.com/5amessi/Enhance-fuzzy-license-plates-images/blob/master/images/plat3/4.jpg)


