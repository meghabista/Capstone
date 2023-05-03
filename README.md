# AN EFFECTIVE APPROACH TO DETECT BREAST CANCER TUMORS USING SEMENTATIONS AND ATTENTION U-NET MODEL 
## Table of Contents
 1. Introduction
 2. Dataset
 3. Problem Formulation
 4. Model
 5. Conclusion and Results
 6. Evaluation
 
 # INTRODUCTION

## Background:
  Breast cancer is a type of cancer that forms in the cells of the breast. It occurs when the normal cells in the breast begin to grow uncontrollably, forming a mass or lump. Over time, the cancer cells may invade nearby tissues and spread to other parts of the body through the lymphatic system or bloodstream. It is also a leading cause of death for women worldwide, but early detection can greatly reduce mortality rates. Medical imaging, such as ultrasound scans, can aid in the early detection of breast cancer. By leveraging machine learning techniques ,breast ultrasound images can be effectively used for classification, detection, and segmentation of breast cancer tumors, leading to improved outcomes for patients. By training machine learning algorithms on large datasets of ultrasound images like the one I'll be using, researchers can develop models such as  Attention U-Net model that can predict the probability of malignancy based on various features of the ultrasound images which can improve the accuracy of tumor detection and reducing false positives.
 
 ## Task:
Build a model that segments the tumor/cancer out of the ultrasound images with high accuracy and performance


## Dataset

## Overview of the Dataset:
* Source: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.
* The data collected at baseline includes breast ultrasound images among women in ages between 25 and 75 years old. 
* This data was collected in 2018. The number of patients are 600 female patients.
* The dataset consists of 780 images with an average image size of 500*500 pixels.
* The images are in PNG format. 
* The ground truth images are presented with original images.
* The images are categorized into three classes, which are normal, benign, and malignant.
 * Normal: 133
 * Benign: 487
 * Maligant: 210
* Each classes also includes two different masks.



## Data visualization
* Merging Images of masks that belong to same class

Example 1:

![image](https://user-images.githubusercontent.com/89595947/236058931-68e1ddcf-7664-48d8-8c0a-38478fd96bbe.png)

![image](https://user-images.githubusercontent.com/89595947/236058950-e2be0b5b-01d2-4ad5-8c66-cfc8bb4cd148.png)

![image](https://user-images.githubusercontent.com/89595947/236058867-ef30ca4c-ced4-4bad-8bb2-437bfefa3e8e.png)

Example 2:

![image](https://user-images.githubusercontent.com/89595947/236059972-4e59f8bc-5e71-4a1e-bc93-eac302b68d19.png)


# Problem Formulation:
 * Resizing the images to 256 x 256 pixels from 500 x 500 pixels
 * Merging images with masks that belong to same class
 * Creating encoder, decoder and attention gate layers and implementing them.
 * Building an Attention U-net model
 * Training the Data

# Model

## Attention U-net model:

The Attention U-Net model is a variant of the U-Net architecture, which is a popular convolutional neural network (CNN) commonly used for image segmentation tasks. The attention mechanism in the Attention U-Net model allows the network to focus on more relevant features during the segmentation process.The U-Net architecture consists of an encoder network that progressively reduces the spatial dimensions of the input image, and a decoder network that expands the feature maps to the original size of the image. The attention mechanism in the Attention U-Net model allows the network to learn to selectively focus on specific parts of the image during the segmentation process. This is achieved by incorporating a gating mechanism that learns to amplify or suppress different parts of the feature maps based on their relevance to the segmentation task.The Attention U-Net model has been shown to perform well on a variety of image segmentation tasks, including medical image segmentation and semantic segmentation of natural images. Its attention mechanism allows the model to better capture fine-grained details and produce more accurate segmentations.
![image](https://user-images.githubusercontent.com/89595947/236074770-b5d916d8-1ff8-46a8-b14e-03cdd08a38a3.png)

U-Net architecture

![image](https://user-images.githubusercontent.com/89595947/236074833-a7b0b6bd-a8a8-40b8-8522-9a405c86b6c1.png)

Breakdown of the attention gates

Source:https://towardsdatascience.com/a-detailed-explanation-of-the-attention-u-net-b371a5590831

# Conclusion and Results:
  * After 12 epochs, the model started outputting what we needed
  
  * The model was easily able to detect black round spots but failed when the shape is         irregular

  * The model also got confused between dark areas. 

Epoch: 1/20
![image](https://user-images.githubusercontent.com/89595947/236066874-229b4391-80ad-405d-afad-7ab9313d9d50.png)

Epoch: 17/20
![image](https://user-images.githubusercontent.com/89595947/236066989-d472de84-25f6-48bf-8004-8aa1f89802a2.png)


# Evaluation
![image](https://user-images.githubusercontent.com/89595947/236067111-4df84ae0-6ca6-4342-adce-6b47ab3fa76b.png)

* The results on validation data are way better than the results on Training Data on IoU. This may indicate that the model can perform way better than what it can do at the current point.
* The Loss is not perfect as it increases in the last. 


