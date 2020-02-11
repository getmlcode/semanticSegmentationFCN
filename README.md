# Semantic Segmentation
This repository has framework for training, validation and inference of semantic segmentation models, but currenty it only supports FCN.
I have used it to train **Fully Convolutional Network** and detect road in a given scene using learnt model.  

`Developed using Tensorflow Framework.`

# Training Progress

#### **`Training Details`**

| Parameter    | Value         |
| ------------- | :-------------:|
**`Optimization Algorithm`**    | **`Adam`**
**`Learning Step Schedule`**    | **`Exponential Decay`**
**`Regularization`**             | **`Dropout = 0.5`**
**`Exploding Gradient Remedy`** | **`Clipping Gradient (Max Norm = 0.1)`**
**`Performance Metric`**         | **`Intersection Over Union (IOU)`**
**`Numper Of Epochs`**           | **`50`**
**`Batch Size`**                 | **`32`**

---

#### `Epoch 5 --> 10 --> 15 on validation data`

|               |		        |
| ------------- |:-------------:|
![](res/Figure_13.gif) | ![](res/Figure_17.gif)
![](res/Figure_9.gif) | ![](res/Figure_6.gif)
![](res/Figure_17.gif) |

---

# Test Results
#### `After training for 50 Epochs`
|       Total 160 Images        |		        
| ------------- |
![](res/Epoch50_Test_Result.gif)|

# Usage Guide

#### `Training`

---

Add Here

---

#### `Inference`

---

Add Here

---

# Future Work
* [ ] Add GUI for both training and inference for user interaction.~~
* [ ] Develop a website to provide this as a web service.
* [ ] Extend/Modify this framework to support other semantic segmentation models.