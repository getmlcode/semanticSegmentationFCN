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

**_Import_**

```python
from ImageSemanticSegmentor.FCN.FCN import FullyConvNet
```

**_Then set following directories_**

| Directory    | Content         |
| ------------- | :-------------:|
**`vggModelDir`**    | **`Pretrained VGG weights`**
**`trainDataDir`**    | **`Training Images`**
**`trainLabelDir`**    | **`Training Image Labels`**
**`validationDir`**    | **`Validation Images`**
**`fcnModelDir`**    | **`Saved Model Weights`**
**`fcnInferDir`**    | **`Model Weights For Inference`**
**`testDataDir`**    | **`Test Images`**
**`testResultDir`**    | **`Inference Results Of Test Images`**

**_Create Object For Training_**

```python
imageSegmenter = FullyConvNet(trainSession, vggModelDir, trainDataDir, trainLabelDir, 
                              validationDir, fcnModelDir, testDataDir, 
                              fcnInferDir, numOfClasses)
```


**_Then Set Optimization Parameters_**

| Parameter    | Purpose         | Note |
| ------------- | :-------------:|:---:|
**`optAlgo`**    | **`Optimization Algoritm`** | **`Only 3 are suppored`**
**`initLearningRate`**    | **`Step Size`**
**`ImgSize`**    | **`Image Dimension To Resize Train Images`**
**`maxGradNorm`**    | **`Maximum Gradient Norm For Clipping Gradient`** | **`Needed To Prevent Exploding Gradient`**

**_Set optimizer_**

```python
imageSegmenter.setOptimizer(optAlgo, initLearningRate, ImgSize, maxGradNorm)
```

**_Then set training parameters_**

| Parameter    | Purpose         |
| ------------- | :-------------:|
**`batchSize`**    | **`Number Of Images In Training Batch`**
**`keepProb`**    | **`Dropout Probability`**
**`metric`**    | **`Performance Metric To Use`**
**`numOfEpochs`**    | **`Number Of Times To Iterate Through Whole Train Data`**
**`saveModel`**    | **`Save Learnt Models ? `**
**`perfThresh`**    | **`Minimum Acceptable Model Performance`**
**`showSegValImages`**    | **`Display Segmented Images During Model Validation`**

**_Start training_**
```python
imageSegmenter.trainFCN(batchSize, keepProb, metric, numOfEpochs, saveModel,
                        perfThresh, showSegValImages)
```

#### *`Then Sit Back And Wait`*

---

#### `Inference`
---



---
# Future Work
* [ ] Add GUI for both training and inference for user interaction.~~
* [ ] Develop a website to provide this as a web service.
* [ ] Extend/Modify this framework to support other semantic segmentation models.

# References
To Be Added