# 50.039 Deep Learning Small Project
---

[TOC]

# Team
**Keith Ng** 1003515
**Li Yuxuan** 1003607
**Ng Jia Yi** 1003696

# About Project

In this repo, you can find the main model running code in `50.039 Small Project 2 Binary.ipynb`. To test the model on the 25 images of the test dataset, find the code in `Binary Classifiers Test.ipynb`. To plot the graph showing distribution of the dataset, find the code in `distribution.ipynb`. 

Results are collected as `.csv` files as well as `.png` under the two folders `Binary Without Preprocessing - Results` and `Binary With Preprocessing - Results`. As names suggested, we ran two experiments, one with the image preprocessing (will be discussed later) and one without image preprocessing. Accuracy, Train Loss and Validation Loss are collected in CSV for graph plotting, and a pre-plotted images have been included in the respective folders.

Looking at the filename, the names are self-explanatory, except for the two symbols `cn` and `ni`. `ni` represents: **Normal-Infected** binary set and `cn` represents **Covid-NonCovid** set.

# Introduction

In this project, we attempt to design a deep learning model, whose task is to assist with the diagnosis of pneumonia, for COVID and non-COVID cases, by using X-ray images of patients. We engineered a lightweight convolutional neural network model that can collectively train/test the given x-ray scans. In this report, we will cover our design process and the challenges that come along with it. 

# Proposed Methodology

## Selected Classifier
Before we decide between multi-class classifiers or 2 binary classifiers, we first take a look at the composition of our dataset. 

![](https://i.imgur.com/jf4hXsX.png)

A multi-class classifier usually favours the largest classes and in this case, it is pretty obvious that there is a bias within our training dataset. The infected, non-covid case, has almost 2x the number of images as compared to the normal and infected and covid training datasets each. Due to the inherit biasness of the given dataset, we have chosen to use the 2 binary classifier instead of the multi-class classifier.

## Image Pre-Processing
We aim to reduce noise from our images with preprocessing techniques. One of the fundamental techniques is edge detection. It locates the borders of objects in images by identifying disjointedness in brightness. Edge detection methods can produce a new set of features. Edges are noted by serious or local changes in the image, and edges normally occur on the borders between two distinct areas in an image. Edges are important features for analysing digital images.

The edge detection method we applied here is called Enhanced Canny edge detection(Ceed-Canny). Our code implementation goes as follows

```
def ceed_canny(image):
    im = io.imread(image)
    enh = enhance_contrast(im, disk(3))
    edges = feature.canny(enh)
    final = img_as_ubyte(edges)
    return final
```

The image first undergoes morphological contrast enhancement. Morphology image processing are processes that manipulate images based on structuring elements. Morphological operations accept an image as input, apply a structuring element to the image, and outputs the resulting image. The pixel of the output image is determined from a comparison between the corresponding pixel of the input image with its neighbours.

Next, it undergoes the canny edge detection. The Canny filter is a multi-stage edge detector. First, noise reduction was performed by implementing Gaussian smoothing. Then, the intensity gradient of the image is computed so that non- maximum suppression can be employed. Likely edges are reduced to 1-pixel curves by discarding non- maximum pixels of the gradient magnitude. Lastly, hysteresis thresholding is performed to determine whether or not to keep the edge pixels.

The results of image preprocessing are as follows.

|   Original   |  MCE    |  Canny    |
| ---- | ---- | ---- |
|   ![](https://i.imgur.com/hjROBKE.png)   |  ![](https://i.imgur.com/60i8fZK.png)    |  ![](https://i.imgur.com/x5EZTQ0.png)    |

<div style="text-align: center"><b>Figure: Step by step image preprocessing result</b></div>
<br/> 

|Class| Before | After |
|----|----|---|
|Normal|![](https://i.imgur.com/3V0NYzg.png)|![](https://i.imgur.com/dUKiQMK.png)|
|Covid|![](https://i.imgur.com/WnbHJ9B.png)|![](https://i.imgur.com/1fqnVnR.png)|
|Non-Covid|![](https://i.imgur.com/jxoNSy6.png)|![](https://i.imgur.com/ssdaKnf.png)|

<div style="text-align: center"><b>Figure: Before vs After per class</b></div>
<br/> 

From the contour lines, we can clearly differentiate the three different classes as most noises are removed and the outlines are clearly presented.

## Image Augmentation

We decided that the given dataset was too small and we would not be able to train our model sufficiently. To cope with this, data augmentation was utilized to expand our training and validation sets. The following code depicts our code implementation.
```
train_transformer = transforms.Compose([transforms.RandomApply([
        transforms.RandomRotation(20, fill=(0,)),
        ],0.7)
  ])

```
Each imaged passed to our transformer has a 70% chance of going through a **random rotation**. We initially implemented other techniques such as horizontal flipping and zooming, but we realized that did not work so well.

```
if self.groups[0] == "train":
    for _ in range(2):
        img_Temp = img
        for _ in range(5):
            img_Temp = train_transformer(img_Temp)
            image = np.asarray(img_Temp) / 255
            image = transforms.functional.to_tensor(np.array(image)).float()
            imgs.append(image)
```

Every image will go through the transformer 2x5 times, and at each iteration, we save the resultant image. Hence, every 1 image produces 10 augmented images in total. Below is an example of how 10 augmented images(derived from the same original image) turns out.


![](https://i.imgur.com/4Z448E1.png)

<div style="text-align: center"><b>Figure: Resultant images undergone Data Augmentation</b></div>
<br/> 

## Selected Model
We started off by researching about the choices of models on doing COVID-19 chest X-ray image classifications. From one of the research papers that we found, we found that the comparison results show that the performance of VGG16 model on both original and augmented X-Ray dataset is better than Resnet50. (https://www.biorxiv.org/content/biorxiv/early/2020/07/17/2020.07.15.205567.full.pdf) For Train-Test ratio 90%-10%, for original, VGG16 achieves 99.25% accuracy whereas Resnet50 achieves 50% accuracy. For augmented, VGG16 achieves 93.84% accuracy whereas Resnet50 achieves 93.38% accuracy.

![](https://i.imgur.com/ZQMVusZ.png)

![](https://i.imgur.com/Y5fHXVS.png)

![](https://i.imgur.com/leGUUzM.png)

<section style="text-align:center">VGG16 architecture (<a href="https://neurohive.io/en/popular-networks/vgg16/">https://neurohive.io/en/popular-networks/vgg16/</a>)</section>

However, due to limited resources, especially the availability of GPUs, training the model on a full VGG16 architecture is too computationally intensive and time-consuming. Therefore, we continue our research further and found out that some papers used simpler models in their designs to achieve high accuracy of prediction as well. 

One of the paper we found (https://www.preprints.org/manuscript/202007.0591/v1/download) used a simple model on Kaggle's Covid19 Chest X-ray CT images dataset and achieved promising results as follows:
![](https://i.imgur.com/YfLqIaA.png)

Another paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7841043/) suggested a different model than the aforementioned ones and the proposed model is trained and tested with 10,000 X-ray images of COVID-19 and normal people combined for two classes. The accuracy evolved as follows:
![](https://i.imgur.com/wVQoMhc.png)

In the end, we decided to follow the first paper's model and designed our model as follows:

|Name|Description|
|----|-----------|
|Input| Size: 1 x 150 x 150|
|Convolution|in_channel=1, out_channel=20, kernel=5, stride=1|
|MaxPooling|kernel=3, stride=2, dilation=1|
|ReLU||
|Convolution|in_channel=20, out_channel=50, kernel=5, stride=1|
|MaxPooling|kernel=3, stride=2, dilation=1|
|ReLU||
|Dropout|rate=0.2|
|Fully Connected|500 neurons|
|Output Layer|2 classes|


# Implementation

We did most of our initial model testings on Google CoLab, before the access to the school GPU was given to us. 

## Custom Dataset

We customized the Dataset object to perform differently for the 2 binary classifiers, one of which is **Normal**,  **Infected**, and another one is  **Covid**, and **Non-Covid**. The Dataset object inherits from the torch dataset class and `__getitem__` method is overwritten to output the augmented image as well as its label. For the first classifier, we use **0** to represent **Normal**, **1** to represent **Infected**. For the second classifier, we use **0** to represents **Covid** and **1** to represents **Non-Covid**.

## Training

During training, we use a batch size of 32 images from the dataset for one training cycle. Each image is pre-processed and transformed randomly into 10 different images. After running 200 batches, we do one validation of the model. We wrote a custom function that will print out a graphical representation of the current performance on the validation of the model, shown as follows:

![](https://i.imgur.com/z2r8Fbh.jpg)


The column label represents the model prediction whereas the row label is the ground true label. If the prediction is correct, meaning that predicted label is the same as ground truth, it will be colored green. If not, it will be colored red. Additionally, we can see that the data augmentation has taken effect on some of the images, which are rotated, flipped and zoomed in.

With the 2 binary classifiers, we train each classifier independently. Each takes in a different DatasetLoader object. For **Normal-Infected** classifier as the first layer, the dataset consists of all the training dataset, but Covid and Non-Covid datasets are combined to form 1. For **Covid-NonCovid** classifier as the second layer, the dataset only consists of Covid and Non-Covid data without the Normal datasets. We customized our code such that user can supply different `mode` to activate different classifier training.

```:python3
# IMPORTANT! Input mode -- 0: Normal-Infected 1: Covid-NonCovid
mode = 0

if mode == 0:
    train_loader = train_loader_NI
    test_loader = test_loader_NI
    val_loader = val_loader_NI
elif mode == 1:
    train_loader = train_loader_CN
    test_loader = test_loader_CN
    val_loader = val_loader_CN
else:
    print("Mode can only be 0 or 1!")
    sys.exit(1) 
```

## Saving and Re-training

When we migrated our training onto the school GPU, it allowed us to train with less time and larger batch size of 32 (since originally we trained on Colab with batch size of 8 only). However, due to the complicated layers and large number of parameters, our training will crash when going into the 3rd epochs as CUDA GPU ran out of memory. Hence, we implemented a saving mechanism that during each training (process until it crashed), the model is constantly saved whenever the accuracy reached a higher value. This ensures that we always have the best model to train with at the start each time we re-run the training. Moreover, for visual representation, the training loss and validation loss, and accuracy are saved for each batch and saved in separate CSV to plot graphs. In the event of crash, the re-run will append to the previous accuracy, train loss and validation loss CSV so in the end we can plot an overall graph about how the three metrics evolve over time (number of batches).

# Results

In our final experiment, we ran 2 binary classifiers separately, each with batch size of **64**, with **data preprocessing** and **data augmentation** as discussed earlier, with validation after every **40 batches** of training, and running over 10 epochs. And then we tested our models with our test script, which chains the two models together. First model will predict the image is **Normal** or **Infected**, the **Infected** ones will then be passed on to the second model to be further classified into **Covid** or **Non-Covid**. Here are the results we obtained:

## Predictions Grid
![](https://i.imgur.com/TLn051v.png)


## Loss Profile
![](https://i.imgur.com/yO2wj5f.png)


## Accuracy Profile
![](https://i.imgur.com/iq96FYQ.png)

## Metrics

|Name|Value|
|:--:|:---:|
|F1 Score| 0.665688|
|Recall|0.680000|
|Accuracy|0.680|

## Confusion Matrix
![](https://i.imgur.com/GVAyZhw.png)

## Results Analysis

From the confusion matrix above, we can easily visualize that our model is able to differentiate between **Normal** and **Infected** clearly except for **1** case which is non-covid but our model thinks it is normal. However, most of the inaccuracy comes at the second part of the classification, where our second model fails to differentiate between covid and non-covid cases. This phenomenon is commonly faced as discussed later in the discussion section.

# Discussion
**Q**: You might find it more difficult to differentiate between non-covid and covid x-rays, rather than between normal x-rays and infected (both covid and non-covid) people x-rays. Was that something to be expected? Discuss.

**A**: This is expected as for covid vs non-covid, it is not recommended to be tested with x-rays. This is because a imaging findings are not specific enough to confirm Covid-19. They can only point to signs of an infection, that could be due to other causes such as seasonal flu. [https://blog.radiology.virginia.edu/covid-19-and-imaging/]

On the other hand, however, x-rays is an important test for making diagnosis of pneumonia. It can reveal areas of opacity (seen as white) which represent inflammation.

---

**Q**: Would it be better to have a model with high overall accuracy or low true negatives/false positives rates on certain classes? Discuss.

**A**: It will be good for prediction like our small project to have a low true negatives/false positive rates. For example:
![](https://i.imgur.com/hVTpo2A.png)

Will give a confusion matrix as follows:

![](https://i.imgur.com/5akpTds.png)

As for the accuracy, it will give:

$correct=TP+TN=5+940=945$
$incorrect=TP+FN=45+10=55$
$accuracy=\frac{correct}{correct+incorrect}=\frac{945}{945+55}=0.945$

As such, the accuracy seems like a high number, however, it will leave out a lot of patients susceptible to extreme losses. Take for example the FN cases where 10 people would be relieved that the doctor has told them that they did not get covid-19 but they actually did, and will do around spreading. Therefore, in this case, a low FP/FN will be more useful than a high accuracy score.

# Conclusion
We have preprocessed our image with **edge detection** to reduce noise and produce a new set of features. On top of that, we have done **data augmentation** to increase the size of our train dataset. We have picked a binary class classifier, where there are two separate models. For the first one, it is trained with **normal** vs **infected**, while for the second one, the model is trained with **covid** vs **non-covid**. After-which, we have obtained our results in terms of F1 score, recall, accuracy as well as confusion matrix. Due to the nature of our project (to predict infected vs normal and covid vs non-covid), we have concluded that it is better to present our data in a confusion matrix, shown in the results above.