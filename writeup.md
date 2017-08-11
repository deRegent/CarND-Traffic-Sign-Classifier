#**Traffic Sign Recognition** 

##Writeup 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/data_set_exploration_classes.png "Dataset examples"
[image2]: ./writeup/train_dataset_distribution.png "Distribution of the train set"
[image3]: ./writeup/validation_dataset_distribution.png "Distribution of the validation set"
[image4]: ./writeup/test_dataset_distribution.png "Distribution of the test set"
[image5]: ./writeup/data_generation.png "Examples of generated data"
[image6]: ./writeup/generated_data_distribution.png "Extended dataset distribution"
[image7]: ./writeup/preprocessing_example.png "Image preprocessing example"
[image8]: ./writeup/train_valid_accuracy.png "Image preprocessing example"
[image9]: ./writeup/my_dataset_grayscale.png "Random images after preprocessing"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Dataset Exploration

#### Dataset Summary 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Exploratory Visualization

Here is an exploratory visualization of the data set. 
This is a grid with a single example per each of 43 classes from the dataset.

![alt text][image1]

I also calculated distributions of classes in training set, validation set and test set. The distributions are showed below:

![alt text][image2]
![alt text][image3]
![alt text][image4]

Dataset exploration summary

The dataset does not contain images with specific standard of image brightness and contrast, so it does not seem to be a good idea to train classifier on raw images. We should normalize all the images, so network may generalize well.

The classes of signs are not equally distributed, so network may not generalise well for the classes which have a small amount of examples. In order to make network more robust dataset should become more balanced which leads us to the problem of generation additional data.

### Design and Test a Model Architecture

I highly relied on Pierre Sermanet and Yann LeCun's paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" in the process of designing my network. I also found Tommy Mulc's explanation of Inception modules very helpful and I used preprocessing technique described in Alex Staravoitau's blog. 

- Pierre Sermanet and Yann LeCun's paper: http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
- Tommy Mulc's blog: https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/
- Alex Staravoitau's blog: https://navoshta.com/traffic-signs-classification/

I also used links from https://github.com/frankkanis/CarND-Student-Blogs to get an insight on how other students approached the problem. In the end, I've designed a model which performs classification by using the following techniques:

- 'Data augmentation'. Using the same techniques as described in LeCun's paper I extend my dataset so it becomes 2 times larger and the distribution of classes becomes more balanced.
- 'Data preprocessing'. Using the approach from Alex Staravoitau's blog I normalize my dataset, convert it to grayscale and perform histogram equalization on each image.
- 'Training convolutional neural network'. My classifier contains one inception module, like the ones in Google LeNet, one convolutional layer, one fully connected layer and an output layer.  

#### Preprocessing

First of all, I decided to extend my dataset and equilize distributions of classes. I used a set of different generation techniques, which included translation, rotation, scaling, adding noise to image and blurring. Adding noise and blurring does not seem to improve performance of my network, so in the end I used translation, rotation and scaling as described in LeCun's paper with the same parametres. Examples of these generation techniques on the random image from the dataset are shown below:

![alt text][image5]

After generating new data I achieved training set of size 81807 which is two times bigger than original set. The distribution of classes in the new dataset is shown below:

![alt text][image6]

My second call was to preprocess dataset further with normalization and grayscaling as it was suggested in the paper.  I also found approach with histogram equalization very helpful as it increased accuracy on the test set from 93% to 96%. 

Here is an example of an original image and a preprocessed image:

![alt text][image7] 


#### Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        									| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   											| 
| Inception module     		| concatenate[conv1x1+(conv1x1+conv3x3)+(conv1x1+conv5x5)+(max_pool3x3+conv1)]; Outputs 32x32x64		|
| RELU				|													|
| Convolution 5x5	      	| 	Outputs 16x16x64 										|
| Fully connected 		|       Outputs 1024 neurons 										|
| Fully connected 		|       Outputs 43 neurons 										|
| Softmax			| 	Outputs probabilities        									|
 


#### Model Training

To train the model, I used an AdamOptimizer as suggested in the lab from "Convolutional Neural Networks" section. I used 100 epochs to train my model and the batch size was 128 to fit my model into memory.

#### Solution Approach

In the process of implementation of this project I used 3 different architectures: 
- LeNet, which was provided in CarND "Convolutional networks" section. I've got accuracy on test set around 89%.
- A network with 3 convolutional layers and 2 fully connected layers. This network provided accuracy of 93% on the test set.
- The final version of my classifier which uses one "Inception module", one convolutional layer and one fully connected layer. This solution provided accuracy of 96,3% on the test set.

My decision of final network architecture was influenced by my desire to understand Inception modules and trying to implement a network, similar to Google LeNet. Unfortunately, I could not train such a deep network on my GPU, so I used a smaller model insted.

I used learning rate 0.0001, which is smaller than it was in the lab, because smaller rate provided me an accuracy boost.

My final model results are:
* training set accuracy of 100%
* validation set accuracy of 98%
* test set accuracy of 96,3%

Training set and validation set accuracies are shown on the chart below:

![alt text][image8]

Before submission I trained model with different epochs and learning rates and it always reached something around 98% accuracy, so I put a stop condition dependent on it.

### Test a Model on New Images

#### Acquiring New Images

Here are five German traffic signs that I found on the web:

I've found 10 different images of traffic signs on the internet using Google images search. I did not wanted to hardcode images into this file, so I put pairs "image_path"-"class" into a json file, which was used as my local database of images.

My data .json file:
{"array":[
{"path":"signs/ahead_only.jpg", "class":35},
{"path":"signs/bumpy_road.jpg","class":22},
{"path":"signs/general_caution.jpg","class":18},
{"path":"signs/keep_left.jpg","class":39},
{"path":"signs/no_entry.jpg","class":17},
{"path":"signs/priority_road.jpg","class":12},
{"path":"signs/speed_limit_30.jpg","class":1},
{"path":"signs/speed_limit_60.jpg","class":3},
{"path":"signs/stop.jpg","class":14},
{"path":"signs/turn_left_ahead.jpg","class":34}
]}

Preprocessed images are shown below:

![alt text][image9]

The images which may be difficult to classify are #1, #3, #7 and #9. Images #1 and #9 have text below them which is unusual for the original dataset. Image #7 has unusual point of view and image #3 even contains a sign of different shape than examples in the German Traffic Signs dataset.

#### Performance on New Images

Accuracy on my small dataset is 70%
Complete log on dataset accuracy with top-5 probabilities is shown below:

1. Correct class of the random "Ahead only" image is 35, prediction is correct:  True
- probability of "Ahead only" is 93.100351
- probability of "Speed limit (70km/h)" is 5.900278
- probability of "General caution" is 0.948107
- probability of "Pedestrians" is 0.026920
- probability of "Traffic signals" is 0.020229

2. Correct class of the random "Bumpy road" image is 22, prediction is correct:  False
- probability of "Priority road" is 42.093447
- probability of "No passing" is 29.786175
- probability of "Speed limit (60km/h)" is 24.397101
- probability of "Bicycles crossing" is 2.343925
- probability of "Slippery road" is 1.335996

3. Correct class of the random "General caution" image is 18, prediction is correct:  True
- probability of "General caution" is 99.999082
- probability of "Road narrows on the right" is 0.000923
- probability of "Pedestrians" is 0.000003
- probability of "Traffic signals" is 0.000000
- probability of "Bicycles crossing" is 0.000000

4. Correct class of the random "Keep left" image is 39, prediction is correct:  False
- probability of "End of all speed and passing limits" is 99.996471
- probability of "Dangerous curve to the right" is 0.002489
- probability of "Right-of-way at the next intersection" is 0.000929
- probability of "End of speed limit (80km/h)" is 0.000101
- probability of "Children crossing" is 0.000005

5. Correct class of the random "No entry" image is 17, prediction is correct:  True
- probability of "No entry" is 99.999928
- probability of "No passing" is 0.000067
- probability of "Bumpy road" is 0.000000
- probability of "Turn right ahead" is 0.000000
- probability of "Slippery road" is 0.000000

6. Correct class of the random "Priority road" image is 12, prediction is correct:  True
- probability of "Priority road" is 100.000000
- probability of "Roundabout mandatory" is 0.000000
- probability of "No passing" is 0.000000
- probability of "No vehicles" is 0.000000
- probability of "Speed limit (100km/h)" is 0.000000

7. Correct class of the random "Speed limit (30km/h)" image is 1, prediction is correct:  True
- probability of "Speed limit (30km/h)" is 99.831980
- probability of "Speed limit (50km/h)" is 0.100625
- probability of "Speed limit (60km/h)" is 0.057590
- probability of "Speed limit (80km/h)" is 0.009063
- probability of "Stop" is 0.000732

8. Correct class of the random "Speed limit (60km/h)" image is 3, prediction is correct:  True
- probability of "Speed limit (60km/h)" is 99.998450
- probability of "Speed limit (20km/h)" is 0.000995
- probability of "Speed limit (50km/h)" is 0.000392
- probability of "Speed limit (80km/h)" is 0.000170
- probability of "No passing" is 0.000000

9. Correct class of the random "Stop" image is 14, prediction is correct:  True
- probability of "Stop" is 99.993706
- probability of "Speed limit (120km/h)" is 0.006221
- probability of "Speed limit (20km/h)" is 0.000069
- probability of "Speed limit (60km/h)" is 0.000003
- probability of "Speed limit (80km/h)" is 0.000002

10. Correct class of the random "Turn left ahead" image is 34, prediction is correct:  False
- probability of "Dangerous curve to the right" is 60.627204
- probability of "Slippery road" is 31.083193
- probability of "Children crossing" is 8.182074
- probability of "Road work" is 0.082017
- probability of "Speed limit (20km/h)" is 0.014404

#### Summary on the model certainty

My model showed performance of 70% on the random images which were found on the Internet. Images "Ahead only", "General caution", "No entry", "Priority road", "Speed limit (30km/h)", "Speed limit (60km/h)", "Stop" were classified correctly with the very high probability of 90% or above.
The model could not classify "Keep left" image which was visualized above with index of 3. The unusual shape of traffic sign leads network to a conclusion that it is a "End of all speed and passing limits".
The network could not identify "Turn left ahead" and "Bumpy road" signs. As we can see, additional text in the image leads to uncertainty of the classifier.


