# **Image Classification with Multiple Classes using CNN** 
---

**Build a Image Classification Project**

The goals / steps of this project are the following:

* In this project I build a CNN, use it to make the 10 classes Image classification
* First I will do some data preprocessing with [CIFAR 10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) and make the data split step
* Then I will play around with different parameters with tiny section of dataset, to try to find the relatively best choice of parameters 
* Then I will use the training set of processed CIFAR-10 dataset to train my network and at the same time make the validation during training in order to avoid overfitting
* After training with my network, I will use the saved model to test with test set, and get the final results of my network


[//]: # (Image References)

[image1]: ./examples/CIFAR-10.jpg "CIFAR-10"
[image2]: ./examples/after_pre_process.png "Grayscaling"
[image4]: ./examples/test_1.png "Traffic Sign 1"
[image5]: ./examples/test_2.png "Traffic Sign 2"
[image6]: ./examples/test_3.png "Traffic Sign 3"
[image7]: ./examples/test_4.png "Traffic Sign 4"
[image8]: ./examples/test_5.png "Traffic Sign 5"
[image9]: ./examples/dataset_visul.png "dataset samples"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

This project is accomplished using a [ipyng](https://github.com/lc8631058/Image_Classification_with_Multiple_Classes_using_CNN/blob/master/dlnd_image_classification.ipynb) file, in order to visualize the details.

### Data Set Summary & Exploration
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. Here is some simple samples from this dataset:

![alt text][image1]

I used the pandas library to calculate summary statistics of the CIFAR-10 data set:

* The size of training set is ? 
```python
n_train = len(X_train)
==> 60000
```
* The size of the validation set is ?
```python
n_validation = len(X_valid)
==> 6000
```

#### 1. Data Preprocessing.

(1) First we nomalize the RGB data from value range 0 to 255 to range 0-1, this simple step is realized by function `normalize`.

(2) I use `LabelBinarizer()` from sklearn to implement the `one_hot_encode` function, to realize one hot encode for labels. Just give a Map which includes the classes your labels have, and the `LabelBinarizer` will fit the class by itself.

(3) For data randomization, because the images in dataset are already randomized, so it's not necessary to randomize it again.

(4) The function `preprocess_and_save_data` in helper.py combined all data-preprocessing functions together and after processing it will save the data as pickle file. 

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because this can eliminate the effect of colors, but after training, I found that the RGB data have better result than the grayscale data. So finally, I decided to use RGB data. 

(1) In the functions `neural_net_image_input`, `neural_net_label_input`, `neural_net_keep_prob_input` I set 3 placeholders as the inputs, labels, and keep_probability of dropout layer.

(2) In function `conv2d_maxpool` I implement conv layer and max-pool layer.

(3) Function `flatten` will flatten all the images into one vector~

(4) Function `fully_conn` realize the final fully connected layer.

(5) And we also have `output` function to generate the final output.

#### 1. Describe final model architecture.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x40 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x40 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 32x32x80 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x80				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 32x32x160 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x160 |
| Fully connected		| 512 hidden units	|
| RELU					|												|
| Fully connected		| 256 hidden units	|
| RELU					|												|
| Fully connected		| 128 hidden units	|
| RELU					|												|
| Fully connected		| 10 hidden units	|
| Softmax				| 10 hidden units	|
 


#### 2. Describe how to train the model.

To train the model, I use the model described in this [paper](https://github.com/lc8631058/SDCND/blob/master/P2-Traffic-Sign-Classifier/Traffic%20Sign%20Recognition%20with%20Multi-Scale%20Convolutional%20Networks.pdf), so I use 3 convlutional layers and 3 fully-connected layers, each convolutional part is composed of convolution, relu, max_pool, batch_normalization, dropout techniche in order. The dropout probability is 0.5, and I add dropout and batch-normalization after each layer except for the last output layer. I set batch_size to 80, cause that's the maximum batch_size that my GPU resource could afford. As for learning-rate I choose 0.001 by experience.

#### 4. Describe the approach taken for finding a solution. 

My final model results were:
* training set accuracy of 85.3%
* validation set accuracy of 80.5% 
* test set accuracy of 80.2%

### Test a Model on New Images

#### 1. Here are the test results: 

The second image might be difficult to classify because as you can see the brightness is very low, it's hard to recognize the sign by eyes. The others should be easier to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)     		| Speed limit (50km/h)   									| 
| End of speed limit (80km/h)     			| End of speed limit (80km/h) 										|
| No entry					| No entry											|
| Dangerous curve to the left	      		| Dangerous curve to the left					 				|
| Traffic signals			| Traffic signals      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.29%.

 Actually I choose totally 45 images as the new images download from web, and the final mean accuracy of the 45 new images is 96.97%. I test them by setting evey 5 images as a batch, some of batches get 80% accuracy and some get 100%. So the above 5 images are the first batch of 45 images, and fortunately they have been predicted 100% correct. 
 
#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 43th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is Dangerous curve to the right sign (probability of 9.99915719e-01), and the image does exist Dangerous curve to the right sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99915719e-01        			| Dangerous curve to the right  									| 
| 4.76271380e-05     				| Children crossing    										|
| 2.77236813e-05					| End of all speed and passing limits  										|
| 5.58202373e-06	      			| Pedestrians  					 				|
| 1.41421299e-06			    | Speed limit (20km/h)         							|
