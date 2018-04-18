# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Behavioral_cloning.png "Model Visualization"
[image2]: ./examples/central_driving.png "Grayscaling"
[image3]: ./examples/crop_1.png "crop Image"
[image4]: ./examples/crop_2.png "crop Image"
[image6]: ./examples/flip_1.png "Flipped Image"
[image7]: ./examples/flip_2.png "Flipped Image"
[image8]: ./examples/case_1.png 
[image9]: ./examples/case_2.png 
[image10]: ./examples/recovery_1.png 
[image11]: ./examples/recovery_2.png 
[image12]: ./examples/model_2_curve.png 
[image13]: ./examples/simulator_effect.jpg 


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 36 (model.py lines 100-106) 

The model includes RELU layers to introduce nonlinearity (code line 100), and the data is normalized in the model using a Keras lambda layer (code line 96), using Cropping2D function I cut the input images from size 160x320x3 to 74x320x3 (code line 97). Here are some examples of cropped image:

![alt text][image3]
![alt text][image4]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 102, 106). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 72-82). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 113).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road to train my model. I added a correction value to the steering angle of left-side images and right-side images, this correction value is 0.25. I didn't collect data by myself, cause I used to use some data collected by myself, but the result is really bad, so I think collect data by ourselves could create many uncertain factors, especitalluy our keyboard only have 4 keys to control the direction, but the directions should be various value, cause this is a regression problem. The udacity's data has no recovery data, the recovery data represents the data collected when the car runs away the road, we recover the car from side of the road to the middle of the road. So we use the left- and right- images collected by side cameras in front of the car to replace the recovery date, we also add a correction to these side-images' steering angle. At final we flipped all data in order to augment the data, and then these steering angles are set as the steering angle of flipped data by multiply by -1.0. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to read the [Nvidia's paper](https://github.com/lc8631058/SDCND/blob/master/P3_Behavioral%20Cloning/End%20to%20End%20Learning%20for%20Self-Driving%20Cars.pdf) provided by udacity.

My first step was to use a convolution neural network model similar to the [Nvidia's architecture](https://github.com/lc8631058/SDCND/blob/master/P3_Behavioral%20Cloning/End%20to%20End%20Learning%20for%20Self-Driving%20Cars.pdf). I thought this model might not be appropriate because I only use the data provided by udacity, it's not a big dataset, so I need to reduce the convolutional layers as well as dense layers describe in Nvidia's architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added BatchNormalization function after each layer eacept for last dense layer(code line 98, 101, 105), as well as dropout layers described above.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track as shown in below images. 

![alt text][image8]
![alt text][image9]
To improve the driving behavior in these cases, I re-train the model to further decrease the loss, here is the loss_curve:
![alt text][image12]
And I found that, the graphic quality and screen resolution of the simulator also influent the test results, for a same model in higher effect my car will press the lines, but with lowest effect it won’t. So I turn donw the graphic quality and screen resolution to lowest, then my car can run through whole track smoothly.
![alt text][image13]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image							| 
| Cropping         		| 74x320x3 RGB image							| 
| Batch Normalization         		| 						| 
| Convolution 5x5    	| 2x2 stride, valid padding, outputs 35x158x24 	|
| Batch Normalization         		| 						| 
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 16x77x36 	|
| Batch Normalization         		| 						| 
| RELU					|												|
| Fully connected		| 100 hidden units	|
| Batch Normalization         		| 						| 
| RELU					|												|
| Fully connected		| 1 hidden units	|

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recovery from side to center. These images show what a recovery looks like starting from right side and left side of a road :

![alt text][image10]
![alt text][image11]

I didnt's repeated this process on track two, cause collect data in track two by myself will create many biases, so I think it's not necessary right now, I don't want to bring some misleading to my network.

After the collection process, I had 9942 number of data points. I then preprocessed this data by the approach describe above.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 60 as evidenced by training. I used an adam optimizer so that manually training the learning rate wasn't necessary.
