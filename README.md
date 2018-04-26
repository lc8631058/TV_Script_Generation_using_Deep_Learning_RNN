# **TV_Script_Generation_using_Deep_Learning_RNN** 

---

**TV Script Generation using Deep Learning (RNN)**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/one_hot_encoding 
[image2]: ./examples/lookup_matrix 
[image4]: ./examples/tokenize_lookup 


### Data Exploration and Preprocessing

#### 1. Explore the Data

In the cell wth title `Explore the Data`, I show the dataset status like below:

```python
Roughly the number of unique words: 11492
Number of scenes: 262
Average number of sentences in each scene: 15.248091603053435
Number of lines: 4257
Average number of words in each line: 11.50434578341555
```

And some sentences from dataset are like:

```
Moe_Szyslak: (INTO PHONE) Moe's Tavern. Where the elite meet to drink.
Bart_Simpson: Eh, yeah, hello, is Mike there? Last name, Rotch.
Moe_Szyslak: (INTO PHONE) Hold on, I'll check. (TO BARFLIES) Mike Rotch. Mike Rotch. Hey, has anybody seen Mike Rotch, lately?
Moe_Szyslak: (INTO PHONE) Listen you little puke. One of these days I'm gonna catch you, and I'm gonna carve my name on your back with an ice pick.
Moe_Szyslak: What's the matter Homer? You're not your normal effervescent self.
Homer_Simpson: I got my problems, Moe. Give me another one.
Moe_Szyslak: Homer, hey, you should not drink to forget your problems.
Barney_Gumble: Yeah, you should only drink to enhance your social skills.

```

#### 2. Data Preprocessing

After showing the basic information of dataset, I make the data preprocessing:

```python 
Lookup Table
Tokenize Punctuation
```

The `Lookup Table` contains following functions:
```python
Dictionary to go from the words to an id, we'll call vocab_to_int
Dictionary to go from the id to word, we'll call int_to_vocab
```

We'll be splitting the script into a word array using spaces as delimiters. However, punctuations like periods and exclamation marks make it hard for the neural network to distinguish between the word "bye" and "bye!". So the function `token_lookup` returns a dict that will be used to tokenize symbols like `!` into `||Exclamation_Mark||`. Create a dictionary for the following symbols where the symbol is the key and value is the token:
```python
Period ( . )
Comma ( , )
Quotation Mark ( " )
Semicolon ( ; )
Exclamation mark ( ! )
Question mark ( ? )
Left Parentheses ( ( )
Right Parentheses ( ) )
Dash ( -- )
Return ( \n )
```

After that, preprocess all data and save them.

### Build RNN

### Build RNN Cell and Initialize

Stack one or more BasicLSTMCells in a MultiRNNCell.
The Rnn size should be set using rnn_size
Initalize Cell State using the MultiRNNCell's zero_state() function
Apply the name "initial_state" to the initial state using tf.identity()
Return the cell and initial state in the following tuple (Cell, InitialState)

Above steps are implemented by `get_init_cell` function.

#### Word Embedding
When you're dealing with words in text, you end up with tens of thousands of classes to predict, one for each word. Trying to one-hot encode these words is massively inefficient, you'll have one element set to 1 and the other 50,000 set to 0. The matrix multiplication going into the first hidden layer will have almost all of the resulting values be zero. This a huge waste of computation.

![alt text][image1]

To solve this problem and greatly increase the efficiency of our networks, we use what are called embeddings. Embeddings are just a fully connected layer like you've seen before. We call this layer the embedding layer and the weights are embedding weights. We skip the multiplication into the embedding layer by instead directly grabbing the hidden layer values from the weight matrix. We can do this because the multiplication of a one-hot encoded vector with a matrix returns the row of the matrix corresponding the index of the "on" input unit.

Instead of doing the matrix multiplication, we use the weight matrix as a lookup table. We encode the words as integers, for example "heart" is encoded as 958, "mind" as 18094. Then to get hidden layer values for "heart", you just take the 958th row of the embedding matrix. This process is called an embedding lookup and the number of hidden units is the embedding dimension.

Embeddings aren't only used for words of course. You can use them for any model where you have a massive number of classes. A particular type of model called Word2Vec uses the embedding layer to find vector representations of words that contain semantic meaning.

Apply embedding to input_data using TensorFlow. Return the embedded sequence.

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
