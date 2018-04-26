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

[image1]: ./examples/one_hot_encoding.png
[image2]: ./examples/lookup_matrix.png
[image3]: ./examples/tokenize_lookup.png


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

### 1. Build RNN Cell and Initialize

Stack one or more BasicLSTMCells in a MultiRNNCell.
The Rnn size should be set using rnn_size
Initalize Cell State using the MultiRNNCell's zero_state() function
Apply the name "initial_state" to the initial state using tf.identity()
Return the cell and initial state in the following tuple (Cell, InitialState)

Above steps are implemented by `get_init_cell` function.

#### 2. Word Embedding
When you're dealing with words in text, you end up with tens of thousands of classes to predict, one for each word. Trying to one-hot encode these words is massively inefficient, you'll have one element set to 1 and the other 50,000 set to 0. The matrix multiplication going into the first hidden layer will have almost all of the resulting values be zero. This a huge waste of computation.

![alt text][image1]

To solve this problem and greatly increase the efficiency of our networks, we use what are called embeddings. Embeddings are just a fully connected layer like you've seen before. We call this layer the embedding layer and the weights are embedding weights. We skip the multiplication into the embedding layer by instead directly grabbing the hidden layer values from the weight matrix. We can do this because the multiplication of a one-hot encoded vector with a matrix returns the row of the matrix corresponding the index of the "on" input unit.

![alt text][image2]

Instead of doing the matrix multiplication, we use the weight matrix as a lookup table. We encode the words as integers, for example "heart" is encoded as 958, "mind" as 18094. Then to get hidden layer values for "heart", you just take the 958th row of the embedding matrix. This process is called an embedding lookup and the number of hidden units is the embedding dimension.

![alt text][image3]

Embeddings aren't only used for words of course. You can use them for any model where you have a massive number of classes. A particular type of model called Word2Vec uses the embedding layer to find vector representations of words that contain semantic meaning.

Function `get_embed`: Apply embedding to input_data using TensorFlow. Return the embedded sequence. 

#### 3. Build RNN

I created a RNN Cell in the get_init_cell() function. Time to use the cell to create a RNN.
Build the RNN using the tf.nn.dynamic_rnn()
Apply the name "final_state" to the final state using tf.identity()
Return the outputs and final_state state in the following tuple (Outputs, FinalState)

### Neural Network Training

#### 1. Hyperparameters

I used the following hyperparameters to train my network, these parameters are chosen empirically and experimentally:

```python
# Number of Epochs
num_epochs = 300
# Batch Size
batch_size = 250
# RNN Size
rnn_size = 500
# Embedding Dimension Size
embed_dim = 256
# Sequence Length
seq_length = 20
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = 7

# number of lstm layers
num_layers = 2
```

### Test the trained model

Generate TV Script:

I give some input words：'moe_szyslak', and my model generate something like this, they looks good:

```
moe_szyslak:(into phone) gotcha ya down for forty bucks. good luck your eminence.
moe_szyslak: sorry, my sweet like is.
moe_szyslak: homer, i got the right back to the world world the game!(sobs)
homer_simpson:(to homer, slightly sobs, then i'm a huge loser.
moe_szyslak: sorry, moe. this is just the name on the middle of moe's.
homer_simpson: i saw this.
lenny_leonard: no.
homer_simpson:(excited) oh you, homer, listen up, lenny.
lenny_leonard: great. i know, it's a beer, moe.
lenny_leonard: oh, how can you go up a worthless things who got a step things about bring that the one of the way?
hans: yeah. no.


homer_simpson: hey, homer.
moe_szyslak: homer, i can't see you that little girl?


kent_brockman:(excited) oh, this is a guy. i got my big thing.(smug chuckle)


moe_szyslak:(reading)" the springfield

```




### Summary

The generated sentences are not always grammatically right, because the word dict size are a little big and by the way I need to use more data to train it, this will be updated later, this is the complete dataset will be used to train my network later: [another dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data).  
