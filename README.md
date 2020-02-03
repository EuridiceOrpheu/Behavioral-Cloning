# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Data Visualization"
[image2]: ./examples/left.jpg "Left image"
[image3]: ./examples/right.jpg "Right Image"
[image4]: ./examples/cropp.png "Cropped image"
[image5]: ./examples/amount_of_data.png "Data vs Performance"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  


### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model2.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* run1.mp4 video driving car in autonomous mode 

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Other useful commands are :
```sh
python drive.py model.h5 run1
```
 This can be used to create the video recording  in autonomous mode.The fourth argument, ```sh run1``` , is the directory in which to save the images seen by the agent.
 ```sh
python video.py run1 --fps 48
```
This creates a video based on images found in the run1 directory(the video will run at 48 FPS).

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of  5 convolution layers with 3x3,5X5 filter sizes and depths between 24 and 64 (model2.py lines 89-93) 
Also model contains 4 fully-connected layers  leading to an output control walue which is steering angle.
The model includes RELU layers to introduce nonlinearity (code line  89-99), and the data is normalized in the model using a Keras lambda layer (code line 85). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model2.py lines 97). 
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 106-110). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model2.py line 103).

#### 4. Appropriate training data

I choose to use the sample driving data given by Udacity.The training data was chosen appropriately from my view of point 
because after data augmentation,data preprocessing and training steps I achieved good results and the car successfully drive around the track one.
Of course to collect additional data would increase the perfomance of the model.
Collecting data from track two will also help my model generalize better.
![alt text][image5]
'Why deep learning?' Slide by Andrew Ng


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to build a behavior cloning network.
First I started to use the LeNet arhitecture, but the model didn’t work as expected. So, I changed to Nvidia CNN arhitecture which is a more powerful and is well suited for this project.The system that they built learns to drive in traffic on local roads with or without lane markings and on highways.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

The next  step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track :
-at the curves in many cases 
to improve the driving behavior in these cases, I incresead my dataset using augumentation techniques and adding a regularization method.

To combat the overfitting, I modified the model by adding a Dropout layer(regularization method) after first fully connected layer.I choose the droput rate=0.25,this value was the best.

I've added the following changes  to the Nvidia arhitecture:
-I added Lamda layer to normalize input images
-I added Cropping Layer (to select the region of interest )
-I included a regularization method (Dropout layer to avoid overfitting)
-I've also included ELU  activation function layers to introduce non-linearity.


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model2.py lines 83-100) consisted of a convolution neural network with the following layers and layer sizes:

Image normalization(Lambda)
Cropping2D(cropping=((50,25),(0,0)
Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
Fully connected: neurons: 100, activation: ELU
Dropout (0.25)
Fully connected: neurons: 50, activation: EL
Fully connected: neurons: 10, activation: ELU
Fully connected: neurons: 1 (output)


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 
I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to revore from mistakes.
I increased  training data using data augmentation.
![alt text][image1]			![alt text][image2]			![alt text][image3]

**Data Augmentation**

For training data I used the following augumentation technique:
-adding to my data left and center images (from left and right cameras) with suitable steering angle(I apply a correction 
+-0.2 )
-I flipped the images captured by the center camera
First approach help my model to generalize better but  the second approach did not improve much my model.
Images from  center ,left and right cameras:

![alt text][image1]			![alt text][image2]			![alt text][image3]


After the collection process, I had 8036x3 number of data points. Then I preprocessed this data by adding in the lambda layer two steps:
-normalizing the data ( divide each element by 255)
-mean centering the data (by substracting 0.5)

After preprocessing step, I cropp images 50 rows pixels from the top of the image and 20 rows pixels from the bottom of the image.The model trains  faster if we select the region of interest.In our case this is the bottom portion of the image.
![alt text][image4]

I finally randomly shuffled the data set and put Y% of the data into a validation set. 
I also used a Python generator in order to generate data for training than storing the training data in  memory.
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by value of loss fucntion. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The main focus on training the model is to have a very small value for loss function( Mean squared error ) for the validation set.
For validation set I obtain loss=0.0157 and for training set  loss= 0,0120.
