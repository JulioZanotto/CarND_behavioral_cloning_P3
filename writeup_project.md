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

[image1]: ./examples/model_arch.png "Model Visualization"
[image2]: ./examples/center_roi.png "Center and ROI"
[image3]: ./examples/recover_1.png "Recovery Image"
[image4]: ./examples/recover_2.png "Recovery Image"
[image5]: ./examples/distrib.png "Distribution"
[image6]: ./examples/epochs.png "Epochs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_generator.py containing the script to create and train the model locally
* model.py containing the script to create and train the model on the Udacity workspace
* drive.py for driving the car in autonomous mode
* model_gen22.h5 containing a trained convolution neural network for my local simulator
* model_trained.h5 containing a trained model for the Udacity workspace
* writeup_report.md summarizing the results
* run1.mp4 where I recorded the autonomous driving for the first track
* run2.mp4 where I recorded the autonomous driving for the second track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
Here I can point out the even using the Keras version 2.2.4 I could not use my locally trained model to run the simulator on the Project workspace, and use the Simulator on the workspace was too laggy, couldnt drive ok, but locally I managed to drive and record on both tracks, it was anwesome.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a similar architecture from the NVIDIA model, where the convolution neural network has filter sizes of 3x3 and 5x5 and depths between 24 and 64 (model_generator.py lines 118-132) 

The model includes RELU layers to introduce nonlinearity (code line 139 and also the activation inside the conv2Ds like line 118), and the data is normalized in the model using a Keras lambda layer (code line 113).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model_generator.py lines 140 and 144). 

The model was trained and validated on different data sets to ensure that the model was not overfitting, chose a percentage of 25% of the dataset to validation (code line 79). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 157).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road by driving on a snake like moviments on some parts, and also used the tip of a corrected steering angle for the left and right images. Also as I really enjoyed driving on training mode, I drove with mouse and keyboard, on the forward and backward path ( because of it I didnt use the flip augmentation, as I had enough data for both steering sides), and where the model was predicting poorly, I collect more data, for example on specific curves.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to first start with the LeNet architecture. I knew it could help a little or at least check if the was enough for learn.

My first step was to use a convolution neural network model similar to the LeNet, a somewhat shallow architecture, as I thought this model might be appropriate because it could show that the model would underfit or overfit, mre probably underfit as for its size, but was a good starting point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. So the LeNet was clearly underfitting and not leraning much showing room for a bigger model, so I switched to the NVIDIA model and made a copy with a smaller fully connected layer, without regularization or Pooling layers at first. At this point the model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added DropOut layers on the fully connected.

Then I trained again the model, but it was a little slow and too much parameters, so I added the Pooling Layers, which helped with the speed of training and was still with good MSE.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track or couldnt do the turn, like the muddy place right after the bridge, to improve the driving behavior in these cases, I collected more data there using the keyboard, and drove back and forward to have info from the curve on both sides.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 109-147) consisted of a convolution neural network with the following layers and layer sizes:

It began with a Conv2D with 24 filters 5x5, then 36 filters also 5x5, and at this step I added a MaxPooling with 2x2 filter size and stride, the standard on the MaxPooling2D layer. Then I added 48 filters 5x5 and ended with 2 filters of 64 with 3x3 size also with MaxPooling.

The fully connected layer consisted of after the flatten a hidden layer of 200 neurons with ReLU activation and DropOut of 50% randomly chosen neurons, from here to 100 neurons also with RELU and same probabily DropOut and ended on the single neuron output for the angle prediction.

Here is a visualization of the architecture using the model.summary() of keras methods:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving and the ROI (better model performance with only whats needed from the picture for the prediction):

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover in case of getting too near the edge, These images show what a recovery looks like, getting close to the side and returning, a snake like moviment on some parts:

![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points. Here it was more challenging to drive but I managed to collect about 4 laps, divided on center drive, recovering, on regular way and opposite.

As I drove a lot, and in both ways in both tracks, I had a really good amount of data, distributed on both steering angle, as it is possible to look at, on this distribution:

![alt text][image5]

After the collection process, I had 69741 number of data points. I then preprocessed this data by normalizing it on the lambda layer. Here I did not used any augmentation because of the number of data points, I also used the ImageDataGenerator on my notebook, as it has 64GB of RAM, I could store it all and use the ImageDataGenerator without any issue, here is an example of the code I used to try it:

```python
# Image data Generator with augmentation
train_datagen = ImageDataGenerator(rotation_range=15,
                            brightness_range=[0.4,1.5],
                            channel_shift_range=20.0,
                            fill_mode='nearest',
                            data_format='channels_last',
                            )

valid_datagen = ImageDataGenerator()

# Training
# Splits for generators
batch_size = 64


train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size=batch_size) # set as training data

validation_generator = valid_datagen.flow(
    X_val,
    y_val,
    batch_size=batch_size) # set as validation data

optim = Adam(learning_rate=0.0001)
model.compile(loss='mse', optimizer=optim, metrics=['mse'])

model.fit_generator(
    train_generator,
    validation_data = validation_generator, 
    epochs = 5)
```

Here it did not show much difference from the non augmented dataset, maintaining and mse of 0.07.

I finally randomly shuffled the data set and put 25% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the image below, where the mse dropped too litle, about 0.001, and I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image6]

By the end I had a model which I could gladly watch it goes around the whole track, for the first and the hardest track, endlessly !!
