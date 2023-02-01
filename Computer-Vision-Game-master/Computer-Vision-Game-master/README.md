# Rock Paper Scissors Game
The goal of this project is to create a Rock, Scissor, Paper game in python, 
which will be processed in a live video and it will be able to give instant results to the users. 
The main idea is to create a program, which turns on the camera, creates two boxes which are the points of interest and then the two players put their hands inside, 
having Rock, Paper or Scissor displayed. 
Then the program will be able to recognize these gestures and it will print the result of the winner. 
This is not an easy task, because the computer cannot recognize human hands or specific signs that a human’s hand can create.
Some difficulties that might occur are the variety of lighting conditions, the different backgrounds, the hand gesture identification and more

A lot of similar research can be found around this topic and there is a big variety of approaches and methods for the same problem


## Two different methods were used for the hand/sign detection.

The two sub methods were used parallel to each other, the first for player 1 and the second for player 2.

### First Approach
![image](https://user-images.githubusercontent.com/82097084/164471035-c3323f2a-2e44-4dcd-8554-5c09e9499b01.png)

The first step was to create a method that identifies a region where the user may put in her hand and also recognizes that a hand has entered this region.
To achieve that, two Regions of Interest were identified for each hand (ROI_1, ROI_2).

*Method 1:*
The program updates a running average of the background values in the ROIs. More specifically, for differentiating between the background, it calculates the
accumulated weighted avg for the background and then subtract this from the frames that contain some object in front of the background that can be distinguished as foreground. 
This is done by calculating the accumulated weight for the first 60 frames. 
After the accumulated average for the background is calculated it is subtracted from every frame taken after the first 60 to find any object that covers the background. 
This will allow the program to detect new objects such as a hand entering the ROIs.
To do that first the program calculates the absolute difference between the background and the frame and then apply Binary Threshold in order to grab the contours from the image.

Once the hand enters the ROI, then the program can detect the change and apply some thresholding techniques to isolate the hand and the hand segment.
Binary thresholding is used to grab the hand segment from the ROI, in order to calculate the contour around the white hand against a black background.
Once we have a thresholded hand segment the Convex Hull method is used to draw a polygon around the hand.

![image](https://user-images.githubusercontent.com/82097084/164473462-f419f3c0-8ad2-4edf-a140-5d5d4ffd516d.png)

Then the center of the hand is calculated against the angle of the outer points of the polygon to infer a finger count.
Each point of the polygon ideally should count for a finger. By calculating the distance of the point from the center, the program identifies if the finger is extended.
Depending on whether the fingers are extended those points are closer or further away from the center of the hand.
In order to account for lines coming from the wrist (the points towards the bottom of the polygon) the most extreme points of the polygon are calculated (top, bottom, left, right).
Then the intersection of the extreme points is calculated as the center of the hand. 
Next the distance for the extreme point furthest away from the center of the hand is calculated. By using a ratio (0.8) of that distance as a radius, a circle is created around the circle.

Any points outside if the circle and far away from the bottom, count as extended fingers.

### Second Approach

For Player 2 a model was developed by using Convolutional Neural Networks.
The model is divided into three parts 
-	Creating the dataset, 
-	Training the model on the dataset 
-	and using the model in order to predict the data.

It would be better to develop the training and testing dataset instead of downloading one from the internet, in order to have more control over the model and understand the process better.

**Dataset Creation**

For the creation of the dataset *create_gesture_data.py* can be used in order to capture multiple hand screenshots for each hand sign and create a training and a testing dataset.
The first step was to create a ROI where the user would put in her hand in order to take the screenshots.
After that, the same methodologies from model 1 (described earlier) were used in order to calculate the background and grab the segment of the hand.
Once the hand is isolated the program starts saving the images in a local folder (The user can specify how many images/frames wants to capture).

![image](https://user-images.githubusercontent.com/82097084/164475824-6ae05fb9-d16a-48c8-8087-839867824491.png)

*Templates generated to be used for training and testing the CNN model*


**Training the model**

For the training ImageDataGenerator of Keras was used in order to load the train and test data. 
To classify the data the names and numbers of the folders where the data were saved were used.
The model that was created for the training of the dataset is presented below 

![image](https://user-images.githubusercontent.com/82097084/164476363-7bfdc2ea-2982-40a6-80b7-118614609eba.png)

*Folders and Subfolders for the CNN model per class*

``` 
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(128,activation ="relu"))
model.add(Dense(6,activation ="softmax"))
```

The ***training parameters*** of the model are the following:

- *A simple sequential model was used for training the model. Because a sequential model is more appropriate for a plain stack of layers where each layer has exactly one input tensor and one output*
- *The model consists of 7 layers including the output layer*
- *With regards to the Conv2D parameters layers closer to the input image use fewer convolutional filters while layers closer to the prediction use more filters, doubling the number of filters for each layer*
- *Filters determine the number of kernels to convolve with the input volume. Each of these operations produces a 2D activation map*

![image](https://user-images.githubusercontent.com/82097084/164479477-ea46c899-b8e8-4e72-a14b-9cdeb31e409d.png)

- *For each Conv2D a max pooling is used afterwards to reduce the spatial dimensions*
- *A kernel size of 3x3 was chosen due to the small size of the input images (200, 200)*
- *Keras has a variety of activation functions as can be seen in the following figure*

![image](https://user-images.githubusercontent.com/82097084/164479831-61341f72-b65e-4dec-a21a-97cdc15f8a60.png)

- *Rectified Linear Units or “relu” was used as it is a widely used activation function in CNN. Any other activation function could be chosen without expecting to alter the accuracy of the model predictions significantly*
- *Then the data are flattened in order to be passed through a dense layer. 4 Dense layers are added with the last Dense layer having 6 nodes which is equal to the number of labels of the dataset*

**The following code presents the rest of the parameters of the model.**

```
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0005)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

history2 = model.fit(train_batches, epochs=6, callbacks=[reduce_lr, early_stop],  validation_data = test_batches)
imgs, labels = next(train_batches)
```
**As can be seen from code block above for compiling the model:**
- *Two different optimization algorithms are used – SGD (stochastic gradient descent, that means the weights are updated at every training instance) and Adam (combination of Adagrad and RMSProp) is used. SGD observed to give higher accuracies*
- *Because of the categories of the hand gestures (0,1,2,3,4,5) a categorical_crossentropy was used*
- *6 Epochs were chosen for training the model but more could also be used in order to increase accuracy*
- *Reduce learning rate function of keras was used to measure if the model has stopped improving after patience =1 number of epochs. In which case the learning rate will be reduced to 0.0001*
- *EarlyStopping’s key objective is to minimize the loss as the model.fit() checks at the end of every epoch whether the loss is no longer decreasing in which case it terminates*
- *The overall accuracy of the model was 83.33%*

**Application of both models to the final algorithm**

As was explained earlier a different methodology is used for each player. 
Method 1 is used for player 1 and method 2 is used for player 2. Both methods “read” the player’s gesture inside their Region of Interest and predict the gesture according to their equivalent model used. 

It is important to notice that even if the purpose of the game is to play Rock Paper Scissors both models have been created to predict the number of fingers of a hand. 
In order to tailor the results of the model to the needs of the game, an improvisation was made in order to create a way that imitates the hand gestures of Rock, Paper and Scissors. 
More specifically, 0 and 1 fingers stand for Rock, 2 and 3 stand for Scissors and 4 and 5 stand for Paper.

When the program starts, a countdown appears that starts from 10 seconds. 
When the clock is at the 2nd second a message appears on the screen (“Play”) and the players can make their choices. When the countdown reaches to 0 then the game terminates, and the winner is announced. 

An assumption is made that the players will not change their hand gesture in the last second of the game.
The last frame of the game decides the winner. The rules are the same as the traditional game (Paper beats Rock e.t.c).
Function winner_check() checks which one of the players wins according to their hand gestures of the last frame. A message appears on the screen with the winner and the program ends.




Folder "Methods" contains the dataset and the two Python scripts that were ysed for gathering the data and training the model 
rock_papper_scissors.py is Main program to run for playing the game no changes in the code required for the program to run
rock_papper_scissors_model.h5 is the trained CNN model that needs to be included in rock_papper_scissors.py
in order for rock_papper_scissors.py to run properly both rock_papper_scissors_model.h5 and rock_papper_scissors.py need to be in the same directory 
