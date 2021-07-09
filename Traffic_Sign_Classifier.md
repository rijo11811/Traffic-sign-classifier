# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/dataset_dist.png "Visualization"
[image2]: ./output/image_dark.png "Dark Image"
[image3]: ./output/image_bright.png "Bright Image"
[image4]: ./output/image_rot_-+15.png "Rotation -15 and +15 D"
[image5]: ./output/dataset_uni.png "Post fake data addition"
[image6]: ./test_images/sign1.JPG " Wild animals crossing "
[image7]: ./test_images/sign2.JPG " Speed limit (20km/h)"
[image8]: ./test_images/sign3.JPG "Road narrows on the right"
[image9]: ./test_images/sign4.JPG "Bicycles crossing"
[image10]: ./test_images/sign5.JPG "Turn right ahead"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


Here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the pandas and numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data.Its is clear from the graph that certaing classes have low count in the training datset. This could result in biased model if left unfixed ...

![alt text][image1]

Below is the key for interpreting this graph:\
0 - Speed limit (20km/h) --> 180\
19 - Dangerous curve to the left --> 180\
37 - Go straight or left --> 180\
27 - Pedestrians --> 210\
32 - End of all speed and passing limits --> 210\
41 - End of no passing --> 210\
42 - End of no passing by vehicles over 3.5 metric tons --> 210\
24 - Road narrows on the right --> 240\
29 - Bicycles crossing --> 240\
21 - Double curve --> 270\
39 - Keep left --> 270\
20 - Dangerous curve to the right --> 300\
40 - Roundabout mandatory --> 300\
22 - Bumpy road --> 330\
36 - Go straight or right --> 330\
6 - End of speed limit (80km/h) --> 360\
16 - Vehicles over 3.5 metric tons prohibited --> 360\
34 - Turn left ahead --> 360\
30 - Beware of ice/snow --> 390\
23 - Slippery road --> 450\
28 - Children crossing --> 480\
15 - No vehicles --> 540\
26 - Traffic signals --> 540\
33 - Turn right ahead --> 599\
14 - Stop --> 690\
31 - Wild animals crossing --> 690\
17 - No entry --> 990\
18 - General caution --> 1080\
35 - Ahead only --> 1080\
11 - Right-of-way at the next intersection --> 1170\
3 - Speed limit (60km/h) --> 1260\
8 - Speed limit (120km/h) --> 1260\
7 - Speed limit (100km/h) --> 1290\
9 - No passing --> 1320\
25 - Road work --> 1350\
5 - Speed limit (80km/h) --> 1650\
4 - Speed limit (70km/h) --> 1770\
10 - No passing for vehicles over 3.5 metric tons --> 1800\
38 - Keep right --> 1860\
12 - Priority road --> 1890\
13 - Yield --> 1920\
1 - Speed limit (30km/h) --> 1980\
2 - Speed limit (50km/h) --> 2010\

### Design and Test a Model Architecture

#### 1. Preprocessing data:
I found that the lot of training images have ligting issues which added a potential of misguiding the final model. To resolve this issue I decided to scale all the pixel colour chanels to a value between 0 and 255. This imporoved the clarity of all the images. Below is an example of before and after view.

![alt text][image2]
![alt text][image3]

I normalized the image data to center the training data around the mean which will make the training process to be efficient.

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques suggested in the papaer [Traffic Sign Recognition with Multi-Scale Convolutional Networks]http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf:

* Perturbation in scale ([.9,1.1] ratio) 
* Perturbation in rotation ([-15,+15] degrees)
* Perturbation in rotation ([-10,+10] degrees)

Here is an example of an original image rotated +15 and -15 degree:

![alt text][image4]

The difference between the original data set and the augmented data set is the following ... 
Perturbation in scale ([.9,1.1] ratio)  was done for all classes having training set size less than 800. Along with this classes with count less than 300 were strengthen with images perturbated in rotation ([-15,+15] degrees) and ([-10,+10] degrees). This resulter in additional training 27838 images.

This also resulted in a more uniform training set as shown in the below bar graph.

![alt text][image5]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 24x24x20   | 
| RELU					|												|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 20x20x34   | 
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 10x10x34 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 6x6x50     | 
| RELU					|												|
| Fully connected		| 1800x600        								|
| RELU					|												|
| DROPOUT				|												|
| Fully connected		| 600x180       								|
| RELU					|												|
| DROPOUT				|												|
| Fully connected		| 180x43        								|
| Softmax				|           									|

I four convolution layers to add more depth into my model with minimal weights. Since the amount of training data is large, I decided to use pooling only between the 3rd and 4th layer. The model has 3 fully connected layers. I decided to use dropout between 1st,2nd and 3rd layers inorder to make the model more robust.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model using Adam optimizer. Adam optimiser supports Adaptive Gradient which takes of the learning rate. As the training datset is of size 62637, I decided to use batch size of 128. I intialised learning rate as 0.001 whic I found to be effective via trial and error. I set the keep probability for droupout as 0.7 to help make the model more robust.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.976
* test set accuracy of 0.957

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
My first approch was a network with 3 convolution layers and two fully connected layers. 

* What were some problems with the initial architecture?
Validation accuracy was low in this iteration. I found some issues in the image pre processing, which greatly improved the result. Also, decided to add more convolution layers and make use of other tensorflow functions like dropout for better results.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

A model with more convolution layers and less pooling produced a better result. Large enough datset was the main factor which inspired me to limit pooling layers.

* Which parameters were tuned? How were they adjusted and why?

I was able to get good results in 20 Epochs. Tried different learning rates and found 0.001 to be most effective. The result was good enough and I did not try to change the keep probability from 0.7.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The most important design choice is the used of 4 convolution layers. This increased the effective depth of the model with minimal weights. stride of 1 in colutioin layers ensured that minimal information is lost.

Limiting the use of pooling helped in preserving information.

Use of dropout between the 3 fully connected layers ensured that the weights are robust and overfitting is avoided.

If a well known architecture was chosen:
* What architecture was chosen? Architecture is inspired by LeNet

* Why did you believe it would be relevant to the traffic sign application? Lenet worked well for Mnist datset. I tuned it for making the model appropriate for traffic sign application.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? The training , validation and test set accuarecy are 0.999,0.976 and 0.957 respectively which are good.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                 |      Prediction	        					| 
|:------------------------------:|:--------------------------------------------:| 
| Wild animals crossing          | Wild animals crossing					    | 
| Speed limit (20km/h)           | Speed limit (20km/h) 						|
| Road narrows on the right		 | Road narrows on the right					|
| Bicycles crossing    		     | Speed limit (60km/h)					 		|
| Turn right ahead		         | Turn right ahead      						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 95.7. 
I noticed that an important factor for the proper prediction of a sign is its ratio to the size of the image. If this ratio is small, then there is a high chance of misclasification. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 29th cell of the Ipython notebook.

For all the positive predictions, the Probability is ~1. The fourth image is misclassified as "Speed limit (60km/h)". However, when run repeatedly, in certain instances, I got all the images classified properly.


Below are the results of Prediction:
Image1:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Dangerous curve to the left   				| 

Image2:
| Probability           |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.981         	    | Speed limit (20km/h)           				| 

Image3:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         		    | Road narrows on the right     				| 

Image4:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.985        			| Speed limit (60km/h)   				        | 

Image5:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Turn right ahead   				            | 




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


