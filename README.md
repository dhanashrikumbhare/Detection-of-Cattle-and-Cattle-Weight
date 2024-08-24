# Detection-of-Cattle-and-Cattle-Weight

Topic : 
Cattle Weight Detection using Convolutional Neural Network as well as Artificial Neural Network.

Objective : 
To detect the presence or absence of cattle within images captured by a smartphone's camera. The objective for the combined Cattle Weight Estimation and Cattle Detection model using a low-cost smartphone is to provide a comprehensive solution for cattle management and monitoring. It will assist farmers in making informed decisions related to cattle management, including feeding, healthcare, and breeding also it will reduce the stress and inconvenience associated with traditional cattle weighing methods.

Input Set : 
For input set of these model we are providing the pictures of cows. There we have created two classes of identifying them as underweight and healthy using a single layer perceptron model of neural networks. First we have pre-processed the data to obtain details of photos. After that, we have set the initial weights and bias to 0 and them updating it based on the input we get and trained the model using an input set. Now, we have calculated the Cattle weight using the following formula : Cattle Weight (in kgs) = ((Heart girth^2) X body length) ÷ 300 

Output Targeted : 
As an output we get the classification of the cattle if they are underweight or healthy based on the input we provide. We are also calculating the cattle weight by applying above formula.

STEP 1 : Data Set Gathering:
For gathering the data set we have taken images of cow containing rear view and side view. A set of 30 input each are trained for both healthy and underweight categories.

STEP 2 : Preprocessing Data:
To preprocess data, it loads the image from the given path and resize it to a specific size and then normalizes the pixel value of the image to a range of [0,1]. It then convert the image data to NumPy array and calculate the values hearth girth and body length. Finally, it returns the processed image, its class, and the calculated hearth girth and body length.

STEP 3:  Data Labelling:
We have spilt the input dataset into two classes:
•	Underweight
•	Healthy
Cows below a specified threshold value are labelled as underweight (or class label = 0) and healthy (or class label = 1) depending upon their weights.

STEP 4:  Data Split:
We have divided the dataset into three sets:
•	a larger training set for model training,
•	 a validation set for tuning model parameters, 
•	and a testing set for evaluating the final model's performance.
‘train_test_split’ function is used to split the data into training, validation and testing set. 20% of the data will be used for testing, and the remaining 80% will be used for training and validation. Randoms state is used to seed the random number generator. Setting it to a specific value ensures that the data split will be reproducible.
