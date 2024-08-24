import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

IMG_SIZE = 777

# Define the Multi-Layer Perceptron class
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = None
        self.bias_hidden = None
        self.weights_hidden_output = None
        self.bias_output = None

    def initialize_weights(self):
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))

    def forward(self, X):
        # Forward pass through the hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.relu(self.hidden_input)
        
        # Forward pass through the output layer
        self.output = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        
        return self.output

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def fit(self, X, y):
        self.initialize_weights()
        for _ in range(self.n_iters):
            # Shuffle the data to introduce variability
            shuffled_indices = np.random.permutation(len(X))
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            for i in range(len(X_shuffled)):
                # Forward pass
                output = self.forward(X_shuffled[i])

                loss = 0.5 * np.clip((y_shuffled[i] - output) ** 2, -1e10, 1e10)


                # Backpropagation
                delta_output = -(y_shuffled[i] - output)
                delta_hidden = np.nan_to_num(delta_output.dot(self.weights_hidden_output.T)) * self.relu_derivative(self.hidden_output)  # Handle NaN values

                # Update weights and biases
                self.weights_hidden_output -= self.lr * self.hidden_output.T.dot(delta_output)
                self.bias_output -= self.lr * delta_output
                self.weights_input_hidden -= self.lr * X_shuffled[i].reshape(-1, 1).dot(delta_hidden)
                self.bias_hidden -= self.lr * delta_hidden


    def predict(self, X):
        output = self.forward(X)
        return output

# Calculate hearth girth from an image
def calculate_hearth_girth(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to find contours
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it represents the cow's outline)
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the approximate hearth girth (circumference)
    hearth_girth_value = 2 * (w + h)

    return hearth_girth_value

# Calculate body length from an image
def calculate_body_length(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to find contours
    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it represents the cow's outline)
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the approximate body length (assuming it's the longer dimension)
    body_length_value = max(w, h)

    return body_length_value

# Directory paths for healthy and underweight images
healthy_folder = 'C:/Users/HP/OneDrive/Documents/OpenCV/input_dataset/healthy'
underweight_folder = 'C:/Users/HP/OneDrive/Documents/OpenCV/input_dataset/underweight'

# Process images in the 'healthy' folder
for filename in os.listdir(healthy_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(healthy_folder, filename)
        image = cv2.imread(image_path)
        if image is not None:
            hearth_girth = calculate_hearth_girth(image)
            body_length = calculate_body_length(image)
            print(f'Image: {filename}, Hearth Girth: {hearth_girth:.2f}, Body Length: {body_length:.2f}')

# Process images in the 'underweight' folder
for filename in os.listdir(underweight_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(underweight_folder, filename)
        image = cv2.imread(image_path)
        if image is not None:
            hearth_girth = calculate_hearth_girth(image)
            body_length = calculate_body_length(image)
            print(f'Image: {filename}, Hearth Girth: {hearth_girth:.2f}, Body Length: {body_length:.2f}')
#In this code, we specify the paths to the "healthy" and "underweight" folders, and then we loop through the images in each folder separately. For 

# Calculate cattle weight based on hearth girth and body length
def calculate_cattle_weight(hearth_girth, body_length):
    cattle_weight = (hearth_girth * body_length) / 3000
    return cattle_weight

# Load and preprocess your dataset as you've done.
# Add the data loading code here
dataset_folder = 'C:/Users/HP/OneDrive/Documents/OpenCV/input_dataset'

image_paths = []
labels = []

# Assuming you have two subfolders 'healthy' and 'underweight'
for class_folder in ['healthy', 'underweight']:
    class_path = os.path.join(dataset_folder, class_folder)
    for filename in os.listdir(class_path):
        if filename.endswith('.jpg'):
            image_paths.append(os.path.join(class_path, filename))
            labels.append(1 if class_folder == 'healthy' else -1)

# Define binary labels for healthy (1) and underweight (-1)
labels = np.array(labels)

def preprocess_image(image_path, class_label):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to load image at path: {image_path}")
        return None, None
    
    # Data Augmentation: Randomly rotate the image by up to 20 degrees
    angle = random.uniform(-20, 20)
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows))
    
    # Data Augmentation: Randomly flip the image horizontally
    if random.choice([True, False]):
        img = cv2.flip(img, 1)  # 1 indicates horizontal flip

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = img.astype(np.float32)
    img = img.flatten()  # Flatten the image

    
    return img, class_label

images = []
preprocessed_labels = []  # Renamed from 'labels' to avoid conflict

for i in range(len(image_paths)):
    img, label = preprocess_image(image_paths[i], labels[i])  # Use 'labels' here
    if img is not None and label is not None:
        images.append(img)
        preprocessed_labels.append(label)  # Use 'preprocessed_labels' here

# Convert images and labels to numpy arrays
images = np.array(images)
preprocessed_labels = np.array(preprocessed_labels)  # Rename 'labels' to 'preprocessed_labels'

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(images, preprocessed_labels, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

# Define the correct input size based on the flattened image size
input_size = X_train.shape[1]


# Initialize the MLP with the correct input size
mlp = MLP(input_size=input_size, hidden_size=128, output_size=1, learning_rate=0.001)

# Ensure your input data X is correctly shaped as (batch_size, input_size)
# Here, we assume X_train is a 2D array with shape (number_of_samples, input_size)
X_train = X_train.reshape(X_train.shape[0], input_size)




# Train the model on the training set for a specified number of epochs
epochs = 60
for epoch in range(epochs):
    mlp.fit(X_train, y_train)

    # Calculate training accuracy
    train_preds = np.where(mlp.predict(X_train) >= 0.5, 1, -1)
    train_acc = accuracy_score(y_train, train_preds)

    # Calculate validation accuracy
    val_preds = np.where(mlp.predict(X_val) >= 0.5, 1, -1)
    val_acc = accuracy_score(y_val, val_preds)

    print(f"Epoch {epoch + 1}: Training Accuracy: {train_acc * 100:.2f}%, Validation Accuracy: {val_acc * 100:.2f}%")

# Evaluate the model on the test set
test_preds = np.where(mlp.predict(X_test) >= 0.5, 1, -1)
test_acc = accuracy_score(y_test, test_preds)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Now, calculate hearth girth and body length for each cow
hearth_girths = []
body_lengths = []

for image_path in image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is not None:
        hearth_girth = calculate_hearth_girth(image)
        body_length = calculate_body_length(image)
        hearth_girths.append(hearth_girth)
        body_lengths.append(body_length)

# Calculate cattle weights for each cow
cattle_weights = []

for hearth_girth, body_length in zip(hearth_girths, body_lengths):
    cattle_weight = calculate_cattle_weight(hearth_girth, body_length)
    cattle_weights.append(cattle_weight)

# Print cattle weights for each cow
for i, weight in enumerate(cattle_weights):
    print(f"Cow {i + 1} - Cattle Weight: {weight:.2f} kg")
