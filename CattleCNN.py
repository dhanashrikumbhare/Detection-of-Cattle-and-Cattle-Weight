import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

IMG_SIZE = 224
NUM_CLASSES = 2 

class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, image):
        self.last_input = image
        h, w = image.shape
        out = np.zeros((h - 2, w - 2, self.num_filters))
        
        for im_region, i, j in self.iterate_regions(image):
            for f in range(self.num_filters):
                out[i, j, f] = np.sum(im_region * self.filters[f])
                
        return out

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        h, w = self.last_input.shape

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        return None

def calculate_hearth_girth(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    hearth_girth_value = 2 * (w + h)

    return hearth_girth_value

def calculate_body_length(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, threshold1=30, threshold2=100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    body_length_value = max(w, h)

    return body_length_value

def calculate_cattle_weight(hearth_girth, body_length):
    cattle_weight = (hearth_girth * body_length) / 3000
    return cattle_weight

def preprocess_image(image_path, class_label):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to load image at path: {image_path}")
        return None, None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0 
    
    img = img.astype(np.float32)
    return img, class_label

healthy_cows_folder = 'C:/Users/HP/OneDrive/Documents/OpenCV/input_dataset/healthy'
underweight_cows_folder = 'C:/Users/HP/OneDrive/Documents/OpenCV/input_dataset/underweight'

images = []
labels = []
image_paths = []

for idx, filename in enumerate(os.listdir(healthy_cows_folder)):
    if filename.endswith('.jpg'):
        image_path = os.path.join(healthy_cows_folder, filename)
        img, label = preprocess_image(image_path, class_label=1)
        if img is not None and label is not None:
            images.append(img)
            labels.append(label)
            image_paths.append(image_path)

for idx, filename in enumerate(os.listdir(underweight_cows_folder)):
    if filename.endswith('.jpg'):
        image_path = os.path.join(underweight_cows_folder, filename)
        img, label = preprocess_image(image_path, class_label=0)
        if img is not None and label is not None:
            images.append(img)
            labels.append(label)
            image_paths.append(image_path)

images = np.array(images)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')]) 

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])



epochs = 30
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))

val_accuracy = history.history['val_accuracy']
train_accuracy = history.history['accuracy']


for epoch, (train_acc, val_acc) in enumerate(zip(train_accuracy, val_accuracy), 1):
    print(f"Epoch {epoch}: Training Accuracy: {train_acc * 100:.2f}%, Validation Accuracy: {val_acc * 100:.2f}%")

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

for idx, image_path in enumerate(image_paths):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is not None:
        hearth_girth = calculate_hearth_girth(image)
        body_length = calculate_body_length(image)
        cattle_weight = calculate_cattle_weight(hearth_girth, body_length)
        print(f"Hearth Girth: {hearth_girth:.2f}, Body Length: {body_length:.2f}, Cattle Weight: {cattle_weight:.2f} kg")


# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_accuracy, label='Training Accuracy', marker='o')
plt.plot(range(1, epochs + 1), val_accuracy, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
