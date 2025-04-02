
import streamlit as st
import pickle
import numpy as np
import cv2
from skimage.feature import hog
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

st.title("MNIST Digit Classifier 🎨")
st.write("Select an index from the MNIST dataset to classify the digit.")

# Load the trained model
loaded_model = pickle.load(open("mnist_model.pkl", "rb"))

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (28, 28))  # Resize to 28x28
    features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return features.reshape(1, -1)  # Reshape for prediction

# Select an index from the MNIST dataset
index = st.slider("Select an index from MNIST dataset", min_value=0, max_value=len(x_test)-1)

# Get the corresponding image and label
image = x_test[index]
label = y_test[index]

# Preprocess and predict
features = preprocess_image(image)
prediction = loaded_model.predict(features)[0]

# Display results
st.write(f"**True Label:** {label}")
st.write(f"**Predicted Digit:** {prediction}")

# Display the image
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
ax.axis('off')
st.pyplot(fig)

