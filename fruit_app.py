import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model("fruit_model.h5")

# Class names
class_names = ['apple', 'banana', 'orange']

# Streamlit app
st.title("Fruit Classifier")
st.write("Upload an image of a fruit and the model will predict whether it's an apple, banana, or orange.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    img_height, img_width = 32, 32
    img = image.convert('RGB').resize((img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Predict the class
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Display the prediction
    st.write(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )