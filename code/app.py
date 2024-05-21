import streamlit as st
import tensorflow as tf
import numpy as np
import gradio
import gradio as gr
import cv2
import joblib

# Load the model
#model = joblib.load("model.h5")
model = tf.keras.models.load_model("model.h5")
def cancer_predict(image_stream):
    # Read the image from the file stream using OpenCV
    img = cv2.imdecode(np.fromstring(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize the image to match the model input shape
    img = cv2.resize(img, (100, 100))
    
    # Preprocess the image
    img = img / 255.0  # Normalize the pixel values (assuming the model expects values between 0 and 1)
    img = img.reshape(1, 100, 100, 3)  # Reshape to match the model input shape

    prediction = model.predict(img).tolist()[0]
    class_names = ["actinic keratosis","dermatofibroma","basal cell carcinoma","melanoma","nevus","pigmented benign keratosis","seborrheic keratosis","vascular lesion","squamous cell carcinoma"]
    
    return class_names[prediction.index(max(prediction))]

st.title("Skin Cancer Classification App")

# Add an image uploader
image = st.file_uploader("Upload an image")

# Make a prediction if an image is uploaded
if image is not None:
    prediction = cancer_predict(image)
    st.write(f"The predicted skin cancer is: {prediction}")
