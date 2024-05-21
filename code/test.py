import tensorflow as tf
import numpy as np
import gradio
import gradio as gr
from tensorflow import keras
#create a function to make predictions
#return a dictionary of labels and probabilities
model = tf.keras.models.load_model("model.h5")
#def cancer_predict(img):
    #img = img.reshape(1, 100, 100, 1)
    #prediction = model.predict(img).tolist()[0]
    #class_names = ["actinic keratosis","dermatofibroma","basal cell carcinoma",
                   #"melanoma","nevus","pigmented benign keratosis","seborrheic keratosis","vascular lesion","squamous cell carcinoma"]
    #return {class_names[i]: prediction[i] for i in range(2)} 
import cv2
classname_value = 0
prediction_value = 0
def cancer_predict(img):
    # Convert grayscale image to color (RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_rgb = img_rgb.reshape(1, 100, 100, 3)  # Reshape to match model input shape
    prediction = model.predict(img_rgb).tolist()[0]
    class_names = ["actinic keratosis","dermatofibroma","basal cell carcinoma",
                   "melanoma","nevus","pigmented benign keratosis","seborrheic keratosis","vascular lesion","squamous cell carcinoma"]
    return {class_names[i]: prediction[i] for i in range(2)} 
    #for i in range(2):
        #classname_value += class_names[i]
        #prediction_value += prediction[i]
    #return{classname_value:prediction_value}

    #return {class_name: round(prob * 100, 2) for class_name, prob in zip(class_names, prediction[0])}
    
#set the user uploaded image as the input array
#match same shape as the input shape in the model
im = gradio.inputs.Image(shape=(100, 100), image_mode='L', invert_colors=False, source="upload")

#setup the interface
iface = gr.Interface(
    fn = cancer_predict, 
    inputs = im, 
    outputs = gradio.outputs.Label(),
)
iface.launch(share=True)
