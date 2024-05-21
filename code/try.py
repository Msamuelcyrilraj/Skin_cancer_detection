import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np


model = tf.keras.models.load_model('model.h5')


def classify_skin_cancer(image):
    # Preprocess the input image
    image_copy = np.copy(image)#copy uploaded img

    # for resizing and converting to nparray
    image_resized = np.array(Image.fromarray(image_copy).resize((100, 100)))

 
    prediction = model.predict(np.expand_dims(image_resized, axis=0))

    
    class_mapping = ["actinic keratosis", "dermatofibroma", "basal cell carcinoma", "melanoma", "nevus", "pigmented benign keratosis", "seborrheic keratosis", "vascular lesion", "squamous cell carcinoma"]
    predicted_class = class_mapping[prediction.argmax()]

    return predicted_class


iface = gr.Interface(
    fn=classify_skin_cancer,
    inputs="image",  
    outputs="text",  
    live=True  
)


iface.launch(debug =  True)