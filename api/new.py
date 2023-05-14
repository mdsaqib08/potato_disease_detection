from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import streamlit as st
import pandas as pd
from io import StringIO
import base64


MODEL = tf.keras.models.load_model("../saved_models/3")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
  image = np.array(Image.open(BytesIO(data)))
  return image

def predict():
  
  uploaded_file = st.file_uploader("Choose a file")
  if uploaded_file is not None:
    image = read_file_as_image(uploaded_file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    st.image(uploaded_file)
    st.write(f"Label : {predicted_class}")
    st.write(f"Confidence : {confidence:.lf}%")

predict()

#streamlit run new.py