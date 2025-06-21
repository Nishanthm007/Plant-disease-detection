import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Load model
model_path = './model/plant_disease_model.h5'
model = tf.keras.models.load_model(model_path)

# Get class labels from training
class_labels = sorted(os.listdir('./data'))  # folder names = class labels

# App UI
st.title("🌿 Plant Disease Detection")
st.write("Upload an image of a plant leaf to detect the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Uploaded Leaf Image", use_column_width=True)

    # Preprocess image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Show prediction
    st.success(f"✅ Predicted Disease: **{predicted_class}**")

    # Optional confidence score
    confidence = np.max(prediction) * 100
    st.info(f"🔍 Confidence: {confidence:.2f}%")
