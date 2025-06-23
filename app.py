import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Load model
model_path = './model/plant_disease_model.h5'
model = tf.keras.models.load_model(model_path)

# Get class labels from training (sorted so 0 = healthy, 1 = crop-disease)
class_labels = sorted(os.listdir('./data'))  # folder names = class labels

# App UI
st.title("🌿 Plant Disease Detection")
st.write("Upload an image of a plant leaf to detect the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.convert("RGB")  # Ensure 3 channels
    st.image(img, caption="🖼️ Uploaded Leaf Image", use_container_width=True)  # use_container_width instead of use_column_width

    # Preprocess image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]  # Get scalar value
    predicted_index = int(prediction >= 0.5)     # 0 for healthy, 1 for crop-disease
    predicted_class = class_labels[predicted_index]

    # Show prediction
    st.success(f"✅ Predicted: **{predicted_class}**")

    # Confidence score (for binary, show probability for predicted class)
    confidence = prediction if predicted_index == 1 else 1 - prediction
    st.info(f"🔍 Confidence: {confidence * 100:.2f}%")
