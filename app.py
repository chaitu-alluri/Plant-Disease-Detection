import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
import time

# Set the page layout and title
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌱", layout="wide")

# Custom CSS for a cleaner look without any background color
st.markdown("""
    <style>
    .main {
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        color: white;  /* Use default text color */
    }
    h1 {
        color: #2E8B57;  /* Light green color for title */
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .prediction-card {
        background-color: black;  /* Slightly lighter background for card */
        color: black;  /* Black text for readability */
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .footer {
        font-size: 12px;
        text-align: center;
        color: #808080;
        margin-top: 50px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
MODEL_PATH = 'models/plant_disease_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
with open('models/class_indices.json', 'r') as f:
    class_indices = json.load(f)
idx_to_class = {int(v): k for k, v in class_indices.items()}

# Main title
st.title("🌿 Plant Disease Detection")

st.markdown("""
    **Upload a picture of a plant leaf** and our model will predict if the plant is diseased. This tool is trained on various plant species and disease types.
""")

# Upload image
uploaded_file = st.file_uploader("📷 Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image in a smaller size
    image = Image.open(uploaded_file)

    # Create a layout with two columns
    col1, col2 = st.columns([1, 2])

    # Display the image in the first column with a smaller width
    with col1:
        st.image(image, caption='🖼️ Uploaded Image.', width=300)  # Adjust the width

    # Analyze and predict in the second column
    with col2:
        with st.spinner('🔍 Analyzing the image...'):
            time.sleep(2)  # Simulate loading time

            # Preprocess the image
            image = image.resize((150, 150))  # Resize to match training
            img_array = np.array(image) / 255.0  # Rescale
            if img_array.shape == (150, 150, 4):  # Handle RGBA images
                img_array = img_array[..., :3]
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction) * 100

            # Get the disease name
            disease_name = idx_to_class.get(predicted_class, "Unknown")

        # Display the result in a card-like format
        st.markdown(f"""
            <div class="prediction-card">
                <h3>🌟 Prediction: {disease_name}</h3>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        © 2024 Plant Disease Detection
    </div>
    """, unsafe_allow_html=True)
