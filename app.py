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
        background-color: black;  /* Dark background for card */
        color: white;  /* White text for readability */
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

# Dictionary mapping disease names to prevention and precaution information
disease_info = {
    "Apple___Apple_scab": {
        "Prevention": "Plant resistant apple varieties. Apply fungicides during the growing season.",
        "Precautions": "Prune infected leaves and branches. Avoid overhead watering."
    },
    "Apple___Black_rot": {
        "Prevention": "Apply fungicides in early spring. Remove mummified fruits and cankered branches.",
        "Precautions": "Ensure good air circulation around trees and remove debris from the orchard."
    },
    "Apple___Cedar_apple_rust": {
        "Prevention": "Use resistant varieties. Apply fungicides during spring.",
        "Precautions": "Remove nearby cedar trees if possible."
    },
    "Apple___Healthy": {
        "Prevention": "Regular monitoring and proper orchard management.",
        "Precautions": "Keep the trees healthy through proper watering, pruning, and fertilization."
    },
    "Blueberry___Healthy": {
        "Prevention": "Ensure plants get full sun and are spaced properly.",
        "Precautions": "Avoid waterlogged soils and manage pests."
    },
    "Cherry___Healthy": {
        "Prevention": "Plant disease-resistant varieties.",
        "Precautions": "Use good irrigation practices, and mulch trees to conserve moisture."
    },
    "Cherry___Powdery_mildew": {
        "Prevention": "Apply fungicides early in the growing season.",
        "Precautions": "Prune trees to increase air circulation. Water at the base to avoid wetting leaves."
    },
    "Corn___Cercospora_leaf_spot": {
        "Prevention": "Use resistant corn varieties and rotate crops.",
        "Precautions": "Avoid overhead watering. Remove infected plants immediately."
    },
    "Corn___Common_rust": {
        "Prevention": "Use resistant hybrids. Apply fungicides if necessary.",
        "Precautions": "Crop rotation and timely planting can help reduce rust occurrence."
    },
    "Corn___Healthy": {
        "Prevention": "Maintain soil health through crop rotation and appropriate fertilization.",
        "Precautions": "Monitor plants regularly for pests and diseases."
    },
    "Corn___Northern_leaf_blight": {
        "Prevention": "Use resistant corn varieties.",
        "Precautions": "Apply fungicides and ensure proper crop rotation."
    },
    "Grape___Black_rot": {
        "Prevention": "Apply fungicides early in the season.",
        "Precautions": "Remove and destroy infected vines and fruits."
    },
    "Grape___Esca_(Black_measles)": {
        "Prevention": "Avoid mechanical injuries and overwatering.",
        "Precautions": "Prune infected branches and remove debris."
    },
    "Grape___Healthy": {
        "Prevention": "Use proper irrigation techniques and regular pruning.",
        "Precautions": "Keep vines free from pests and diseases through regular monitoring."
    },
    "Grape___Leaf_blight_(Isariopsis)": {
        "Prevention": "Use fungicides during early stages of growth.",
        "Precautions": "Increase airflow around vines by proper spacing and pruning."
    },
    "Orange___Citrus_greening": {
        "Prevention": "Control the psyllid insect vector with insecticides.",
        "Precautions": "Use certified disease-free plants. Remove and destroy infected trees."
    },
    "Peach___Bacterial_spot": {
        "Prevention": "Apply copper-based fungicides.",
        "Precautions": "Prune trees to improve airflow and avoid overhead watering."
    },
    "Peach___Healthy": {
        "Prevention": "Ensure proper fertilization and irrigation.",
        "Precautions": "Monitor regularly for signs of disease or pests."
    },
    "Bell_pepper___Bacterial_spot": {
        "Prevention": "Use disease-free seeds and resistant varieties.",
        "Precautions": "Avoid overhead watering. Prune infected leaves immediately."
    },
    "Bell_pepper___Healthy": {
        "Prevention": "Space plants to improve airflow and avoid overcrowding.",
        "Precautions": "Regularly check for pests and diseases."
    },
    "Potato___Early_blight": {
        "Prevention": "Use certified seed potatoes and practice crop rotation.",
        "Precautions": "Apply fungicides if necessary. Remove infected plant debris."
    },
    "Potato___Healthy": {
        "Prevention": "Ensure proper soil health through crop rotation and fertilization.",
        "Precautions": "Keep the plants well-watered and free from pests."
    },
    "Potato___Late_blight": {
        "Prevention": "Use disease-resistant varieties and fungicides.",
        "Precautions": "Remove infected plants immediately and avoid overhead watering."
    },
    "Raspberry___Healthy": {
        "Prevention": "Use disease-free plants and ensure proper spacing.",
        "Precautions": "Monitor regularly for pests and diseases."
    },
    "Soybean___Healthy": {
        "Prevention": "Practice crop rotation and use disease-resistant varieties.",
        "Precautions": "Monitor for pests and ensure proper fertilization."
    },
    "Squash___Powdery_mildew": {
        "Prevention": "Use resistant varieties. Apply fungicides early.",
        "Precautions": "Ensure good air circulation and avoid overhead watering."
    },
    "Strawberry___Healthy": {
        "Prevention": "Plant in well-drained soil and ensure proper spacing.",
        "Precautions": "Monitor for pests and diseases, and remove any infected plants."
    },
    "Strawberry___Leaf_scorch": {
        "Prevention": "Apply fungicides and prune infected leaves.",
        "Precautions": "Ensure good airflow and avoid overcrowding plants."
    },
    "Tomato___Bacterial_spot": {
        "Prevention": "Use copper-based fungicides and disease-resistant varieties.",
        "Precautions": "Avoid overhead watering and remove infected leaves."
    },
    "Tomato___Early_blight": {
        "Prevention": "Use resistant varieties and apply fungicides.",
        "Precautions": "Practice crop rotation and avoid waterlogged soils."
    },
    "Tomato___Healthy": {
        "Prevention": "Ensure good airflow and water plants at the base.",
        "Precautions": "Regularly monitor for signs of pests and diseases."
    },
    "Tomato___Late_blight": {
        "Prevention": "Apply fungicides and use resistant varieties.",
        "Precautions": "Remove and destroy infected plants."
    },
    "Tomato___Leaf_mold": {
        "Prevention": "Apply fungicides and space plants properly.",
        "Precautions": "Prune to improve air circulation and avoid wetting leaves."
    },
    "Tomato___Septoria_leaf_spot": {
        "Prevention": "Use disease-free seeds and fungicides.",
        "Precautions": "Remove infected leaves and avoid overhead watering."
    },
    "Tomato___Spider_mites": {
        "Prevention": "Use insecticidal soaps or miticides.",
        "Precautions": "Keep plants well-watered and monitor regularly for mite activity."
    },
    "Tomato___Target_spot": {
        "Prevention": "Apply fungicides and use resistant varieties.",
        "Precautions": "Remove infected leaves and avoid overhead watering."
    },
    "Tomato___Tomato_mosaic_virus": {
        "Prevention": "Use disease-resistant seeds.",
        "Precautions": "Avoid handling plants when wet and sanitize tools."
    },
    "Tomato___Tomato_yellow_leaf_curl_virus": {
        "Prevention": "Use insecticides to control whitefly vectors.",
        "Precautions": "Remove and destroy infected plants."
    }
}


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

        # Check if the disease name exists in the dictionary and display information
        if disease_name in disease_info:
            prevention = disease_info[disease_name]["Prevention"]
            precautions = disease_info[disease_name]["Precautions"]

            st.markdown(f"""
                <div class="prediction-card">
                    <h3>🛡️ Prevention and Precautions</h3>
                    <p><b>Prevention:</b> {prevention}</p>
                    <p><b>Precautions:</b> {precautions}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="prediction-card">
                    <h3>🛡️ Prevention and Precautions</h3>
                    <p>Sorry, we don't have prevention and precaution information for this disease.</p>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        © 2024 Plant Disease Detection
    </div>
    """, unsafe_allow_html=True)
