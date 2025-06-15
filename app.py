import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import os
import gdown # New import for downloading from Google Drive

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="auto"
)

# Ensure the model and class_labels.json are in the same directory as this app.py file.
MODEL_PATH = "plant_disease_detector_vgg19.h5"
LABELS_PATH = "class_labels.json"

# --- Google Drive File ID for your model ---
# IMPORTANT: REPLACE THIS WITH YOUR ACTUAL GOOGLE DRIVE FILE ID
# Ensure the file is shared publicly ("Anyone with the link")
GOOGLE_DRIVE_FILE_ID = "1AK0wQjUix8P-D43kbHsoDURF91ZaFPnA" 

@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model():
    """
    Loads the pre-trained Keras model.
    If the model file does not exist locally, it attempts to download it from Google Drive using gdown.
    """
    if not os.path.exists(MODEL_PATH):
        st.info("Model file not found locally. Attempting to download from Google Drive...")
        try:
            # Construct the Google Drive download URL
            gdown_url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(gdown_url, MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model from Google Drive: {e}")
            st.info("Please ensure the GOOGLE_DRIVE_FILE_ID is correct and the file is publicly accessible.")
            return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure 'plant_disease_detector_vgg19.h5' is in the same directory and is a valid Keras model.")
        return None

@st.cache_data # Cache the labels loading
def load_class_labels():
    """Loads the class labels from the JSON file."""
    try:
        with open(LABELS_PATH, "r") as f:
            class_labels = json.load(f)
        return class_labels
    except Exception as e:
        st.error(f"Error loading class labels: {e}")
        st.info("Please make sure 'class_labels.json' is in the same directory.")
        return None

# Load the model and labels
# These calls can happen after set_page_config
model = load_model()
class_labels = load_class_labels()

# --- Rest of your Streamlit App UI ---
st.title("🌿 Plant Leaf Disease Detector")
st.write("Upload an image of a plant leaf to detect potential diseases.")

if model and class_labels:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image for prediction
        img_array = img_to_array(image.resize((256, 256))) # Resize to target size (256, 256)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = tf.keras.applications.vgg19.preprocess_input(img_array) # Preprocess for VGG19

        # Make prediction
        try:
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_labels[str(predicted_class_index)]

            st.success(f"**Prediction: {predicted_class_name}**")
            st.write(f"Confidence: {np.max(predictions) * 100:.2f}%")

            # Display all probabilities (optional)
            st.markdown("---")
            st.subheader("All Probabilities:")
            prob_dict = {class_labels[str(i)]: float(prob) for i, prob in enumerate(predictions[0])}
            # Sort probabilities for better readability
            sorted_prob_dict = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
            st.json(sorted_prob_dict)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Please try another image or ensure the model is correctly loaded.")
else:
    st.warning("Model or class labels could not be loaded. Please check the file paths and ensure the files exist.")

st.markdown("---")
st.markdown("This app uses a VGG19-based transfer learning model to classify plant diseases.")
