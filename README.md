# Plant Leaf Disease Identifier

## Table of Contents
- [Introduction](#1-introduction)
- [Features](#2-features)
- [How it Works](#3-how-it-works)
- [Project Structure](#4-project-structure)
- [Setup and Installation (Local Development)](#5-setup-and-installation-local-development)
- [Model Training & Saving (Google Colab)](#6-model-training--saving-google-colab)
- [Running the Streamlit App Locally](#7-running-the-streamlit-app-locally)
- [Deployment to Streamlit Community Cloud](#8-deployment-to-streamlit-community-cloud)
- [Contributing](#9-contributing)
- [License](#license)

## 1. Introduction
This project implements a web-based Plant Leaf Disease Identifier. It uses a deep learning model (VGG19 with transfer learning) to classify diseases from images of plant leaves. The model is designed to be trained on powerful cloud GPUs (like Google Colab) and then deployed as a user-friendly web application using Streamlit for local execution or cloud sharing.

## 2. Features
- Upload plant leaf images for disease prediction.
- Utilizes a pre-trained VGG19 model for robust feature extraction.
- Provides predicted disease name and confidence score.
- Dynamically downloads the trained model from Google Drive, reducing repository size.
- Easy-to-use Streamlit interface.

## 3. How it Works
- **Training:** Fine-tune VGG19 on a plant disease dataset using `main_ml_code.py`.
- **Model Saving:** Save the model (.h5) and labels (.json).
- **Model Hosting:** Host the model on Google Drive and download it using `gdown`.
- **Deployment:** Use `app.py` to predict diseases from user-uploaded images.

## 4. Project Structure
```
├── app.py                      # Streamlit web application
├── main_ml_code.py             # Model training script
├── class_labels.json           # Label mapping
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

## 5. Setup and Installation (Local Development)
### Prerequisites
- Miniconda
- Git
- VS Code (optional)

### Clone the Repository
```bash
git clone https://github.com/YourGitHubUsername/plant-leaf-disease-identifier.git
cd plant-leaf-disease-identifier
```

### Create and Activate Conda Environment
```bash
conda create -n colab python=3.10 -y
conda activate colab
```

### Install Dependencies
```bash
pip install tensorflow==2.18.0 streamlit gdown Pillow numpy
```

### Verify Setup (Optional)
```bash
python -c "import sys; import tensorflow as tf; import streamlit; import google.protobuf; print(f'Python Version: {sys.version}'); print(f'TensorFlow Version: {tf.__version__}'); print(f'Streamlit Version: {streamlit.__version__}'); print(f'Protobuf Version: {google.protobuf.__version__}'); print(f'GPUs detected: {tf.config.list_physical_devices('GPU')}')"
```

## 6. Model Training & Saving (Google Colab)
### Train the Model
- Use `main_ml_code.py` on Google Colab with GPU runtime enabled.

### Save Model and Labels
```python
with open("class_labels.json", "w") as f:
    json.dump({str(v): k for k, v in train.class_indices.items()}, f)
model.save("plant_disease_detector_vgg19.h5")
```

### Upload to Google Drive and Get File ID
- Upload the model to Google Drive and set access to "Anyone with the link".
- Copy the File ID from the sharing URL.

## 7. Running the Streamlit App Locally
### Update app.py
Replace the line:
```python
GOOGLE_DRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
```
with your actual file ID.

### Run the App
```bash
conda activate colab
streamlit run app.py
```

## 8. Deployment to Streamlit Community Cloud
- Commit changes to GitHub.
- Go to [Streamlit Cloud](https://share.streamlit.io) and deploy your repo.

## 9. Contributing
Fork the repo, make changes, and submit a pull request.

## License
This project is open-source and available under the [MIT License](LICENSE).