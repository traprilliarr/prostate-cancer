import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import base64

# Load the saved models
def load_models():
    with open('models.sav', 'rb') as f:
        models = pickle.load(f)
    return models

def preprocess_data(data):
    # Drop unnecessary columns
    data = data.drop(columns=['Unnamed: 0'], errors='ignore')

    # Check if 'diagnosis_result_1' column exists
    if 'diagnosis_result_1' in data.columns:
        target_column = 'diagnosis_result_1'
    else:
        target_column = 'diagnosis_result'

    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(data.drop(target_column, axis=1))
    scaled_features = scaler.transform(data.drop(target_column, axis=1))
    preprocessed_data = pd.DataFrame(scaled_features, columns=data.columns[:-1])

    return preprocessed_data

# Perform PCA
def perform_pca(data):
    pca = PCA(n_components=8)
    pca_data = pca.fit_transform(data)
    return pca_data

# Predict using the saved models
def predict(data):
    # Load the saved models
    models = load_models()

    # Preprocess the data
    preprocessed_data = preprocess_data(data)
    pca_data = perform_pca(preprocessed_data)

    # Make predictions using the models
    knn_model = models['knn']
    rf_model = models['rf']

    knn_predictions = knn_model.predict(pca_data)
    rf_predictions = rf_model.predict(pca_data)

    return knn_predictions, rf_predictions

# Page for Prediction
def prediction_page():
    st.header("Prediction")
    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    data = None

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.write(data.head())

        knn_predictions, rf_predictions = predict(data)

        # Display predictions
        st.subheader("K-Nearest Neighbors (KNN) Predictions")
        st.write(knn_predictions)

        st.subheader("Random Forest Predictions")
        st.write(rf_predictions)
    else:
        st.write("Please upload a CSV file.")

# Add logo and text in the top left corner
def add_logo_text():
    logo_path = 'C:/Prostate Cancer/logo.png'
    st.markdown(
        f"""
        <style>
        .logo-container {{
            display: flex;
            align-items: center;
        }}
        .logo-image {{
            margin-right: 10px;
        }}
        .logo-text {{
            font-size: 18px;
            font-weight: bold;
        }}
        </style>
        <div class="logo-container">
            <img class="logo-image" src="data:image/png;base64,{get_base64_encoded_image(logo_path)}" alt="Logo" width="50" height="50">
            <div class="logo-text">Retno Tri Aprillia-09021382025160</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Function to get base64 encoded image
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
