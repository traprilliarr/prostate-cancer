import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
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


# Perform KNN classification
def perform_knn_classification(x_train, y_train, x_test, y_test, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return knn, report


# Perform Random Forest classification
def perform_rf_classification(x_train, y_train, x_test, y_test, n_estimators):
    rf = RandomForestClassifier(class_weight="balanced", n_estimators=n_estimators, random_state=1)
    rf.fit(x_train, y_train)
    y_pred_train = rf.predict(x_train)
    y_pred_test = rf.predict(x_test)
    report_train = classification_report(y_train, y_pred_train, output_dict=True)
    report_test = classification_report(y_test, y_pred_test, output_dict=True)
    return rf, report_train, report_test


# Page for KNN Classification Results
def knn_classification_page(models, data):  
    st.header("K-Nearest Neighbors (KNN) Classification")
    k = st.slider("Select the value of K", 1, 20, 5)

    # Preprocess the data
    preprocessed_data = preprocess_data(data)
    pca_data = perform_pca(preprocessed_data)

    # Split data into train and test using train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        pca_data, data['diagnosis_result'], test_size=0.2, random_state=1
    )

    knn_model, knn_report = perform_knn_classification(
        x_train, y_train, x_test, y_test, k
    )
    st.subheader("KNN Classification Report")
    knn_table = pd.DataFrame(knn_report).transpose()
    st.table(knn_table[['precision', 'recall', 'f1-score', 'support']])

    return knn_model


# Page for Random Forest Classification Results
def random_forest_classification_page(models, data):  
    st.header("Random Forest Classification")
    n_estimators = st.slider("Select the number of trees", 10, 1000, 50, step=10)

    # Preprocess the data
    preprocessed_data = preprocess_data(data)
    pca_data = perform_pca(preprocessed_data)

    # Split data into train and test using train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        pca_data, data['diagnosis_result'], test_size=0.2, random_state=1
    )

    rf_model, rf_report_train, rf_report_test = perform_rf_classification(
        x_train, y_train, x_test, y_test, n_estimators
    )

    # Display Random Forest classification report on test data
    st.subheader("Random Forest Classification Test Report")
    rf_test_table = pd.DataFrame(rf_report_test).transpose()
    st.table(rf_test_table[['precision', 'recall', 'f1-score', 'support']])

    return rf_model


# Prediction Page
def prediction_page(models):
    st.header("Prediction")
    radius = st.text_input("Enter the radius value: ")
    texture = st.text_input("Enter the texture value: ")
    perimeter = st.text_input("Enter the perimeter value: ")
    area = st.text_input("Enter the area value: ")
    smoothness = st.text_input("Enter the smoothness value: ")
    compactness = st.text_input("Enter the compactness value: ")
    symmetry = st.text_input("Enter the symmetry value: ")
    fractal_dimension = st.text_input("Enter the fractal dimension value: ")

    detect_button = st.button("Detect")

    if detect_button:
        if radius and texture and perimeter and area and smoothness and compactness and symmetry and fractal_dimension:
            try:
                new_data_input = [[float(radius), float(texture), float(perimeter), float(area), float(smoothness), float(compactness), float(symmetry), float(fractal_dimension)]]
                model = models['Random Forest']
                input_scaled = models['Scaler'].transform(new_data_input)
                input_pca = models['PCA'].transform(input_scaled)
                model_prediction = model.predict(input_pca)

                if model_prediction[0] == 1:
                    st.write("Penderita Kanker Prostat")
                else:
                    st.write("Bukan Penderita Kanker Prostat")

            except ValueError:
                st.write("Input harus berupa angka")
        else:
            st.write("Mohon lengkapi semua nilai input")

    st.write("Link lengkap untuk melihat hasil:")
    st.markdown("[Google Colab](https://colab.research.google.com/drive/1tcCmww4xKxT0uRdWEBreC20nj5vXIw6x?usp=sharing)")


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

# Main function
def main():
    # Load the models
    models = load_models()

    # Sidebar
    st.sidebar.title("Prostate Cancer Classifier")
    page_options = ["KNN Classification", "Random Forest Classification", "Prediction"]
    page_selection = st.sidebar.selectbox("Choose a page", page_options)

    # Read CSV data file
    data = pd.read_csv("data.csv")

    # Main content
    if page_selection == "KNN Classification":
        add_logo_text()
        knn_classification_page(models, data)
    elif page_selection == "Random Forest Classification":
        add_logo_text()
        random_forest_classification_page(models, data)
    elif page_selection == "Prediction":
        prediction_page(models)

# Run the app
if __name__ == "__main__":
    main()
