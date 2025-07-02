import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import base64

class RandomForestClassifierApp:
    def __init__(self):
        self.models = None

    def load_models(self):
        with open('models.sav', 'rb') as f:
            self.models = pickle.load(f)

    def preprocess_data(self, data):
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

    def perform_pca(self, data):
        pca = PCA(n_components=8)
        pca_data = pca.fit_transform(data)
        return pca_data

    def perform_rf_classification(self, x_train, y_train, x_test, y_test, n_estimators):
        rf = RandomForestClassifier(class_weight="balanced", n_estimators=n_estimators, random_state=1)
        rf.fit(x_train, y_train)
        y_pred_train = rf.predict(x_train)
        y_pred_test = rf.predict(x_test)
        report_train = classification_report(y_train, y_pred_train, output_dict=True)
        report_test = classification_report(y_test, y_pred_test, output_dict=True)
        return rf, report_train, report_test

    def random_forest_classification_page(self):
        st.header("Random Forest Classification")
        n_estimators = st.slider("Select the number of trees", 10, 1000, 50, step=10)

        # Upload data
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        data = None

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.write(data.head())

            # Load the saved models
            self.load_models()

            # Preprocess the data
            preprocessed_data = self.preprocess_data(data)
            pca_data = self.perform_pca(preprocessed_data)

            # Split data into train and test using train_test_split
            x_train, x_test, y_train, y_test = train_test_split(
                pca_data, data['diagnosis_result'], test_size=0.2, random_state=1
            )

            rf_model, rf_report_train, rf_report_test = self.perform_rf_classification(
                x_train, y_train, x_test, y_test, n_estimators
            )

            # Display Random Forest classification report on test data
            st.subheader("Random Forest Classification Test Report")
            rf_test_table = pd.DataFrame(rf_report_test).transpose()
            st.table(rf_test_table[['precision', 'recall', 'f1-score', 'support']])
        else:
            st.write("Please upload a CSV file.")

    def add_logo_text(self):
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
                <img class="logo-image" src="data:image/png;base64,{self.get_base64_encoded_image(logo_path)}" alt="Logo" width="50" height="50">
                <div class="logo-text">Retno Tri Aprillia-09021382025160</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    def get_base64_encoded_image(self, image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
