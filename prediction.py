import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Preprocessor:
    @staticmethod
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

class PCAAnalyzer:
    @staticmethod
    def perform_pca(data):
        pca = PCA(n_components=8)
        pca_data = pca.fit_transform(data)
        return pca_data

class PredictionApp:
    def __init__(self, models):
        self.models = models

    def run(self):
        st.header("Testing")
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
                    new_data_input = [[float(radius), 
                                       float(texture), 
                                       float(perimeter), 
                                       float(area), 
                                       float(smoothness), 
                                       float(compactness), 
                                       float(symmetry), 
                                       float(fractal_dimension)]]
                    model = self.models['Random Forest']
                    input_scaled = self.models['Scaler'].transform(new_data_input)
                    input_pca = self.models['PCA'].transform(input_scaled)
                    model_prediction = model.predict(input_pca)

                    if model_prediction[0] == 1:
                        st.markdown("<p style='font-size: 24px; font-weight: bold;'>Diagnosed with Prostate Cancer</p>", unsafe_allow_html=True)
                    else:
                        st.markdown("<p style='font-size: 24px; font-weight: bold;'>Not Diagnosed with Prostate Cancer</p>", unsafe_allow_html=True)

                except ValueError:
                    st.write("The input must be in the form of numbers")
            else:
                st.write("Please complete the data")

def main():
    # Load the saved model and preprocessing artifacts
    with open('models.sav', 'rb') as f:
        models = pickle.load(f)

    preprocessor = Preprocessor()
    pca_analyzer = PCAAnalyzer()

    # Prediction Page
    prediction_app = PredictionApp(models)
    prediction_app.run()

# Run the app
if __name__ == "__main__":
    main()
