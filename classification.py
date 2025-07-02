import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)

class DataHandler:
    @staticmethod
    def load_data_from_csv(file_path):
        data = pd.read_csv(file_path)
        return data

    @staticmethod
    def upload_data():
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        data = None

        if uploaded_file is not None:
            data = DataHandler.load_data_from_csv(uploaded_file)
            st.subheader("Data Preview")
            st.write(data.head())

        return data

class Preprocessor:
    @staticmethod
    def preprocess_data(data):
        # Drop unnecessary columns
        data = data.drop(columns=['Unnamed: 0', 'diagnosis_result', 'id'], errors='ignore')

        # Standardize the data
        scaler = StandardScaler()
        scaler.fit(data)
        scaled_features = scaler.transform(data)
        preprocessed_data = pd.DataFrame(scaled_features, columns=data.columns)

        return preprocessed_data

    @staticmethod
    def perform_pca(data):
        pca = PCA(n_components=8)  # Use all PCA components
        pca_data = pca.fit_transform(data)
        return pca, pca_data

class ClassificationApp:
    def __init__(self):
        self.models = None

    def load_models(self):
        with open('models.sav', 'rb') as f:
            self.models = pickle.load(f)

    def perform_knn_classification(self, x_test_pca, k):
        # Load KNN model from models dictionary
        knn_model = self.models["KNN"]

        # Predict using the specified K value
        knn_model.n_neighbors = k
        y_pred_knn = knn_model.predict(x_test_pca)

        return y_pred_knn

    def perform_rf_classification(self, x_test_pca, n_estimators):
        # Load Random Forest model from models dictionary
        rf_model = self.models["Random Forest"]

        # Predict using the specified number of estimators
        rf_model.n_estimators = n_estimators
        y_pred_rf = rf_model.predict(x_test_pca)

        return y_pred_rf

    def run(self):
        self.load_models()

        # Upload data using custom function
        data = DataHandler.upload_data()

        if data is not None and self.models is not None:
            preprocessor = Preprocessor()

            # Preprocess the data and perform PCA
            preprocessed_data = preprocessor.preprocess_data(data)
            pca, pca_data = preprocessor.perform_pca(preprocessed_data)

            # Split data into train and test sets
            x_train, x_test, y_train, y_test = train_test_split(pca_data, data['diagnosis_result'], test_size=0.25,random_state=42)

            # Input nilai jumlah pohon dan nilai K
            n_estimators = st.slider("Select the number of trees (n_estimators)", 10, 1000, 50, step=10)
            k = st.slider("Select the value of K", 1, 19, 5)

            # Create KNN and Random Forest classifiers with balanced class weights
            knn_classifier = KNeighborsClassifier(n_neighbors=k)
            rf_classifier = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced")

            # Train classifiers
            knn_classifier.fit(x_train, y_train)
            rf_classifier.fit(x_train, y_train)

            # Predictions
            y_pred_knn = knn_classifier.predict(x_test)
            y_pred_rf = rf_classifier.predict(x_test)

            # Calculate and display accuracy
            knn_accuracy = accuracy_score(y_test, y_pred_knn)
            rf_accuracy = accuracy_score(y_test, y_pred_rf)

            # Calculate precision, recall, and F1-score
            knn_precision = precision_score(y_test, y_pred_knn, average='weighted')
            knn_recall = recall_score(y_test, y_pred_knn, average='weighted')
            knn_f1 = f1_score(y_test, y_pred_knn, average='weighted')

            rf_precision = precision_score(y_test, y_pred_rf, average='weighted')
            rf_recall = recall_score(y_test, y_pred_rf, average='weighted')
            rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')

            # Display Random Forest classification report and accuracy
            st.subheader("Random Forest Classification Test Report")
            rf_report_test = classification_report(y_test, y_pred_rf, target_names=['B', 'M'], output_dict=True)
            rf_report_df = pd.DataFrame(rf_report_test).transpose()
            st.write(rf_report_df)
            st.write("Random Forest Accuracy:", rf_accuracy)
            
            # Display KNN classification report and accuracy
            st.subheader("K-Nearest Neighbors (KNN) Classification Report")
            knn_report = classification_report(y_test, y_pred_knn, target_names=['B', 'M'], output_dict=True)
            knn_report_df = pd.DataFrame(knn_report).transpose()
            st.write(knn_report_df)
            st.write("KNN Accuracy:", knn_accuracy)

            # Display comparison table
            comparison_data = {
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'KNN': [knn_accuracy, knn_precision, knn_recall, knn_f1],
                'Random Forest': [rf_accuracy, rf_precision, rf_recall, rf_f1]
            }
            comparison_df = pd.DataFrame(comparison_data).set_index('Metric')
            st.subheader("Comparison Table")
            st.write(comparison_df)

def main():
    st.title("Prostate Cancer Diagnosis")
    classification_app = ClassificationApp()
    classification_app.run()

if __name__ == "__main__":
    main()
