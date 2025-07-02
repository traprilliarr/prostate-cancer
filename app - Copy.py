import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  # Import train_test_split
import matplotlib.pyplot as plt

st.title('Analisis Komparatif Ketepatan Hasil Deteksi Kanker Prostat Menggunakan Metode Random Forest dan K-Nearest Neighbor')

with open('models.sav', 'rb') as f:
    models = pickle.load(f)

cp = pd.read_csv("data.csv")

# Data preprocessing
cp.diagnosis_result = [1 if each == 'M' else 0 for each in cp.diagnosis_result]
cp2 = pd.get_dummies(cp, columns=['diagnosis_result'], drop_first=True)
scaler = StandardScaler()
scaler.fit(cp2.drop('diagnosis_result_1', axis=1))
scaled_features = scaler.transform(cp2.drop('diagnosis_result_1', axis=1))
new_data = pd.DataFrame(scaled_features, columns=cp2.columns[:-1])
x = new_data
y = cp2['diagnosis_result_1']

method = st.sidebar.selectbox("Select Method", ("KNN", "Random Forest"))

if method == "KNN":
    # KNN parameters
    k = st.sidebar.slider("K", 1, 20, 5)

    # KNN classification
    knn = KNeighborsClassifier(n_neighbors=k)

    # Data Splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    knn.fit(x_train, y_train)

    # Classification results
    y_pred = knn.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Display accuracy
    accuracy = report['accuracy']
    st.write("Accuracy: {:.2f}%".format(accuracy * 100))

    # Display precision, recall, f1-score, and support in a table
    table_data = {
        'Class': ['0', '1'],
        'Precision': [report['0']['precision'], report['1']['precision']],
        'Recall': [report['0']['recall'], report['1']['recall']],
        'F1-Score': [report['0']['f1-score'], report['1']['f1-score']],
        'Support': [report['0']['support'], report['1']['support']]
    }
    table_df = pd.DataFrame(table_data)
    st.write("Classification Report:")
    st.dataframe(table_df)

    # Plotting accuracy
    error_rate = []
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train)
        pred_i = knn.predict(x_test)
        error_rate.append(np.mean(pred_i != y_test))

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(range(1, 40), error_rate)
    plt.title("KNN Accuracy")
    plt.xlabel("K")
    plt.ylabel("Error Rate")
    st.pyplot(fig)

elif method == "Random Forest":
    # Random Forest parameters
    n_estimators = st.sidebar.slider("Number of Trees", 1, 100, 10)

    # Splitting the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    # Random Forest classification
    rf = RandomForestClassifier(class_weight="balanced", n_estimators=n_estimators, random_state=1)
    rf.fit(x_train, y_train)

    # Classification results
    y_pred = rf.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Display accuracy
    accuracy = report['accuracy']
    st.write("Accuracy: {:.2f}%".format(accuracy * 100))

    # Display precision, recall, f1-score, and support in a table
    table_data = {
        'Class': ['0', '1'],
        'Precision': [report['0']['precision'], report['1']['precision']],
        'Recall': [report['0']['recall'], report['1']['recall']],
        'F1-Score': [report['0']['f1-score'], report['1']['f1-score']],
        'Support': [report['0']['support'], report['1']['support']]
    }
    table_df = pd.DataFrame(table_data)
    st.write("Classification Report:")
    st.dataframe(table_df)

    # Plotting accuracy
    accuracies = []
    for i in range(1, 101):
        rf = RandomForestClassifier(class_weight="balanced", n_estimators=i, random_state=1)
        rf.fit(x_train, y_train)
        pred_i = rf.predict(x_test)
        accuracy_i = np.mean(pred_i == y_test)
        accuracies.append(accuracy_i)

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(range(1, 101), accuracies)
    plt.title("Random Forest Accuracy")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    st.pyplot(fig)
