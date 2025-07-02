import streamlit as st
from classification import ClassificationApp
from prediction import PredictionApp, Preprocessor, PCAAnalyzer
import base64
import pickle

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

class MainApp:
    def __init__(self):
        self.models = None

    def load_models(self):
        # Load the saved model and preprocessing artifacts
        with open('models.sav', 'rb') as f:
            self.models = pickle.load(f)

    def run(self):
        # Add logo and text
        add_logo_text()

        st.title("Comparative Analysis of Prostate Cancer Detection Accuracy using Random Forest and K-Nearest Neighbor")

        # Sidebar selection
        page = st.sidebar.selectbox("Select a page", ["Testing", "Classification Model"])

        if page == "Testing":
            if self.models is None:
                st.write("Please load the models first.")
            else:
                preprocessor = Preprocessor()
                pca_analyzer = PCAAnalyzer()

                # Prediction Page
                prediction_app = PredictionApp(self.models)
                prediction_app.run()  
        elif page == "Classification Model":
            classification_app = ClassificationApp()
            classification_app.run()
        else:
            st.write("Please select a page.")

# Main function
def main():
    app = MainApp()
    app.load_models()
    app.run()

if __name__ == '__main__':
    main()
