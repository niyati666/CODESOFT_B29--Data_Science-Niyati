import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('iris_model.pkl')

# Set app config
st.set_page_config(page_title="Iris Flower Predictor 🌸", page_icon="🌸")

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'intro'

# Function to switch to input page
def go_to_input():
    st.session_state.page = 'input'

# Page: Intro
if st.session_state.page == 'intro':
    st.title("🌸 Iris Flower Species Predictor")
    
    st.markdown("""
Welcome to the **Iris Flower Predictor App**!


- This app is used to predict the species of an Iris flower based on its measurements.

- Click **Next** below to begin entering your flower's details.
""")



    if st.button("➡️ Next"):
        go_to_input()

# Page: Input + Prediction
elif st.session_state.page == 'input':
    st.title("🌸 Enter the measurements of the flower")

    st.sidebar.header("Input Flower Measurements 🌿")
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.8)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 1.2)

    if st.sidebar.button('🌼 Predict Species'):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]

        species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        species = species_map[prediction]

        st.success(f"🌼 **Predicted Species:** {species}")
        st.info(f"**Your Input:** Sepal Length = {sepal_length} cm, Sepal Width = {sepal_width} cm, Petal Length = {petal_length} cm, Petal Width = {petal_width} cm")

    if st.button("⬅️ Back to Intro"):
        st.session_state.page = 'intro'

# Footer
st.markdown("---")
