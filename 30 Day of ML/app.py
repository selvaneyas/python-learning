import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_iris

# Load Model
model = joblib.load("iris_model.pkl")
iris = load_iris()

# Streamlit UI
st.title("🌺 Iris Flower Classification")
st.write("Enter flower features to predict the species.")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width", 0.1, 2.5, 0.2)

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Species: **{iris.target_names[prediction]}**")
