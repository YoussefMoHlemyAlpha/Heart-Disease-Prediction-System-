import numpy as np
import pickle
import streamlit as st
import os

# Load the model and scaler
loaded_model = pickle.load(open('D:/MLProject/trained_model.sav', 'rb'))
scaler = pickle.load(open('D:/MLProject/scaler.sav', 'rb'))

def Prediction_System(input_data):
    input_data_as_ndarray = np.asarray(input_data)
    input_data_reshaped = input_data_as_ndarray.reshape(1, -1)
    input_data_scaled = scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(input_data_scaled)
    return 'The Person does not have heart disease' if prediction[0] == 0 else 'The Person has heart disease'

def load_html(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            st.markdown(f.read(), unsafe_allow_html=True)
    else:
        st.error("HTML file not found")

def main():
    html_file_path = 'D:/MLProject/index.html'
    load_html(html_file_path)

    st.sidebar.header('Input Features')
    
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, step=1)
    sex = st.sidebar.selectbox('Gender', [0, 1])
    cp = st.sidebar.number_input('Chest Pain Type (cp)', min_value=0, max_value=3, step=1)
    trestbps = st.sidebar.number_input('Resting Blood Pressure (trestbps)', min_value=0)
    chol = st.sidebar.number_input('Serum Cholesterol (chol)', min_value=0)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
    restecg = st.sidebar.number_input('Resting Electrocardiographic Results (restecg)', min_value=0, max_value=2, step=1)
    thalach = st.sidebar.number_input('Maximum Heart Rate Achieved (thalach)', min_value=0)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', [0, 1])
    oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise Relative to Rest (oldpeak)', min_value=0.0, format="%.1f")
    slope = st.sidebar.number_input('Slope of the Peak Exercise ST Segment (slope)', min_value=0, max_value=2, step=1)
    ca = st.sidebar.number_input('Number of Major Vessels (0-3) Colored by Fluoroscopy (ca)', min_value=0, max_value=3, step=1)
    thal = st.sidebar.number_input('Thalassemia (thal)', min_value=0, max_value=3, step=1)

    Diagnosis = ""

    if st.sidebar.button('Test Result'):
        input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        Diagnosis = Prediction_System(input_data)

    st.write(f"**Diagnosis:** {Diagnosis}")

if __name__ == '__main__':
    main()


