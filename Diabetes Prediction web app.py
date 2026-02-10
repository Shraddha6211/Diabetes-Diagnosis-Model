# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 14:47:14 2026

@author: Asus
"""

import numpy as np
import pickle
import streamlit as st


# Loading the saved model
loaded_model = pickle.load(open('D:\Personal\Learnings\Machine Learning\Diabetes Diagnosis/trained_model.sav','rb'))

# creating a function for Prediction

def diabetes_prediction(input_data):
    
    
    # change this data to numpy array
    input_data_as_nparray = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    # if we don't reshape, the model keeps expecting 767 inputs as it was trained on 767 data points
    input_data_reshaped = input_data_as_nparray.reshape(1,-1)

    #Standardize the input as well
    # std_data = scaler.transform(input_data_reshaped)
    # std_data

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return "The person is not Diabetic."
    else:
        return "The person is Diabetic."
    
    
def main():
    
    # Giving a Title for our webpage
    st.title('Diabetes Prediction Web App')
    
    # Getting the input from the user
        
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Blood Glucose Level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('Body Mass Index')
    DiabetesPedigreeFunction = st.text_input('Value of Diabetes Pedigree Function')
    Age = st.text_input('Age')
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for prediction
    
    if st.button("Get my Diagnosis"):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    