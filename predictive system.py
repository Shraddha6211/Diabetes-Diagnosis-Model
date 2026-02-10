# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# Loading the saved model
loaded_model = pickle.load(open('D:\Personal\Learnings\Machine Learning\Diabetes Diagnosis/trained_model.sav','rb'))

input_data = (1,84,66,23,94,28.1,0.127,20)

# change this data to numpy array
input_data_as_nparray = np.asarray(input_data)

# reshape the array as we are predicting for one instance
# if we don't reshape, the model keeps expecting 767 inputs as it was trained on 767 data points
input_data_reshaped = input_data_as_nparray.reshape(1,-1)

#Standardize the input as well
# std_data = scaler.transform(input_data_reshaped)
# std_data

prediction = loaded_model.predict(input_data_reshaped)
prediction

if (prediction[0] == 0):
    print("The person is not Diabetic.")
else:
    print("The person is Diabetic.")
    
    