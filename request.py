#This script sends request to web service which accepts patient info and returns the prediction whether patient will die due to heart failure.

#Before running this script, ensure you have run the following command from command line on the linux server that hosts the code to start the web service OR you have created Docker container that runs this script
#python predict.py

#Import libraries as needed
import pandas as pd
import numpy as np
import requests

#Change the URL if the prediction service is running from Cloud
url = 'http://localhost:9696/predict'

patient1 = {'age': 39, 'anaemia': '0', 'creatinine_phosphokinase': '582', 'diabetes': '0', 'ejection_fraction': '20', 'high_blood_pressure': '1', 'platelets': '265000.00', 'serum_creatinine': '1.9', 'serum_sodium': '130', 'sex': '1', 'smoking': 1, 'time': 4, 'DEATH_EVENT': 1}
patient2 = {'age': 53, 'anaemia': '1', 'creatinine_phosphokinase': '7861', 'diabetes': '0', 'ejection_fraction': '45', 'high_blood_pressure': '0', 'platelets': '263358.03', 'serum_creatinine': '1.9', 'serum_sodium': '129', 'sex': '0', 'smoking': 1, 'time': 6, 'DEATH_EVENT': 1} 
patient3 = {'age': 38, 'anaemia': '0', 'creatinine_phosphokinase': '146', 'diabetes': '0', 'ejection_fraction': '20', 'high_blood_pressure': '0', 'platelets': '162000.00', 'serum_creatinine': '0.8', 'serum_sodium': '140', 'sex': '0', 'smoking': 0, 'time': 6, 'DEATH_EVENT': 1} 
patient4 = {'age': 36, 'anaemia': '0', 'creatinine_phosphokinase': '111', 'diabetes': '0', 'ejection_fraction': '20', 'high_blood_pressure': '0', 'platelets': '210000.00', 'serum_creatinine': '1.7', 'serum_sodium': '120', 'sex': '1', 'smoking': 0, 'time': 7, 'DEATH_EVENT': 0} 
patient = patient4

response = requests.post(url,json=patient).json()
pat_prob = response['death_probability']
pat = response['death']

if pat:
    print(f"Patient is likely to die due to heart failure {pat_prob}.Please call the ambulance")
else:
    print(f"Patient may not die due to heart failure. The probability of this is {pat_prob}. Not required to call the ambulance")