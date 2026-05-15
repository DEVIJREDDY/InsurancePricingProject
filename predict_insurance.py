import joblib
import pandas as pd

# Load model
model = joblib.load("insurance_model.pkl")

# Sample data
sample = pd.DataFrame({
    'Age': [30],
    'Gender': [1],
    'BMI': [28.5],
    'Children': [2],
    'Smoker': [0],
    'Region': [2]
})

# Predict insurance price
prediction = model.predict(sample)

print("Predicted Insurance Price:", prediction[0])