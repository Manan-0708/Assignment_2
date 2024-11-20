import pandas as pd
import joblib  # Use joblib to save and load the model
from sklearn.preprocessing import StandardScaler

# Load the trained model (assuming you've saved it as 'diabetes_model.pkl')
model = joblib.load('Diabetes_model_results.pkl')

# Function to preprocess input data and make predictions
def preprocess_and_predict(input_data):
    # Assuming input_data is a dictionary with the required features
    df = pd.DataFrame([input_data])
    
    # Apply any required preprocessing (e.g., scaling)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Make prediction
    prediction = model.predict(df_scaled)
    return prediction[0]