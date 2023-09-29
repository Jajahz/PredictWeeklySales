import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the saved model from a file
model_filename = 'linear_regression_model.pkl'
loaded_model = joblib.load(model_filename)

# Example DataFrame for prediction
example_data = pd.DataFrame({
    'Store': [2],
    'Holiday_Flag': [1],
    'Temperature': [75.0],
    'Fuel_Price': [2.5],
    'CPI': [140.0],
    'Unemployment': [6.5]
})

# Sidebar for user input
st.sidebar.header('Input Features for Prediction')
store = st.sidebar.number_input('Store', min_value=1)
holiday_flag = st.sidebar.selectbox('Holiday Flag', [0, 1])
temperature = st.sidebar.number_input('Temperature', min_value=-100.0, max_value=100.0, value=75.0)
fuel_price = st.sidebar.number_input('Fuel Price', min_value=0.0, value=2.5)
cpi = st.sidebar.number_input('CPI', min_value=0.0, value=140.0)
unemployment = st.sidebar.number_input('Unemployment', min_value=0.0, value=6.5)

# Create a DataFrame with user input
user_data = pd.DataFrame({
    'Store': [store],
    'Holiday_Flag': [holiday_flag],
    'Temperature': [temperature],
    'Fuel_Price': [fuel_price],
    'CPI': [cpi],
    'Unemployment': [unemployment]
})

# Make predictions using the loaded model
predicted_sales = loaded_model.predict(user_data)

# Display predicted sales
st.subheader('Predicted Weekly Sales:')
st.write(f'{predicted_sales[0]:.2f}')

# (Assuming y_test and predictions are defined in your original code)
# Calculate Mean Squared Error (MSE)
# mse = mean_squared_error(y_test, predictions)
# st.subheader('Mean Squared Error (MSE):')
# st.write(f'{mse:.2f}')

# # Calculate R-squared (R²)
# r_squared = r2_score(y_test, predictions)
# st.subheader('R-squared (R²):')
# st.write(f'{r_squared:.2f}')