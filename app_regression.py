import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the saved model from a file
model_filename = 'linear_regression_model.pkl'
loaded_model = joblib.load(model_filename)

# Assuming 'Walmart.csv' is in the same directory as your Python script
file_path = 'Walmart.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Features (X) and Target (y)
features = df[['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']]
target = df['Weekly_Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

st.title('Weekly Sales Prediction App')

st.header('Input Features for Prediction')

# User input for prediction
store = st.number_input('Store', min_value=1)
holiday_flag = st.selectbox('Holiday Flag', [0, 1])
temperature = st.number_input('Temperature', min_value=-100.0, max_value=100.0, value=75.0)
fuel_price = st.number_input('Fuel Price', min_value=0.0, value=2.5)
cpi = st.number_input('CPI', min_value=0.0, value=140.0)
unemployment = st.number_input('Unemployment', min_value=0.0, value=6.5)

# Create a DataFrame with user input
example_data = pd.DataFrame({
    'Store': [store],
    'Holiday_Flag': [holiday_flag],
    'Temperature': [temperature],
    'Fuel_Price': [fuel_price],
    'CPI': [cpi],
    'Unemployment': [unemployment]
})

# Make predictions using the loaded model
predicted_sales = loaded_model.predict(example_data)

# Display predicted sales
st.subheader('Predicted Weekly Sales:')
st.write(f'{predicted_sales[0]:.2f}')

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, loaded_model.predict(X_test))

# Calculate R-squared (R²)
r_squared = r2_score(y_test, loaded_model.predict(X_test))

# Display MSE and R-squared
# st.subheader('Model Evaluation Metrics:')
# st.write(f'Mean Squared Error (MSE): {mse:.2f}')
# st.write(f'R-squared (R²): {r_squared:.2f}')
