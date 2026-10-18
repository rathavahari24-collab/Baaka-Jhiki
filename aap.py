import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(page_title="Delivery Delay Predictor", layout="centered")

st.title("▒Đ Delivery Delay Prediction App")
st.write("This application uses a Decision Tree model to predict whether a delivery will be delayed based on key operational parameters.")

# Load the saved Decision Tree model
@st.cache_resource
def load_model():
    return joblib.load('best_dt_model.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model. Please ensure 'best_dt_model.joblib' is present. Details: {e}")
    st.stop()

# Create input form
st.subheader("Enter Delivery Details")
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        delivery_distance = st.number_input("Delivery Distance (miles)", min_value=0.0, value=20.0, step=0.1)
        traffic_congestion = st.slider("Traffic Congestion Level (1-5)", min_value=1, max_value=5, value=3)
        weather_condition = st.slider("Weather Condition Level (1-3)", min_value=1, max_value=3, value=1)
        delivery_slot = st.slider("Delivery Slot Time (1-3)", min_value=1, max_value=3, value=2)
        driver_experience = st.number_input("Driver Experience (years)", min_value=0, value=5, step=1)
        num_stops = st.number_input("Number of Stops", min_value=0, value=3, step=1)
        
    with col2:
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, value=3, step=1)
        road_condition_score = st.slider("Road Condition Score (1-5)", min_value=1, max_value=5, value=3)
        package_weight = st.number_input("Package Weight (lbs)", min_value=0.0, value=15.0, step=0.1)
        fuel_efficiency = st.number_input("Fuel Efficiency (mpg)", min_value=0.0, value=12.0, step=0.1)
        warehouse_processing_time = st.number_input("Warehouse Processing Time (minutes)", min_value=0, value=45, step=1)

    submit_button = st.form_submit_button(label="Predict Delay Status")

if submit_button:
    # Assemble input features into a DataFrame with exact column names
    input_data = pd.DataFrame([[ 
        delivery_distance,
        traffic_congestion,
        weather_condition,
        delivery_slot,
        driver_experience,
        num_stops,
        vehicle_age,
        road_condition_score,
        package_weight,
        fuel_efficiency,
        warehouse_processing_time
    ]], columns=[
        'Delivery_Distance',
        'Traffic_Congestion',
        'Weather_Condition',
        'Delivery_Slot',
        'Driver_Experience',
        'Num_Stops',
        'Vehicle_Age',
        'Road_Condition_Score',
        'Package_Weight',
        'Fuel_Efficiency',
        'Warehouse_Processing_Time'
    ])
    
    # Run prediction
    prediction = model.predict(input_data)[0]
    
    # Display predictions
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ Predicted Status: **Delayed**")
    else:
        st.success("✅ Predicted Status: **No Delay**")
