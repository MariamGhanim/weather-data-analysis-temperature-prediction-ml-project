import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Weather Predictor", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { 
        background: rgba(255, 255, 255, 0.7); 
        padding: 20px; 
        border-radius: 15px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.3);
    }
    .predict-card {
        background: linear-gradient(135deg, #007bff, #00d4ff);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    div[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
    h1, h2, h3 { color: #1e3a8a; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_assets():
    model = joblib.load('best_random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model_assets()
except:
    st.error("Model files not found! Please save your model and scaler first")
    st.stop()

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4052/4052984.png", width=100)
st.sidebar.title("Weather Parameters")
st.sidebar.markdown("Adjust the sliders to predict the temperature")

def user_input_features():
    humidity = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 50.0)
    pressure = st.sidebar.slider('Pressure (millibars)', 900.0, 1100.0, 1010.0)
    wind_bearing = st.sidebar.slider('Wind Bearing (degrees)', 0.0, 360.0, 180.0)
    month = st.sidebar.slider('Month', 1, 12, 6)
    hour = st.sidebar.slider('Hour', 0, 23, 12)
    precip_type = st.sidebar.selectbox('Precip Type', options=[('Rain', 0), ('Snow', 1)], format_func=lambda x: x[0])[1]

    data = {
        'Humidity': humidity,
        'Month': month,
        'Precip Type': precip_type,
        'Pressure (millibars)': pressure,
        'Hour': hour,
        'Wind Bearing (degrees)': wind_bearing,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.title("Temperature Prediction Dashboard")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Model Performance")
    st.metric(label="R2 Score", value="0.9200", delta="Excellent")
    st.metric(label="MAE", value="2.0858", delta_color="inverse")
    
    st.info("The Tuned Random Forest model was selected for its high accuracy")

with col2:
    st.subheader("Prediction Result")
    
    # 1. Scaling the input
    input_scaled = scaler.transform(input_df)
    
    # 2. Making the prediction
    prediction = model.predict(input_scaled)[0]
    
    # 3. Visualizing 
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Temperature (°C)"},
        gauge = {
            'axis': {'range': [None, 50]},
            'bar': {'color': "#1e3a8a"},
            'steps' : [
                {'range': [0, 15], 'color': "#e0f2fe"},
                {'range': [15, 30], 'color': "#7dd3fc"},
                {'range': [30, 50], 'color': "#0284c7"}],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
    <div class="predict-card">
        The estimated temperature for the current conditions is: {prediction:.2f}°C
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Developed by Mariam Ghanim | Data Science Epsilon Ai Graduation Project 2026")