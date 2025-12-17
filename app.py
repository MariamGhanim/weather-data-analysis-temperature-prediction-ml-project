import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Temperature Prediction",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for baby blue theme
st.markdown("""
<style>
    .main {
        background-color: #f0f8ff;
    }
    
    .stApp {
        background-color: #f0f8ff;
    }
    
    .stHeader {
        background-color: #87ceeb;
        color: white;
    }
    
    .css-1d391kg {
        background-color: #e6f3ff;
    }
    
    .stSidebar {
        background-color: #e6f3ff;
    }
    
    .stSelectbox > div > div {
        background-color: #b8daff;
    }
    
    .stSlider > div > div > div {
        background-color: #87ceeb;
    }
    
    .metric-container {
        background-color: #cce7ff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #87ceeb;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background-color: #4682b4;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        background-color: #5f9ea0;
    }
    
    .prediction-box {
        background-color: #cce7ff;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #87ceeb;
        text-align: center;
        margin: 1rem 0;
    }
    
    h1, h2, h3 {
        color: #2c5282;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process the weather data"""
    try:
        df = pd.read_csv('Data/weatherHistory.csv')
        
        # Data cleaning
        df_clean = df.copy()
        df_clean = df_clean.drop_duplicates()
        df_clean = df_clean.dropna(subset=['Precip Type'])
        df_clean.drop('Loud Cover', axis=1, inplace=True, errors='ignore')
        df_clean.drop('Daily Summary', axis=1, inplace=True, errors='ignore')
        
        # Convert date
        df_clean['Formatted Date'] = pd.to_datetime(df_clean['Formatted Date'], utc=True)
        
        # Feature engineering
        df_features = df_clean.copy()
        df_features['Year'] = df_features['Formatted Date'].dt.year
        df_features['Month'] = df_features['Formatted Date'].dt.month
        df_features['Day'] = df_features['Formatted Date'].dt.day
        df_features['Hour'] = df_features['Formatted Date'].dt.hour
        df_features['WeekOfYear'] = df_features['Formatted Date'].dt.isocalendar().week.astype(int)
        
        # Season feature
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'
        
        df_features['Season'] = df_features['Month'].apply(get_season)
        
        # Time of day feature
        def get_time_of_day(hour):
            if 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            elif 18 <= hour < 22:
                return 'Evening'
            else:
                return 'Night'
        
        df_features['Time_of_Day'] = df_features['Hour'].apply(get_time_of_day)
        
        # Temperature-related features
        df_features['Temp_Apparent_Diff'] = (
            df_features['Temperature (C)'] - df_features['Apparent Temperature (C)']
        )
        
        # Categorical features
        df_features['Temp_Range_Indicator'] = pd.cut(
            df_features['Temperature (C)'],
            bins=[-50, 0, 10, 20, 30, 50],
            labels=['Very Cold', 'Cold', 'Mild', 'Warm', 'Hot']
        )
        
        df_features['Wind_Speed_Category'] = pd.cut(
            df_features['Wind Speed (km/h)'],
            bins=[0, 5, 15, 25, 40, 100],
            labels=['Calm', 'Light', 'Moderate', 'Strong', 'Very Strong'],
            include_lowest=True
        )
        
        df_features['Humidity_Category'] = pd.cut(
            df_features['Humidity'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low', 'Moderate', 'High', 'Very High'],
            include_lowest=True
        )
        
        df_features['Pressure_Category'] = pd.cut(
            df_features['Pressure (millibars)'],
            bins=[0, 1000, 1013, 1025, 1100],
            labels=['Very Low', 'Below Normal', 'Normal', 'High'],
            include_lowest=True
        )
        
        df_features['Visibility_Category'] = pd.cut(
            df_features['Visibility (km)'],
            bins=[0, 2, 5, 10, 20],
            labels=['Poor', 'Moderate', 'Good', 'Excellent'],
            include_lowest=True
        )
        
        # Comfort index
        def comfort_index(temp, humidity):
            if temp < 10:
                return 'Cold'
            elif temp > 30 and humidity > 0.7:
                return 'Hot_Humid'
            elif temp > 30:
                return 'Hot_Dry'
            elif 20 <= temp <= 25 and 0.4 <= humidity <= 0.6:
                return 'Comfortable'
            else:
                return 'Moderate'
        
        df_features['Comfort_Index'] = df_features.apply(
            lambda row: comfort_index(row['Temperature (C)'], row['Humidity']),
            axis=1
        )
        
        # Categorical encoding
        ordinal_cols = [
            'Season', 'Temp_Range_Indicator', 'Wind_Speed_Category', 'Time_of_Day',
            'Humidity_Category', 'Pressure_Category', 'Visibility_Category',
            'Comfort_Index'
        ]
        
        nominal_cols = ['Precip Type', 'Summary']
        
        df_encoded = df_features.copy()
        
        # Ordinal encoding
        ordinal_mappings = {
            'Season': ['Winter', 'Spring', 'Summer', 'Autumn'],
            'Temp_Range_Indicator': ['Very Cold', 'Cold', 'Mild', 'Warm', 'Hot'],
            'Wind_Speed_Category': ['Calm', 'Light', 'Moderate', 'Strong', 'Very Strong'],
            'Time_of_Day': ['Night', 'Morning', 'Afternoon', 'Evening'],
            'Humidity_Category': ['Low', 'Moderate', 'High', 'Very High'],
            'Pressure_Category': ['Very Low', 'Below Normal', 'Normal', 'High'],
            'Visibility_Category': ['Poor', 'Moderate', 'Good', 'Excellent'],
            'Comfort_Index': ['Cold', 'Moderate', 'Comfortable', 'Hot_Dry', 'Hot_Humid']
        }
        
        for col in ordinal_cols:
            if col in df_encoded.columns:
                ordinal_encoder = OrdinalEncoder(categories=[ordinal_mappings[col]], 
                                               handle_unknown='use_encoded_value', 
                                               unknown_value=-1)
                df_encoded[col + '_encoded'] = ordinal_encoder.fit_transform(df_encoded[[col]]).astype(int)
        
        # One-hot encoding
        for col in nominal_cols:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col.replace(' ', '_'), drop_first=False)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        return df_encoded
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def train_models(df_encoded):
    """Train machine learning models"""
    if df_encoded is None:
        return None, None, None, None
    
    # Prepare features and target
    features_to_exclude = ['Formatted Date', 'Summary', 'Apparent Temperature (C)', 'Temp_Apparent_Diff']
    X = df_encoded.drop(['Temperature (C)'] + features_to_exclude, axis=1)
    y = df_encoded['Temperature (C)']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Feature selection
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns
    X_train_numeric = X_train[numeric_columns]
    X_test_numeric = X_test[numeric_columns]
    
    k_best_features = min(12, X_train_numeric.shape[1])
    selector = SelectKBest(score_func=f_regression, k=k_best_features)
    
    X_train_selected = selector.fit_transform(X_train_numeric, y_train)
    X_test_selected = selector.transform(X_test_numeric)
    
    selected_features = numeric_columns[selector.get_support()]
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Train models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42),
        'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=20, random_state=42, verbose=-1),
        'Linear Regression': LinearRegression()
    }
    
    trained_models = {}
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        trained_models[name] = model
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['R²'])
    best_model = trained_models[best_model_name]
    
    return best_model, scaler, selector, selected_features

def main():
    st.title("Temperature Prediction Model")
    st.markdown("### Weather-based Temperature Forecasting")
    
    # Load data and train models
    with st.spinner("Loading and processing data..."):
        df_encoded = load_and_process_data()
    
    if df_encoded is None:
        st.error("Failed to load data. Please ensure the data file exists.")
        return
    
    with st.spinner("Training machine learning models..."):
        model, scaler, selector, selected_features = train_models(df_encoded)
    
    if model is None:
        st.error("Failed to train models.")
        return
    
    # Sidebar for inputs
    st.sidebar.header("Weather Conditions")
    
    # Input fields
    humidity = st.sidebar.slider("Humidity", 0.0, 1.0, 0.5, 0.01)
    wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 50.0, 10.0, 0.1)
    visibility = st.sidebar.slider("Visibility (km)", 0.0, 20.0, 10.0, 0.1)
    pressure = st.sidebar.slider("Pressure (millibars)", 900.0, 1100.0, 1013.0, 0.1)
    
    # Date and time inputs
    season = st.sidebar.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"], index=1)
    hour = st.sidebar.selectbox("Hour", range(0, 24), index=12)
    
    # Map season to encoded value
    season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}
    season_encoded = season_map[season]
    
    time_of_day_map = {
        range(0, 6): 0,   # Night
        range(6, 12): 1,  # Morning
        range(12, 18): 2, # Afternoon
        range(18, 24): 3  # Evening
    }
    time_of_day = 0
    for hour_range, tod in time_of_day_map.items():
        if hour in hour_range:
            time_of_day = tod
            break
    
    # Create feature vector
    if st.sidebar.button("Predict Temperature", type="primary"):
        try:
            # Create input features based on the model's requirements
            # This is a simplified version - you may need to adjust based on actual selected features
            
            # Calculate derived features
            humidity_cat = 0 if humidity < 0.3 else (1 if humidity < 0.6 else (2 if humidity < 0.8 else 3))
            wind_cat = 0 if wind_speed < 5 else (1 if wind_speed < 15 else (2 if wind_speed < 25 else (3 if wind_speed < 40 else 4)))
            pressure_cat = 0 if pressure < 1000 else (1 if pressure < 1013 else (2 if pressure < 1025 else 3))
            visibility_cat = 0 if visibility < 2 else (1 if visibility < 5 else (2 if visibility < 10 else 3))
            
            # Create feature array (adjust based on your actual features)
            features = np.array([[
                humidity, wind_speed, visibility, pressure,
                season_encoded, hour, time_of_day,
                humidity_cat, wind_cat, pressure_cat, visibility_cat
            ]])
            
            # Ensure we have the right number of features
            if features.shape[1] < len(selected_features):
                # Pad with zeros if needed
                padding = np.zeros((1, len(selected_features) - features.shape[1]))
                features = np.concatenate([features, padding], axis=1)
            elif features.shape[1] > len(selected_features):
                # Truncate if needed
                features = features[:, :len(selected_features)]
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            # Display result
            st.markdown(f"""
            <div class="prediction-box">
                <h2>Predicted Temperature</h2>
                <h1 style="color: #2c5282; font-size: 3rem;">{prediction:.1f}°C</h1>
                <p style="color: #4682b4;">Based on the provided weather conditions</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Temperature (°C)"},
                gauge = {
                    'axis': {'range': [-30, 50]},
                    'bar': {'color': "#4682b4"},
                    'steps': [
                        {'range': [-30, 0], 'color': "#b8daff"},
                        {'range': [0, 20], 'color': "#87ceeb"},
                        {'range': [20, 35], 'color': "#5f9ea0"},
                        {'range': [35, 50], 'color': "#4682b4"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prediction
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor="#f0f8ff",
                plot_bgcolor="#f0f8ff",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    
    # Model information
    st.markdown("### Model Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h4>Model Type</h4>
            <p>Ensemble of ML models with feature selection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h4>Features Used</h4>
            <p>Weather conditions, temporal features, and engineered variables</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h4>Training Data</h4>
            <p>Historical weather data with comprehensive preprocessing</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset overview
    if st.checkbox("Show Dataset Overview"):
        st.subheader("Dataset Information")
        st.write(f"Dataset shape: {df_encoded.shape}")
        st.write("Sample of processed data:")
        st.dataframe(df_encoded.head(), height=300)

if __name__ == "__main__":
    main()