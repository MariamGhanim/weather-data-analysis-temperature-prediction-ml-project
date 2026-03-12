import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download


@st.cache_resource
def load_model_assets():
    model_path = hf_hub_download(
        repo_id="MariamGhanim/Weather-Prediction-1",
        filename="best_random_forest_model.joblib"
    )
    scaler_path = hf_hub_download(
        repo_id="MariamGhanim/Weather-Prediction-1",
        filename="scaler.joblib"
    )
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def run():
    st.markdown(
        """
        <div style="text-align:center; margin-bottom: 20px;">
            <h1 style="color:#1e3a8a;">Temperature Prediction</h1>
            <p style="color:#475569;">Adjust the parameters and get a real-time prediction</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model, scaler = load_model_assets()
    except Exception:
        st.error("Model files not found! Please save your model and scaler first.")
        st.stop()

    # --- Sidebar inputs ---
    st.sidebar.image(
        "https://cdn-icons-png.flaticon.com/512/4052/4052984.png", width=100
    )
    st.sidebar.title("Weather Parameters")
    st.sidebar.markdown("Adjust the sliders to predict the temperature")

    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
    pressure = st.sidebar.slider("Pressure (millibars)", 900.0, 1100.0, 1010.0)
    wind_bearing = st.sidebar.slider("Wind Bearing (degrees)", 0.0, 360.0, 180.0)
    month = st.sidebar.slider("Month", 1, 12, 6)
    hour = st.sidebar.slider("Hour", 0, 23, 12)
    precip_type = st.sidebar.selectbox(
        "Precip Type",
        options=[("Rain", 0), ("Snow", 1)],
        format_func=lambda x: x[0],
    )[1]

    input_df = pd.DataFrame(
        {
            "Humidity": [humidity],
            "Month": [month],
            "Precip Type": [precip_type],
            "Pressure (millibars)": [pressure],
            "Hour": [hour],
            "Wind Bearing (degrees)": [wind_bearing],
        }
    )

    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Model Performance")
        st.metric(label="R² Score", value="0.9200", delta="Excellent")
        st.metric(label="MAE", value="2.0858", delta_color="inverse")
        st.info(
            "The Tuned Random Forest model was selected for its high accuracy."
        )

    with col2:
        st.subheader("Prediction Result")

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prediction,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Predicted Temperature (°C)"},
                gauge={
                    "axis": {"range": [None, 50]},
                    "bar": {"color": "#1e3a8a"},
                    "steps": [
                        {"range": [0, 15], "color": "#e0f2fe"},
                        {"range": [15, 30], "color": "#7dd3fc"},
                        {"range": [30, 50], "color": "#0284c7"},
                    ],
                },
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
        <div class="predict-card">
            The estimated temperature for the current conditions is: {prediction:.2f}°C
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Footer ---
    st.markdown("---")
    st.caption(
        "Developed by Mariam Ghanim | Data Science Epsilon Ai Graduation Project 2026"
    )
