import streamlit as st
from pages import overview, eda, prediction

st.set_page_config(
    page_title="Weather Predictor",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric {
        background: rgba(255, 255, 255, 0.7);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(10px);
    }
    .predict-card {
        background: linear-gradient(135deg, #007bff, #00d4ff);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        backdrop-filter: blur(12px);
    }
    div[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border-right: 1px solid #e0e0e0;
    }
    h1, h2, h3 { color: #1e3a8a; }
    .stExpander {
        background: rgba(255, 255, 255, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(8px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

pg = st.navigation(
    [
        st.Page(overview.run, title="Project Overview", url_path="overview"),
        st.Page(eda.run, title="Exploratory Data Analysis", url_path="eda"),
        st.Page(prediction.run, title="Temperature Prediction", url_path="prediction"),
    ]
)
pg.run()