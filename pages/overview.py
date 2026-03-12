import streamlit as st


def run():
    st.markdown(
        """
        <div style="text-align:center; margin-bottom: 30px;">
            <img src="https://cdn-icons-png.flaticon.com/512/4052/4052984.png" width="100">
            <h1 style="color:#1e3a8a;">Weather Temperature Predictor</h1>
            <p style="font-size:18px; color:#475569;">
                 Temperature Prediction Using Machine Learning
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # --- Project Overview ---
    st.subheader("Project Overview")
    st.markdown(
        """
        This project predicts **ambient temperature (°C)** from historical weather
        observations using a **Random Forest Regressor** trained on real-world data.

        The dataset spans multiple years of hourly weather records and includes
        parameters such as **humidity, atmospheric pressure, wind speed,
        wind bearing, precipitation type, and visibility**.

        The final model was selected after comparing seven regression algorithms
        and fine-tuning the top two through grid-search cross-validation.
        """
    )

    # --- Dataset at a Glance ---
    st.markdown("---")
    st.subheader("Dataset at a Glance")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Records", "~96,453")
    with col2:
        st.metric("Features", "12 columns")
    with col3:
        st.metric("Time Span", "2006 – 2016")

    st.markdown(
        """
        | Feature | Description |
        |---|---|
        | Temperature (C) | Target variable – ambient temperature |
        | Humidity | Relative humidity (0 – 1) |
        | Pressure (millibars) | Atmospheric pressure |
        | Wind Speed (km/h) | Surface wind speed |
        | Wind Bearing (degrees) | Wind direction (0° – 360°) |
        | Visibility (km) | Horizontal visibility |
        | Precip Type | Rain or Snow |
        | Summary | Short weather description |
        """
    )

    # --- Project Workflow ---
    st.markdown("---")
    st.subheader("Project Workflow")

    steps = [
        ("1. Data Loading & Inspection", "Loaded the CSV, checked shape, types, and missing values."),
        ("2. Data Cleaning", "Removed duplicates, handled missing Precip Type (~0.02 %), dropped constant column (Loud Cover), and converted date to datetime."),
        ("3. Exploratory Data Analysis", "Univariate & bivariate analysis with skewness, box-plots, bar charts, scatter plots, and a correlation heatmap."),
        ("4. Feature Engineering", "Extracted Month & Hour from the date, label-encoded Precip Type, one-hot-encoded Summary, and dropped highly correlated Apparent Temperature."),
        ("5. Feature Selection", "Used Random Forest feature importances to drop low-impact features (< 0.5 % importance)."),
        ("6. Model Training", "Trained 7 models: Linear, Ridge, Lasso, Decision Tree, Random Forest, XGBoost, KNN."),
        ("7. Hyperparameter Tuning", "Grid-searched Random Forest & XGBoost; Tuned Random Forest achieved the best balance."),
        ("8. Deployment", "Final model and scaler saved with joblib; served through this Streamlit app."),
    ]

    for title, desc in steps:
        with st.expander(title):
            st.write(desc)

    # --- Model Performance ---
    st.markdown("---")
    st.subheader("Final Model Performance")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Model", "Tuned Random Forest")
    with m2:
        st.metric("R² Score", "0.9200", delta="Excellent")
    with m3:
        st.metric("MAE", "2.0858 °C")

    st.info(
        "The Tuned Random Forest Regressor was selected for its high accuracy "
        "and strong generalisation on unseen data."
    )

    # --- Footer ---
    st.markdown("---")
    st.caption(
        "Developed by Mariam Ghanim | Data Science Epsilon Ai Graduation Project 2026"
    )
