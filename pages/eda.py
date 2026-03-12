import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_data():
    df = pd.read_csv("Data/weatherHistory.csv")

    # --- cleaning (mirrors notebook) ---
    df = df.drop_duplicates()
    df = df.dropna(subset=["Precip Type"])
    df.drop("Loud Cover", axis=1, inplace=True)

    # datetime features
    df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], utc=True)
    df["Month"] = df["Formatted Date"].dt.month
    df["Hour"] = df["Formatted Date"].dt.hour

    # drop heavy-text column
    df.drop("Daily Summary", axis=1, inplace=True)
    return df


def run():
    st.markdown(
        """
        <div style="text-align:center; margin-bottom: 20px;">
            <h1 style="color:#1e3a8a;">Exploratory Data Analysis</h1>
            <p style="color:#475569;">Interactive visualisations built with Plotly Express</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_data()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 3.1  UNIVARIATE ANALYSIS
    st.header("3.1 — Univariate Analysis")

    # 3.1.2 Boxplots — outlier detection
    st.subheader("3.1.2 Outlier Detection (Box Plots)")
    selected_box = st.selectbox(
        "Select a feature", numerical_cols, key="box_feature"
    )
    fig_box = px.box(
        df,
        y=selected_box,
        title=f"Box Plot – {selected_box}",
        color_discrete_sequence=["#1e3a8a"],
    )
    fig_box.update_layout(template="plotly_white")
    st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("Observation"):
        st.markdown(
            "Outliers are present in **Temperature, Apparent Temperature, "
            "Humidity, and Wind Speed**."
        )

    # 3.1.3 Summary distribution
    st.subheader("3.1.3 Weather Summary Distribution")
    summary_counts = df["Summary"].value_counts().reset_index()
    summary_counts.columns = ["Summary", "Count"]
    fig_sum = px.bar(
        summary_counts,
        x="Summary",
        y="Count",
        title="Weather Summary Distribution",
        color_discrete_sequence=["#1e3a8a"],
    )
    fig_sum.update_layout(template="plotly_white", xaxis_tickangle=-45)
    st.plotly_chart(fig_sum, use_container_width=True)

    with st.expander("Observation"):
        st.markdown(
            "Most days are **Partly Cloudy**, followed by **Mostly Cloudy**, "
            "**Overcast**, and **Foggy**."
        )

    # 3.1.4 Precipitation type
    st.subheader("3.1.4 Precipitation Type Distribution")
    precip_counts = df["Precip Type"].value_counts().reset_index()
    precip_counts.columns = ["Precip Type", "Count"]
    fig_precip = px.bar(
        precip_counts,
        x="Precip Type",
        y="Count",
        title="Precipitation Type Distribution",
        color_discrete_sequence=["#1e3a8a"],
    )
    fig_precip.update_layout(template="plotly_white")
    st.plotly_chart(fig_precip, use_container_width=True)

    with st.expander("Observation"):
        st.markdown(
            "The dataset has only two precipitation types: **rain** and **snow**. "
            "Most days experienced **rain**."
        )

    # ------------------------------------------------------------------ #
    # 3.2  BIVARIATE ANALYSIS
    # ------------------------------------------------------------------ #
    st.header("3.2 — Bivariate Analysis")

    # 3.2.1 Summary vs Precip Type
    st.subheader("3.2.1 Summary vs Precipitation Type")
    fig_scatter = px.strip(
        df,
        x="Precip Type",
        y="Summary",
        title="Summary vs Precipitation Type",
        color="Precip Type",
        color_discrete_sequence=["#1e3a8a", "#0284c7"],
    )
    fig_scatter.update_layout(template="plotly_white")
    st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("Observation"):
        st.markdown(
            """
            - **Rain** can occur under all types of weather in the dataset.
            - **Snow** is only observed under specific conditions: cloudy, foggy,
              or windy days.
            """
        )

    # 3.2.2 Precipitation Type vs Numerical Variables
    st.subheader("3.2.2 Precipitation Type vs Numerical Variables")
    numeric_no_date = [
        c
        for c in numerical_cols
        if c not in ("Month", "Hour")
    ]
    selected_num = st.selectbox(
        "Select a numerical feature", numeric_no_date, key="precip_num"
    )
    fig_precip_num = px.box(
        df,
        x="Precip Type",
        y=selected_num,
        color="Precip Type",
        title=f"{selected_num} by Precipitation Type",
        color_discrete_sequence=["#1e3a8a", "#0284c7"],
    )
    fig_precip_num.update_layout(template="plotly_white")
    st.plotly_chart(fig_precip_num, use_container_width=True)

    with st.expander("Observations — Snow vs Rain"):
        st.markdown(
            """
            | Variable | Snow vs Rain |
            |---|---|
            | Temperature & Apparent Temperature | Lower on snowy days |
            | Humidity | Higher on snowy days |
            | Wind Speed | Lower on snowy days |
            | Visibility | Lower on snowy days |
            | Pressure | No significant difference |
            """
        )

    # 3.2.3 Summary vs Numerical Variables
    st.subheader("3.2.3 Summary vs Numerical Variables")
    selected_num2 = st.selectbox(
        "Select a numerical feature", numeric_no_date, key="summary_num"
    )
    fig_summary_num = px.box(
        df,
        x="Summary",
        y=selected_num2,
        color="Summary",
        title=f"{selected_num2} by Weather Summary",
    )
    fig_summary_num.update_layout(
        template="plotly_white", xaxis_tickangle=-45, showlegend=False
    )
    st.plotly_chart(fig_summary_num, use_container_width=True)

    with st.expander("Observations"):
        st.markdown(
            """
            - **Temperature:** Highest on dry days; lowest on foggy & breezy days.
            - **Humidity:** High on foggy days.
            - **Wind Speed:** High on dangerously windy & partially cloudy days.
            - **Visibility:** Low on foggy days.
            - **Pressure:** Low on windy & breezy days.
            """
        )

    # 3.2.4 Correlation Heatmap
    st.subheader("3.2.4 Correlation Heatmap")
    numerical_df = df.select_dtypes(include=[np.number])
    corr = numerical_df.corr()
    # Mask the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = corr.where(~mask)

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=corr_masked.values,
            x=corr.columns,
            y=corr.columns,
            colorscale="Blues",
            text=np.round(corr_masked.values, 2),
            texttemplate="%{text}",
            zmin=-1,
            zmax=1,
        )
    )
    fig_heat.update_layout(
        title="Correlation Heatmap of Numerical Features",
        template="plotly_white",
        height=600,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("Correlation Summary"):
        st.markdown(
            """
            **Extremely Strong Positive Correlation**
            - Temperature (C) & Apparent Temperature (C): **0.99**

            **Strong Negative Correlation**
            - Temperature (C) & Humidity: **−0.63**

            **Moderate Positive Correlations**
            - Temperature (C) & Visibility (km): **0.39**
            - Apparent Temperature (C) & Visibility (km): **0.38**

            **Weak / Negligible Correlations**
            - Wind Speed & Humidity: 0.22 (weak positive)
            - Pressure (millibars): almost no correlation with other features.
            """
        )

    # ------------------------------------------------------------------ #
    # Q & A Summary
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.header("Key Questions & Answers from the EDA")

    qa_pairs = [
        (
            "Q1: What is the most common weather condition in the dataset?",
            "**Partly Cloudy** is the most frequent weather summary, followed by "
            "Mostly Cloudy, Overcast, and Foggy.",
        ),
        (
            "Q2: What types of precipitation are recorded, and which is more common?",
            "Only **rain** and **snow** are recorded. Rain dominates the dataset "
            "by a large margin.",
        ),
        (
            "Q3: Under what weather conditions does snow occur?",
            "Snow is only observed on **cloudy, foggy, or windy** days, "
            "while rain can occur under all weather types in the dataset.",
        ),
        (
            "Q4: How does temperature differ between rainy and snowy days?",
            "**Temperature and Apparent Temperature are significantly lower** "
            "on snowy days compared to rainy days.",
        ),
        (
            "Q5: How does humidity behave on snowy vs rainy days?",
            "**Humidity is higher on snowy days** on average, which makes sense "
            "given that snow tends to occur in colder, moisture-laden conditions.",
        ),
        (
            "Q6: Is there a difference in visibility between snow and rain?",
            "Yes — **visibility is noticeably lower on snowy days**, likely due "
            "to snowfall reducing horizontal sight distance.",
        ),
        (
            "Q7: Which weather type has the highest temperatures?",
            "**Dry days** record the highest temperatures, while **foggy and "
            "breezy days** have the lowest.",
        ),
        (
            "Q8: When is wind speed the highest?",
            "Wind speed peaks on **dangerously windy** and **partially cloudy** "
            "days according to the weather summary grouping.",
        ),
        (
            "Q9: What is the strongest correlation in the dataset?",
            "**Temperature and Apparent Temperature** have an almost perfect "
            "correlation of **0.99**, meaning they move nearly identically.",
        ),
        (
            "Q10: Which feature has the strongest negative relationship with temperature?",
            "**Humidity** has a strong negative correlation of **−0.63** with "
            "temperature — as temperature rises, humidity tends to drop.",
        ),
        (
            "Q11: Does atmospheric pressure correlate with other weather features?",
            "**Pressure is nearly independent** of all other features in the "
            "dataset — it shows almost no meaningful correlation.",
        ),
        (
            "Q12: Are there outliers in the data?",
            "Yes — box-plot analysis revealed outliers in **Temperature, "
            "Apparent Temperature, Humidity, and Wind Speed**.",
        ),
    ]

    for question, answer in qa_pairs:
        with st.expander(question):
            st.markdown(answer)

    # --- Footer ---
    st.markdown("---")
    st.caption(
        "Developed by Mariam Ghanim | Data Science Epsilon Ai Graduation Project 2026"
    )
