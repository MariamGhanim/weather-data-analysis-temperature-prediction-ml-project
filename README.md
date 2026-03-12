<p align="center">
  <img src="https://cdn-icons-png.flaticon.com/512/4052/4052984.png" width="120" alt="Weather Icon"/>
</p>

<h1 align="center">Temperature Prediction using Machine Learning</h1>

<p align="center">
  A machine learning project that predicts temperature based on weather parameters, with an interactive Streamlit dashboard for real-time predictions.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-1.52-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Model-Random%20Forest-green" alt="Model"/>
</p>

---

## Overview

This project builds a regression model to predict **temperature (°C)** from historical weather data. The full pipeline includes data cleaning, exploratory data analysis, feature engineering, model comparison, hyperparameter tuning, and deployment as a web app.

## Dataset

- **Source:** [Kaggle — Weather Dataset](https://www.kaggle.com/datasets/muthuj7/weather-dataset/)
- **Records:** ~96,000 hourly weather observations
- **Target Variable:** `Temperature (C)`
- **Features Used:** Humidity, Pressure, Wind Bearing, Precipitation Type, Month, Hour

## Project Pipeline

```
Data Loading → Cleaning → EDA & Visualization → Feature Engineering
    → Train/Test Split (80/20) → Scaling → Model Training → Tuning → Deployment
```

### Models Trained

Seven regression models were trained and evaluated: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, XGBoost, and KNN. After comparison, the **Tuned Random Forest** was selected as the best model via `GridSearchCV` (3-fold CV), achieving an **R² of 0.9200** and **MAE of 2.0858**.

### Hyperparameter Tuning (Random Forest)

```
n_estimators: [100, 200]
max_depth: [10, 20]
min_samples_split: [2, 5]
max_features: ['sqrt']
```

## Web Application

An interactive **Streamlit** dashboard allows users to adjust weather parameters via sliders and get real-time temperature predictions displayed on a gauge chart.

**Features:**
- Sidebar controls for Humidity, Pressure, Wind Bearing, Month, Hour, and Precipitation Type
- Gauge visualization powered by Plotly
- Model performance metrics (R² and MAE) displayed on the dashboard

### Run the App

```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                          # Streamlit web application
├── Notebook.ipynb                  # Full ML pipeline (EDA, training, tuning)
├── best_random_forest_model.joblib # Trained model
├── scaler.joblib                   # Fitted StandardScaler
├── requirements.txt                # Python dependencies
├── Data/
│   ├── weatherHistory.csv          # Raw dataset
│   └── Data and Github Links.txt   # Source links
└── README.md
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/MariamGhanim/weather-data-analysis-temperature-prediction-ml-project.git
cd weather-data-analysis-temperature-prediction-ml-project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Dashboard

```bash
streamlit run app.py
```

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| ML | scikit-learn, XGBoost |
| Deployment | Streamlit |

## Author

**Mariam Ghanim**
Data Science Epsilon AI — Graduation Project 2026
