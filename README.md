# Weather Data Analysis & Temperature Prediction ML Project

![Application Screenshot](appPic.PNG)

## Overview
This is a Data Analysis and ML project that implements temperature prediction based on weather conditions. The project features complete data exploration, feature engineering, model comparison, and deployment through an interactive Streamlit web application.

**Key Components:**
- **Data Analysis**: Comprehensive exploratory data analysis with statistical insights and visualizations
- **Machine Learning**: Multiple model comparison (Linear Regression, Random Forest, LightGBM) with hyperparameter tuning
- **Web Application**: Interactive Streamlit interface for real-time temperature predictions

**Note: The weather data used in this project is synthetic and not real-world data.**

## Live Application
Access the deployed application here: https://temperature-prediction-mariamghanim.streamlit.app/

## Features
- Interactive Streamlit web interface
- Temperature prediction using Random Forest model
- Comprehensive Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing pipeline
- Real-time predictions with user-friendly controls

## Model Overview
The project trained and compared three different machine learning models to find the best performer:

### Models Compared:
1. **Linear Regression**: Simple baseline model
2. **Random Forest Regressor**: Tree-based ensemble method
3. **LightGBM**: Gradient boosting framework

### Best Model Selected: Random Forest Regressor
After comprehensive evaluation, Random Forest was chosen as the final model due to its superior performance:

### Model Performance After Training:
- **Model Type**: Random Forest Regressor (Selected from 3 models)
- **Best Parameters**: n_estimators=100, max_depth=20
- **Cross-Validation RMSE**: ~2.5°C
- **Test Set Performance**:
  - MAE (Mean Absolute Error): ~1.8°C
  - RMSE (Root Mean Square Error): ~2.4°C
  - R² Score: ~0.95
- **Features Used**: 12 selected features including humidity, wind speed, visibility, pressure, temporal features, and engineered variables

## Data Analysis and Visualizations

### Key Graphs and Analysis from the Notebook:

#### 1. Data Skewness Analysis
Bar chart showing the skewness distribution of all numerical attributes to identify data distribution patterns.

#### 2. Outlier Detection (Box Plots)
Multiple box plots for each numerical variable (Temperature, Humidity, Wind Speed, Visibility, Pressure) revealing outliers in the dataset.

#### 3. Weather Summary Distribution
Bar chart displaying the frequency of different weather conditions, showing that partly cloudy days are most common.

#### 4. Precipitation Type Analysis
Bar chart comparing rain vs snow frequency, demonstrating that rain is more prevalent in the dataset.

#### 5. Summary vs Precipitation Type (Scatter Plot)
Scatter plot revealing the relationship between weather conditions and precipitation types.

#### 6. Precipitation Type vs Numerical Variables
Multiple bar plots showing how temperature, humidity, wind speed, and other variables differ between rainy and snowy days.

#### 7. Weather Summary vs Numerical Variables
Extensive bar plot analysis showing how different weather summaries (dry, foggy, windy) relate to various meteorological measurements.

#### 8. Correlation Heatmap
Heatmap visualization of correlations between all numerical features, highlighting the strong relationship between temperature and apparent temperature (0.99) and the negative correlation between temperature and humidity (-0.63).

## Data Analysis and Observations

### Key Findings from Exploratory Data Analysis:

#### Skewness Analysis:
- Most weather attributes show relatively normal distribution
- Some outliers present in Temperature, Apparent Temperature, Humidity, and Wind Speed

#### Weather Patterns:
- **Most Common Weather**: Partly cloudy days dominate the dataset, followed by mostly cloudy, overcast, and foggy conditions
- **Precipitation Types**: Only two types observed - rain and snow, with rain being more frequent

#### Temperature Correlations:
- **Strong Positive Correlation (0.99)**: Temperature and Apparent Temperature are almost perfectly linked
- **Strong Negative Correlation (-0.63)**: Temperature and Humidity - as temperature rises, humidity typically decreases
- **Moderate Positive Correlations**: Temperature with Visibility (0.39) - warmer weather tends to improve visibility

#### Precipitation Analysis:
- **Rain**: Can occur under all weather conditions in the dataset
- **Snow**: Only observed during cloudy, foggy, or windy conditions
- **Temperature Differences**: Snow days show lower temperatures and higher humidity compared to rainy days
- **Visibility**: Snow days have reduced visibility compared to rainy days

#### Seasonal Weather Characteristics:
- **Dry Days**: Highest temperatures recorded
- **Foggy/Breezy Days**: Lowest temperatures
- **Humidity Patterns**: Highest during foggy conditions
- **Wind Speed**: Peaks during dangerously windy and partially cloudy days
- **Pressure**: Lower during windy and breezy conditions

## Technical Implementation

### Data Preprocessing:
- Duplicate removal and missing value handling
- DateTime feature extraction (Year, Month, Day, Hour, Week)
- Categorical encoding (Ordinal and One-Hot encoding)
- Feature scaling using StandardScaler
- Feature selection using SelectKBest with F-regression

### Feature Engineering:
- Seasonal categorization
- Time of day classification
- Temperature range indicators
- Wind speed categories
- Humidity and pressure categories
- Visibility categories
- Comfort index calculation

### Model Training:
- Train-test split (80-20)
- Feature selection (top 12 features)
- Hyperparameter tuning with GridSearchCV
- Cross-validation for model evaluation

## Application Interface
The Streamlit app provides intuitive controls for:
- Humidity adjustment (0.0 - 1.0)
- Wind Speed input (0-50 km/h)
- Visibility setting (0-20 km)
- Pressure configuration (900-1100 millibars)
- Season selection (Winter, Spring, Summer, Autumn)
- Month selection (based on chosen season)
- Hour of day selection (0-23)

## Installation and Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

## Project Structure
- `app.py` - Streamlit web application
- `Notebook.ipynb` - Data analysis and model development
- `Data/weatherHistory.csv` - Weather dataset
- `requirements.txt` - Python dependencies

## Technologies Used
- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Plotly**: Interactive visualizations
- **LightGBM**: Additional model comparison
