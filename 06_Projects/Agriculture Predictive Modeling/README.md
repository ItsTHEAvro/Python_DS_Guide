# Agricultural Yield Prediction Project

## Project Overview

This project focuses on building a predictive model for crop yield based on various environmental and soil measurements. By analyzing historical agricultural data, we aim to develop a model that can assist farmers in making data-driven decisions about planting, fertilization, and irrigation.

## Business Context

Agriculture faces increasing challenges from climate change, water scarcity, and the need to feed a growing global population. Predictive modeling can help optimize resource use, increase crop yields, and reduce environmental impact by enabling precision agriculture practices.

## Dataset

The `soil_measures.csv` dataset contains the following variables:

- `soil_temperature`: Temperature of soil in Celsius
- `soil_moisture`: Percentage of water content in soil
- `soil_ph`: pH level of soil (acidity/alkalinity)
- `nitrogen_level`: Nitrogen content in soil (ppm)
- `phosphorus_level`: Phosphorus content in soil (ppm)
- `potassium_level`: Potassium content in soil (ppm)
- `rainfall`: Average rainfall in mm during growing season
- `sunlight_hours`: Average daily sunlight in hours
- `irrigation_amount`: Amount of irrigation water applied (liters per square meter)
- `fertilizer_used`: Amount of fertilizer applied (kg per hectare)
- `pesticide_used`: Amount of pesticides applied (kg per hectare)
- `crop_variety`: Type of crop planted
- `planting_density`: Number of plants per square meter
- `previous_crop`: Crop planted in the previous season
- `yield`: Crop yield in tons per hectare (target variable)

## Project Goals

1. Develop a machine learning model to predict crop yield based on environmental and agricultural factors
2. Identify the most influential factors affecting crop yield
3. Provide actionable insights for optimizing agricultural practices

## Analysis Plan

### 1. Data Preprocessing

- Clean the dataset and handle missing values
- Explore data distributions and outliers
- Encode categorical variables
- Create relevant features (feature engineering)

### 2. Exploratory Data Analysis (EDA)

- Analyze correlations between variables
- Visualize relationships between factors and yield
- Investigate seasonal patterns if applicable
- Identify potential interactions between variables

### 3. Model Selection and Training

- Split data into training and test sets
- Compare multiple regression algorithms:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Regression
- Implement cross-validation

### 4. Model Evaluation

- Assess models using multiple metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R-squared
- Analyze prediction errors

### 5. Feature Importance

- Determine which factors most strongly influence yield
- Create visualizations to illustrate important relationships

### 6. Recommendations

- Develop practical recommendations for farmers
- Suggest optimal conditions for maximizing yield
- Identify potential areas for resource optimization

## Technical Requirements

- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn
- Jupyter Notebook for analysis

## Project Deliverables

- Jupyter Notebook with complete analysis
- Agricultural yield prediction model
- Summary report with findings and recommendations
- Visualizations of key insights

## Getting Started

1. Clone this repository
2. Install the required dependencies: `pip install -r requirements.txt`
3. Open the Jupyter Notebook: `yield_prediction_analysis.ipynb`
4. Follow the step-by-step analysis process

## Advanced Extensions (Optional)

- Incorporate weather forecast data for future yield predictions
- Create an interactive dashboard for farmers to input their conditions and receive predictions
- Expand the model to account for different geographical regions
- Add time-series analysis to capture seasonal effects
