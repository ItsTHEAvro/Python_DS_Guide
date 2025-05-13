# Agriculture Predictive Modeling

## Project Overview

This project explores predictive modeling techniques for agricultural applications. You'll use various soil measurements to predict crop yield, identify optimal growing conditions, and develop models that could help farmers make data-driven decisions. Through this project, you'll gain experience applying machine learning to real-world agricultural problems.

## Learning Objectives

- Apply regression and classification techniques to agricultural data
- Evaluate model performance and select appropriate metrics for agricultural predictions
- Interpret model outcomes in the context of farming decisions
- Handle environmental and seasonal time-series data
- Communicate findings in a way that's actionable for non-technical stakeholders

## Key Topics Covered

- Data preprocessing for environmental measurements
- Feature engineering for agricultural predictors
- Regression modeling for yield prediction
- Feature importance and model interpretation
- Handling geospatial components in predictive models
- Time-series aspects of crop growth cycles

## Dataset Description

The `soil_measures.csv` file contains various soil measurements from different agricultural plots along with corresponding crop yields and other farming parameters:

- `plot_id`: Unique identifier for each agricultural plot
- `soil_type`: Classification of soil type (e.g., clay, loam, sandy)
- `ph_level`: Soil pH level
- `nitrogen_kg_ha`: Nitrogen content (kg per hectare)
- `phosphorus_kg_ha`: Phosphorus content (kg per hectare)
- `potassium_kg_ha`: Potassium content (kg per hectare)
- `organic_matter_pct`: Organic matter percentage in soil
- `moisture_pct`: Soil moisture percentage
- `temperature_c`: Soil temperature in Celsius
- `rainfall_mm`: Seasonal rainfall in millimeters
- `irrigation_type`: Type of irrigation system used
- `crop_type`: Type of crop grown
- `planting_date`: Date when crop was planted
- `harvest_date`: Date when crop was harvested
- `yield_tons_ha`: Crop yield in tons per hectare
- `previous_crop`: Type of crop grown in previous season
- `fertilizer_used`: Type of fertilizer used
- `pesticide_used`: Whether pesticides were used (yes/no)
- `region`: Geographical region where the plot is located
- `elevation_m`: Elevation of the plot in meters

## Project Tasks

### 1. Data Loading and Initial Exploration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from sklearn.inspection import permutation_importance

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Load the dataset
soil_data = pd.read_csv('soil_measures.csv')

# Initial exploration
print(f"Dataset shape: {soil_data.shape}")
print(soil_data.info())

# Summary statistics
print(soil_data.describe())

# Check for missing values
print("\nMissing values per column:")
print(soil_data.isna().sum())

# Display a few rows
print("\nSample data:")
print(soil_data.head())

# Check categorical variable distributions
for col in ['soil_type', 'irrigation_type', 'crop_type', 'region']:
    print(f"\n{col} distribution:")
    print(soil_data[col].value_counts())

# Preview correlations between numeric variables and yield
numeric_cols = soil_data.select_dtypes(include=['float64', 'int64']).columns
correlation = soil_data[numeric_cols].corr()['yield_tons_ha'].sort_values(ascending=False)
print("\nCorrelations with yield:")
print(correlation)
```

### 2. Data Cleaning and Preprocessing

```python
# Create a clean copy of the dataset
agriculture_data = soil_data.copy()

# Convert date columns to datetime
agriculture_data['planting_date'] = pd.to_datetime(agriculture_data['planting_date'])
agriculture_data['harvest_date'] = pd.to_datetime(agriculture_data['harvest_date'])

# Calculate growing period length (days)
agriculture_data['growing_period'] = (agriculture_data['harvest_date'] -
                                      agriculture_data['planting_date']).dt.days

# Extract month from planting date as a feature
agriculture_data['planting_month'] = agriculture_data['planting_date'].dt.month

# Handle missing values
# For numeric columns, use the median
num_cols = agriculture_data.select_dtypes(include=['float64', 'int64']).columns
agriculture_data[num_cols] = agriculture_data[num_cols].fillna(agriculture_data[num_cols].median())

# For categorical columns, use the most frequent value
cat_cols = agriculture_data.select_dtypes(include=['object']).columns
for col in cat_cols:
    agriculture_data[col] = agriculture_data[col].fillna(agriculture_data[col].mode()[0])

# Create NPK ratio feature (nitrogen:phosphorus:potassium)
agriculture_data['n_p_ratio'] = agriculture_data['nitrogen_kg_ha'] / agriculture_data['phosphorus_kg_ha']
agriculture_data['n_k_ratio'] = agriculture_data['nitrogen_kg_ha'] / agriculture_data['potassium_kg_ha']
agriculture_data['p_k_ratio'] = agriculture_data['phosphorus_kg_ha'] / agriculture_data['potassium_kg_ha']

# Create a feature for water availability (combining rainfall and irrigation)
agriculture_data['water_availability'] = agriculture_data['rainfall_mm']
# Adjust water availability based on irrigation type (simplified approach)
irrigation_efficiency = {'drip': 0.9, 'sprinkler': 0.7, 'flood': 0.5, 'none': 0}
agriculture_data['irrigation_efficiency'] = agriculture_data['irrigation_type'].map(irrigation_efficiency)

# Convert categorical features to dummy variables
agriculture_data = pd.get_dummies(agriculture_data, columns=['soil_type', 'irrigation_type',
                                                         'crop_type', 'previous_crop',
                                                         'fertilizer_used', 'region'],
                              drop_first=True)

# Convert binary variables
agriculture_data['pesticide_used'] = agriculture_data['pesticide_used'].map({'yes': 1, 'no': 0})

# Drop unnecessary columns for modeling
cols_to_drop = ['plot_id', 'planting_date', 'harvest_date']
agriculture_data_model = agriculture_data.drop(cols_to_drop, axis=1)

# Check the processed data
print(f"Processed data shape: {agriculture_data_model.shape}")
print(agriculture_data_model.columns)
```

### 3. Exploratory Data Analysis

```python
# Visualize distribution of the target variable (yield)
plt.figure(figsize=(10, 6))
sns.histplot(agriculture_data['yield_tons_ha'], kde=True)
plt.title('Distribution of Crop Yield', fontsize=16)
plt.xlabel('Yield (tons per hectare)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualize relationship between key soil properties and yield
key_soil_props = ['ph_level', 'nitrogen_kg_ha', 'phosphorus_kg_ha',
                  'potassium_kg_ha', 'organic_matter_pct', 'moisture_pct']

plt.figure(figsize=(18, 10))
for i, feature in enumerate(key_soil_props):
    plt.subplot(2, 3, i+1)
    sns.scatterplot(x=feature, y='yield_tons_ha', hue='crop_type',
                    data=soil_data, alpha=0.7)
    plt.title(f'Yield vs. {feature}', fontsize=14)
    plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Analyze yield by crop type
plt.figure(figsize=(12, 6))
sns.boxplot(x='crop_type', y='yield_tons_ha', data=soil_data)
plt.title('Yield Distribution by Crop Type', fontsize=16)
plt.xlabel('Crop Type', fontsize=12)
plt.ylabel('Yield (tons per hectare)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Analyze yield by soil type
plt.figure(figsize=(12, 6))
sns.boxplot(x='soil_type', y='yield_tons_ha', data=soil_data)
plt.title('Yield Distribution by Soil Type', fontsize=16)
plt.xlabel('Soil Type', fontsize=12)
plt.ylabel('Yield (tons per hectare)', fontsize=12)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Analyze yield by region
plt.figure(figsize=(14, 6))
sns.boxplot(x='region', y='yield_tons_ha', data=soil_data)
plt.title('Yield Distribution by Region', fontsize=16)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Yield (tons per hectare)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Correlation heatmap for numeric variables
plt.figure(figsize=(14, 12))
corr = soil_data.select_dtypes(include=['float64', 'int64']).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.tight_layout()
plt.show()

# Analyze growing period vs. yield by crop type
plt.figure(figsize=(12, 6))
sns.scatterplot(x='growing_period', y='yield_tons_ha', hue='crop_type',
                data=agriculture_data, alpha=0.7)
plt.title('Yield vs. Growing Period by Crop Type', fontsize=16)
plt.xlabel('Growing Period (days)', fontsize=12)
plt.ylabel('Yield (tons per hectare)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(title='Crop Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

### 4. Model Development and Training

```python
# Prepare data for modeling
X = agriculture_data_model.drop('yield_tons_ha', axis=1)
y = agriculture_data_model['yield_tons_ha']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Return results
    return {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'model': model
    }

# Create and evaluate models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Evaluate all models
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)

    print(f"  Train RMSE: {results[name]['train_rmse']:.4f}")
    print(f"  Test RMSE: {results[name]['test_rmse']:.4f}")
    print(f"  Train R²: {results[name]['train_r2']:.4f}")
    print(f"  Test R²: {results[name]['test_r2']:.4f}")
    print(f"  Train MAE: {results[name]['train_mae']:.4f}")
    print(f"  Test MAE: {results[name]['test_mae']:.4f}")
    print()

# Visualize model performance comparison
metrics = ['test_rmse', 'test_r2', 'test_mae']
model_names = list(results.keys())

plt.figure(figsize=(15, 12))
for i, metric in enumerate(metrics):
    plt.subplot(3, 1, i+1)
    values = [results[model][metric] for model in model_names]

    # For R², higher is better, for RMSE and MAE, lower is better
    bars = plt.bar(model_names, values, color='skyblue')

    if metric == 'test_r2':
        # Add a line at R² = 0 (no predictive power)
        plt.axhline(y=0, color='red', linestyle='--')
        # Highlight the best model
        bars[np.argmax(values)].set_color('green')
    else:
        # Highlight the best model (lowest error)
        bars[np.argmin(values)].set_color('green')

    plt.title(f'Model Comparison - {metric}', fontsize=14)
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

### 5. Feature Importance and Model Interpretation

```python
# Get the best performing model (assuming it's Random Forest or XGBoost)
best_model = results['Random Forest']['model']  # Change to the actual best model

# For tree-based models, we can directly get feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15))
    plt.title('Top 15 Most Important Features for Yield Prediction', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Top 15 most important features:")
    print(feature_importances.head(15))
else:
    # For non-tree models, use permutation importance
    perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)

    # Plot permutation importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15))
    plt.title('Top 15 Features (Permutation Importance)', fontsize=16)
    plt.xlabel('Importance (Increase in Error When Permuted)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Top 15 most important features (permutation importance):")
    print(feature_importances.head(15))

# Make predictions on test data
y_pred = best_model.predict(X_test)

# Compare predictions with actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs. Predicted Yield', fontsize=16)
plt.xlabel('Actual Yield (tons per hectare)', fontsize=12)
plt.ylabel('Predicted Yield (tons per hectare)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs. Predicted Values', fontsize=16)
plt.xlabel('Predicted Yield (tons per hectare)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Residuals distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals', fontsize=16)
plt.xlabel('Residual Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 6. Crop-Specific Modeling and Insights

```python
# Analyze which factors matter most for different crop types
crop_types = soil_data['crop_type'].unique()

for crop in crop_types:
    # Filter data for this crop
    crop_data = soil_data[soil_data['crop_type'] == crop].copy()

    if len(crop_data) < 10:  # Skip if too few samples
        print(f"Skipping {crop} - not enough data ({len(crop_data)} samples)")
        continue

    print(f"\n--- Analysis for {crop} ---")

    # Correlation with yield for this crop
    crop_corr = crop_data.select_dtypes(include=['float64', 'int64']).corr()['yield_tons_ha']
    print(f"Top factors correlated with {crop} yield:")
    print(crop_corr.drop('yield_tons_ha').sort_values(ascending=False).head(5))

    # Create visualizations for top factors
    top_factors = crop_corr.drop('yield_tons_ha').sort_values(ascending=False).head(2).index

    plt.figure(figsize=(14, 6))
    for i, factor in enumerate(top_factors):
        plt.subplot(1, 2, i+1)
        sns.scatterplot(x=factor, y='yield_tons_ha', data=crop_data)
        plt.title(f'{crop}: {factor} vs. Yield', fontsize=14)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Analyze optimal conditions for each crop
print("\nOptimal conditions for each crop type:")
for crop in crop_types:
    crop_data = soil_data[soil_data['crop_type'] == crop].copy()

    if len(crop_data) < 10:  # Skip if too few samples
        continue

    # Get top 10% yielding plots for this crop
    top_yield_threshold = crop_data['yield_tons_ha'].quantile(0.9)
    top_plots = crop_data[crop_data['yield_tons_ha'] >= top_yield_threshold]

    print(f"\n{crop} - Optimal conditions (based on top 10% yielding plots):")

    # Show average values for key factors
    key_factors = ['ph_level', 'nitrogen_kg_ha', 'phosphorus_kg_ha',
                  'potassium_kg_ha', 'organic_matter_pct', 'moisture_pct',
                  'temperature_c', 'rainfall_mm']

    for factor in key_factors:
        avg_value = top_plots[factor].mean()
        std_value = top_plots[factor].std()
        print(f"  {factor}: {avg_value:.2f} ± {std_value:.2f}")

    # Most common soil type
    print(f"  Most common soil type: {top_plots['soil_type'].mode()[0]}")

    # Most common irrigation type
    print(f"  Most common irrigation type: {top_plots['irrigation_type'].mode()[0]}")
```

### 7. Creating a Simple Yield Prediction Tool

```python
# Function to predict yield for given input conditions
def predict_yield(model, soil_type, ph, nitrogen, phosphorus, potassium,
                 organic_matter, moisture, temperature, rainfall,
                 irrigation_type, crop_type, growing_days, region):
    """
    Predicts crop yield based on input conditions.

    Parameters:
    -----------
    model : trained machine learning model
    soil_type : str, soil type
    ph : float, soil pH level
    nitrogen : float, nitrogen content (kg/ha)
    phosphorus : float, phosphorus content (kg/ha)
    potassium : float, potassium content (kg/ha)
    organic_matter : float, organic matter percentage
    moisture : float, soil moisture percentage
    temperature : float, soil temperature in Celsius
    rainfall : float, seasonal rainfall in mm
    irrigation_type : str, type of irrigation
    crop_type : str, type of crop
    growing_days : int, expected growing period in days
    region : str, geographical region

    Returns:
    --------
    float : Predicted yield in tons per hectare
    """
    # Create a DataFrame with one row for the input conditions
    input_data = pd.DataFrame({
        'ph_level': [ph],
        'nitrogen_kg_ha': [nitrogen],
        'phosphorus_kg_ha': [phosphorus],
        'potassium_kg_ha': [potassium],
        'organic_matter_pct': [organic_matter],
        'moisture_pct': [moisture],
        'temperature_c': [temperature],
        'rainfall_mm': [rainfall],
        'growing_period': [growing_days],
        'soil_type': [soil_type],
        'irrigation_type': [irrigation_type],
        'crop_type': [crop_type],
        'region': [region]
    })

    # Calculate derived features
    input_data['n_p_ratio'] = input_data['nitrogen_kg_ha'] / input_data['phosphorus_kg_ha']
    input_data['n_k_ratio'] = input_data['nitrogen_kg_ha'] / input_data['potassium_kg_ha']
    input_data['p_k_ratio'] = input_data['phosphorus_kg_ha'] / input_data['potassium_kg_ha']
    input_data['planting_month'] = 3  # Default to March if not provided

    # Convert categorical variables to match model training format
    input_data_encoded = pd.get_dummies(input_data, columns=['soil_type', 'irrigation_type',
                                                         'crop_type', 'region'])

    # Ensure all columns from training are present
    for col in X.columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0

    # Keep only columns used during model training
    input_data_final = input_data_encoded[X.columns]

    # Make prediction
    yield_prediction = model.predict(input_data_final)[0]

    return yield_prediction

# Example usage of the prediction function
example_yield = predict_yield(
    best_model,
    soil_type="loam",
    ph=6.5,
    nitrogen=120,
    phosphorus=60,
    potassium=80,
    organic_matter=3.5,
    moisture=30,
    temperature=22,
    rainfall=800,
    irrigation_type="drip",
    crop_type="corn",
    growing_days=120,
    region="midwest"
)

print(f"Predicted yield for example scenario: {example_yield:.2f} tons/ha")

# Create a function to show yield sensitivity to changes in key parameters
def plot_yield_sensitivity(model, base_params, param_to_vary, range_values):
    """
    Creates a plot showing how yield predictions change when varying a single parameter.

    Parameters:
    -----------
    model : trained machine learning model
    base_params : dict, base parameter values
    param_to_vary : str, parameter to vary
    range_values : list, values to try for the parameter
    """
    yields = []

    for value in range_values:
        # Create a copy of base parameters and update the one to vary
        params = base_params.copy()
        params[param_to_vary] = value

        # Make prediction
        yield_value = predict_yield(model, **params)
        yields.append(yield_value)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range_values, yields, 'o-', linewidth=2, markersize=8)
    plt.title(f'Yield Sensitivity to {param_to_vary}', fontsize=16)
    plt.xlabel(param_to_vary, fontsize=12)
    plt.ylabel('Predicted Yield (tons/ha)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example base parameters
base_params = {
    "soil_type": "loam",
    "ph": 6.5,
    "nitrogen": 120,
    "phosphorus": 60,
    "potassium": 80,
    "organic_matter": 3.5,
    "moisture": 30,
    "temperature": 22,
    "rainfall": 800,
    "irrigation_type": "drip",
    "crop_type": "corn",
    "growing_days": 120,
    "region": "midwest"
}

# Example sensitivity analysis for nitrogen levels
nitrogen_range = [40, 60, 80, 100, 120, 140, 160, 180, 200]
plot_yield_sensitivity(best_model, base_params, "nitrogen", nitrogen_range)

# Example sensitivity analysis for pH levels
ph_range = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
plot_yield_sensitivity(best_model, base_params, "ph", ph_range)
```

## Mini Practice Task

After completing the guided analysis, explore these additional questions:

1. Develop a model to predict the optimal planting date for a specific crop based on the soil and weather conditions.

2. Investigate whether there's an interaction effect between soil pH and nutrient levels on crop yield. For example, does the impact of nitrogen levels change at different pH levels?

3. Create a predictive model that recommends the ideal fertilizer type and application rate based on soil measurements and desired crop.

## Conclusion

By the end of this project, you'll have gained practical experience in applying data science techniques to agricultural challenges. You'll understand how different soil properties, environmental conditions, and farming practices affect crop yields. This knowledge can help inform data-driven decisions in agriculture, ultimately leading to more efficient resource use and higher productivity.

## Next Steps

Consider extending your analysis by:

- Incorporating satellite imagery or drone data to capture more spatial information
- Adding weather forecast data to create predictive models for future growing seasons
- Developing an interactive dashboard for farmers to input their field conditions and receive recommendations
- Creating an optimization model that suggests the best crop rotation strategy over multiple seasons
