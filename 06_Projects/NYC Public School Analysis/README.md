# NYC Public School Test Result Analysis

## Project Overview

This project explores test score performance across New York City's diverse public schools. You'll analyze standardized test results to identify patterns, disparities, and potential relationships between school characteristics and academic performance.

## Learning Objectives

- Apply data aggregation and grouping techniques
- Calculate and interpret summary statistics
- Create informative visualizations to highlight disparities
- Formulate meaningful conclusions from statistical analysis

## Key Topics Covered

- Data cleaning and preparation
- Statistical analysis and hypothesis testing
- Grouped aggregations and comparisons
- Data visualization for educational metrics

## Dataset Description

The `schools.csv` file contains standardized test results and demographic information for NYC public schools with the following columns:

- `school_name`: Name of the school
- `borough`: NYC borough (Manhattan, Bronx, Brooklyn, Queens, Staten Island)
- `building_code`: Unique building identifier
- `average_math`: Average math score (scale of 200-800)
- `average_reading`: Average reading score (scale of 200-800)
- `average_writing`: Average writing score (scale of 200-800)
- `percent_tested`: Percentage of students who took the test
- `student_count`: Total number of students enrolled
- `asian_percent`: Percentage of Asian students
- `black_percent`: Percentage of Black/African American students
- `hispanic_percent`: Percentage of Hispanic/Latino students
- `white_percent`: Percentage of White students
- `free_lunch`: Percentage of students eligible for free lunch (economic need indicator)

## Project Tasks

### 1. Data Loading and Initial Exploration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Load the dataset
schools = pd.read_csv('schools.csv')

# Initial data exploration
print(f"Dataset shape: {schools.shape}")
print(schools.info())
print(schools.describe())

# Check for missing values
print("\nMissing values per column:")
print(schools.isna().sum())
```

### 2. Data Cleaning and Preparation

```python
# Handle missing values
schools_clean = schools.copy()

# For numeric columns, replace NaNs with median values
numeric_cols = ['average_math', 'average_reading', 'average_writing', 'percent_tested']
for col in numeric_cols:
    schools_clean[col] = schools_clean[col].fillna(schools_clean[col].median())

# Create a new 'total_score' column (sum of all three test scores)
schools_clean['total_score'] = schools_clean['average_math'] + \
                                schools_clean['average_reading'] + \
                                schools_clean['average_writing']

# Categorize schools by size
def categorize_size(count):
    if count < 1000:
        return 'Small'
    elif count < 2000:
        return 'Medium'
    else:
        return 'Large'

schools_clean['size_category'] = schools_clean['student_count'].apply(categorize_size)

# Show the first few rows of the cleaned dataset
print(schools_clean.head())
```

### 3. Summary Statistics by Borough

```python
# Calculate summary statistics by borough
borough_stats = schools_clean.groupby('borough').agg({
    'average_math': 'mean',
    'average_reading': 'mean',
    'average_writing': 'mean',
    'total_score': 'mean',
    'school_name': 'count'
}).rename(columns={'school_name': 'num_schools'}).round(2)

print("Summary statistics by borough:")
print(borough_stats)

# Visualize average scores by borough
plt.figure(figsize=(12, 6))
borough_stats[['average_math', 'average_reading', 'average_writing']].plot(
    kind='bar',
    color=['#1f77b4', '#ff7f0e', '#2ca02c']
)
plt.title('Average Test Scores by Borough', fontsize=14)
plt.xlabel('Borough', fontsize=12)
plt.ylabel('Average Score', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.legend(title='Subject')
plt.tight_layout()
plt.show()
```

### 4. Analyzing Economic Factors and Performance

```python
# Calculate correlation between free lunch percentage and test scores
correlations = schools_clean[['free_lunch', 'average_math', 'average_reading',
                               'average_writing', 'total_score']].corr()
print("Correlation between economic need and test scores:")
print(correlations.loc['free_lunch'])

# Create scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='free_lunch', y='total_score', data=schools_clean, scatter_kws={'alpha':0.5})
plt.title('Relationship Between Economic Need and Test Performance', fontsize=14)
plt.xlabel('Percentage of Students Eligible for Free Lunch', fontsize=12)
plt.ylabel('Total Score (Math + Reading + Writing)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Compare high vs. low economic need schools
high_need = schools_clean[schools_clean['free_lunch'] > 75]
low_need = schools_clean[schools_clean['free_lunch'] < 25]

print(f"Number of high economic need schools: {len(high_need)}")
print(f"Number of low economic need schools: {len(low_need)}")

# Perform t-test to compare means
t_stat, p_value = stats.ttest_ind(high_need['total_score'], low_need['total_score'],
                                   equal_var=False)

print(f"T-test results: t-statistic = {t_stat:.2f}, p-value = {p_value:.6f}")
```

### 5. Demographic Analysis

```python
# Analyze demographic makeup and performance
demographic_cols = ['asian_percent', 'black_percent', 'hispanic_percent', 'white_percent']

# Calculate correlation between demographics and performance
demographic_corr = schools_clean[demographic_cols + ['total_score']].corr()
print("Correlation between demographics and total score:")
print(demographic_corr['total_score'].sort_values(ascending=False))

# Create a visualization of demographic distribution across boroughs
plt.figure(figsize=(14, 8))

# Calculate average demographic percentages by borough
demo_by_borough = schools_clean.groupby('borough')[demographic_cols].mean()

# Create stacked bar chart
demo_by_borough.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Demographic Composition by Borough', fontsize=14)
plt.xlabel('Borough', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.xticks(rotation=45)
plt.grid(False)
plt.legend(title='Demographic')
plt.tight_layout()
plt.show()
```

### 6. School Size Analysis

```python
# Analyze performance by school size
size_performance = schools_clean.groupby('size_category').agg({
    'average_math': 'mean',
    'average_reading': 'mean',
    'average_writing': 'mean',
    'total_score': 'mean',
    'school_name': 'count'
}).rename(columns={'school_name': 'num_schools'})

print("Performance by school size:")
print(size_performance)

# Visualize performance by school size
plt.figure(figsize=(10, 6))
sns.boxplot(x='size_category', y='total_score', data=schools_clean,
            order=['Small', 'Medium', 'Large'])
plt.title('Test Performance by School Size', fontsize=14)
plt.xlabel('School Size', fontsize=12)
plt.ylabel('Total Score', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

## Mini Practice Task

After completing the guided analysis, tackle these questions:

1. Identify the top 5 and bottom 5 schools based on total test scores. What characteristics do schools in each group share?
2. Create a more complex categorical variable that combines school size and borough. Analyze how this interaction affects test performance.
3. Investigate if there's a significant difference in test scores between schools with high vs. low percentages of students taking the tests.

## Conclusion

By the end of this project, you'll have gained insights into educational disparities in NYC public schools and factors that may influence academic performance. You'll have developed skills in statistical analysis, data aggregation, and visualization that can be applied to many other domains.

## Next Steps

Consider extending your analysis by:

- Incorporating additional datasets like teacher-student ratios or school funding
- Building predictive models to identify at-risk schools
- Creating an interactive dashboard to visualize the findings
- Analyzing trends over multiple years if historical data becomes available
