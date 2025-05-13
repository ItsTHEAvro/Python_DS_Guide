# Visualizing the History of Nobel Prize Winners

## Project Overview

This project explores a dataset containing information about Nobel Prize laureates from its inception in 1901 through recent years. You'll create compelling data visualizations to uncover patterns and trends in the awards, focusing on demographics, geographic distribution, and historical changes.

## Learning Objectives

- Apply advanced data visualization techniques
- Analyze temporal trends and patterns
- Create insightful visualizations that tell a cohesive story
- Practice data preparation for effective visualization

## Key Topics Covered

- Data wrangling and preparation
- Historical trend visualization
- Geographical data representation
- Demographic analysis through visualization
- Creating publication-quality graphics

## Dataset Description

The `nobel.csv` file contains data on Nobel Prize winners with the following columns:

- `year`: Year the prize was awarded
- `category`: Category of the prize (Chemistry, Economics, Literature, Medicine, Peace, or Physics)
- `prize`: Full name of the prize
- `motivation`: Description of the achievement
- `prize_share`: Share of the prize awarded (1, 1/2, 1/3, or 1/4)
- `laureate_id`: Unique identifier for the laureate
- `laureate_type`: Organization or Individual
- `full_name`: Full name of the laureate
- `birth_date`: Birth date of the laureate
- `birth_city`: City of birth
- `birth_country`: Country of birth
- `gender`: Gender of the laureate
- `organization_name`: Organization the laureate was affiliated with at the time of the award
- `organization_city`: City of the organization
- `organization_country`: Country of the organization
- `death_date`: Death date of the laureate, if applicable

## Project Tasks

### 1. Data Loading and Initial Exploration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# Set visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

# Load the dataset
nobel = pd.read_csv('nobel.csv')

# Initial exploration
print(f"Dataset shape: {nobel.shape}")
print(nobel.info())

# Summary statistics
print(nobel.describe(include='all'))

# Check for missing values
print("\nMissing values per column:")
print(nobel.isna().sum())
```

### 2. Data Cleaning and Preparation

```python
# Create a clean copy of the dataset
nobel_df = nobel.copy()

# Convert birth_date and death_date to datetime
nobel_df['birth_date'] = pd.to_datetime(nobel_df['birth_date'], errors='coerce')
nobel_df['death_date'] = pd.to_datetime(nobel_df['death_date'], errors='coerce')

# Calculate age at time of award
nobel_df['award_year'] = pd.to_datetime(nobel_df['year'], format='%Y')
nobel_df['age_at_award'] = nobel_df.apply(
    lambda x: np.nan if pd.isna(x['birth_date']) else (x['award_year'] - x['birth_date']).days / 365.25,
    axis=1
)

# Clean country names for better visualization
# Map historical countries to their modern equivalents
country_mapping = {
    'Russian Empire': 'Russia',
    'USSR': 'Russia',
    'East Germany': 'Germany',
    'West Germany': 'Germany',
    'Schleswig (now Germany)': 'Germany',
    'Prussia': 'Germany',
    'Austria-Hungary': 'Austria',
    'United Kingdom of Great Britain and Ireland': 'United Kingdom',
    'Czechoslovakia': 'Czech Republic',
    'Siam': 'Thailand',
    'Bengal, India (now Bangladesh)': 'Bangladesh'
}

nobel_df['birth_country'] = nobel_df['birth_country'].replace(country_mapping)
nobel_df['organization_country'] = nobel_df['organization_country'].replace(country_mapping)

# Create decade column for aggregation
nobel_df['decade'] = (nobel_df['year'] // 10) * 10

# Print shape of processed data
print(f"Processed dataset shape: {nobel_df.shape}")
```

### 3. Nobel Prize Categories Over Time

```python
# Analyze distribution by category
category_counts = nobel_df['category'].value_counts()
print("Number of prizes by category:")
print(category_counts)

# Visualize prizes awarded by category over time
plt.figure(figsize=(14, 8))
category_by_decade = nobel_df.groupby(['decade', 'category']).size().unstack()
category_by_decade.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Nobel Prizes Awarded by Category and Decade', fontsize=16)
plt.xlabel('Decade', fontsize=14)
plt.ylabel('Number of Prizes', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Category', title_fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Line plot showing trend for each category
plt.figure(figsize=(14, 8))
for category in nobel_df['category'].unique():
    category_data = nobel_df[nobel_df['category'] == category]
    yearly_counts = category_data.groupby('year').size().rolling(window=10).mean()
    plt.plot(yearly_counts.index, yearly_counts.values, linewidth=2, label=category)

plt.title('10-Year Rolling Average of Nobel Prizes by Category', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Number of Prizes (10-year average)', fontsize=14)
plt.legend(title='Category')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 4. Geographical Distribution of Nobel Laureates

```python
# Count prizes by birth country and organization country
birth_country_counts = nobel_df['birth_country'].value_counts().reset_index()
birth_country_counts.columns = ['country', 'laureates_born']

org_country_counts = nobel_df['organization_country'].value_counts().reset_index()
org_country_counts.columns = ['country', 'laureates_worked']

# Merge the two dataframes
country_comparison = pd.merge(birth_country_counts, org_country_counts,
                               on='country', how='outer').fillna(0)

# Calculate net brain gain/drain
country_comparison['net_flow'] = country_comparison['laureates_worked'] - country_comparison['laureates_born']
country_comparison = country_comparison.sort_values('net_flow', ascending=False)

# Display top 10 countries with most Nobel laureates
print("Top 10 countries by number of Nobel laureates born:")
print(birth_country_counts.head(10))

# Create a world map visualization using Plotly
fig = px.choropleth(birth_country_counts,
                    locations='country',
                    locationmode='country names',
                    color='laureates_born',
                    hover_name='country',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title='Number of Nobel Laureates by Birth Country')
fig.update_layout(geo=dict(showframe=False,
                          showcoastlines=True))
fig.show()

# Visualize brain gain/drain
plt.figure(figsize=(14, 8))
top_countries = country_comparison.head(15)
sns.barplot(x='net_flow', y='country', data=top_countries,
           palette='RdBu_r', orient='h')
plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)
plt.title('Top 15 Countries by Nobel Prize "Brain Drain/Gain"', fontsize=16)
plt.xlabel('Net Flow (Positive = Brain Gain, Negative = Brain Drain)', fontsize=14)
plt.ylabel('Country', fontsize=14)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
```

### 5. Gender Distribution Analysis

```python
# Analyze gender distribution
gender_counts = nobel_df['gender'].value_counts()
print("\nGender distribution:")
print(gender_counts)

# Gender distribution over time
gender_by_decade = nobel_df.groupby(['decade', 'gender']).size().unstack().fillna(0)

if 'Female' not in gender_by_decade.columns:
    gender_by_decade['Female'] = 0

# Calculate percentage of female laureates by decade
gender_by_decade['Female_pct'] = (gender_by_decade['Female'] /
                                  gender_by_decade.sum(axis=1) * 100)

# Plot gender trends
fig, ax1 = plt.subplots(figsize=(14, 8))

# Bar chart for counts
ax1.bar(gender_by_decade.index, gender_by_decade['Male'], label='Male', color='skyblue')
ax1.bar(gender_by_decade.index, gender_by_decade['Female'], bottom=gender_by_decade['Male'],
        label='Female', color='pink')
ax1.set_xlabel('Decade', fontsize=14)
ax1.set_ylabel('Number of Laureates', fontsize=14)
ax1.tick_params(axis='y')
ax1.legend(loc='upper left')

# Line chart for percentage
ax2 = ax1.twinx()
ax2.plot(gender_by_decade.index, gender_by_decade['Female_pct'],
        color='red', marker='o', linestyle='-', linewidth=2)
ax2.set_ylabel('Percentage of Female Laureates', fontsize=14, color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0, 30)

plt.title('Gender Distribution of Nobel Laureates by Decade', fontsize=16)
plt.xticks(rotation=45)
plt.grid(False)
plt.tight_layout()
plt.show()

# Gender distribution by category
plt.figure(figsize=(14, 8))
gender_by_category = nobel_df.groupby(['category', 'gender']).size().unstack().fillna(0)
gender_by_category['Total'] = gender_by_category.sum(axis=1)
gender_by_category['Female_pct'] = (gender_by_category['Female'] / gender_by_category['Total'] * 100)
gender_by_category = gender_by_category.sort_values('Female_pct', ascending=False)

# Create a horizontal bar chart
sns.set_color_codes("pastel")
sns.barplot(x="Male", y=gender_by_category.index, data=gender_by_category,
            label="Male", color="b")
sns.set_color_codes("muted")
sns.barplot(x="Female", y=gender_by_category.index, data=gender_by_category,
            label="Female", color="r")

# Add a legend and labels
plt.legend(ncol=2, loc="lower right", frameon=True)
plt.title('Gender Distribution by Nobel Prize Category', fontsize=16)
plt.xlabel('Number of Laureates', fontsize=14)
plt.ylabel('Category', fontsize=14)

# Add text with percentage of female laureates
for i, category in enumerate(gender_by_category.index):
    plt.text(gender_by_category.loc[category, 'Total'] + 5, i,
             f"{gender_by_category.loc[category, 'Female_pct']:.1f}% Female",
             va='center')

plt.tight_layout()
plt.show()
```

### 6. Age Analysis of Nobel Laureates

```python
# Analyze age distribution
plt.figure(figsize=(14, 8))
sns.histplot(nobel_df['age_at_award'].dropna(), bins=20, kde=True)
plt.title('Age Distribution of Nobel Laureates at Time of Award', fontsize=16)
plt.xlabel('Age (years)', fontsize=14)
plt.ylabel('Number of Laureates', fontsize=14)
plt.axvline(nobel_df['age_at_award'].dropna().mean(), color='red', linestyle='--',
           label=f"Mean Age: {nobel_df['age_at_award'].dropna().mean():.1f} years")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Age trends by category
plt.figure(figsize=(14, 8))
sns.boxplot(x='category', y='age_at_award', data=nobel_df)
plt.title('Age Distribution by Category', fontsize=16)
plt.xlabel('Category', fontsize=14)
plt.ylabel('Age at Award (years)', fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Age trends over time
plt.figure(figsize=(14, 8))
age_by_decade = nobel_df.groupby(['decade', 'category'])['age_at_award'].mean().unstack()
age_by_decade.plot(marker='o')
plt.title('Average Age of Nobel Laureates by Decade and Category', fontsize=16)
plt.xlabel('Decade', fontsize=14)
plt.ylabel('Average Age at Award', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(title='Category')
plt.tight_layout()
plt.show()
```

## Mini Practice Task

After completing the guided analysis, explore these additional questions:

1. Identify the top universities or research institutions associated with Nobel laureates. Create a visualization showing the top 10 institutions and their distribution across prize categories.
2. Analyze the time between birth and death for deceased laureates. Has the average lifespan of Nobel laureates changed over time? Does it differ by category?
3. Create a network visualization showing connections between countries (where laureates were born vs. where they worked when receiving the prize).

## Conclusion

By the end of this project, you'll have gained insights into historical patterns of Nobel Prize awards and developed advanced data visualization skills. You'll be able to tell compelling stories with data using a variety of visualization techniques.

## Next Steps

Consider extending your analysis by:

- Incorporating text analysis of the prize motivations to identify key research themes over time
- Creating an interactive dashboard using tools like Plotly or Tableau
- Comparing Nobel laureates to other prestigious award recipients (Fields Medal, Turing Award, etc.)
- Analyzing collaboration patterns among Nobel Prize winners who shared awards
