# Los Angeles Crime Analysis

## Project Overview

This project explores crime data from Los Angeles to uncover spatial and temporal patterns in criminal activity. You'll analyze when and where different types of crimes occur, identifying hotspots and seasonal trends that could help inform public safety initiatives.

## Learning Objectives

- Apply data analysis techniques to spatio-temporal data
- Create effective visualizations of crime patterns
- Identify trends and seasonality in time series data
- Combine spatial and temporal analysis to generate insights

## Key Topics Covered

- Time series analysis and visualization
- Geospatial data processing and mapping
- Categorical data analysis
- Advanced aggregation and grouping operations
- Interactive visualization techniques

## Setup Instructions

1. **Environment Setup:**

   - Make sure Python 3.8+ is installed on your system
   - Required libraries: pandas, numpy, matplotlib, seaborn, folium, plotly
   - Install dependencies: `pip install pandas numpy matplotlib seaborn folium plotly`

2. **Dataset:**

   - The `crimes.csv` file should be in the same directory as your notebook
   - Ensure you have internet access for the interactive map components

3. **Getting Started:**
   - Create a new Jupyter notebook in this directory
   - Follow the project tasks outlined below
   - Execute each code cell and analyze the results

## Dataset Description

The `crimes.csv` file contains reported crime incidents in Los Angeles with the following columns:

- `DR_NO`: Division of Records Number (unique identifier)
- `Date Rptd`: Date the crime was reported
- `DATE OCC`: Date the crime occurred
- `TIME OCC`: Time the crime occurred (24-hour format)
- `AREA NAME`: Name of the LAPD reporting district
- `AREA`: LAPD reporting district number
- `Rpt Dist No`: Reporting District Number
- `Part 1-2`: Crime category
- `Crm Cd`: Crime code
- `Crm Cd Desc`: Crime code description
- `Vict Age`: Victim age
- `Vict Sex`: Victim sex
- `Vict Descent`: Victim descent/ethnicity
- `Premis Cd`: Premise code
- `Premis Desc`: Premise description
- `Weapon Used Cd`: Weapon used code
- `Weapon Desc`: Weapon description
- `Status`: Status of the case
- `Status Desc`: Status description
- `Crm Cd 1`: Crime code 1
- `Crm Cd 2`: Crime code 2
- `Crm Cd 3`: Crime code 3
- `Crm Cd 4`: Crime code 4
- `LOCATION`: Location of crime
- `Cross Street`: Cross street
- `LAT`: Latitude
- `LON`: Longitude

## Project Tasks

### 1. Data Loading and Initial Exploration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = [12, 8]

# Load the dataset with appropriate date parsing
crimes = pd.read_csv('crimes.csv', parse_dates=['Date Rptd', 'DATE OCC'])

# Initial exploration
print(f"Dataset shape: {crimes.shape}")
display(crimes.info())  # Using display() for better notebook output

# Summary statistics with improved formatting
display(crimes.describe(include='all').round(2))

# Check for missing values with visual representation
missing_values = crimes.isna().sum().sort_values(ascending=False)
missing_pct = (missing_values / len(crimes) * 100).round(2)
missing_df = pd.DataFrame({'Count': missing_values, 'Percent': missing_pct})
missing_df = missing_df[missing_df['Count'] > 0]

print("\nMissing values per column:")
display(missing_df)

if not missing_df.empty:
    plt.figure(figsize=(12, 6))
    plt.bar(missing_df.index, missing_df['Percent'], color='crimson')
    plt.title('Percentage of Missing Values by Column', fontsize=16)
    plt.xlabel('Columns', fontsize=14)
    plt.ylabel('Missing Values (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Display a few rows to understand the data
print("\nSample data:")
display(crimes.head())
```

### 2. Data Cleaning and Preparation

```python
# Create a clean copy of the dataset
la_crimes = crimes.copy()

# Convert date columns to datetime - already done in read_csv with parse_dates
# Extract time components using more efficient pandas methods
la_crimes = la_crimes.assign(
    Year=la_crimes['DATE OCC'].dt.year,
    Month=la_crimes['DATE OCC'].dt.month,
    Day=la_crimes['DATE OCC'].dt.day,
    Hour=la_crimes['TIME OCC'] // 100,
    Minute=la_crimes['TIME OCC'] % 100,
    Day_of_Week=la_crimes['DATE OCC'].dt.day_name()
)

# Create report delay column (days between occurrence and report)
la_crimes['Report_Delay'] = (la_crimes['Date Rptd'] - la_crimes['DATE OCC']).dt.days

# Filter out records with missing location data
la_crimes = la_crimes.dropna(subset=['LAT', 'LON'])

# Check for unreasonable values in latitude and longitude - using query for better readability
la_crimes = la_crimes.query('33 < LAT < 35 and -119 > LON > -117')

# Create simplified crime category based on the crime description using more efficient vectorized approach
crime_categories = {
    'theft': 'Theft',
    'stolen': 'Theft',
    'shoplifting': 'Theft',
    'robbery': 'Robbery/Burglary',
    'burglary': 'Robbery/Burglary',
    'assault': 'Assault/Battery',
    'battery': 'Assault/Battery',
    'fight': 'Assault/Battery',
    'veh': 'Vehicle-Related',
    'vehicle': 'Vehicle-Related',
    'narcotic': 'Drug-Related',
    'drug': 'Drug-Related',
    'homicide': 'Homicide',
    'murder': 'Homicide',
    'weapon': 'Weapon',
    'firearm': 'Weapon',
    'gun': 'Weapon'
}

def assign_category(description):
    desc_lower = description.lower()
    for key, category in crime_categories.items():
        if key in desc_lower:
            return category
    return 'Other'

la_crimes['Crime_Category'] = la_crimes['Crm Cd Desc'].apply(assign_category)

# Print shape of processed data
print(f"Processed dataset shape: {la_crimes.shape}")
```

### 3. Temporal Crime Analysis

```python
# Analyze crime counts by year with improved visualization
yearly_counts = la_crimes.groupby('Year').size()

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(yearly_counts.index, yearly_counts.values, color=sns.color_palette("viridis", len(yearly_counts)))

# Add value annotations on top of each bar
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10)

ax.set_title('Number of Crimes by Year', fontsize=16)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Number of Crimes', fontsize=14)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Analyze crime patterns by month with improved seasonal visualization
monthly_counts = la_crimes.groupby(['Year', 'Month']).size().unstack(level=0)

# Add month names for better readability
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_counts.index = month_names[:len(monthly_counts)]

fig, ax = plt.subplots(figsize=(14, 7))
monthly_counts.plot(marker='o', ax=ax, linewidth=2)

# Add annotations for peaks and valleys
# Find max and min for each year
for year in monthly_counts.columns:
    max_month = monthly_counts[year].idxmax()
    min_month = monthly_counts[year].idxmin()
    max_val = monthly_counts[year].max()
    min_val = monthly_counts[year].min()

    ax.annotate(f'Max: {max_val:,}',
                xy=(monthly_counts.index.get_loc(max_month), max_val),
                xytext=(0, 10),
                textcoords="offset points",
                ha='center',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

    ax.annotate(f'Min: {min_val:,}',
                xy=(monthly_counts.index.get_loc(min_month), min_val),
                xytext=(0, -20),
                textcoords="offset points",
                ha='center',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

ax.set_title('Monthly Crime Counts by Year', fontsize=16)
ax.set_xlabel('Month', fontsize=14)
ax.set_ylabel('Number of Crimes', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(title='Year')
plt.tight_layout()
plt.show()

# Analyze crime patterns by day of week with improved visualization
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = la_crimes['Day_of_Week'].value_counts().reindex(day_order)

# Create a more informative visualization with percentages
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(day_counts.index, day_counts.values,
        color=sns.color_palette("viridis", len(day_counts)))

# Add percentage labels
total = day_counts.sum()
for bar in bars:
    height = bar.get_height()
    percentage = (height / total) * 100
    ax.annotate(f'{height:,}\n({percentage:.1f}%)',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

ax.set_title('Number of Crimes by Day of Week', fontsize=16)
ax.set_xlabel('Day of Week', fontsize=14)
ax.set_ylabel('Number of Crimes', fontsize=14)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Analyze crime patterns by hour with improved visualization
hourly_counts = la_crimes['Hour'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(hourly_counts.index, hourly_counts.values, marker='o',
        linewidth=2, color='mediumseagreen')

# Add morning, afternoon, evening, night annotations
periods = [
    {'name': 'Night', 'start': 0, 'end': 5, 'y_pos': hourly_counts[2:6].mean()},
    {'name': 'Morning', 'start': 6, 'end': 11, 'y_pos': hourly_counts[8:12].mean()},
    {'name': 'Afternoon', 'start': 12, 'end': 17, 'y_pos': hourly_counts[14:18].mean()},
    {'name': 'Evening', 'start': 18, 'end': 23, 'y_pos': hourly_counts[20:24].mean()}
]

for period in periods:
    middle = (period['start'] + period['end']) / 2
    ax.annotate(period['name'],
                xy=(middle, period['y_pos']),
                xytext=(0, 30),
                textcoords="offset points",
                ha='center',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc='lightyellow', ec="orange", alpha=0.8))

# Add peak time annotation
peak_hour = hourly_counts.idxmax()
peak_value = hourly_counts.max()
ax.annotate(f'Peak: {peak_hour}:00 ({peak_value:,} crimes)',
            xy=(peak_hour, peak_value),
            xytext=(20, 0),
            textcoords="offset points",
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
            bbox=dict(boxstyle="round,pad=0.3", fc='red', alpha=0.2))

ax.set_title('Number of Crimes by Hour of Day', fontsize=16)
ax.set_xlabel('Hour (24-hour format)', fontsize=14)
ax.set_ylabel('Number of Crimes', fontsize=14)
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Create a heatmap showing crime patterns by hour and day of week with improved readability
hour_day_counts = pd.crosstab(la_crimes['Hour'], la_crimes['Day_of_Week'])
hour_day_counts = hour_day_counts.reindex(columns=day_order)

plt.figure(figsize=(14, 10))
sns.heatmap(hour_day_counts, cmap='viridis',
            annot=True, fmt=',d', cbar_kws={'label': 'Number of Crimes'},
            linewidths=0.5)
plt.title('Crime Patterns by Hour and Day of Week', fontsize=16)
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Hour of Day', fontsize=14)
plt.tight_layout()
plt.show()

# Create an interactive visualization using Plotly
hour_day_fig = px.imshow(hour_day_counts,
                         labels=dict(x="Day of Week", y="Hour of Day", color="Number of Crimes"),
                         title="Crime Patterns by Hour and Day of Week (Interactive)",
                         color_continuous_scale='viridis')
hour_day_fig.update_layout(width=800, height=600)
hour_day_fig.show()
```

### 4. Crime Type Analysis

```python
# Analyze most common crime types with improved visualization
crime_type_counts = la_crimes['Crime_Category'].value_counts()

# Plot using Plotly for interactivity
crime_type_fig = px.bar(x=crime_type_counts.values,
                         y=crime_type_counts.index,
                         orientation='h',
                         labels={'x': 'Number of Crimes', 'y': 'Crime Category'},
                         title='Number of Crimes by Category',
                         color=crime_type_counts.values,
                         color_continuous_scale='viridis')

crime_type_fig.update_layout(
    width=800,
    height=500,
    xaxis_title='Number of Crimes',
    yaxis_title='Crime Category',
    coloraxis_showscale=False
)

crime_type_fig.show()

# Analyze crime types by time of day with improved visualization
crime_hour = pd.crosstab(la_crimes['Hour'], la_crimes['Crime_Category'])

# Plot top 5 crime types
top_5_crimes = crime_type_counts.nlargest(5).index

fig, ax = plt.subplots(figsize=(16, 10))

for crime_type in top_5_crimes:
    ax.plot(crime_hour.index, crime_hour[crime_type],
            marker='o', linewidth=2, label=crime_type)

# Add shaded areas for different times of day
ax.axvspan(0, 6, alpha=0.2, color='navy', label='Night (12AM-6AM)')
ax.axvspan(6, 12, alpha=0.2, color='gold', label='Morning (6AM-12PM)')
ax.axvspan(12, 18, alpha=0.2, color='lightblue', label='Afternoon (12PM-6PM)')
ax.axvspan(18, 24, alpha=0.2, color='purple', label='Evening (6PM-12AM)')

ax.set_title('Crime Frequency by Hour of Day', fontsize=16)
ax.set_xlabel('Hour of Day', fontsize=14)
ax.set_ylabel('Number of Crimes', fontsize=14)
ax.legend(title='Crime Category')
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Create interactive version with Plotly
crime_hour_data = []

for crime_type in top_5_crimes:
    crime_hour_data.append(
        go.Scatter(
            x=crime_hour.index,
            y=crime_hour[crime_type],
            mode='lines+markers',
            name=crime_type
        )
    )

crime_hour_layout = go.Layout(
    title='Crime Frequency by Hour of Day (Interactive)',
    xaxis=dict(title='Hour of Day'),
    yaxis=dict(title='Number of Crimes'),
    width=900,
    height=600,
)

crime_hour_fig = go.Figure(data=crime_hour_data, layout=crime_hour_layout)
crime_hour_fig.show()

# Analyze crime types by day of week with improved stacked bar chart
crime_day = pd.crosstab(la_crimes['Day_of_Week'], la_crimes['Crime_Category'])
crime_day = crime_day.reindex(day_order)

# Calculate proportions for better comparison
crime_day_pct = crime_day.div(crime_day.sum(axis=1), axis=0) * 100

# Create the visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Plot counts (left)
crime_day[top_5_crimes].plot(kind='bar', stacked=True, ax=ax1)
ax1.set_title('Crime Types by Day of Week (Counts)', fontsize=16)
ax1.set_xlabel('Day of Week', fontsize=14)
ax1.set_ylabel('Number of Crimes', fontsize=14)
ax1.legend(title='Crime Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# Plot percentages (right)
crime_day_pct[top_5_crimes].plot(kind='bar', stacked=True, ax=ax2)
ax2.set_title('Crime Types by Day of Week (%)', fontsize=16)
ax2.set_xlabel('Day of Week', fontsize=14)
ax2.set_ylabel('Percentage of Crimes', fontsize=14)
ax2.legend(title='Crime Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Analyze seasonal patterns in crime types with improved visualization
crime_month = pd.crosstab(la_crimes['Month'], la_crimes['Crime_Category'])
crime_month.index = month_names[:len(crime_month)]  # Use month names for clarity

# Create a visualization with both absolute numbers and normalized trends
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))

# Plot absolute numbers
for crime_type in top_5_crimes:
    ax1.plot(crime_month.index, crime_month[crime_type],
             marker='o', linewidth=2, label=crime_type)

ax1.set_title('Seasonal Patterns in Crime Types (Absolute)', fontsize=16)
ax1.set_xlabel('Month', fontsize=14)
ax1.set_ylabel('Number of Crimes', fontsize=14)
ax1.legend(title='Crime Category')
ax1.grid(True, alpha=0.3)

# Plot normalized trends (percentage of yearly total)
crime_month_pct = crime_month.copy()
for col in crime_month.columns:
    crime_month_pct[col] = crime_month_pct[col] / crime_month_pct[col].sum() * 100

for crime_type in top_5_crimes:
    ax2.plot(crime_month_pct.index, crime_month_pct[crime_type],
             marker='o', linewidth=2, label=crime_type)

ax2.set_title('Seasonal Patterns in Crime Types (% of Yearly Total)', fontsize=16)
ax2.set_xlabel('Month', fontsize=14)
ax2.set_ylabel('Percentage of Yearly Crimes', fontsize=14)
ax2.legend(title='Crime Category')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 5. Geospatial Analysis

```python
# Create a base map centered on Los Angeles with Folium improvements
la_map = folium.Map(
    location=[34.05, -118.25],
    zoom_start=10,
    tiles='CartoDB positron'  # Cleaner map style
)

# Add a heatmap layer showing crime density with optimized parameters
heat_data = la_crimes[['LAT', 'LON']].sample(min(10000, len(la_crimes))).values.tolist()  # Sample for performance
HeatMap(
    heat_data,
    radius=15,
    blur=20,
    gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
).add_to(la_map)

# Add a title to the map
title_html = '''
    <h3 align="center" style="font-size:16px"><b>Los Angeles Crime Heatmap</b></h3>
'''
la_map.get_root().html.add_child(folium.Element(title_html))

# Save the map as an HTML file
la_map.save('la_crime_heatmap.html')

# Create separate heatmaps for different crime types with enhanced styling
for crime_type in top_5_crimes:
    # Filter data for the specific crime type
    crime_subset = la_crimes[la_crimes['Crime_Category'] == crime_type]

    # Sample for better performance if dataset is large
    if len(crime_subset) > 10000:
        crime_subset = crime_subset.sample(10000)

    # Create map with customized style based on crime type
    crime_map = folium.Map(
        location=[34.05, -118.25],
        zoom_start=10,
        tiles='CartoDB positron'
    )

    # Add title with crime type and count
    crime_title = f'''
        <h3 align="center" style="font-size:16px">
        <b>Los Angeles {crime_type} Crimes</b><br>
        <span style="font-size:12px">Total: {len(crime_subset):,} incidents</span>
        </h3>
    '''
    crime_map.get_root().html.add_child(folium.Element(crime_title))

    # Add heatmap layer with custom gradient based on crime type
    color_map = {
        'Theft': {0.4: 'blue', 0.65: 'yellow', 1: 'red'},
        'Robbery/Burglary': {0.4: 'purple', 0.65: 'orange', 1: 'red'},
        'Assault/Battery': {0.4: 'green', 0.65: 'orange', 1: 'red'},
        'Vehicle-Related': {0.4: 'blue', 0.65: 'green', 1: 'red'},
        'Drug-Related': {0.4: 'green', 0.65: 'lime', 1: 'red'},
        'Homicide': {0.4: 'black', 0.65: 'purple', 1: 'red'},
        'Weapon': {0.4: 'blue', 0.65: 'purple', 1: 'red'},
        'Other': {0.4: 'blue', 0.65: 'green', 1: 'red'}
    }

    # Get color map or use default if not found
    gradient = color_map.get(crime_type, {0.4: 'blue', 0.65: 'lime', 1: 'red'})

    heat_data = crime_subset[['LAT', 'LON']].values.tolist()
    HeatMap(heat_data, radius=15, blur=20, gradient=gradient).add_to(crime_map)

    # Save the map
    crime_map.save(f'la_crime_heatmap_{crime_type.lower().replace("/", "_")}.html')

# Analyze crime by area name with improved visualization
area_counts = la_crimes.groupby('AREA NAME').size().sort_values(ascending=False)

plt.figure(figsize=(14, 8))
ax = sns.barplot(
    x=area_counts.index[:15],
    y=area_counts.values[:15],
    palette=sns.color_palette("viridis", 15)
)

# Add data labels on top of bars
for i, v in enumerate(area_counts.values[:15]):
    ax.text(i, v + 100, f"{v:,}", ha='center', fontsize=10)

plt.title('Top 15 Areas by Number of Crimes', fontsize=16)
plt.xlabel('Area Name', fontsize=14)
plt.ylabel('Number of Crimes', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Create interactive version with Plotly
area_fig = px.bar(
    x=area_counts.index[:15],
    y=area_counts.values[:15],
    labels={'x': 'Area Name', 'y': 'Number of Crimes'},
    title='Top 15 Areas by Number of Crimes (Interactive)',
    color=area_counts.values[:15],
    color_continuous_scale='viridis'
)
area_fig.update_layout(
    width=800,
    height=500,
    xaxis_tickangle=-45,
    coloraxis_showscale=False
)
area_fig.show()

# Analyze crime types by area with improved visualization
area_crime_type = pd.crosstab(la_crimes['AREA NAME'], la_crimes['Crime_Category'])
area_crime_type_pct = area_crime_type.div(area_crime_type.sum(axis=1), axis=0) * 100

# Plot area crime composition for top 10 areas using Plotly for interactivity
top_10_areas = area_counts[:10].index
area_crime_plot = px.bar(
    area_crime_type_pct.loc[top_10_areas, top_5_crimes].reset_index().melt(id_vars='AREA NAME'),
    x='AREA NAME',
    y='value',
    color='Crime_Category',
    title='Crime Type Distribution in Top 10 Areas (%)',
    labels={'AREA NAME': 'Area Name', 'value': 'Percentage of Crimes'},
    height=600
)
area_crime_plot.show()
```

### 6. Interactive Visualization with Plotly

```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create time series visualization using Plotly
crime_over_time = la_crimes.groupby(['Year', 'Month']).size().reset_index()
crime_over_time.columns = ['Year', 'Month', 'Count']
crime_over_time['Date'] = pd.to_datetime(crime_over_time[['Year', 'Month']].assign(DAY=1))

# Create interactive time series with annotations for major trends
fig = px.line(crime_over_time, x='Date', y='Count',
              title='Crime in Los Angeles Over Time (Interactive)',
              labels={'Count': 'Number of Crimes', 'Date': 'Date'},
              line_shape='spline', # smoother line
              render_mode='svg')  # better for line charts

# Add range slider and buttons for time range selection
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

# Add annotations for major events or trends if any significant peaks/valleys exist
# Example (adjust based on your actual data):
local_max = crime_over_time.loc[crime_over_time['Count'].idxmax()]
local_min = crime_over_time.loc[crime_over_time['Count'].idxmin()]

fig.add_annotation(
    x=local_max['Date'],
    y=local_max['Count'],
    text=f"Peak: {local_max['Count']:,} crimes",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="#636363"
)

fig.add_annotation(
    x=local_min['Date'],
    y=local_min['Count'],
    text=f"Low: {local_min['Count']:,} crimes",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="#636363"
)

fig.update_layout(
    hovermode="x unified",
    width=900,
    height=500,
)

fig.show()

# Create interactive heatmap of crimes by hour and day of week
hour_day_fig = px.imshow(
    hour_day_counts,
    labels=dict(x="Day of Week", y="Hour of Day", color="Number of Crimes"),
    title="Crime Patterns by Hour and Day of Week (Interactive)",
    color_continuous_scale='viridis',
    text_auto=True,  # Show values on cells
    aspect="auto"
)

hour_day_fig.update_layout(
    xaxis={'side': 'top'},
    width=800,
    height=600,
    hovermode="closest"
)

hour_day_fig.show()

# Create interactive choropleth map of crimes by area
# First, aggregate crimes by area
area_crime_counts = la_crimes.groupby('AREA NAME').agg(
    Crime_Count=('DR_NO', 'count'),
    Latitude=('LAT', 'mean'),
    Longitude=('LON', 'mean')
).reset_index()

# Create a bubble map of crime by area
area_map = px.scatter_mapbox(
    area_crime_counts,
    lat="Latitude",
    lon="Longitude",
    size="Crime_Count",  # Size bubbles by crime count
    color="Crime_Count",  # Color bubbles by crime count
    hover_name="AREA NAME",
    hover_data=["Crime_Count"],
    zoom=9,
    height=600,
    title="Los Angeles Crimes by Area",
    color_continuous_scale=px.colors.sequential.Viridis
)

area_map.update_layout(
    mapbox_style="open-street-map",
    margin={"r":0,"t":50,"l":0,"b":0}
)

area_map.show()

# Create sample-based scatter plot of crime locations by type
# Sample data for better performance
sample_size = min(5000, len(la_crimes))
crime_sample = la_crimes.sample(sample_size)

# Create scatter plot color-coded by crime type
scatter_map = px.scatter_mapbox(
    crime_sample,
    lat="LAT",
    lon="LON",
    color="Crime_Category",
    hover_name="Crm Cd Desc",
    hover_data=["DATE OCC", "AREA NAME", "Vict Age"],
    zoom=10,
    height=700,
    title=f"Los Angeles Crime Map by Category (Sample of {sample_size:,} Crimes)",
    color_discrete_sequence=px.colors.qualitative.Bold
)

scatter_map.update_layout(
    mapbox_style="open-street-map",
    margin={"r":0,"t":50,"l":0,"b":0},
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0.8)"
    )
)

scatter_map.show()

# Create an interactive dashboard with multiple views
# Use plotly subplots for a dashboard-like layout
dashboard = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Crime by Day of Week",
        "Crime by Hour of Day",
        "Top 5 Crime Categories",
        "Monthly Crime Trends"
    ),
    specs=[
        [{"type": "bar"}, {"type": "scatter"}],
        [{"type": "bar"}, {"type": "scatter"}]
    ],
    vertical_spacing=0.1,
    horizontal_spacing=0.05
)

# 1. Crime by Day of Week (top-left)
day_data = go.Bar(
    x=day_counts.index,
    y=day_counts.values,
    marker=dict(color=day_counts.values, colorscale="Viridis")
)
dashboard.add_trace(day_data, row=1, col=1)

# 2. Crime by Hour (top-right)
hour_data = go.Scatter(
    x=hourly_counts.index,
    y=hourly_counts.values,
    mode='lines+markers',
    line=dict(shape='spline', color='mediumseagreen', width=3),
    marker=dict(size=8)
)
dashboard.add_trace(hour_data, row=1, col=2)

# 3. Top Crime Categories (bottom-left)
top_crimes = crime_type_counts.nlargest(5)
category_data = go.Bar(
    x=top_crimes.index,
    y=top_crimes.values,
    marker=dict(color=top_crimes.values, colorscale="Viridis")
)
dashboard.add_trace(category_data, row=2, col=1)

# 4. Monthly Crime Trends (bottom-right)
monthly_trend = go.Scatter(
    x=crime_over_time['Date'],
    y=crime_over_time['Count'],
    mode='lines',
    line=dict(shape='spline', width=3, color='royalblue')
)
dashboard.add_trace(monthly_trend, row=2, col=2)

# Update layout and formatting
dashboard.update_layout(
    title="Los Angeles Crime Dashboard",
    showlegend=False,
    height=800,
    width=1100,
    template="plotly_white"
)

# Update axes formatting
dashboard.update_xaxes(title_text="Day of Week", row=1, col=1)
dashboard.update_yaxes(title_text="Number of Crimes", row=1, col=1)

dashboard.update_xaxes(title_text="Hour of Day", row=1, col=2)
dashboard.update_yaxes(title_text="Number of Crimes", row=1, col=2)

dashboard.update_xaxes(title_text="Crime Category", row=2, col=1)
dashboard.update_yaxes(title_text="Number of Crimes", row=2, col=1)

dashboard.update_xaxes(title_text="Date", row=2, col=2)
dashboard.update_yaxes(title_text="Number of Crimes", row=2, col=2)

dashboard.show()
```

## Mini Practice Task

After completing the guided analysis, explore these additional questions:

1. Analyze the reporting delay (difference between when a crime occurred and when it was reported) across different crime categories. Do certain types of crimes have longer reporting delays?

2. Create a time series decomposition for a specific crime type, breaking down the trend, seasonality, and residual components.

3. Investigate the relationship between crime types and victim demographics (age, sex, descent). Are certain demographic groups more vulnerable to particular types of crimes?

## Conclusion

By the end of this project, you'll have gained insights into crime patterns in Los Angeles and developed skills in spatio-temporal data analysis. You'll be able to identify when and where different crime types occur most frequently, which could inform policy decisions and resource allocation for public safety.

## Next Steps

Consider extending your analysis by:

1. **Advanced Prediction**: Build predictive models to forecast crime hotspots using machine learning techniques such as Random Forest or XGBoost.

2. **Demographic Integration**: Incorporate additional datasets like population density, income levels, or police station locations to analyze socioeconomic factors.

3. **Real-Time Dashboard**: Create an interactive dashboard for real-time crime monitoring using Dash or Streamlit.

4. **Pattern Recognition**: Perform cluster analysis to identify similar crime patterns across different areas of Los Angeles.

5. **Temporal Anomaly Detection**: Develop algorithms to detect unusual spikes or changes in crime patterns that deviate from historical trends.

## References

- Los Angeles Open Data Portal: https://data.lacity.org/
- Folium Documentation: https://python-visualization.github.io/folium/
- Plotly Documentation: https://plotly.com/python/
- Pandas Time Series Analysis: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
