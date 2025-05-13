# Investigating Netflix Movies

## Project Overview

This project examines the trends in movie duration over time using a dataset of Netflix movies. You'll analyze whether movies have indeed been getting shorter over the years, and explore potential factors that might influence movie duration.

## Learning Objectives

- Apply pandas for data manipulation and cleaning
- Create meaningful visualizations to identify trends over time
- Practice exploratory data analysis techniques
- Formulate and test hypotheses about factors affecting movie duration

## Key Topics Covered

- Data loading and preprocessing
- Temporal analysis and trend identification
- Categorical data analysis
- Data visualization for insights

## Dataset Description

The `netflix_data.csv` file contains information about movies on Netflix with the following columns:

- `show_id`: The unique identifier for each movie
- `type`: Type of content (Movie or TV Show)
- `title`: Title of the movie
- `director`: Director of the movie
- `cast`: Main cast members
- `country`: Country of production
- `date_added`: Date when added to Netflix
- `release_year`: Year of release
- `rating`: Content rating (PG, R, etc.)
- `duration`: Duration in minutes
- `listed_in`: Genre categories
- `description`: Brief description of the content

## Project Tasks

### 1. Data Loading and Exploration

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
netflix_df = pd.read_csv('netflix_data.csv')

# Explore the dataset
print(netflix_df.info())
print(netflix_df.head())

# Filter for movies only
netflix_movies = netflix_df[netflix_df['type'] == 'Movie']

# Check for missing values in key columns
print(netflix_movies[['title', 'release_year', 'duration']].isna().sum())
```

### 2. Data Cleaning and Preparation

```python
# Extract numeric duration from the 'duration' column
netflix_movies['duration_minutes'] = netflix_movies['duration'].str.extract('(\d+)').astype(int)

# Create a subset with necessary columns
movies_subset = netflix_movies[['title', 'country', 'release_year', 'duration_minutes', 'listed_in']]

# Create a color column based on genre
def assign_genre_color(genre_list):
    if 'Children' in genre_list:
        return 'yellow'
    elif 'Documentaries' in genre_list:
        return 'green'
    elif any(x in genre_list for x in ['Action', 'Thrillers']):
        return 'red'
    else:
        return 'blue'

movies_subset['genre_color'] = movies_subset['listed_in'].apply(assign_genre_color)
```

### 3. Analyzing Duration Trends

```python
# Create a scatter plot of movie duration by release year
plt.figure(figsize=(12, 8))
sns.scatterplot(x='release_year', y='duration_minutes',
                data=movies_subset,
                hue='genre_color',
                alpha=0.7)

plt.title('Movie Duration by Release Year')
plt.xlabel('Release Year')
plt.ylabel('Duration (minutes)')
plt.legend(title='Genre Category', labels=['Action/Thriller', 'Children', 'Documentary', 'Other'])
plt.grid(True, alpha=0.3)

# Add a trend line
sns.regplot(x='release_year', y='duration_minutes',
            data=movies_subset,
            scatter=False,
            line_kws={"color": "black"})

plt.show()

# Calculate correlation between release year and duration
correlation = movies_subset['release_year'].corr(movies_subset['duration_minutes'])
print(f"Correlation between release year and duration: {correlation:.4f}")
```

### 4. Analyzing Additional Factors

```python
# Analyze duration by genre categories
genre_duration = movies_subset.groupby('genre_color')['duration_minutes'].agg(['mean', 'median', 'count'])
print("Duration statistics by genre:")
print(genre_duration)

# Analyze trends by decade
movies_subset['decade'] = (movies_subset['release_year'] // 10) * 10
decade_trends = movies_subset.groupby('decade')['duration_minutes'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='decade', y='duration_minutes', data=decade_trends)
plt.title('Average Movie Duration by Decade')
plt.xlabel('Decade')
plt.ylabel('Average Duration (minutes)')
plt.show()
```

## Mini Practice Task

After completing the guided analysis, try to answer these questions:

1. How does movie duration vary by country of production? Identify the top 5 countries with the longest and shortest average movie durations.
2. Is there a relationship between the release year and movie duration for specific genres? Create separate trend lines for each genre category.
3. Create a visualization showing how the distribution of movie durations has changed across decades using box plots or violin plots.

## Conclusion

By the end of this project, you should be able to draw conclusions about whether movies have been getting shorter over time and identify factors that contribute to movie duration. You'll have gained practical experience in data manipulation with pandas and creating informative visualizations.

## Next Steps

Consider extending your analysis by:

- Incorporating text analysis on movie descriptions
- Building a predictive model for movie duration
- Comparing Netflix movie durations with those from other streaming platforms
