# Data Manipulation with Python

## Learning Objectives

By the end of this section, you will:

- Master pandas for efficient data manipulation
- Learn techniques for cleaning and preprocessing data
- Understand how to transform, filter, and aggregate datasets
- Develop skills in combining data from multiple sources
- Work effectively with date and time data

## Key Topics Covered

### 1. Introduction to Pandas

- Series and DataFrame objects
- Creating and importing data
- Basic operations and attributes
- Data exploration methods

```python
import pandas as pd
import numpy as np

# Creating a simple DataFrame
data = {
    'name': ['John', 'Anna', 'Peter', 'Linda'],
    'age': [28, 34, 29, 42],
    'city': ['New York', 'Paris', 'Berlin', 'London'],
    'salary': [65000, 70000, 59000, 85000]
}

df = pd.DataFrame(data)
print(df.head())

# Basic exploration
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")
print(df.describe())
print(df.info())
```

**Mini Practice Task:** Create a pandas DataFrame from a dictionary containing information about 5 different books (title, author, year, pages, genre). Display the basic information and statistics about your DataFrame.

### 2. Data Transformation

- Column operations and creation
- Applying functions
- Type conversion
- String manipulations
- Handling missing data

```python
# Adding and modifying columns
df['salary_thousands'] = df['salary'] / 1000
df['is_senior'] = df['age'] > 30

# Applying functions
def get_first_name(name):
    return name.split()[0]

df['first_name'] = df['name'].apply(get_first_name)

# Handling missing values
df_with_missing = df.copy()
df_with_missing.loc[1, 'salary'] = np.nan
df_with_missing.loc[3, 'city'] = np.nan

# Filling missing values
df_filled = df_with_missing.fillna({
    'salary': df['salary'].mean(),
    'city': 'Unknown'
})

print(df_filled)
```

**Mini Practice Task:** Create a DataFrame with some intentionally missing values. Write code to identify the missing values, then fill them with appropriate replacements (mean for numeric columns, "Unknown" for text columns).

### 3. Data Aggregation and Grouping

- GroupBy operations
- Aggregate functions
- Pivot tables and cross-tabulations
- Reshaping data

```python
# Sample data with categories
data = {
    'department': ['IT', 'HR', 'Sales', 'IT', 'HR', 'Sales', 'IT'],
    'employee_name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace'],
    'salary': [75000, 65000, 80000, 70000, 60000, 95000, 82000],
    'years_experience': [5, 3, 7, 4, 2, 9, 6]
}

employees = pd.DataFrame(data)

# GroupBy operations
dept_stats = employees.groupby('department').agg({
    'salary': ['mean', 'min', 'max'],
    'years_experience': ['mean', 'min', 'max'],
    'employee_name': 'count'
})

print(dept_stats)

# Pivot table
pivot = employees.pivot_table(
    index='department',
    values=['salary', 'years_experience'],
    aggfunc=['mean', 'count']
)

print(pivot)
```

**Mini Practice Task:** Using the employees DataFrame from above, create a pivot table that shows the average salary and count of employees by department. Add a column showing the salary per year of experience for each department.

### 4. Slicing and Indexing

- Setting and resetting indexes
- Hierarchical/multi-level indexing
- Advanced selection methods
- Boolean indexing and filtering
- Conditional selection

```python
# Setting indexes
indexed_df = employees.set_index('employee_name')
print(indexed_df.loc['Alice'])

# Multi-level indexing
multi_index = employees.set_index(['department', 'employee_name'])
print(multi_index.loc['IT'])
print(multi_index.loc[('IT', 'Alice')])

# Boolean indexing
high_salary = employees[employees['salary'] > 70000]
print(high_salary)

# Complex filtering
experienced_it = employees[
    (employees['department'] == 'IT') &
    (employees['years_experience'] > 4)
]
print(experienced_it)
```

**Mini Practice Task:** Create a DataFrame with data about different products (product_id, category, price, stock_quantity). Use boolean indexing to find all products that are low in stock (less than 10) or expensive (more than $100).

### 5. Joining and Merging Data

- Concatenating DataFrames
- Merge operations (inner, outer, left, right)
- Joining on indexes
- Handling merge conflicts

```python
# Creating sample DataFrames
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['John', 'Anna', 'Peter', 'Linda', 'Michael'],
    'city': ['New York', 'Paris', 'Berlin', 'London', 'Tokyo']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105],
    'customer_id': [1, 3, 5, 6, 7],
    'amount': [150, 200, 300, 120, 250],
    'order_date': pd.date_range(start='2023-01-01', periods=5)
})

# Different types of merges
inner_merge = pd.merge(customers, orders, on='customer_id', how='inner')
left_merge = pd.merge(customers, orders, on='customer_id', how='left')
right_merge = pd.merge(customers, orders, on='customer_id', how='right')
outer_merge = pd.merge(customers, orders, on='customer_id', how='outer')

print("Inner merge (only matching records):")
print(inner_merge)
print("\nLeft merge (all customers, matching orders):")
print(left_merge)
```

**Mini Practice Task:** Create two DataFrames: one for employees (employee_id, name, department) and one for projects (project_id, project_name, employee_id). Perform different types of merges to find: 1) All employees and their projects, 2) Only employees with assigned projects.

### 6. Cleaning Data

- Detecting and handling duplicates
- Outlier detection and handling
- Data normalization and standardization
- Data integrity checks
- Advanced missing data imputation

```python
# Sample data with issues
data = {
    'id': [1, 2, 2, 3, 4, 5],
    'value': [100, 200, 200, 300, 9999, 500],
    'category': ['A', 'B', 'B', 'C', 'D', 'E']
}

dirty_df = pd.DataFrame(data)

# Finding and removing duplicates
print(f"Duplicates: {dirty_df.duplicated().sum()}")
clean_df = dirty_df.drop_duplicates()

# Detecting outliers using Z-score
from scipy import stats

z_scores = stats.zscore(clean_df['value'])
outliers = abs(z_scores) > 3
print("Outliers:")
print(clean_df[outliers])

# Normalizing data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
clean_df['value_normalized'] = scaler.fit_transform(clean_df[['value']])
print(clean_df)
```

**Mini Practice Task:** Create a DataFrame with some duplicated rows and numerical outliers. Write a function to clean the data by removing duplicates and replacing outliers with the median value.

### 7. Working with Dates and Times

- Date/time objects and operations
- Time series manipulation
- Resampling and frequency conversion
- Rolling and expanding windows
- Date range generation

```python
# Working with datetime data
import pandas as pd
import numpy as np

# Creating date ranges
date_range = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')

# Time series data
ts_data = pd.DataFrame({
    'date': date_range,
    'value': np.random.randn(31).cumsum()  # Random walk
})
ts_data.set_index('date', inplace=True)

# Date extraction
ts_data['year'] = ts_data.index.year
ts_data['month'] = ts_data.index.month
ts_data['day'] = ts_data.index.day
ts_data['day_of_week'] = ts_data.index.dayofweek

# Resampling (e.g., to weekly)
weekly_data = ts_data.resample('W').mean()
print(weekly_data)

# Rolling windows
rolling_mean = ts_data['value'].rolling(window=7).mean()
print("7-day rolling average:")
print(rolling_mean.tail())
```

**Mini Practice Task:** Create a DataFrame with daily sales data for the last 30 days. Calculate weekly totals, identify the day of the week with the highest average sales, and create a 3-day moving average of sales.

## Resources for Further Learning

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Python for Data Analysis](https://wesmckinney.com/book/) by Wes McKinney
- [Pandas Cookbook](https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html)

## Next Steps

After mastering data manipulation techniques, proceed to [Data Visualization](../03_Visualization/README.md) to learn how to effectively visualize your processed data.
