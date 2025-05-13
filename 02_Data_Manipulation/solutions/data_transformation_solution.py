# Solution for Data Transformation Mini Practice Task
import pandas as pd
import numpy as np

# Create a DataFrame with some intentionally missing values
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'age': [25, 30, np.nan, 42, 38],
    'city': ['New York', np.nan, 'San Francisco', np.nan, 'Boston'],
    'salary': [75000, 82000, 68000, np.nan, 93000],
    'department': ['IT', 'Marketing', 'IT', 'Finance', np.nan]
}

df = pd.DataFrame(data)

# Display the original DataFrame
print("Original DataFrame with missing values:")
print(df)

# Identify the missing values
print("\nMissing value counts per column:")
print(df.isna().sum())

# Display percentage of missing values per column
print("\nPercentage of missing values per column:")
missing_percentage = (df.isna().sum() / len(df) * 100).round(2)
print(missing_percentage)

# Display rows with at least one missing value
print("\nRows with at least one missing value:")
print(df[df.isna().any(axis=1)])

# Fill missing values - numeric columns with mean, text columns with "Unknown"
df_filled = df.copy()

# For numeric columns, fill with mean
numeric_columns = df.select_dtypes(include=['number']).columns
for col in numeric_columns:
    df_filled[col] = df_filled[col].fillna(df[col].mean())

# For text columns, fill with "Unknown"
text_columns = df.select_dtypes(include=['object']).columns
for col in text_columns:
    df_filled[col] = df_filled[col].fillna("Unknown")

# Display the filled DataFrame
print("\nDataFrame after filling missing values:")
print(df_filled)

# Verify no missing values remain
print("\nMissing value counts after filling:")
print(df_filled.isna().sum())

# Additional data transformation - creating new columns
df_filled['salary_thousands'] = (df_filled['salary'] / 1000).round(1)
df_filled['is_senior'] = df_filled['age'] > 35

# Display final transformed DataFrame
print("\nFinal transformed DataFrame:")
print(df_filled)

# Example output:
# Original DataFrame with missing values:
#       name   age           city    salary department
# 0    Alice  25.0       New York  75000.0         IT
# 1      Bob  30.0           NaN  82000.0  Marketing
# 2  Charlie   NaN  San Francisco  68000.0         IT
# 3    David  42.0           NaN      NaN    Finance
# 4      Eva  38.0        Boston  93000.0        NaN
# 
# Missing value counts per column:
# name          0
# age           1
# city          2
# salary        1
# department    1
# dtype: int64
# 
# Percentage of missing values per column:
# name          0.0
# age          20.0
# city         40.0
# salary       20.0
# department   20.0
# dtype: float64
# 
# Rows with at least one missing value:
#       name   age           city    salary department
# 1      Bob  30.0           NaN  82000.0  Marketing
# 2  Charlie   NaN  San Francisco  68000.0         IT
# 3    David  42.0           NaN      NaN    Finance
# 4      Eva  38.0        Boston  93000.0        NaN
# 
# DataFrame after filling missing values:
#       name   age           city    salary department
# 0    Alice  25.0       New York  75000.0         IT
# 1      Bob  30.0        Unknown  82000.0  Marketing
# 2  Charlie  33.8  San Francisco  68000.0         IT
# 3    David  42.0        Unknown  79500.0    Finance
# 4      Eva  38.0        Boston  93000.0    Unknown
# 
# Missing value counts after filling:
# name          0
# age           0
# city          0
# salary        0
# department    0
# dtype: int64
# 
# Final transformed DataFrame:
#       name   age           city    salary department  salary_thousands  is_senior
# 0    Alice  25.0       New York  75000.0         IT              75.0      False
# 1      Bob  30.0        Unknown  82000.0  Marketing              82.0      False
# 2  Charlie  33.8  San Francisco  68000.0         IT              68.0      False
# 3    David  42.0        Unknown  79500.0    Finance              79.5       True
# 4      Eva  38.0        Boston  93000.0    Unknown              93.0       True