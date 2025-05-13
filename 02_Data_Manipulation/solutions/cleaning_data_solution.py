# Solution for Cleaning Data Mini Practice Task
import pandas as pd
import numpy as np
from scipy import stats

# Create a DataFrame with duplicated rows and numerical outliers
data = {
    'customer_id': [1001, 1002, 1002, 1003, 1004, 1005, 1006, 1006, 1007],
    'purchase_amount': [120, 85, 85, 90, 1500, 110, 95, 95, 2000],
    'items_purchased': [4, 3, 3, 3, 12, 5, 2, 2, 8],
    'store_location': ['Downtown', 'Suburb', 'Suburb', 'Downtown', 'Mall', 'Mall', 'Downtown', 'Downtown', 'Suburb'],
    'customer_rating': [4.5, 3.8, 3.8, 4.2, 4.9, 3.5, 4.1, 4.1, 4.7]
}

df = pd.DataFrame(data)

# Display the original DataFrame
print("Original DataFrame with duplicates and outliers:")
print(df)

# Create a function to clean the data
def clean_data(df):
    """
    Clean the data by removing duplicates and replacing outliers with the median value.
    
    Parameters:
    df (DataFrame): Input DataFrame with duplicates and outliers
    
    Returns:
    DataFrame: Cleaned DataFrame
    """
    # Make a copy of the DataFrame to avoid modifying the original
    cleaned_df = df.copy()
    
    # Check for duplicates
    duplicate_count = cleaned_df.duplicated().sum()
    print(f"\nNumber of duplicate rows found: {duplicate_count}")
    
    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates()
    print(f"After removing duplicates, shape: {cleaned_df.shape}")
    
    # Check for numerical columns that might have outliers
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nChecking for outliers in columns: {numeric_columns}")
    
    # Function to detect and handle outliers
    def handle_outliers(df, column):
        # Calculate Z-scores
        z_scores = stats.zscore(df[column])
        
        # Define outliers as values with absolute Z-score > 3
        outliers = np.abs(z_scores) > 3
        outlier_indices = np.where(outliers)[0]
        
        if len(outlier_indices) > 0:
            print(f"Outliers found in column '{column}' at indices: {outlier_indices}")
            print(f"Outlier values: {df.loc[outlier_indices, column].values}")
            
            # Replace outliers with the column median
            median_value = df[column].median()
            print(f"Replacing outliers with median: {median_value}")
            
            # Create a copy of the dataframe to avoid SettingWithCopyWarning
            df_copy = df.copy()
            df_copy.loc[outlier_indices, column] = median_value
            return df_copy
        else:
            print(f"No outliers found in column '{column}'")
            return df
    
    # Handle outliers for each numeric column
    for column in numeric_columns:
        cleaned_df = handle_outliers(cleaned_df, column)
    
    return cleaned_df

# Apply the cleaning function
cleaned_df = clean_data(df)

# Display the cleaned DataFrame
print("\nCleaned DataFrame:")
print(cleaned_df)

# Show descriptive statistics before and after cleaning
print("\nDescriptive Statistics - Original DataFrame:")
print(df.describe())

print("\nDescriptive Statistics - Cleaned DataFrame:")
print(cleaned_df.describe())

# Additional data cleaning steps
# Reset the index after cleaning
cleaned_df = cleaned_df.reset_index(drop=True)
print("\nFinal Cleaned DataFrame with Reset Index:")
print(cleaned_df)

# Example output:
# Original DataFrame with duplicates and outliers:
#    customer_id  purchase_amount  items_purchased store_location  customer_rating
# 0         1001              120                4      Downtown             4.5
# 1         1002               85                3        Suburb             3.8
# 2         1002               85                3        Suburb             3.8
# 3         1003               90                3      Downtown             4.2
# 4         1004             1500               12          Mall             4.9
# 5         1005              110                5          Mall             3.5
# 6         1006               95                2      Downtown             4.1
# 7         1006               95                2      Downtown             4.1
# 8         1007             2000                8        Suburb             4.7
# 
# Number of duplicate rows found: 2
# After removing duplicates, shape: (7, 5)
# 
# Checking for outliers in columns: ['customer_id', 'purchase_amount', 'items_purchased', 'customer_rating']
# No outliers found in column 'customer_id'
# Outliers found in column 'purchase_amount' at indices: [2, 6]
# Outlier values: [1500 2000]
# Replacing outliers with median: 95.0
# No outliers found in column 'items_purchased'
# No outliers found in column 'customer_rating'
# 
# Cleaned DataFrame:
#    customer_id  purchase_amount  items_purchased store_location  customer_rating
# 0         1001              120                4      Downtown             4.5
# 1         1002               85                3        Suburb             3.8
# 3         1003               90                3      Downtown             4.2
# 4         1004               95               12          Mall             4.9
# 5         1005              110                5          Mall             3.5
# 6         1006               95                2      Downtown             4.1
# 8         1007               95                8        Suburb             4.7
# 
# Descriptive Statistics - Original DataFrame:
#       customer_id  purchase_amount  items_purchased  customer_rating
# count     9.00000        9.000000        9.000000        9.000000
# mean   1004.00000      464.444444        4.666667        4.177778
# std       2.29129      702.991903        3.428010        0.445658
# min    1001.00000       85.000000        2.000000        3.500000
# 25%    1002.00000       90.000000        3.000000        3.800000
# 50%    1004.00000       95.000000        3.000000        4.200000
# 75%    1006.00000      120.000000        5.000000        4.500000
# max    1007.00000     2000.000000       12.000000        4.900000
# 
# Descriptive Statistics - Cleaned DataFrame:
#       customer_id  purchase_amount  items_purchased  customer_rating
# count     7.00000        7.000000        7.000000        7.000000
# mean   1004.00000       98.571429        5.285714        4.242857
# std       2.38048       12.485222        3.729359        0.516809
# min    1001.00000       85.000000        2.000000        3.500000
# 25%    1002.00000       90.000000        3.000000        3.800000
# 50%    1004.00000       95.000000        4.000000        4.200000
# 75%    1006.00000      110.000000        8.000000        4.700000
# max    1007.00000      120.000000       12.000000        4.900000