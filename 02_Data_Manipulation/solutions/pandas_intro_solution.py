# Solution for Introduction to Pandas Mini Practice Task
import pandas as pd

# Create a pandas DataFrame from a dictionary containing information about 5 different books
books_data = {
    'title': [
        'The Great Gatsby', 
        'To Kill a Mockingbird', 
        'Dune', 
        '1984', 
        'Pride and Prejudice'
    ],
    'author': [
        'F. Scott Fitzgerald', 
        'Harper Lee', 
        'Frank Herbert', 
        'George Orwell', 
        'Jane Austen'
    ],
    'year': [1925, 1960, 1965, 1949, 1813],
    'pages': [180, 281, 412, 328, 279],
    'genre': ['Classic', 'Fiction', 'Science Fiction', 'Dystopian', 'Romance']
}

# Create the DataFrame
books_df = pd.DataFrame(books_data)

# Display the DataFrame
print("Books DataFrame:")
print(books_df)

# Display basic information about the DataFrame
print("\nDataFrame Information:")
print(f"Shape: {books_df.shape}")
print(f"Columns: {list(books_df.columns)}")

# Display basic statistics
print("\nBasic Statistics:")
print(books_df.describe())

# Display the info (types, non-null values)
print("\nDataFrame Info:")
books_df.info()

# Additional analysis
print("\nOldest book:", books_df.loc[books_df['year'].idxmin()]['title'])
print("Most recent book:", books_df.loc[books_df['year'].idxmax()]['title'])
print("Average pages:", round(books_df['pages'].mean(), 2))
print("Genres breakdown:", books_df['genre'].value_counts().to_dict())

# Example output:
# Books DataFrame:
#                 title               author  year  pages           genre
# 0     The Great Gatsby  F. Scott Fitzgerald  1925    180         Classic
# 1  To Kill a Mockingbird          Harper Lee  1960    281         Fiction
# 2                 Dune        Frank Herbert  1965    412  Science Fiction
# 3                 1984        George Orwell  1949    328       Dystopian
# 4    Pride and Prejudice         Jane Austen  1813    279         Romance
#
# DataFrame Information:
# Shape: (5, 5)
# Columns: ['title', 'author', 'year', 'pages', 'genre']
#
# Basic Statistics:
#               year        pages
# count     5.000000     5.000000
# mean   1922.400000   296.000000
# std      62.416338    86.093514
# min    1813.000000   180.000000
# 25%     1925.000000   279.000000
# 50%     1949.000000   281.000000
# 75%     1960.000000   328.000000
# max     1965.000000   412.000000
#
# DataFrame Info:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 5 entries, 0 to 4
# Data columns (total 5 columns):
#  #   Column  Non-Null Count  Dtype 
# ---  ------  --------------  ----- 
#  0   title   5 non-null      object
#  1   author  5 non-null      object
#  2   year    5 non-null      int64 
#  3   pages   5 non-null      int64 
#  4   genre   5 non-null      object
# dtypes: int64(2), object(3)
# memory usage: 328.0+ bytes
#
# Oldest book: Pride and Prejudice
# Most recent book: Dune
# Average pages: 296.0
# Genres breakdown: {'Classic': 1, 'Fiction': 1, 'Science Fiction': 1, 'Dystopian': 1, 'Romance': 1}