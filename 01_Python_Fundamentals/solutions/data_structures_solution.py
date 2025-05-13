# Solution for Python Lists and Data Structures Mini Practice Task

# Create a dictionary representing a dataset
dataset = {
    'name': 'Customer Churn Dataset',
    'rows': 5000,
    'columns': 12,
    'missing_values': 145
}

# Print the original dictionary
print("Original dataset dictionary:")
for key, value in dataset.items():
    print(f"{key}: {value}")

# Add a new key 'data_types' with a list of common data types
dataset['data_types'] = ['int', 'float', 'string', 'boolean', 'datetime']

# Print the updated dictionary
print("\nUpdated dataset dictionary with data types:")
for key, value in dataset.items():
    print(f"{key}: {value}")

# Example output:
# Original dataset dictionary:
# name: Customer Churn Dataset
# rows: 5000
# columns: 12
# missing_values: 145
#
# Updated dataset dictionary with data types:
# name: Customer Churn Dataset
# rows: 5000
# columns: 12
# missing_values: 145
# data_types: ['int', 'float', 'string', 'boolean', 'datetime']