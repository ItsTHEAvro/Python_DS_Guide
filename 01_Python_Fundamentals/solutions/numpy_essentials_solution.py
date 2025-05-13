# Solution for NumPy Essentials Mini Practice Task
import numpy as np

# Create a 3x3 NumPy array of random integers between 1 and 100
random_array = np.random.randint(1, 101, size=(3, 3))
print("3x3 array of random integers:")
print(random_array)

# Calculate the row means
row_means = np.mean(random_array, axis=1)
print("\nRow means:")
for i, mean in enumerate(row_means):
    print(f"Row {i+1}: {mean}")

# Calculate the column means
column_means = np.mean(random_array, axis=0)
print("\nColumn means:")
for i, mean in enumerate(column_means):
    print(f"Column {i+1}: {mean}")

# Find the maximum value and its position
max_value = np.max(random_array)
max_position = np.unravel_index(np.argmax(random_array), random_array.shape)

print(f"\nMaximum value: {max_value}")
print(f"Position of maximum value (row, column): {max_position}")

# Additional calculations to demonstrate NumPy capabilities
print("\nAdditional statistics:")
print(f"Array sum: {np.sum(random_array)}")
print(f"Array standard deviation: {np.std(random_array)}")
print(f"Array median: {np.median(random_array)}")

# Reshaping and manipulating the array
flattened = random_array.flatten()
print("\nFlattened array:")
print(flattened)

# Example output (numbers will vary due to randomness):
# 3x3 array of random integers:
# [[45 23 89]
#  [56 78 12]
#  [34 67 90]]
# 
# Row means:
# Row 1: 52.33333333333333
# Row 2: 48.666666666666664
# Row 3: 63.666666666666664
# 
# Column means:
# Column 1: 45.0
# Column 2: 56.0
# Column 3: 63.666666666666664
# 
# Maximum value: 90
# Position of maximum value (row, column): (2, 2)
# 
# Additional statistics:
# Array sum: 494
# Array standard deviation: 27.266742954147804
# Array median: 56.0
# 
# Flattened array:
# [45 23 89 56 78 12 34 67 90]