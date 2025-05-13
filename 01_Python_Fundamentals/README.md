# Python Fundamentals for Data Science

## Learning Objectives

By the end of this section, you will:

- Master Python syntax and fundamental programming concepts
- Understand how to work with various data structures in Python
- Learn to write efficient functions and use popular data science packages
- Gain proficiency with NumPy for numerical computing
- Develop skills in advanced Python techniques like list comprehensions and error handling

## Key Topics Covered

### 1. Python Basics

- Python syntax and data types
- Variables and assignments
- Basic operators
- Control flow statements (if, else, elif)
- Print statements and string formatting

```python
# Basic variable assignment and operations
name = "Data Scientist"
years_experience = 5
is_python_user = True

# String formatting
print(f"A {name} with {years_experience} years of experience")

# Conditional logic
if years_experience > 3 and is_python_user:
    print("You're an experienced Python data scientist!")
elif years_experience <= 3 and is_python_user:
    print("You're building good Python experience!")
else:
    print("Time to learn Python!")
```

**Mini Practice Task:** Create variables for your name, age, and favorite programming language. Write a conditional statement that prints different messages based on your favorite language.

### 2. Python Lists and Data Structures

- Lists creation and manipulation
- List methods and operations
- Dictionaries, tuples, and sets
- Indexing and slicing

```python
# Creating and manipulating lists
languages = ['Python', 'R', 'SQL', 'Java']
languages.append('Julia')
print(languages[0])  # Output: Python
print(languages[-1])  # Output: Julia
print(languages[1:3])  # Output: ['R', 'SQL']

# Dictionaries
data_scientist = {
    'name': 'Alex',
    'skills': ['Python', 'Machine Learning', 'SQL'],
    'experience_years': 4
}
print(f"Name: {data_scientist['name']}")
print(f"First skill: {data_scientist['skills'][0]}")
```

**Mini Practice Task:** Create a dictionary representing a dataset with keys for 'name', 'rows', 'columns', and 'missing_values'. Write code to add a new key 'data_types' with a list of common data types.

### 3. Functions and Packages

- Function definition and calling
- Parameters and return values
- Default arguments
- Importing and using modules
- Essential data science packages overview

```python
# Defining and calling functions
def calculate_mean(numbers):
    """Calculate the mean of a list of numbers."""
    return sum(numbers) / len(numbers)

data = [12, 15, 23, 7, 42]
mean_value = calculate_mean(data)
print(f"Mean: {mean_value}")

# Using libraries
import numpy as np
from matplotlib import pyplot as plt

# Creating an array with NumPy
array = np.array([1, 2, 3, 4, 5])
print(f"Mean using NumPy: {np.mean(array)}")
```

**Mini Practice Task:** Write a function called `describe_data` that takes a list as input and returns a dictionary containing the mean, median, minimum, and maximum values. Test it with a sample list.

### 4. NumPy Essentials

- NumPy arrays vs. Python lists
- Array creation and operations
- Broadcasting
- Statistical functions
- Array manipulation

```python
import numpy as np

# Creating arrays
array_1d = np.array([1, 2, 3, 4, 5])
array_2d = np.array([[1, 2, 3], [4, 5, 6]])

# Array operations
print(array_1d * 2)  # Output: [2 4 6 8 10]
print(array_1d.mean())  # Output: 3.0

# Broadcasting
height = np.array([1.7, 1.8, 1.6, 1.9, 1.5])
weight = np.array([65, 75, 55, 80, 50])
bmi = weight / (height ** 2)
print(f"BMI values: {bmi}")
```

**Mini Practice Task:** Create a 3x3 NumPy array of random integers between 1 and 100. Calculate the row and column means, find the maximum value and its position in the array.

### 5. Advanced Python Concepts

- List comprehensions and generators
- Lambda functions
- Error handling with try-except
- Working with files
- Scope and namespaces

```python
# List comprehensions
squares = [x**2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Lambda functions
double = lambda x: x * 2
print(double(5))  # Output: 10

# Error handling
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
finally:
    print("This always executes")
```

**Mini Practice Task:** Use a list comprehension to create a list of tuples containing numbers from 1 to 10 and their cubes. Then write a lambda function that takes a number and returns whether it's even or odd.

## Resources for Further Learning

- [Official Python Documentation](https://docs.python.org/3/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Python for Data Analysis](https://wesmckinney.com/book/) by Wes McKinney

## Next Steps

After mastering Python fundamentals, proceed to [Data Manipulation with pandas](../02_Data_Manipulation/README.md) to learn how to effectively work with structured data.
