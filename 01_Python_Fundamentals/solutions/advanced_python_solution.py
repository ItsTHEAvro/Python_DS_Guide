# Solution for Advanced Python Concepts Mini Practice Task

# Using list comprehension to create a list of tuples containing numbers from 1 to 10 and their cubes
numbers_and_cubes = [(num, num**3) for num in range(1, 11)]

print("List of tuples with numbers and their cubes:")
for number, cube in numbers_and_cubes:
    print(f"{number} cubed: {cube}")

# Lambda function that takes a number and returns whether it's even or odd
is_even = lambda x: "Even" if x % 2 == 0 else "Odd"

# Test the lambda function with each number
print("\nTesting if numbers are even or odd:")
for number, _ in numbers_and_cubes:
    print(f"{number} is {is_even(number)}")

# Additional demonstration of advanced Python concepts
print("\nDemonstrating other advanced Python concepts:")

# Generator expression (memory efficient)
# Just to demonstrate, we'll calculate squares using a generator
def squares_generator(n):
    for i in range(1, n+1):
        yield i**2

print("\nFirst 5 squares using a generator:")
squares_gen = squares_generator(5)
for square in squares_gen:
    print(square)

# Error handling demonstration
print("\nDemonstrating error handling:")
numbers = [1, 2, 0, 4, 5]
for number in numbers:
    try:
        result = 10 / number
        print(f"10 / {number} = {result}")
    except ZeroDivisionError:
        print(f"Cannot divide 10 by {number} (division by zero)")

# File operations example (commented out to avoid creating actual files)
"""
# Writing to a file
with open('example.txt', 'w') as file:
    file.write('This is an example of file handling in Python.')

# Reading from a file
with open('example.txt', 'r') as file:
    content = file.read()
    print(f"\nFile content: {content}")
"""

# Example output:
# List of tuples with numbers and their cubes:
# 1 cubed: 1
# 2 cubed: 8
# 3 cubed: 27
# 4 cubed: 64
# 5 cubed: 125
# 6 cubed: 216
# 7 cubed: 343
# 8 cubed: 512
# 9 cubed: 729
# 10 cubed: 1000
#
# Testing if numbers are even or odd:
# 1 is Odd
# 2 is Even
# 3 is Odd
# 4 is Even
# 5 is Odd
# 6 is Even
# 7 is Odd
# 8 is Even
# 9 is Odd
# 10 is Even
#
# Demonstrating other advanced Python concepts:
#
# First 5 squares using a generator:
# 1
# 4
# 9
# 16
# 25
#
# Demonstrating error handling:
# 10 / 1 = 10.0
# 10 / 2 = 5.0
# Cannot divide 10 by 0 (division by zero)
# 10 / 4 = 2.5
# 10 / 5 = 2.0