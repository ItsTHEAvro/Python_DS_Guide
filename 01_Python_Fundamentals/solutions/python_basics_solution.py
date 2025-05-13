# Solution for Python Basics Mini Practice Task

# Create variables for name, age, and favorite programming language
name = "Alex"
age = 28
favorite_language = "Python"

# Conditional statement that prints different messages based on favorite language
if favorite_language.lower() == "python":
    print(f"Hello {name}! Python is a great choice for data science!")
elif favorite_language.lower() == "r":
    print(f"Hello {name}! R is excellent for statistical analysis!")
elif favorite_language.lower() == "julia":
    print(f"Hello {name}! Julia is gaining popularity for its speed in numerical computing!")
else:
    print(f"Hello {name}! {favorite_language} is an interesting choice. Have you considered Python for data science?")

# Example output with current values:
# Hello Alex! Python is a great choice for data science!