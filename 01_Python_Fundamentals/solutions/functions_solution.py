"""
Solution to the 'describe_data' function mini-practice task.

Task: Write a function called `describe_data` that takes a list as input 
and returns a dictionary containing the mean, median, minimum, and maximum values.
"""

def describe_data(numbers):
    """
    Calculate descriptive statistics for a list of numbers.
    
    Parameters:
    numbers (list): A list of numeric values
    
    Returns:
    dict: A dictionary containing mean, median, minimum, and maximum values
    """
    # Check if the input is valid
    if not numbers:
        return {
            'mean': None,
            'median': None,
            'min': None,
            'max': None
        }
    
    # Calculate statistics
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    
    # Mean calculation
    mean_value = sum(sorted_numbers) / n
    
    # Median calculation
    if n % 2 == 0:  # Even number of elements
        median_value = (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    else:  # Odd number of elements
        median_value = sorted_numbers[n//2]
    
    # Min and Max
    min_value = sorted_numbers[0]
    max_value = sorted_numbers[-1]
    
    # Return statistics as a dictionary
    return {
        'mean': mean_value,
        'median': median_value,
        'min': min_value,
        'max': max_value
    }

# Example usage
if __name__ == "__main__":
    # Test with a sample list
    sample_data = [12, 15, 23, 7, 42, 19, 30]
    stats = describe_data(sample_data)
    
    print("Sample data:", sample_data)
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test with edge cases
    print("\nEmpty list:", describe_data([]))
    print("Single element:", describe_data([5]))
    print("Duplicate elements:", describe_data([3, 3, 3, 3, 3]))