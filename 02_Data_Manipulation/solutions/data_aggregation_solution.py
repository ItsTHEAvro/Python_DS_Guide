# Solution for Data Aggregation and Grouping Mini Practice Task
import pandas as pd
import numpy as np

# Using the employees DataFrame from the example
data = {
    'department': ['IT', 'HR', 'Sales', 'IT', 'HR', 'Sales', 'IT'],
    'employee_name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace'],
    'salary': [75000, 65000, 80000, 70000, 60000, 95000, 82000],
    'years_experience': [5, 3, 7, 4, 2, 9, 6]
}

employees = pd.DataFrame(data)

# Display the original DataFrame
print("Original Employees DataFrame:")
print(employees)

# Create a pivot table showing the average salary and count of employees by department
pivot_table = employees.pivot_table(
    index='department',
    values=['salary', 'years_experience'],
    aggfunc=['mean', 'count']
)

print("\nPivot Table - Average Salary and Employee Count by Department:")
print(pivot_table)

# For easier column access, flatten the hierarchical column index
pivot_table.columns = ['_'.join(col).strip() for col in pivot_table.columns.values]
print("\nPivot Table with Flattened Column Names:")
print(pivot_table)

# Add a column showing the salary per year of experience for each department
# First, let's create a groupby object for the calculation
dept_stats = employees.groupby('department').agg({
    'salary': 'mean',
    'years_experience': 'mean'
})

# Calculate salary per year of experience
dept_stats['salary_per_year_experience'] = (
    dept_stats['salary'] / dept_stats['years_experience']
).round(2)

print("\nDepartment Statistics with Salary per Year of Experience:")
print(dept_stats)

# Alternative approach: Adding the calculation directly to the pivot table
pivot_table['salary_per_year_experience'] = (
    pivot_table['mean_salary'] / pivot_table['mean_years_experience']
).round(2)

print("\nPivot Table with Salary per Year of Experience:")
print(pivot_table)

# Additional analysis: Comparing departments
print("\nDepartment Comparison:")
print(f"Highest average salary: {dept_stats['salary'].idxmax()} (${dept_stats['salary'].max()})")
print(f"Most experienced department: {dept_stats['years_experience'].idxmax()} ({dept_stats['years_experience'].max()} years)")
print(f"Highest salary per year of experience: {dept_stats['salary_per_year_experience'].idxmax()} (${dept_stats['salary_per_year_experience'].max()})")

# Example output:
# Original Employees DataFrame:
#   department employee_name  salary  years_experience
# 0         IT        Alice   75000                 5
# 1         HR          Bob   65000                 3
# 2      Sales      Charlie   80000                 7
# 3         IT        David   70000                 4
# 4         HR          Eva   60000                 2
# 5      Sales        Frank   95000                 9
# 6         IT        Grace   82000                 6
# 
# Pivot Table - Average Salary and Employee Count by Department:
#             mean                    count          
#           salary years_experience salary years_experience
# department                                               
# HR         62500             2.5      2                 2
# IT         75667             5.0      3                 3
# Sales      87500             8.0      2                 2
# 
# Pivot Table with Flattened Column Names:
#           mean_salary mean_years_experience count_salary count_years_experience
# department                                                                     
# HR             62500                   2.5            2                      2
# IT             75667                   5.0            3                      3
# Sales          87500                   8.0            2                      2
# 
# Department Statistics with Salary per Year of Experience:
#           salary  years_experience  salary_per_year_experience
# department                                                    
# HR         62500               2.5                    25000.00
# IT         75667               5.0                    15133.40
# Sales      87500               8.0                    10937.50
# 
# Pivot Table with Salary per Year of Experience:
#           mean_salary mean_years_experience count_salary count_years_experience  salary_per_year_experience
# department                                                                                                
# HR             62500                   2.5            2                      2                    25000.00
# IT             75667                   5.0            3                      3                    15133.40
# Sales          87500                   8.0            2                      2                    10937.50
# 
# Department Comparison:
# Highest average salary: Sales ($87500.0)
# Most experienced department: Sales (8.0 years)
# Highest salary per year of experience: HR ($25000.0)