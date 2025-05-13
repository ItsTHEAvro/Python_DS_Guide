# Solution for Joining and Merging Data Mini Practice Task
import pandas as pd
import numpy as np

# Create the first DataFrame for employees
employees_data = {
    'employee_id': [101, 102, 103, 104, 105],
    'name': ['Alice Johnson', 'Bob Smith', 'Charlie Davis', 'Diana Miller', 'Edward Wilson'],
    'department': ['Engineering', 'Marketing', 'Finance', 'HR', 'Engineering']
}

# Create the second DataFrame for projects
projects_data = {
    'project_id': [1001, 1002, 1003, 1004, 1005],
    'project_name': ['Website Redesign', 'Data Analytics Tool', 'Budget Planning', 'Recruitment System', 'Mobile App'],
    'employee_id': [101, 101, 103, 106, 107]
}

# Create the DataFrames
employees_df = pd.DataFrame(employees_data)
projects_df = pd.DataFrame(projects_data)

# Display the original DataFrames
print("Employees DataFrame:")
print(employees_df)
print("\nProjects DataFrame:")
print(projects_df)

# 1. Perform an inner merge to find employees with assigned projects
employees_with_projects = pd.merge(
    employees_df, 
    projects_df, 
    on='employee_id', 
    how='inner'
)

print("\n1. Inner Merge - Employees with assigned projects:")
print(employees_with_projects)

# 2. Perform a left merge to show all employees and their projects (if any)
all_employees_with_projects = pd.merge(
    employees_df,
    projects_df,
    on='employee_id',
    how='left'
)

print("\n2. Left Merge - All employees and their projects (if any):")
print(all_employees_with_projects)

# 3. Perform a right merge to show all projects and their assigned employees (if any)
all_projects_with_employees = pd.merge(
    employees_df,
    projects_df,
    on='employee_id',
    how='right'
)

print("\n3. Right Merge - All projects and their assigned employees (if any):")
print(all_projects_with_employees)

# 4. Perform an outer merge to include all employees and projects
all_employees_and_projects = pd.merge(
    employees_df,
    projects_df,
    on='employee_id',
    how='outer'
)

print("\n4. Outer Merge - All employees and projects:")
print(all_employees_and_projects)

# Additional analysis
# Count number of projects per employee
projects_per_employee = all_employees_with_projects.groupby('name').size().reset_index(name='project_count')
print("\nNumber of projects per employee:")
print(projects_per_employee)

# Find employees without any projects
employees_without_projects = all_employees_with_projects[all_employees_with_projects['project_id'].isna()]
print("\nEmployees without any projects:")
print(employees_without_projects if not employees_without_projects.empty else "All employees have projects")

# Find projects without assigned employees
projects_without_employees = all_projects_with_employees[all_projects_with_employees['name'].isna()]
print("\nProjects without assigned employees:")
print(projects_without_employees if not projects_without_employees.empty else "All projects have assigned employees")

# Example output:
# Employees DataFrame:
#    employee_id           name   department
# 0         101  Alice Johnson  Engineering
# 1         102     Bob Smith    Marketing
# 2         103  Charlie Davis     Finance
# 3         104   Diana Miller          HR
# 4         105  Edward Wilson  Engineering
# 
# Projects DataFrame:
#    project_id         project_name  employee_id
# 0        1001     Website Redesign         101
# 1        1002  Data Analytics Tool         101
# 2        1003      Budget Planning         103
# 3        1004   Recruitment System         106
# 4        1005           Mobile App         107
# 
# 1. Inner Merge - Employees with assigned projects:
#    employee_id           name   department  project_id         project_name
# 0         101  Alice Johnson  Engineering       1001     Website Redesign
# 1         101  Alice Johnson  Engineering       1002  Data Analytics Tool
# 2         103  Charlie Davis     Finance       1003      Budget Planning
# 
# 2. Left Merge - All employees and their projects (if any):
#    employee_id           name   department  project_id         project_name
# 0         101  Alice Johnson  Engineering     1001.0     Website Redesign
# 1         101  Alice Johnson  Engineering     1002.0  Data Analytics Tool
# 2         102     Bob Smith    Marketing        NaN                  NaN
# 3         103  Charlie Davis     Finance     1003.0      Budget Planning
# 4         104   Diana Miller          HR        NaN                  NaN
# 5         105  Edward Wilson  Engineering        NaN                  NaN
# 
# 3. Right Merge - All projects and their assigned employees (if any):
#    employee_id           name   department  project_id         project_name
# 0       101.0  Alice Johnson  Engineering       1001     Website Redesign
# 1       101.0  Alice Johnson  Engineering       1002  Data Analytics Tool
# 2       103.0  Charlie Davis     Finance       1003      Budget Planning
# 3         NaN           NaN          NaN       1004   Recruitment System
# 4         NaN           NaN          NaN       1005           Mobile App
# 
# 4. Outer Merge - All employees and projects:
#    employee_id           name   department  project_id         project_name
# 0       101.0  Alice Johnson  Engineering     1001.0     Website Redesign
# 1       101.0  Alice Johnson  Engineering     1002.0  Data Analytics Tool
# 2       102.0     Bob Smith    Marketing        NaN                  NaN
# 3       103.0  Charlie Davis     Finance     1003.0      Budget Planning
# 4       104.0   Diana Miller          HR        NaN                  NaN
# 5       105.0  Edward Wilson  Engineering        NaN                  NaN
# 6         NaN           NaN          NaN     1004.0   Recruitment System
# 7         NaN           NaN          NaN     1005.0           Mobile App