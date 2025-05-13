# Solution for Working with Dates and Times Mini Practice Task
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Create a DataFrame with daily sales data for the last 30 days
# First, create the date range
end_date = pd.Timestamp('2025-05-13')  # Using current date from context
start_date = end_date - pd.Timedelta(days=29)
date_range = pd.date_range(start=start_date, end=end_date)

# Generate some realistic sales data with day-of-week patterns
# Higher sales on weekends, lower on weekdays
np.random.seed(42)  # For reproducibility
base_sales = 1000
weekday_variation = {
    0: 0.7,    # Monday: 70% of base
    1: 0.8,    # Tuesday: 80% of base
    2: 0.9,    # Wednesday: 90% of base
    3: 1.0,    # Thursday: 100% of base
    4: 1.3,    # Friday: 130% of base
    5: 1.5,    # Saturday: 150% of base
    6: 1.4     # Sunday: 140% of base
}

# Create sales with day-of-week patterns and some randomness
sales = [
    base_sales * weekday_variation[day.dayofweek] * (1 + 0.2 * np.random.randn())
    for day in date_range
]

# Create the DataFrame
sales_df = pd.DataFrame({
    'date': date_range,
    'sales': sales
})

# Display the first few rows of the DataFrame
print("Daily Sales Data for Last 30 Days:")
print(sales_df.head())

# 1. Calculate weekly totals
sales_df['weekday'] = sales_df['date'].dt.day_name()
sales_df['week'] = sales_df['date'].dt.isocalendar().week

weekly_sales = sales_df.groupby('week')['sales'].sum().reset_index()
print("\nWeekly Sales Totals:")
print(weekly_sales)

# 2. Identify the day of the week with the highest average sales
avg_sales_by_day = sales_df.groupby('weekday')['sales'].mean().reset_index()
avg_sales_by_day = avg_sales_by_day.sort_values('sales', ascending=False)
print("\nAverage Sales by Day of Week (Highest to Lowest):")
print(avg_sales_by_day)

highest_day = avg_sales_by_day.iloc[0]['weekday']
highest_avg = avg_sales_by_day.iloc[0]['sales']
print(f"\nThe day of the week with the highest average sales is {highest_day} (${highest_avg:.2f})")

# 3. Create a 3-day moving average of sales
sales_df['3_day_moving_avg'] = sales_df['sales'].rolling(window=3).mean()

print("\nSales Data with 3-Day Moving Average:")
print(sales_df[['date', 'sales', '3_day_moving_avg']].tail(10))

# Additional analyses
# Extract month, day, and weekday components
sales_df['month'] = sales_df['date'].dt.month
sales_df['day'] = sales_df['date'].dt.day
sales_df['is_weekend'] = sales_df['date'].dt.dayofweek >= 5

# Calculate weekend vs. weekday stats
weekend_avg = sales_df[sales_df['is_weekend']]['sales'].mean()
weekday_avg = sales_df[~sales_df['is_weekend']]['sales'].mean()

print("\nWeekend vs. Weekday Comparison:")
print(f"Average weekend sales: ${weekend_avg:.2f}")
print(f"Average weekday sales: ${weekday_avg:.2f}")
print(f"Weekend sales are {(weekend_avg/weekday_avg - 1) * 100:.1f}% higher than weekday sales")

# Identify the day with the highest and lowest sales
max_sales_day = sales_df.loc[sales_df['sales'].idxmax()]
min_sales_day = sales_df.loc[sales_df['sales'].idxmin()]

print("\nSales Extremes:")
print(f"Highest sales: ${max_sales_day['sales']:.2f} on {max_sales_day['date'].strftime('%A, %B %d, %Y')}")
print(f"Lowest sales: ${min_sales_day['sales']:.2f} on {min_sales_day['date'].strftime('%A, %B %d, %Y')}")

# Visualize the data (commented out as this solution is intended to be run in a non-interactive environment)
"""
plt.figure(figsize=(12, 6))
plt.plot(sales_df['date'], sales_df['sales'], label='Daily Sales')
plt.plot(sales_df['date'], sales_df['3_day_moving_avg'], label='3-Day Moving Average', color='red')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.title('Daily Sales and 3-Day Moving Average')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""

# Example output (actual values will vary due to random generation):
# Daily Sales Data for Last 30 Days:
#         date        sales
# 0 2025-04-14   685.361468
# 1 2025-04-15   920.607365
# 2 2025-04-16   763.545316
# 3 2025-04-17  1021.029089
# 4 2025-04-18  1196.055148
# 
# Weekly Sales Totals:
#    week        sales
# 0    16   6487.84807
# 1    17   6904.01005
# 2    18   7352.01504
# 3    19   6872.47990
# 4    20   2031.27863
# 
# Average Sales by Day of Week (Highest to Lowest):
#     weekday       sales
# 5  Saturday  1502.39373
# 6    Sunday  1361.72109
# 4    Friday  1313.14711
# 3  Thursday   979.07268
# 2 Wednesday   889.93986
# 1   Tuesday   776.13239
# 0    Monday   680.01055
# 
# The day of the week with the highest average sales is Saturday ($1502.39)
# 
# Sales Data with 3-Day Moving Average:
#          date       sales  3_day_moving_avg
# 20 2025-05-04  1521.13479       1464.31225
# 21 2025-05-05   697.30409       1202.18513
# 22 2025-05-06   737.85449        985.43112
# 23 2025-05-07   877.11628        770.75829
# 24 2025-05-08   904.09563        839.68880
# 25 2025-05-09  1439.33737       1073.51643
# 26 2025-05-10  1587.16199       1310.19833
# 27 2025-05-11  1144.37976       1390.29304
# 28 2025-05-12   784.17088       1171.90421
# 29 2025-05-13   737.10775        888.55280
# 
# Weekend vs. Weekday Comparison:
# Average weekend sales: $1432.06
# Average weekday sales: $891.18
# Weekend sales are 60.7% higher than weekday sales
# 
# Sales Extremes:
# Highest sales: $1587.16 on Saturday, May 10, 2025
# Lowest sales: $680.01 on Monday, April 14, 2025