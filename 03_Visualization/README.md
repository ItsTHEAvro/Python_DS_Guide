# Data Visualization with Python

## Learning Objectives

By the end of this section, you will:

- Master fundamental visualization techniques using Matplotlib
- Create statistical visualizations with Seaborn
- Learn best practices for creating effective data visualizations
- Understand how to choose the right visualization for different types of data
- Develop skills in customizing and styling visualizations for clear communication

## Key Topics Covered

### 1. Introduction to Matplotlib

- The Matplotlib object hierarchy
- Figure and Axes objects
- Basic plot types
- Customizing plot appearance
- Saving and exporting visualizations

```python
import matplotlib.pyplot as plt
import numpy as np

# Create data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# First subplot
ax1.plot(x, y1, 'b-', label='Sine')
ax1.set_title('Sine Function')
ax1.set_xlabel('x')
ax1.set_ylabel('sin(x)')
ax1.grid(True)
ax1.legend()

# Second subplot
ax2.plot(x, y2, 'r-', label='Cosine')
ax2.set_title('Cosine Function')
ax2.set_xlabel('x')
ax2.set_ylabel('cos(x)')
ax2.grid(True)
ax2.legend()

# Adjust layout and title
plt.tight_layout()
fig.suptitle('Trigonometric Functions', fontsize=16, y=1.05)
plt.show()

# Save figure
fig.savefig('trig_functions.png', dpi=300, bbox_inches='tight')
```

**Mini Practice Task:** Create a figure with two subplots side by side. In the first subplot, show a quadratic function (y = x²). In the second, show an exponential function (y = e^x). Add appropriate titles, labels, and legends to each plot.

### 2. Plot Types in Matplotlib

- Line plots and scatter plots
- Bar charts and histograms
- Pie charts
- Box plots and violin plots
- Heatmaps
- 3D plotting basics

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sample data
categories = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E']
values = [24, 17, 32, 15, 22]
data = pd.DataFrame({
    'category': categories,
    'value': values
})

# Create a figure with multiple plot types
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Bar chart (top left)
axs[0, 0].bar(categories, values, color='skyblue')
axs[0, 0].set_title('Bar Chart')
axs[0, 0].set_xlabel('Category')
axs[0, 0].set_ylabel('Value')
axs[0, 0].tick_params(axis='x', rotation=45)

# Pie chart (top right)
axs[0, 1].pie(values, labels=categories, autopct='%1.1f%%',
              startangle=90, shadow=True)
axs[0, 1].set_title('Pie Chart')
axs[0, 1].axis('equal')  # Equal aspect ratio ensures the pie is circular

# Histogram (bottom left)
random_data = np.random.normal(100, 15, 200)
axs[1, 0].hist(random_data, bins=20, color='lightgreen', edgecolor='black')
axs[1, 0].set_title('Histogram')
axs[1, 0].set_xlabel('Value')
axs[1, 0].set_ylabel('Frequency')

# Scatter plot (bottom right)
x = np.random.rand(50) * 10
y = x + np.random.randn(50) * 2
axs[1, 1].scatter(x, y, color='coral', alpha=0.7)
axs[1, 1].set_title('Scatter Plot')
axs[1, 1].set_xlabel('X Value')
axs[1, 1].set_ylabel('Y Value')

plt.tight_layout()
plt.show()
```

**Mini Practice Task:** Create a visualization that includes a histogram and a box plot of the same dataset (you can use numpy's random functions to create sample data). Compare what insights each visualization provides.

### 3. Statistical Visualization with Seaborn

- Seaborn themes and styles
- Distribution plots
- Categorical plots
- Regression plots
- Matrix plots
- Multi-plot grids

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set the Seaborn theme
sns.set_theme(style="whitegrid")

# Generate some sample data
tips = sns.load_dataset('tips')

# Create a figure with multiple Seaborn plots
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Distribution plot (top left)
sns.histplot(tips['total_bill'], kde=True, ax=axs[0, 0])
axs[0, 0].set_title('Distribution of Bill Amounts')

# Categorical plot (top right)
sns.boxplot(x='day', y='total_bill', data=tips, ax=axs[0, 1])
axs[0, 1].set_title('Bill Amount by Day')

# Regression plot (bottom left)
sns.regplot(x='total_bill', y='tip', data=tips, ax=axs[1, 0])
axs[1, 0].set_title('Tip vs. Total Bill')

# Complex categorical plot (bottom right)
sns.violinplot(x='day', y='total_bill', hue='sex', split=True, data=tips, ax=axs[1, 1])
axs[1, 1].set_title('Bill Distribution by Day and Gender')

plt.tight_layout()
plt.show()

# FacetGrid example
g = sns.FacetGrid(tips, col="time", row="sex", height=4)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip")
g.add_legend()
plt.show()
```

**Mini Practice Task:** Load one of Seaborn's sample datasets (like 'iris', 'planets', or 'diamonds') and create a pair plot to explore relationships between variables. Add a custom color palette and adjust the appearance to improve readability.

### 4. Customizing and Styling Visualizations

- Color palettes and colormaps
- Plot styles and themes
- Annotations and text
- Legends and colorbars
- Custom layouts and grids
- Adjusting aesthetics for publication

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set a modern style
sns.set_style("darkgrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Create sample data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=60)
data = pd.DataFrame({
    'date': dates,
    'metric_a': np.random.normal(100, 15, 60).cumsum(),
    'metric_b': np.random.normal(50, 10, 60).cumsum(),
    'metric_c': np.random.normal(75, 20, 60).cumsum()
})

# Create a styled figure
fig, ax = plt.subplots(figsize=(12, 7))

# Plot lines with custom colors and styles
sns.lineplot(x='date', y='metric_a', data=data, linewidth=2.5,
             color='#1f77b4', marker='o', markersize=6, label='Metric A')
sns.lineplot(x='date', y='metric_b', data=data, linewidth=2.5,
             color='#ff7f0e', marker='s', markersize=6, label='Metric B')
sns.lineplot(x='date', y='metric_c', data=data, linewidth=2.5,
             color='#2ca02c', marker='^', markersize=6, label='Metric C')

# Add annotations
plt.annotate('Key Event', xy=(dates[30], data['metric_a'][30]),
             xytext=(dates[35], data['metric_a'][30] + 50),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
             fontsize=12)

# Customize grid, labels, and title
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Cumulative Value', fontsize=14)
ax.set_title('Trend Analysis of Key Metrics', fontsize=16, fontweight='bold', pad=20)

# Customize legend
ax.legend(fontsize=12, frameon=True, fancybox=True, framealpha=0.7,
          loc='upper left', title='Metrics', title_fontsize=13)

# Customize axes appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=12)

# Add a text box with insights
textbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.05, 'Metrics A and C show strong positive correlation.\nMetric B shows more volatility.',
        transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=textbox_props)

# Format date ticks
fig.autofmt_xdate(rotation=45)

plt.tight_layout()
plt.show()
```

**Mini Practice Task:** Choose a simple dataset and create a highly customized visualization that emphasizes a key insight. Include custom colors, annotations, a styled legend, and remove unnecessary chart elements for clarity.

### 5. Data Communication and Storytelling

- Principles of effective data visualization
- Choosing the right visualization
- Creating visualization narratives
- Avoiding common visualization pitfalls
- Audience-centered design

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set the visual style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'

# Create storytelling data - company quarterly revenue
quarters = ['Q1 2022', 'Q2 2022', 'Q3 2022', 'Q4 2022', 'Q1 2023', 'Q2 2023']
revenue = np.array([3.2, 3.5, 3.8, 4.3, 4.0, 4.8])
costs = np.array([2.5, 2.7, 2.8, 2.9, 3.0, 3.3])
profit = revenue - costs

# Create a multi-part narrative visualization
fig, axs = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [2, 1]})

# Part 1: Revenue vs Costs stacked bar chart
width = 0.35
x = np.arange(len(quarters))
axs[0].bar(x, costs, width, label='Costs', color='#ff9999')
axs[0].bar(x, profit, width, bottom=costs, label='Profit', color='#66b3ff')
axs[0].set_title('Quarterly Revenue Breakdown', fontsize=16, pad=20)
axs[0].set_ylabel('Amount (in millions $)', fontsize=14)
axs[0].set_xticks(x)
axs[0].set_xticklabels(quarters, fontsize=12)
axs[0].legend(fontsize=12, loc='upper left')

# Add annotations on key points
axs[0].annotate('Record Profit', xy=(5, revenue[5] - 0.2),
                xytext=(5, revenue[5] + 0.5),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12, ha='center')

# Add text explaining the trend
axs[0].text(0.02, 0.95, 'Revenue and profit showing steady growth across quarters',
            transform=axs[0].transAxes, fontsize=12, va='top')

# Part 2: Profit margin line chart
profit_margin = (profit / revenue) * 100
axs[1].plot(x, profit_margin, 'o-', linewidth=3, color='#2ca02c')
axs[1].set_title('Profit Margin Percentage', fontsize=16, pad=20)
axs[1].set_ylabel('Profit Margin (%)', fontsize=14)
axs[1].set_xlabel('Quarter', fontsize=14)
axs[1].set_xticks(x)
axs[1].set_xticklabels(quarters, fontsize=12)
axs[1].grid(True, linestyle='--', alpha=0.7)

# Highlight best and worst quarters for profit margin
best_idx = np.argmax(profit_margin)
worst_idx = np.argmin(profit_margin)
axs[1].annotate(f'Highest: {profit_margin[best_idx]:.1f}%',
                xy=(best_idx, profit_margin[best_idx]),
                xytext=(best_idx-0.2, profit_margin[best_idx]+3),
                arrowprops=dict(facecolor='green', shrink=0.05),
                fontsize=12)
axs[1].annotate(f'Lowest: {profit_margin[worst_idx]:.1f}%',
                xy=(worst_idx, profit_margin[worst_idx]),
                xytext=(worst_idx+0.2, profit_margin[worst_idx]-3),
                arrowprops=dict(facecolor='red', shrink=0.05),
                fontsize=12)

# Add an overall title with insights
fig.suptitle('Company Financial Performance Analysis', fontsize=18, fontweight='bold', y=0.98)
fig.text(0.5, 0.925, 'Despite increasing costs, both revenue and profit margins have improved year-over-year',
         ha='center', fontsize=14, style='italic')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
```

**Mini Practice Task:** Design a visualization that tells a clear story about a dataset of your choice. Include an explicit headline that conveys the main insight, visual cues that direct attention to important points, and explanatory text that helps interpret the data.

## Resources for Further Learning

- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/tutorial.html)
- [Data Visualization: A Practical Introduction](https://socviz.co/) by Kieran Healy
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) by Claus O. Wilke

## Next Steps

After mastering data visualization techniques, move on to [Statistics](../04_Statistics/README.md) to develop a strong foundation in statistical analysis of your data.
