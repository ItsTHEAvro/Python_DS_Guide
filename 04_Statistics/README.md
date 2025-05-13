# Statistics in Python

## Learning Objectives

By the end of this section, you will:

- Understand fundamental statistical concepts and their implementation in Python
- Apply descriptive and inferential statistics to analyze data
- Master probability distributions and their applications
- Design and analyze experiments using statistical methods
- Conduct hypothesis testing and make data-driven decisions

## Key Topics Covered

### 1. Descriptive Statistics

- Measures of central tendency
- Measures of dispersion
- Quartiles and percentiles
- Skewness and kurtosis
- Correlation analysis

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data
np.random.seed(42)
data = np.random.normal(100, 15, 1000)

# Basic descriptive statistics
print(f"Mean: {np.mean(data):.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"Std Dev: {np.std(data, ddof=1):.2f}")
print(f"Min: {np.min(data):.2f}")
print(f"Max: {np.max(data):.2f}")

# Using pandas for descriptive stats
df = pd.DataFrame({'values': data})
print("\nPandas Description:")
print(df.describe())

# Quartiles and percentiles
q1, q2, q3 = np.percentile(data, [25, 50, 75])
print(f"\nQuartiles: Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}")
print(f"IQR: {q3 - q1:.2f}")

# Skewness and kurtosis
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)
print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurtosis:.4f}")

# Visualizing the distribution
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(np.mean(data), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(data):.2f}')
plt.axvline(np.median(data), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(data):.2f}')
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
sns.boxplot(x=data)
plt.title('Box Plot')
plt.xlabel('Value')

plt.tight_layout()
plt.show()
```

**Mini Practice Task:** Generate two different datasets: one with a symmetric distribution and one with a skewed distribution. Calculate and compare their descriptive statistics (mean, median, standard deviation, skewness, and kurtosis). Visualize both distributions using histograms and box plots.

### 2. Probability and Random Variables

- Probability concepts
- Random variables and distributions
- Common probability distributions
- Sampling distributions
- Central Limit Theorem

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Set up the figure
plt.figure(figsize=(15, 12))

# 1. Normal Distribution
x = np.linspace(-4, 4, 1000)
plt.subplot(2, 2, 1)
plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', lw=2, label='Standard Normal')
plt.fill_between(x, stats.norm.pdf(x, 0, 1), where=(x>-2)&(x<2), alpha=0.3)
plt.title('Normal Distribution\n(68% of data within 1 std dev)', fontsize=12)
plt.xlabel('z-score')
plt.ylabel('Probability Density')
plt.legend()

# 2. Binomial Distribution
n, p = 20, 0.3
x = np.arange(0, n+1)
plt.subplot(2, 2, 2)
plt.bar(x, stats.binom.pmf(x, n, p), alpha=0.7, color='skyblue', edgecolor='black')
plt.title(f'Binomial Distribution\n(n={n}, p={p})', fontsize=12)
plt.xlabel('Number of Successes')
plt.ylabel('Probability')

# 3. Poisson Distribution
lam = 5
x = np.arange(0, 15)
plt.subplot(2, 2, 3)
plt.bar(x, stats.poisson.pmf(x, lam), alpha=0.7, color='lightgreen', edgecolor='black')
plt.title(f'Poisson Distribution\n(λ={lam})', fontsize=12)
plt.xlabel('Number of Events')
plt.ylabel('Probability')

# 4. Central Limit Theorem Demonstration
sample_means = []
# Take 1000 samples of size 30 from an exponential distribution
# and calculate the mean of each sample
for _ in range(1000):
    sample = np.random.exponential(size=30)
    sample_means.append(np.mean(sample))

plt.subplot(2, 2, 4)
sns.histplot(sample_means, kde=True, color='purple', alpha=0.6)
plt.axvline(np.mean(sample_means), color='red', linestyle='dashed',
            linewidth=2, label=f'Mean: {np.mean(sample_means):.2f}')
plt.title('Central Limit Theorem\nSampling Distribution of Means', fontsize=12)
plt.xlabel('Sample Means')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()
```

**Mini Practice Task:** Generate random samples from three different distributions (Normal, Uniform, and Exponential). For each distribution, create 500 samples of size 50, compute the mean of each sample, and plot the distribution of these means. Observe how the Central Limit Theorem applies regardless of the original distribution.

### 3. Statistical Inference and Hypothesis Testing

- Point and interval estimation
- Confidence intervals
- Hypothesis testing fundamentals
- One-sample and two-sample tests
- ANOVA

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
np.random.seed(42)
group1 = np.random.normal(100, 10, 30)  # Control group
group2 = np.random.normal(105, 10, 30)  # Treatment group

# 1. Confidence Interval for a Mean
mean = np.mean(group1)
std_error = stats.sem(group1)
confidence = 0.95
degrees_freedom = len(group1) - 1

conf_interval = stats.t.interval(confidence, degrees_freedom, mean, std_error)
print(f"95% Confidence interval for Group 1 mean: {conf_interval}")

# 2. One sample t-test
# Testing if Group 1's mean is different from 95
tstat, pval = stats.ttest_1samp(group1, 95)
print(f"\nOne-sample t-test for Group 1 vs 95:")
print(f"t-statistic: {tstat:.4f}, p-value: {pval:.4f}")
print(f"Result: {'Reject H₀' if pval < 0.05 else 'Fail to reject H₀'}")

# 3. Two-sample t-test
# Testing if Group 1 and Group 2 have different means
tstat, pval = stats.ttest_ind(group1, group2, equal_var=True)
print(f"\nTwo-sample t-test for Group 1 vs Group 2:")
print(f"t-statistic: {tstat:.4f}, p-value: {pval:.4f}")
print(f"Result: {'Reject H₀' if pval < 0.05 else 'Fail to reject H₀'}")

# 4. ANOVA example
# Create three groups for ANOVA
group3 = np.random.normal(110, 10, 30)
all_data = [group1, group2, group3]
group_labels = ['Control', 'Treatment A', 'Treatment B']

# Perform ANOVA
fstat, pval = stats.f_oneway(group1, group2, group3)
print(f"\nANOVA for all three groups:")
print(f"F-statistic: {fstat:.4f}, p-value: {pval:.4f}")
print(f"Result: {'Reject H₀' if pval < 0.05 else 'Fail to reject H₀'}")

# Visualize the data for comparison
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.boxplot(all_data, labels=group_labels)
plt.title('Group Comparisons')
plt.ylabel('Values')

plt.subplot(1, 2, 2)
# Create a combined dataframe for easy plotting
combined_data = []
for i, group in enumerate(all_data):
    for value in group:
        combined_data.append({'Group': group_labels[i], 'Value': value})
df = pd.DataFrame(combined_data)

# Plot with Seaborn
sns.violinplot(x='Group', y='Value', data=df, inner='quartile')
plt.title('Violin Plot of Groups')

plt.tight_layout()
plt.show()
```

**Mini Practice Task:** Generate three datasets representing different treatment groups. Perform the following analyses:

1. Calculate 95% confidence intervals for each group.
2. Test if each group's mean is significantly different from a hypothesized value.
3. Perform pairwise t-tests between groups.
4. Conduct an ANOVA to test if there's a significant difference among all groups.

### 4. Correlation and Regression Analysis

- Correlation coefficients (Pearson, Spearman)
- Simple linear regression
- Multiple regression
- Model evaluation and diagnostics
- Non-linear relationships

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Generate correlated data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 2 * x + 3 + np.random.normal(0, 1.5, len(x))  # y = 2x + 3 + noise

# Create a DataFrame
df = pd.DataFrame({'x': x, 'y': y})

# 1. Calculate correlation coefficient
pearson_corr, p_value = stats.pearsonr(df['x'], df['y'])
print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {p_value:.4f})")

spearman_corr, p_value = stats.spearmanr(df['x'], df['y'])
print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {p_value:.4f})")

# 2. Simple Linear Regression
X = df[['x']]  # Independent variable (features)
y = df['y']    # Dependent variable (target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Model evaluation
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f"\nLinear Regression Results:")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Training R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")

# Visualize the results
plt.figure(figsize=(12, 10))

# Scatterplot with regression line
plt.subplot(2, 2, 1)
sns.regplot(x='x', y='y', data=df, scatter_kws={'alpha':0.6},
            line_kws={'color':'red'})
plt.title(f'Linear Regression\ny = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
plt.xlabel('x')
plt.ylabel('y')

# Residual plot
plt.subplot(2, 2, 2)
residuals = y - model.predict(X)
sns.residplot(x=df['x'], y=residuals, lowess=True,
              line_kws={'color':'red', 'lw':1})
plt.title('Residual Plot')
plt.xlabel('x')
plt.ylabel('Residuals')
plt.axhline(y=0, color='grey', linestyle='--')

# Distribution of residuals
plt.subplot(2, 2, 3)
sns.histplot(residuals, kde=True, color='skyblue')
plt.title('Distribution of Residuals')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')

# Actual vs. Predicted
plt.subplot(2, 2, 4)
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Actual vs. Predicted')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()

# 3. Multiple Regression Example
# Adding another variable
df['x2'] = x**2 * 0.1 + np.random.normal(0, 1, len(x))

# Correlation matrix
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Fit multiple regression
X_multi = df[['x', 'x2']]
X_train, X_test, y_train, y_test = train_test_split(X_multi, y, test_size=0.3, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Evaluate multiple regression
y_pred_multi = model_multi.predict(X_test)
r2_multi = r2_score(y_test, y_pred_multi)
mse_multi = mean_squared_error(y_test, y_pred_multi)

print("\nMultiple Regression Results:")
print(f"Coefficients: {model_multi.coef_}")
print(f"Intercept: {model_multi.intercept_:.4f}")
print(f"Test R² Score: {r2_multi:.4f}")
print(f"Test MSE: {mse_multi:.4f}")
```

**Mini Practice Task:** Create a dataset with one dependent variable and three independent variables, with varying degrees of correlation. Fit simple linear regression models for each predictor separately, and then a multiple regression model with all predictors. Compare the models' performance and interpret the results.

### 5. Sampling Methods and Experimental Design

- Sampling techniques
- Sample size determination
- Design of experiments
- Factorial designs
- Randomization and blocking
- Bias and variance

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Create a population
np.random.seed(42)
population_size = 10000
population = np.random.normal(50, 15, population_size)

# Add some skew to make it more realistic
population = np.exp(np.random.normal(3.5, 0.4, population_size))

# True population parameters
pop_mean = np.mean(population)
pop_median = np.median(population)
pop_std = np.std(population)

print(f"Population Mean: {pop_mean:.2f}")
print(f"Population Median: {pop_median:.2f}")
print(f"Population Std Dev: {pop_std:.2f}")

# 1. Simple Random Sampling
def sample_and_estimate(sample_size, num_samples=1000):
    """Draw multiple samples and return means and confidence intervals"""
    sample_means = []
    confidence_intervals = []

    for _ in range(num_samples):
        # Draw a random sample
        sample = np.random.choice(population, size=sample_size, replace=False)

        # Calculate sample mean
        sample_mean = np.mean(sample)
        sample_means.append(sample_mean)

        # Calculate 95% confidence interval
        ci = stats.norm.interval(0.95, loc=sample_mean,
                                scale=np.std(sample)/np.sqrt(sample_size))
        confidence_intervals.append(ci)

    return sample_means, confidence_intervals

# Compare different sample sizes
sample_sizes = [30, 100, 500]
all_sample_means = []
all_cis = []

plt.figure(figsize=(15, 10))

for i, size in enumerate(sample_sizes):
    means, cis = sample_and_estimate(size)
    all_sample_means.append(means)
    all_cis.append(cis)

    # Plot distribution of sample means
    plt.subplot(2, 2, i+1)
    sns.histplot(means, kde=True, color=f'C{i}')
    plt.axvline(pop_mean, color='red', linestyle='--',
                label=f'Pop Mean: {pop_mean:.2f}')
    plt.title(f'Distribution of Sample Means\n(n={size}, {len(means)} samples)')
    plt.xlabel('Sample Mean')
    plt.ylabel('Frequency')
    plt.legend()

# Calculate coverage probability (% of CIs that contain true mean)
coverage_probs = []
for size, cis in zip(sample_sizes, all_cis):
    contains_mean = [ci[0] <= pop_mean <= ci[1] for ci in cis]
    coverage_prob = np.mean(contains_mean)
    coverage_probs.append(coverage_prob)
    print(f"Sample size {size}: {coverage_prob:.2%} of CIs contain the true mean")

# Plot sample size vs. standard error
std_errors = [np.std(means) for means in all_sample_means]

plt.subplot(2, 2, 4)
plt.plot(sample_sizes, std_errors, 'o-', color='purple')
plt.title('Sample Size vs. Standard Error')
plt.xlabel('Sample Size')
plt.ylabel('Standard Error of Mean')
plt.xscale('log')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 2. Demonstrating Different Sampling Techniques with a Stratified Population

# Create a stratified population
strata_means = [30, 50, 70]
strata_stds = [5, 8, 10]
strata_sizes = [3000, 5000, 2000]  # 30%, 50%, 20%
strata_labels = ['Group A', 'Group B', 'Group C']

stratified_pop = []
strata_indx = []

for i in range(len(strata_means)):
    stratum = np.random.normal(strata_means[i], strata_stds[i], strata_sizes[i])
    stratified_pop.extend(stratum)
    strata_indx.extend([i] * strata_sizes[i])

stratified_pop = np.array(stratified_pop)
strata_indx = np.array(strata_indx)

# Create a DataFrame
df = pd.DataFrame({
    'value': stratified_pop,
    'stratum': [strata_labels[i] for i in strata_indx]
})

total_size = len(stratified_pop)
overall_mean = np.mean(stratified_pop)

print(f"\nStratified Population:")
print(f"Overall Mean: {overall_mean:.2f}")
for i in range(len(strata_means)):
    print(f"{strata_labels[i]}: {np.mean(df[df['stratum'] == strata_labels[i]]['value']):.2f} " +
          f"(proportion: {strata_sizes[i]/total_size:.2f})")

# Visualize the stratified population
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df, x='value', hue='stratum', kde=True, element='step')
plt.title('Distribution by Stratum')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.boxplot(x='stratum', y='value', data=df)
plt.title('Box Plots by Stratum')
plt.xlabel('Stratum')
plt.ylabel('Value')

plt.tight_layout()
plt.show()

# Compare sampling techniques
sample_size = 300

# Simple Random Sample
srs_indices = np.random.choice(range(total_size), size=sample_size, replace=False)
srs_sample = df.iloc[srs_indices]
srs_mean = np.mean(srs_sample['value'])

# Stratified Sample
strat_sample = pd.DataFrame()
for stratum in strata_labels:
    stratum_data = df[df['stratum'] == stratum]
    stratum_prop = len(stratum_data) / total_size
    stratum_sample_size = int(sample_size * stratum_prop)
    stratum_sample = stratum_data.sample(stratum_sample_size, random_state=42)
    strat_sample = pd.concat([strat_sample, stratum_sample])
strat_mean = np.mean(strat_sample['value'])

# Cluster Sample (simulate by selecting whole "blocks")
# For demonstration, create 20 clusters and select 6 randomly
num_clusters = 20
cluster_size = total_size // num_clusters
cluster_ids = np.repeat(range(num_clusters), cluster_size)
df['cluster'] = cluster_ids[:total_size]  # In case of rounding issues

selected_clusters = np.random.choice(range(num_clusters), size=6, replace=False)
cluster_sample = df[df['cluster'].isin(selected_clusters)]
cluster_mean = np.mean(cluster_sample['value'])

# Systematic Sample
step = total_size // sample_size
start = np.random.randint(0, step)
sys_indices = np.arange(start, total_size, step)
sys_sample = df.iloc[sys_indices]
sys_mean = np.mean(sys_sample['value'])

# Print results
print("\nSampling Techniques Comparison (sample size: 300):")
print(f"Population Mean: {overall_mean:.4f}")
print(f"Simple Random Sample Mean: {srs_mean:.4f} (Error: {abs(srs_mean - overall_mean):.4f})")
print(f"Stratified Sample Mean: {strat_mean:.4f} (Error: {abs(strat_mean - overall_mean):.4f})")
print(f"Cluster Sample Mean: {cluster_mean:.4f} (Error: {abs(cluster_mean - overall_mean):.4f})")
print(f"Systematic Sample Mean: {sys_mean:.4f} (Error: {abs(sys_mean - overall_mean):.4f})")
```

**Mini Practice Task:** Create a simulated population with distinct subgroups. Compare the accuracy and precision of estimates from different sampling techniques: simple random sampling, stratified sampling, cluster sampling, and systematic sampling. Evaluate which technique works best for your simulated population.

## Resources for Further Learning

- [Think Stats](https://greenteapress.com/wp/think-stats-2e/) by Allen B. Downey
- [Statistical Inference in Python](https://inferentialthinking.com/) by Ani Adhikari & John DeNero
- [An Introduction to Statistical Learning](https://www.statlearning.com/) by James, Witten, Hastie, Tibshirani
- [SciPy Documentation - Statistics Module](https://docs.scipy.org/doc/scipy/reference/stats.html)

## Next Steps

After mastering statistics fundamentals, move on to [Machine Learning](../05_Machine_Learning/README.md) to learn how to apply these statistical concepts to building predictive models.
