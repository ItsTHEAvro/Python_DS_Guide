# Antarctic Penguin Species Clustering

## Project Overview

This project explores how unsupervised machine learning techniques can be used to cluster penguin data based on physical characteristics. You'll apply dimensionality reduction and clustering algorithms to analyze a dataset of penguin measurements from three Antarctic species, and compare the algorithm-derived clusters with the actual taxonomic classifications.

## Learning Objectives

- Apply unsupervised learning techniques to biological data
- Implement and evaluate clustering algorithms
- Visualize high-dimensional data using dimensionality reduction
- Validate clustering results against known classifications
- Interpret cluster characteristics in a biological context

## Key Topics Covered

- Data preprocessing for clustering
- Dimensionality reduction with PCA
- K-means, hierarchical, and DBSCAN clustering
- Cluster validation and evaluation metrics
- Visualization of clusters in reduced dimensions

## Setup Instructions

1. **Environment Setup:**

   - Make sure Python 3.8+ is installed on your system
   - Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy
   - Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn scipy`

2. **Dataset:**

   - The `penguins.csv` file should be in the same directory as your notebook
   - No additional data processing is required before starting

3. **Getting Started:**
   - Create a new Jupyter notebook in this directory
   - Follow the project tasks outlined below
   - Execute each code cell and analyze the results

## Dataset Description

The `penguins.csv` file contains measurements of Antarctic penguins collected from three species (Adelie, Chinstrap, and Gentoo) with the following columns:

- `species`: Penguin species (Adelie, Chinstrap, Gentoo)
- `island`: Island where the penguin was observed (Dream, Biscoe, Torgersen)
- `bill_length_mm`: Bill length in millimeters
- `bill_depth_mm`: Bill depth in millimeters
- `flipper_length_mm`: Flipper length in millimeters
- `body_mass_g`: Body mass in grams
- `sex`: Sex of the penguin (male, female)
- `year`: Year of observation

## Project Tasks

### 1. Data Loading and Initial Exploration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.manifold import TSNE

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

# Load the dataset
penguins = pd.read_csv('penguins.csv')

# Initial exploration
print(f"Dataset shape: {penguins.shape}")
print(penguins.info())

# Summary statistics
print(penguins.describe())

# Check for missing values
print("\nMissing values per column:")
print(penguins.isna().sum())

# Display a few rows to understand the data
print("\nSample data:")
print(penguins.head())

# Check species distribution
species_counts = penguins['species'].value_counts()
print("\nSpecies distribution:")
print(species_counts)
```

### 2. Data Cleaning and Preparation

```python
# Create a clean copy of the dataset
penguins_clean = penguins.copy()

# Handle missing values (if any)
penguins_clean = penguins_clean.dropna()

# Select features for clustering
feature_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = penguins_clean[feature_cols]

# Store the actual species labels for later comparison
species_labels = penguins_clean['species']

# Map species to numeric values for comparison metrics
species_map = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
y_true = penguins_clean['species'].map(species_map)

# Standardize features (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a DataFrame with scaled features for easier plotting
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

# Print shape of processed data
print(f"Processed dataset shape: {X_scaled_df.shape}")
```

### 3. Exploratory Data Visualization

```python
# Create pairplot to visualize relationships between features
plt.figure(figsize=(12, 10))
sns.pairplot(penguins_clean, hue='species', vars=feature_cols,
             plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k', 'linewidth': 0.5})
plt.suptitle('Pairwise Relationships between Penguin Features by Species',
             y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# Create boxplots to compare distributions across species
plt.figure(figsize=(16, 12))
for i, feature in enumerate(feature_cols):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=feature, data=penguins_clean)
    plt.title(f'{feature} by Species', fontsize=14)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Create a correlation heatmap
plt.figure(figsize=(10, 8))
correlation = X.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Penguin Features', fontsize=16)
plt.tight_layout()
plt.show()
```

### 4. Dimensionality Reduction with PCA

```python
# Apply PCA for dimensionality reduction
pca = PCA()
pca_result = pca.fit_transform(X_scaled)

# Print explained variance ratio
print("Explained variance ratio by principal component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")
print(f"Total explained variance with 2 components: {pca.explained_variance_ratio_[:2].sum()*100:.2f}%")

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_)
plt.ylabel('Explained Variance Ratio', fontsize=12)
plt.xlabel('Principal Component', fontsize=12)
plt.title('Explained Variance by Principal Component', fontsize=16)
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Visualize data in the first two principal components
plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1],
                      c=y_true, cmap='viridis',
                      s=100, alpha=0.8, edgecolor='k')

plt.title('PCA of Penguin Data Colored by Species', fontsize=16)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
plt.grid(True, alpha=0.3)

# Create a legend with species names
species_names = list(species_map.keys())
handles, _ = scatter.legend_elements()
plt.legend(handles, species_names, title="Species")

# Plot component loadings (feature importance in each PC)
plt.figure(figsize=(12, 8))
loading_vectors = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3', 'PC4'],
                               index=feature_cols)
sns.heatmap(loading_vectors, annot=True, cmap='coolwarm', cbar=True)
plt.title('Feature Loadings for Principal Components', fontsize=16)
plt.tight_layout()
plt.show()
```

### 5. K-means Clustering

```python
# Determine optimal number of clusters using the elbow method
inertia = []
silhouette_scores = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

    # Calculate silhouette score (skip k=1)
    if k > 1:
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot elbow method
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'o-', markersize=8)
plt.title('Elbow Method for K-means', fontsize=16)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(k_range)

# Plot silhouette scores
plt.subplot(1, 2, 2)
plt.plot(list(k_range)[1:], silhouette_scores, 'o-', markersize=8)
plt.title('Silhouette Score for K-means', fontsize=16)
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(list(k_range)[1:])

plt.tight_layout()
plt.show()

# Apply K-means with the optimal k (in this case, likely k=3 based on domain knowledge)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original data
penguins_clean['kmeans_cluster'] = kmeans_labels

# Visualize clusters in PCA space
plt.figure(figsize=(14, 6))

# Plot clusters
plt.subplot(1, 2, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels,
            cmap='viridis', s=100, alpha=0.8, edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='X', s=200, c='red', label='Centroids')
plt.title('K-means Clusters in PCA Space', fontsize=16)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Plot actual species
plt.subplot(1, 2, 2)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y_true,
            cmap='viridis', s=100, alpha=0.8, edgecolor='k')
plt.title('Actual Penguin Species in PCA Space', fontsize=16)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Evaluate clustering performance
ari = adjusted_rand_score(y_true, kmeans_labels)
print(f"Adjusted Rand Index (ARI): {ari:.4f} (1.0 is perfect match)")

# Compare cluster assignments with true species labels
cluster_species_crosstab = pd.crosstab(penguins_clean['kmeans_cluster'],
                                        penguins_clean['species'],
                                        normalize='index') * 100

print("\nCluster composition (% of each species within clusters):")
print(cluster_species_crosstab)

# Visualize cluster composition
plt.figure(figsize=(10, 6))
cluster_species_crosstab.plot(kind='bar', stacked=True)
plt.title('Species Composition of Each Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.legend(title='Species')
plt.tight_layout()
plt.show()
```

### 6. Hierarchical Clustering

```python
# Compute linkage matrix for hierarchical clustering
Z = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(16, 10))
plt.title('Hierarchical Clustering Dendrogram', fontsize=16)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Distance', fontsize=12)
dendrogram(
    Z,
    leaf_rotation=90.,
    leaf_font_size=10.,
)
plt.axhline(y=5, color='r', linestyle='--')  # Draw a line where we might cut the tree
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Cut the dendrogram to get cluster labels
from scipy.cluster.hierarchy import fcluster
hierarchical_labels = fcluster(Z, t=3, criterion='maxclust')  # Cut for 3 clusters

# Add hierarchical cluster labels to the original data
penguins_clean['hierarchical_cluster'] = hierarchical_labels

# Visualize hierarchical clusters in PCA space
plt.figure(figsize=(14, 6))

# Plot hierarchical clusters
plt.subplot(1, 2, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=hierarchical_labels,
            cmap='viridis', s=100, alpha=0.8, edgecolor='k')
plt.title('Hierarchical Clusters in PCA Space', fontsize=16)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
plt.grid(True, alpha=0.3)

# Plot actual species again for comparison
plt.subplot(1, 2, 2)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y_true,
            cmap='viridis', s=100, alpha=0.8, edgecolor='k')
plt.title('Actual Penguin Species in PCA Space', fontsize=16)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Mini Practice Task

After completing the guided analysis, explore these additional questions:

1. Use t-SNE (t-Distributed Stochastic Neighbor Embedding) as an alternative dimensionality reduction technique and visualize the clusters. Compare the results with PCA.

2. Try different distance metrics (Euclidean, Manhattan, cosine) for hierarchical clustering and evaluate which one produces clusters that best match the actual penguin species.

3. Implement Gaussian Mixture Models (GMM) as another clustering approach and compare its performance with K-means, hierarchical, and DBSCAN.

## Conclusion

By the end of this project, you'll have gained hands-on experience applying unsupervised learning techniques to biological data. You'll understand how different clustering algorithms perform on real-world data and how to validate and interpret the results. You'll also appreciate how machine learning can help identify natural groupings in biological measurements that correspond to taxonomic classifications.

## Next Steps

Consider extending your analysis by:

1. **Advanced Visualization**: Create an interactive dashboard using Plotly or Bokeh to explore the clustering results dynamically.

2. **Feature Engineering**: Derive new features from existing measurements (e.g., bill shape ratio) and analyze their impact on clustering performance.

3. **Additional Algorithms**: Experiment with Spectral Clustering or OPTICS algorithms to see if they better capture the natural penguin groupings.

4. **Environmental Analysis**: Analyze how environmental factors (like island location) influence the clustering results.

5. **Hybrid Approach**: Build a semi-supervised classification model using the discovered clusters as features.

## References

- Palmer Penguins Dataset: https://allisonhorst.github.io/palmerpenguins/
- Scikit-learn Documentation: https://scikit-learn.org/stable/modules/clustering.html
- Clustering Algorithm Comparison: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
