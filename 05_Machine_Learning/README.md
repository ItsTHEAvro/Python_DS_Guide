# Machine Learning with Python

## Learning Objectives

By the end of this section, you will:

- Understand fundamental machine learning concepts and workflows
- Implement supervised and unsupervised learning algorithms
- Evaluate and improve machine learning models
- Apply feature engineering and selection techniques
- Develop skills in advanced modeling techniques like ensemble methods

## Key Topics Covered

### 1. Introduction to Machine Learning

- Types of machine learning
- The ML workflow
- Training and testing methodology
- Bias-variance tradeoff
- Model selection and evaluation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load sample data
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Basic data exploration
print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a simple model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names,
            yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

**Mini Practice Task:** Load a dataset of your choice (e.g., iris, wine, or digits from sklearn.datasets), split it into training and testing sets, train a classifier, and evaluate its performance. Experiment with different train/test split ratios to see how they affect model performance.

### 2. Supervised Learning: Classification

- Classification algorithms (Logistic Regression, KNN, SVM, Decision Trees)
- Multi-class classification
- Imbalanced classes
- Probability calibration
- Decision boundaries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.metrics import plot_confusion_matrix

# Generate synthetic data for classification
X, y = make_classification(
    n_samples=1000, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classifiers to compare
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(random_state=42, probability=True),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Train classifiers and collect results
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': clf,
        'accuracy': accuracy
    }
    print(f"{name} - Accuracy: {accuracy:.4f}")

# Visualize decision boundaries
def plot_decision_boundaries(X, y, models, names):
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))

    # Define mesh grid for plotting
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    for i, (name, model) in enumerate(zip(names, models)):
        if len(models) == 1:
            ax = axes
        else:
            ax = axes[i]

        # Plot the decision boundary
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3)

        # Plot the data points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')

    plt.tight_layout()
    plt.show()

# Plot decision boundaries for selected classifiers
selected_models = [
    results['Logistic Regression']['model'],
    results['K-Nearest Neighbors']['model'],
    results['Support Vector Machine']['model']
]
selected_names = ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine']

plot_decision_boundaries(X_test_scaled, y_test, selected_models, selected_names)

# ROC curve comparison
plt.figure(figsize=(10, 8))
for name, result in results.items():
    model = result['model']
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()
```

**Mini Practice Task:** Create a dataset with imbalanced classes (e.g., 90% vs. 10%). Train different classifiers on this dataset and compare their performance using appropriate metrics like precision, recall, F1-score, and ROC-AUC. Implement techniques for handling imbalanced data (e.g., class weights, SMOTE) and observe the improvement.

### 3. Supervised Learning: Regression

- Linear regression and variants
- Regularization techniques
- Polynomial regression
- Decision trees for regression
- Evaluation metrics for regression

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# Generate synthetic data with non-linear pattern
np.random.seed(42)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel() + 0.1 * np.random.randn(200)

# Add some outliers
y[::10] += 2 * (0.5 - np.random.rand(20))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define regression models
regressors = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Lasso Regression': Lasso(alpha=0.1, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    'Polynomial Regression (degree=3)': make_pipeline(
        PolynomialFeatures(degree=3), StandardScaler(), LinearRegression()
    ),
    'Decision Tree Regressor': DecisionTreeRegressor(max_depth=5, random_state=42)
}

# Train models and collect results
results = {}
for name, reg in regressors.items():
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {
        'model': reg,
        'mse': mse,
        'r2': r2
    }
    print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")

# Visualize predictions
plt.figure(figsize=(15, 10))
# Sort test data for smooth plotting
X_test_sorted_idx = np.argsort(X_test.ravel())
X_test_sorted = X_test[X_test_sorted_idx]
y_test_sorted = y_test[X_test_sorted_idx]

# Plot training and test data
plt.subplot(2, 1, 1)
plt.scatter(X_train, y_train, color='black', s=10, label='Training data')
plt.scatter(X_test, y_test, color='red', s=10, alpha=0.5, label='Test data')
plt.title('Regression Models Comparison')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Plot model predictions for selected models
plt.subplot(2, 1, 2)
plt.scatter(X_test, y_test, color='black', s=10, label='Test data')

colors = ['red', 'blue', 'green', 'orange']
selected_models = ['Linear Regression', 'Polynomial Regression (degree=3)',
                   'Ridge Regression', 'Decision Tree Regressor']

for i, name in enumerate(selected_models):
    model = results[name]['model']
    y_pred = model.predict(X_test_sorted)
    plt.plot(X_test_sorted, y_pred, color=colors[i], linewidth=2, label=name)

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.tight_layout()
plt.show()

# Compare model performance
metrics = ['mse', 'r2']
fig, axes = plt.subplots(1, len(metrics), figsize=(12, 5))

for i, metric in enumerate(metrics):
    values = [results[name][metric] for name in results.keys()]

    if metric == 'mse':
        title = 'Mean Squared Error (lower is better)'
    else:
        title = 'R² Score (higher is better)'

    axes[i].bar(range(len(results)), values, color='skyblue')
    axes[i].set_title(title)
    axes[i].set_xticks(range(len(results)))
    axes[i].set_xticklabels(results.keys(), rotation=45, ha='right')
    axes[i].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

**Mini Practice Task:** Create a dataset with a non-linear relationship between features and the target. Experiment with different regression techniques and feature transformations to find the best model. Use cross-validation to tune hyperparameters for the most promising models and compare their performance.

### 4. Model Evaluation and Validation

- Cross-validation techniques
- Hyperparameter tuning
- Learning curves
- Feature importance
- Model interpretation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Load data
data = load_diabetes()
X = data.data
y = data.target
feature_names = data.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Cross-validation with different models
models = {
    'Ridge Regression': Ridge(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

for name, model in models.items():
    # 5-fold cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'
    )
    # Convert negative MSE to positive for easier interpretation
    mse_scores = -cv_scores

    print(f"{name} - Cross-validation MSE: {mse_scores.mean():.2f} ± {mse_scores.std():.2f}")

# 2. Hyperparameter tuning with GridSearchCV
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=42))
])

param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5,
    scoring='neg_mean_squared_error', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print(f"Best cross-validation score: {-grid_search.best_score_:.2f} MSE")

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print(f"Test set MSE: {test_mse:.2f}")
print(f"Test set R²: {test_r2:.2f}")

# 3. Learning curves
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=5,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(8, 6))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, scoring='neg_mean_squared_error',
                       return_times=True)

    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    axes.legend(loc="best")

    return axes

plt.figure(figsize=(12, 6))
plot_learning_curve(
    best_model, 'Learning Curve (RandomForestRegressor)',
    X_train, y_train, cv=5
)
plt.show()

# 4. Feature Importance
rf_model = best_model.named_steps['regressor']
feature_importances = rf_model.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})
importance_df = importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# 5. Actual vs. Predicted
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
plt.tight_layout()
plt.show()
```

**Mini Practice Task:** Choose a regression or classification dataset and implement a complete model evaluation workflow:

1. Split the data into training and testing sets.
2. Implement 5-fold cross-validation to evaluate baseline performance.
3. Use GridSearchCV or RandomizedSearchCV to find the best hyperparameters.
4. Create learning curves to diagnose bias and variance.
5. Identify and interpret the most important features.

### 5. Unsupervised Learning

- Clustering algorithms (K-means, DBSCAN, hierarchical)
- Dimensionality reduction (PCA, t-SNE)
- Anomaly detection
- Topic modeling
- Association rules

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Set random seed for reproducibility
np.random.seed(42)

# 1. Generate synthetic clustering data
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.7, random_state=42)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# 3. DBSCAN Clustering
dbscan = DBSCAN(eps=0.3, min_samples=10)
y_dbscan = dbscan.fit_predict(X_scaled)

# 4. Hierarchical Clustering
agg_clust = AgglomerativeClustering(n_clusters=4)
y_agg = agg_clust.fit_predict(X_scaled)

# 5. Plot clustering results
plt.figure(figsize=(15, 5))

# Plot original clusters
plt.subplot(1, 3, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.8)
plt.title('Original Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot K-Means clustering results
plt.subplot(1, 3, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', s=50, alpha=0.8)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=100)
plt.title(f'K-Means Clustering\nSilhouette Score: {silhouette_score(X_scaled, y_kmeans):.3f}')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot DBSCAN results
plt.subplot(1, 3, 3)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_dbscan, cmap='viridis', s=50, alpha=0.8)
noise = y_dbscan == -1
plt.scatter(X_scaled[noise, 0], X_scaled[noise, 1], c='black', marker='x', s=50)
plt.title('DBSCAN Clustering\n(X marks noise points)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

# 6. Dimensionality Reduction Example
# Load digits dataset (64 features)
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# Standardize data
X_digits_scaled = StandardScaler().fit_transform(X_digits)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_digits_scaled)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_digits_scaled)

# Plot dimensionality reduction results
plt.figure(figsize=(16, 7))

# Plot PCA results
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_digits, cmap='viridis',
                      alpha=0.8, s=50)
plt.colorbar(scatter)
plt.title('PCA Projection of Digits Dataset')
plt.xlabel(f'PC1 (Variance Explained: {pca.explained_variance_ratio_[0]:.2f})')
plt.ylabel(f'PC2 (Variance Explained: {pca.explained_variance_ratio_[1]:.2f})')

# Plot t-SNE results
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits, cmap='viridis',
                     alpha=0.8, s=50)
plt.colorbar(scatter)
plt.title('t-SNE Projection of Digits Dataset')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')

plt.tight_layout()
plt.show()

# 7. Finding optimal number of clusters with Elbow method
inertia = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

plt.figure(figsize=(12, 5))

# Elbow method plot
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, 'o-', color='blue')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True, alpha=0.3)

# Silhouette score plot
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, 'o-', color='green')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Mini Practice Task:** Load a high-dimensional dataset and apply dimensionality reduction techniques (PCA, t-SNE) to visualize it. Then implement at least two different clustering algorithms and compare their results. Use silhouette scores and visualization to determine the optimal number of clusters.

### 6. Feature Engineering and Selection

- Feature extraction
- Feature transformation
- Feature selection methods
- Automated feature engineering
- Handling categorical variables

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load California housing dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

print("Dataset shape:", X.shape)
print("Feature names:", X.columns.tolist())

# Add some categorical features for demonstration
X['Ocean_Proximity'] = pd.cut(
    X['MedInc'],
    bins=[0, 2, 4, 6, 10, 20],
    labels=['<2', '2-4', '4-6', '6-10', '>10']
)
X['Size_Category'] = pd.cut(
    X['AveRooms'],
    bins=[0, 4, 6, 8, 20],
    labels=['Small', 'Medium', 'Large', 'Very Large']
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Basic Feature Engineering

# Separate numerical and categorical features
numerical_features = X.columns[:-2].tolist()
categorical_features = ['Ocean_Proximity', 'Size_Category']

print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)

# 2. Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# 3. Feature Selection Methods

# Filter method: SelectKBest
X_num = X_train[numerical_features].values
selector = SelectKBest(f_regression, k=5)
X_filtered = selector.fit_transform(X_num, y_train)

feature_scores = pd.DataFrame({
    'Feature': numerical_features,
    'Score': selector.scores_
})
feature_scores = feature_scores.sort_values('Score', ascending=False)

print("\nFeature selection scores (filter method):")
print(feature_scores)

# Wrapper method: Recursive Feature Elimination
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=5)
X_rfe = selector.fit_transform(X_num, y_train)

feature_ranking = pd.DataFrame({
    'Feature': numerical_features,
    'Ranking': selector.ranking_
})
feature_ranking = feature_ranking.sort_values('Ranking')

print("\nFeature ranking (RFE method - lower is better):")
print(feature_ranking)

# 4. Feature Engineering Pipeline
def evaluate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return scores.mean()

# Define different feature engineering strategies
feature_engineering_pipelines = {
    'Original Features': Pipeline([
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=1.0))
    ]),
    'Polynomial Features': Pipeline([
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('model', Ridge(alpha=1.0))
    ]),
    'SelectKBest': Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scale', StandardScaler()),
                ('select', SelectKBest(f_regression, k=5))
            ]), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])),
        ('model', Ridge(alpha=1.0))
    ]),
    'RFE': Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([
                ('scale', StandardScaler()),
                ('select', RFE(LinearRegression(), n_features_to_select=5))
            ]), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])),
        ('model', Ridge(alpha=1.0))
    ])
}

# Compare feature engineering strategies
results = {}
for name, pipeline in feature_engineering_pipelines.items():
    score = evaluate_model(pipeline, X_train, y_train)
    results[name] = score
    print(f"{name}: R² = {score:.4f}")

# 5. Feature Importance from Tree-Based Models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train[numerical_features], y_train)

feature_importances = pd.DataFrame({
    'Feature': numerical_features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature importances (Random Forest):")
print(feature_importances)

# Visualize results
plt.figure(figsize=(14, 10))

# Visualize feature selection scores
plt.subplot(2, 2, 1)
sns.barplot(x='Score', y='Feature', data=feature_scores)
plt.title('Feature Selection Scores (Filter Method)')
plt.tight_layout()

# Visualize feature importances
plt.subplot(2, 2, 2)
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances (Random Forest)')
plt.tight_layout()

# Visualize correlation matrix
plt.subplot(2, 2, 3)
correlation_matrix = X[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')

# Visualize model comparison
plt.subplot(2, 2, 4)
methods = list(results.keys())
scores = list(results.values())
plt.bar(methods, scores, color='skyblue')
plt.ylim(0, 1)
plt.ylabel('R² Score (Cross-Validation)')
plt.title('Feature Engineering Methods Comparison')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# 6. Train final model with best strategy and evaluate
best_pipeline = feature_engineering_pipelines[max(results, key=results.get)]
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

print("\nFinal Model Evaluation:")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Plot predictions vs actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.tight_layout()
plt.show()
```

**Mini Practice Task:** Choose a dataset with a mix of numerical and categorical features. Implement a complete feature engineering workflow that includes:

1. Handling missing values and outliers
2. Creating interaction features and polynomial features
3. Encoding categorical variables using different techniques
4. Selecting the best features using multiple methods
5. Evaluating the impact of your feature engineering steps on model performance

### 7. Ensemble Methods and Advanced Modeling

- Bagging and RandomForest
- Boosting algorithms (AdaBoost, Gradient Boosting, XGBoost)
- Stacking and voting ensembles
- Model persistence and deployment
- Model monitoring and maintenance

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Load breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Define base and ensemble models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
}

# 2. Cross-validation evaluation
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_results[name] = scores
    print(f"{name} - CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# 3. Create and evaluate voting ensemble
voting_clf = VotingClassifier(
    estimators=[
        ('lr', models['Logistic Regression']),
        ('rf', models['Random Forest']),
        ('gb', models['Gradient Boosting'])
    ],
    voting='soft'
)

voting_scores = cross_val_score(
    voting_clf, X_train_scaled, y_train, cv=5, scoring='accuracy'
)
print(f"Voting Classifier - CV Accuracy: {voting_scores.mean():.4f} ± {voting_scores.std():.4f}")
models['Voting'] = voting_clf

# 4. Create and evaluate stacking ensemble
base_learners = [
    ('lr', models['Logistic Regression']),
    ('dt', models['Decision Tree']),
    ('rf', models['Random Forest'])
]
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=GradientBoostingClassifier(n_estimators=100, random_state=42)
)
stacking_scores = cross_val_score(
    stacking_clf, X_train_scaled, y_train, cv=5, scoring='accuracy'
)
print(f"Stacking Classifier - CV Accuracy: {stacking_scores.mean():.4f} ± {stacking_scores.std():.4f}")
models['Stacking'] = stacking_clf

# 5. Train models on full training set and evaluate on test set
test_results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    test_results[name] = accuracy
    print(f"{name} - Test Accuracy: {accuracy:.4f}")

# 6. ROC curve comparison
plt.figure(figsize=(10, 8))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.show()

# 7. Compare model performance with boxplots
plt.figure(figsize=(14, 6))
data_to_plot = [cv_results[model] for model in cv_results.keys()]
labels = list(cv_results.keys())


plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
plt.title('Model Performance Comparison (Cross-Validation)')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 8. Feature importance comparison
plt.figure(figsize=(16, 12))

# Define models with feature importance
importance_models = {
    'Random Forest': models['Random Forest'],
    'Gradient Boosting': models['Gradient Boosting'],
    'XGBoost': models['XGBoost']
}

for i, (name, model) in enumerate(importance_models.items()):
    plt.subplot(2, 2, i+1)

    # Get feature importances
    if name == 'XGBoost':
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_

    # Sort and plot top 10 features
    indices = np.argsort(importances)[-10:]
    plt.title(f'Top 10 Feature Importances - {name}')
    plt.barh(range(10), importances[indices], align='center')
    plt.yticks(range(10), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')

plt.tight_layout()
plt.show()

# 9. Confusion matrix for best model
best_model_name = max(test_results, key=test_results.get)
best_model = models[best_model_name]
y_pred = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names,
            yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()
plt.show()

# 10. Model persistence (saving and loading)
# Save the best model
joblib.dump(best_model, 'best_model.pkl')
print(f"\nBest model ({best_model_name}) saved to 'best_model.pkl'")

# Load the model
loaded_model = joblib.load('best_model.pkl')
loaded_accuracy = accuracy_score(y_test, loaded_model.predict(X_test_scaled))
print(f"Loaded model accuracy: {loaded_accuracy:.4f}")
```

**Mini Practice Task:** Choose a classification dataset and implement an ensemble modeling approach:

1. Train various base models (logistic regression, decision trees, KNN)
2. Create both a voting ensemble and a stacking ensemble
3. Compare the performance of individual models vs. ensembles
4. Analyze which features are most important across different models
5. Save your best model for future use

## Resources for Further Learning

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) by Aurélien Géron
- [Python Machine Learning](https://sebastianraschka.com/books.html) by Sebastian Raschka
- [Introduction to Statistical Learning](https://www.statlearning.com/) by James, Witten, Hastie, Tibshirani
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)

## Next Steps

After mastering these machine learning concepts, you can apply them in the [Projects](../06_Projects/README.md) section to solve real-world problems and build your data science portfolio. Consider also exploring deep learning and specialized areas like natural language processing or computer vision.
