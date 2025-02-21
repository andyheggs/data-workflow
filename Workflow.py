# ----------------------------
# Workflow & Hyperparameter Optimization
# ----------------------------

import pandas as pd
import seaborn as sns
import numpy as np

# The following are classes and functions to split data, standardise features,
# build KNN models, perform hyperparameter searches, and evaluate performance
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import scipy.stats

# -----------------------------------------
# 1. DATA LOADING AND PREPARATION
# -----------------------------------------

# Load raw data from URL; index_col="Id" ensures the 'Id' column is used as the DataFrame index
data = pd.read_csv('https://wagon-public-datasets.s3.amazonaws.com/houses_train_raw.csv', index_col="Id")

# Only keep columns that are numerical
# Then drop any rows that contain NaN values
data = data.select_dtypes(include=np.number).dropna()

# Separate features (X) from the target variable (y)
X = data.drop(columns=['SalePrice'])
y = data['SalePrice']

# -----------------------------------------
# 2. TRAIN/TEST SPLIT
# -----------------------------------------
# The train_test_split function splits X and y into training and testing sets.
# test_size=0.3 -> 30% of the dataset is reserved for testing
# random_state=0 -> ensures reproducibility of the split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# -----------------------------------------
# 3. FEATURE SCALING
# -----------------------------------------
# Many ML algorithms can benefit from having the features scaled.
# StandardScaler() transforms features by removing the mean and scaling to unit variance.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Compute scaling factors using training data, then transform
X_test_scaled = scaler.transform(X_test)        # Use the same scaling factors from training for the test data

# -----------------------------------------
# 4. BASELINE KNN MODEL
# -----------------------------------------
# We'll build a baseline KNN Regressor model to get an initial performance measure.
# Using cross_val_score to check how well KNN performs in 5-fold cross-validation.

baseline_knn = KNeighborsRegressor(n_neighbors=1)
baseline_scores = cross_val_score(baseline_knn, X_train_scaled, y_train, cv=5)
baseline_score_avg = baseline_scores.mean()

# -----------------------------------------
# 5. FIRST GRID SEARCH FOR K
# -----------------------------------------
# We create a parameter grid where n_neighbors takes values [1,5,10,20,50].
# GridSearchCV will try each combination of these parameters
# and find the best one based on cross-validation performance.

param_grid = {'n_neighbors': [1, 5, 10, 20, 50]}
grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Extract the best hyperparameter K and its corresponding performance (mean CV score)
best_k_v1 = grid_search.best_params_['n_neighbors']
best_score_v1 = grid_search.best_score_

# -----------------------------------------
# 6. REFINED GRID SEARCH FOR K
# -----------------------------------------
# We now use a narrower range around the previously found best K: (best_k_v1 - 5) to (best_k_v1 + 5).
# This helps us home in on the optimal neighbour count with finer granularity.

param_grid_refined = {'n_neighbors': range(max(1, best_k_v1 - 5), best_k_v1 + 5)}
grid_search_refined = GridSearchCV(KNeighborsRegressor(), param_grid_refined, cv=5, n_jobs=-1)
grid_search_refined.fit(X_train_scaled, y_train)

best_k_v2 = grid_search_refined.best_params_['n_neighbors']
best_score_v2 = grid_search_refined.best_score_

# -----------------------------------------
# 7. GRID SEARCH FOR MULTIPLE PARAMETERS
# -----------------------------------------
# Here, we search over two hyperparameters: n_neighbors and p.
# p is the power parameter of the Minkowski metric used in KNN.
# p=1 -> Manhattan distance, p=2 -> Euclidean distance, p=3 -> a higher-order Minkowski distance, etc.

param_grid_multiple = {
    'n_neighbors': [1, 5, 10, 20, 50],
    'p': [1, 2, 3]
}
grid_search_multiple = GridSearchCV(KNeighborsRegressor(), param_grid_multiple, cv=5, n_jobs=-1)
grid_search_multiple.fit(X_train_scaled, y_train)

best_params_multiple = grid_search_multiple.best_params_
best_score_multiple = grid_search_multiple.best_score_

# -----------------------------------------
# 8. RANDOMISED SEARCH
# -----------------------------------------
# RandomSearchCV explores randomly chosen points from the specified distributions.
# n_iter=15 controls how many parameter settings are sampled.
# This approach can be more efficient than a full grid search, especially with many hyperparameters.

param_distributions = {
    'n_neighbors': scipy.stats.randint(1, 50),
    'p': [1, 2, 3]
}
random_search = RandomizedSearchCV(KNeighborsRegressor(), param_distributions, n_iter=15, cv=5, n_jobs=-1)
random_search.fit(X_train_scaled, y_train)

best_params_random = random_search.best_params_
best_score_random = random_search.best_score_

# -----------------------------------------
# 9. EVALUATING GENERALISATION ON TEST SET
# -----------------------------------------
# We retrieve the best model from the random search and evaluate it on the test set.
# The r2_score provides a measure of how well future samples are likely to be predicted.

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
r2_test = r2_score(y_test, y_pred)

# -----------------------------------------
# 10. PRINT RESULTS
# -----------------------------------------
print("Baseline KNN Average Score:", baseline_score_avg)
print("Best k from first GridSearch:", best_k_v1, "with score:", best_score_v1)
print("Best k from refined GridSearch:", best_k_v2, "with score:", best_score_v2)
print("Best parameters from multiple parameter GridSearch:", best_params_multiple, "with score:", best_score_multiple)
print("Best parameters from RandomizedSearch:", best_params_random, "with score:", best_score_random)
print("R2 score on test set:", r2_test)
