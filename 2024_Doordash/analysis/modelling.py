#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:23:02 2025
@author: lucie

Loads the cleaned 'model_input' from Script #1 (assumes you have that in memory or 
import it), then runs modeling steps, feature importance, scaling, etc.
"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import tree, svm, neighbors, linear_model
from sklearn.neural_network import MLPRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# Assume 'model_input' is already in memory from Script #1
# or load from a file if you exported it
# ---------------------------------------------------------------------

# For example, if Script #1 saved a cleaned CSV:
# model_input = pd.read_csv('cleaned_model_input.csv')

print("Columns in model_input:", model_input.columns.tolist())

# Make sure there are no NAs
model_input.dropna(inplace=True)

# ---------------------------------------------------------------------
# Define X and Y for initial modeling
# ---------------------------------------------------------------------
# Convert boolean columns to int if any
bool_cols = X.select_dtypes(include='bool').columns.tolist()
X[bool_cols] = X[bool_cols].astype(int)

# The target variable Y is the 'total_duration'
# Y = model_input["total_duration"]

# If 'total_duration' is datetime-like (unlikely here),
# you would convert to numeric:
model_input["total_duration"] = model_input["total_duration"].dt.total_seconds()

model_input["prep_time"] = model_input["total_duration"] - model_input["estimated_store_to_consumer_driving_duration"] - model_input["estimated_order_place_duration"]

to_remove = model_input["prep_time"] < 0 | model_input["prep_time"].isna()
to_remove = to_remove.index[to_remove]
model_input.drop(to_remove, axis=0, inplace=True)


# Identify datetime columns in X_train
datetime_columns = model_input.select_dtypes(include=['datetime64']).columns

# Convert datetime columns to numerical format (e.g., seconds since epoch)
for col in datetime_columns:
    model_input[col] = (model_input[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

Y = model_input["prep_time"]

# Use the same columns from the first script's X
# X = model_input.drop(columns=["total_duration"])
X = model_input.drop(columns=["prep_time"])

# Convert boolean columns into int (if any exist)
bool_cols = X.select_dtypes(include='bool').columns.tolist()
X[bool_cols] = X[bool_cols].astype(int)

# ---------------------------------------------------------------------
# Basic Train/Test Split
# ---------------------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=13
)

# ---------------------------------------------------------------------
# Feature Importance with RandomForest
# ---------------------------------------------------------------------
forest = RandomForestRegressor(random_state=13)
forest.fit(X_train, Y_train)

feats = {}
for feature, importance in zip(X.columns, forest.feature_importances_):
    feats[feature] = importance

importances = pd.DataFrame.from_dict(feats, orient="index").rename(columns={0: "Gini_importance"})
importances.sort_values(by="Gini_importance").plot(kind="bar", rot=90, figsize=(15, 12))
plt.show()

# ---------------------------------------------------------------------
# PCA Example (Optional)
# ---------------------------------------------------------------------
X_Train = np.asarray(X_train.values)
X_std = StandardScaler().fit_transform(X_Train)
pca = PCA().fit(X_std)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Nb of components")
plt.ylabel("Cumulative explained variance")
plt.show()

# ---------------------------------------------------------------------
# Scaling Function
# ---------------------------------------------------------------------
def scale(scaler, x, y):
    """Return scaled x, scaled y, and the fitted scalers."""
    x_scaler = scaler
    x_scaler.fit(x)
    x_scaled = x_scaler.transform(x)

    y_scaler = scaler
    # Must reshape y to fit the scaler
    y_scaler.fit(y.values.reshape(-1, 1))
    y_scaled = y_scaler.transform(y.values.reshape(-1, 1))

    return x_scaled, y_scaled, x_scaler, y_scaler

# Test scaling
x_scaled, y_scaled, x_scaler, y_scaler = scale(StandardScaler(), X, Y)
X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled = train_test_split(
    x_scaled, y_scaled, test_size=0.2, random_state=13
)

# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------
def rmse_with_inv_transformation(scaler, y_test, y_pred_scaled, model_name):
    """
    Calculate RMSE after inverse_transform of scaled predictions.
    """
    y_predict = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    rmse_error = mean_squared_error(y_test, y_predict[:, 0], squared=False)
    print("Error = " + '{:.3f}'.format(rmse_error) + " in " + model_name)
    return rmse_error, y_predict

def make_regression(X_train, Y_train, X_test, Y_test, model, model_name, verbose=True):
    """Train and evaluate a regression model."""
    model.fit(X_train, Y_train)
    Y_predict_train = model.predict(X_train)
    train_error = mean_squared_error(Y_train, Y_predict_train, squared=False)

    Y_predict_test = model.predict(X_test)
    test_error = mean_squared_error(Y_test, Y_predict_test, squared=False)

    if verbose:
        print(f"Train error = {train_error:.4f} in {model_name}")
        print(f"Test error = {test_error:.4f} in {model_name}")

    return model, Y_predict_test, train_error, test_error

# ---------------------------------------------------------------------
# Dictionaries for Models, Features, Scalers
# ---------------------------------------------------------------------
regression_models = {
    'Ridge': linear_model.Ridge(),
    'DecisionTree': tree.DecisionTreeRegressor(max_depth=6),
    'RandomForest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'LGBM': LGBMRegressor(),
    'MLP': MLPRegressor()
}

feature_sets = {
    "all": X.columns.to_list(),
    "selected_40": importances.sort_values(by="Gini_importance").iloc[-40:].index.tolist(),
    "selected_20": importances.sort_values(by="Gini_importance").iloc[-20:].index.tolist(),
    "selected_10": importances.sort_values(by="Gini_importance").iloc[-10:].index.tolist(),
}

scalers = {
    "Std_scaler": StandardScaler(),
    "MinMax_scaler": MinMaxScaler(),
    "No_scaler": None
}

pred_dict = {
    "regression_model": [],
    "feature_set": [],
    "scaler_name": [],
    "RMSE": []
}

# ---------------------------------------------------------------------
# Main Loop Over Feature Sets & Scalers & Models
# ---------------------------------------------------------------------
for feature_set_name, feature_list in feature_sets.items():
    print(f"\n=== Feature Set: {feature_set_name} ===")

    # Subset the columns we need, ignoring if some columns don't exist
    X_subset = X[feature_list].drop(
        columns=["estimated_store_to_consumer_driving_duration", "estimated_order_place_duration"],
        errors="ignore"
    )

    for scaler_name, scaler in scalers.items():
        print(f"  -- Using scaler: {scaler_name} --")

        # Scale or not scale
        if scaler_name == "No_scaler":
            X_data = X_subset.copy()
            y_data = Y.copy()
            y_scaler = None
        else:
            X_data, y_data, x_scaler, y_scaler = scale(scaler, X_subset, Y)

        # Train/Test split for each combination
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
            X_data, y_data, test_size=0.2, random_state=13
        )

        # Optionally ravel all y (if you prefer or only for certain models)
        # For simplicity, we can do it for all models:
        y_train_f = np.ravel(y_train_f)
        y_test_f = np.ravel(y_test_f)

        # Loop over each model
        for model_name, model in regression_models.items():
            trained_model, y_pred, train_error, test_error = make_regression(
                X_train_f, y_train_f, X_test_f, y_test_f, model, model_name, verbose=True
            )

            # If no scaler, we use test_error as RMSE
            if scaler_name == "No_scaler":
                rmse_error = test_error
            else:
                # Inverse-transform predictions to original scale
                rmse_error, y_pred_orig = rmse_with_inv_transformation(
                    y_scaler, y_test_f, y_pred, model_name
                )

            # Store results
            pred_dict["regression_model"].append(model_name)
            pred_dict["feature_set"].append(feature_set_name)
            pred_dict["scaler_name"].append(scaler_name)
            pred_dict["RMSE"].append(rmse_error)

# ---------------------------------------------------------------------
# Convert Results to DataFrame and Plot
# ---------------------------------------------------------------------
pred_df = pd.DataFrame(pred_dict)
print("\nFinal Performance Table:\n", pred_df)

pred_df.plot(kind='bar', figsize=(12, 8))
plt.show()


for model_name in regression_models.keys():
    _, Y_predict, _, _= make_regression(X_train, Y_train, X_test, Y_test,regression_models[model_name], model_name, verbose=False)
    print("RMSE of:",model_name, mean_squared_error(Y_test,Y_predict, squared=False))
    
