#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final pipeline that:
1) Loads 'doordash' and 'model_input'
2) Cleans 'model_input' columns (removes or converts datetime, handles missing)
3) Splits into train/test
4) Trains multiple models and collects row-level predictions
5) Extracts RandomForest feature importances
6) Calls prepare_tableau_data(...) with correct predictions_dict
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn import linear_model, tree
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

##############################################################
# 1. LOAD doordash & model_input
##############################################################

# Example only - replace with your actual loading or in-memory references
# doordash = pd.read_csv("path/doordash.csv", parse_dates=["created_at", "actual_delivery_time"])
# model_input = pd.read_csv("path/model_input.csv")

# In your environment, do something like:
# from your_exploration_script import doordash, model_input

##############################################################
# 2. Clean model_input columns
##############################################################

# 2A. Convert any boolean columns
bool_cols = model_input.select_dtypes(include="bool").columns
if len(bool_cols) > 0:
    model_input[bool_cols] = model_input[bool_cols].astype(int)

# 2B. Ensure any datetime columns you need are dropped or converted to numeric
dt_cols = model_input.select_dtypes(include="datetime64").columns
if len(dt_cols) > 0:
    # Example: either drop them or convert them
    # Here, let's drop them (assuming they aren't needed):
    print("Dropping datetime columns from model_input:", dt_cols)
    model_input.drop(columns=dt_cols, inplace=True)

# 2C. Confirm numeric for the features, especially if you want to avoid object columns
# (If you have leftover object columns, consider encoding them or dropping them)
obj_cols = model_input.select_dtypes(include="object").columns
if len(obj_cols) > 0:
    print("Dropping object columns from model_input:", obj_cols)
    model_input.drop(columns=obj_cols, inplace=True)

# By now, model_input should be purely numeric (except the target if you stored it here).

##############################################################
# 3. Define X (features) and Y (target)
##############################################################
# We assume "prep_time" is the numeric target:
Y = model_input["prep_time"]  # must be numeric, no NaNs
X = model_input.drop(columns=["prep_time"], errors="ignore")

# If you see columns like "created_at", "market_id" as numeric but are meaningless,
# you should drop them as well. For demonstration, let's just keep them if numeric.

##############################################################
# 4. Train/Test Split
##############################################################
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=13
)

# We now handle missing values in X_train, X_test
imputer = SimpleImputer(strategy="mean")  # or "median", etc.
# Fit on X_train
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Convert back to DataFrame if you need column labels
X_columns = X.columns  # store for feature names
X_train = pd.DataFrame(X_train, columns=X_columns)
X_test = pd.DataFrame(X_test, columns=X_columns)

# y might also have NaNs, but presumably not if you filtered earlier

##############################################################
# 5. Define & Fit Models, Collect Predictions
##############################################################
regression_models = {
    'Ridge': linear_model.Ridge(),
    'DecisionTree': tree.DecisionTreeRegressor(max_depth=6),
    'RandomForest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'LGBM': LGBMRegressor(),
    'MLP': MLPRegressor()
}

fitted_models = {}
predictions_dict = {}  # row-level predictions for each model

for model_name, model in regression_models.items():
    print(f"\n--- Training {model_name} ---")
    model.fit(X_train, Y_train)  # now no NaNs => won't crash
    fitted_models[model_name] = model
    # Predict on X_test
    y_pred_test = model.predict(X_test)
    predictions_dict[model_name] = y_pred_test

##############################################################
# 6. Extract RandomForest Feature Importances
##############################################################
rf_model = fitted_models["RandomForest"]  # now guaranteed to exist
rf_importances = rf_model.feature_importances_
feature_importances = pd.DataFrame({
    "feature": X_columns,
    "Gini_importance": rf_importances
}).sort_values(by="Gini_importance", ascending=False)

print("\nFeature importances (RandomForest):")
print(feature_importances.head(10))

##############################################################
# 7. prepare_tableau_data function
##############################################################
def prepare_tableau_data(
    doordash,
    model_input,
    pred_df,  # optional
    feature_importances,
    Y_test,
    predictions_dict,  # row-level predictions
    output_dir="."
):
    """
    Prepares CSV files for Tableau.
    Expects predictions_dict to have arrays matching Y_test length.
    """
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Align doordash & model_input by index
    combined_df = doordash.loc[model_input.index].copy()

    # Copy engineered columns
    for col in ["prep_time", "total_duration", "available_dashers"]:
        if col in model_input.columns:
            combined_df[col] = model_input[col]

    # Convert 'created_at' to datetime if necessary
    if "created_at" in combined_df.columns and combined_df["created_at"].dtype != "datetime64[ns]":
        combined_df["created_at"] = pd.to_datetime(combined_df["created_at"], unit="s", errors="coerce")

    # 1) Market stats
    market_stats = combined_df.groupby('market_id').agg({
        'prep_time': ['count', 'mean', 'std'],
        'total_duration': ['mean', 'std'],
        'available_dashers': 'mean'
    }).round(2)
    market_stats.columns = ['order_count', 'avg_prep_time', 'std_prep_time',
                            'avg_duration', 'std_duration', 'avg_dashers']
    market_stats.to_csv(os.path.join(output_dir, 'tableau_market_stats.csv'), index=False)

    # 2) Time-based analysis
    time_analysis = combined_df.copy()
    if "created_at" in time_analysis.columns and time_analysis["created_at"].dtype == "datetime64[ns]":
        time_analysis["hour"] = time_analysis["created_at"].dt.hour
        time_analysis["day_of_week"] = time_analysis["created_at"].dt.day_name()
        time_analysis["date"] = time_analysis["created_at"].dt.date
    else:
        time_analysis["hour"] = np.nan
        time_analysis["day_of_week"] = np.nan
        time_analysis["date"] = np.nan

    hourly_stats = time_analysis.groupby(["hour", "market_id", "store_primary_category"]).agg({
        "prep_time": ["count", "mean"],
        "available_dashers": "mean"
    }).round(2)
    hourly_stats.columns = [f"{c[0]}_{c[1]}" for c in hourly_stats.columns]
    hourly_stats.reset_index(inplace=True)
    hourly_stats.to_csv(os.path.join(output_dir, 'tableau_hourly_stats.csv'), index=False)

    # 3) Model Performance Data
    model_performance = pd.DataFrame(index=Y_test.index)
    model_performance["actual"] = Y_test.values

    for model_name, preds in predictions_dict.items():
        model_performance[f"{model_name}_pred"] = preds

    if "market_id" in combined_df.columns:
        model_performance["market_id"] = combined_df.loc[Y_test.index, "market_id"].values
    else:
        model_performance["market_id"] = np.nan

    if "store_primary_category" in combined_df.columns:
        model_performance["store_category"] = combined_df.loc[Y_test.index, "store_primary_category"].values
    else:
        model_performance["store_category"] = np.nan

    if "created_at" in combined_df.columns and combined_df["created_at"].dtype == "datetime64[ns]":
        model_performance["hour"] = combined_df.loc[Y_test.index, "created_at"].dt.hour.values
    else:
        model_performance["hour"] = np.nan

    model_performance.reset_index(drop=True, inplace=True)
    model_performance.to_csv(os.path.join(output_dir, 'tableau_model_performance.csv'), index=False)

    # 4) Feature Importance
    feature_importances.to_csv(os.path.join(output_dir, 'tableau_feature_importance.csv'), index=False)

    # 5) Store Category Analysis
    if "store_primary_category" in combined_df.columns:
        store_stats = combined_df.groupby("store_primary_category").agg({
            "prep_time": ["count", "mean", "std"],
            "total_duration": ["mean", "std"]
        }).round(2)
        store_stats.columns = ["order_count", "avg_prep_time", "std_prep_time", "avg_duration", "std_duration"]
        store_stats.to_csv(os.path.join(output_dir, 'tableau_store_stats.csv'), index=False)
    else:
        # No store_primary_category -> Make an empty CSV
        pd.DataFrame().to_csv(os.path.join(output_dir, 'tableau_store_stats.csv'), index=False)

    # 6) Driver Availability Analysis
    keep_cols = []
    for c in ["available_dashers", "prep_time", "total_duration", "market_id", "store_primary_category"]:
        if c in combined_df.columns:
            keep_cols.append(c)

    driver_analysis = combined_df[keep_cols].copy()
    driver_analysis.reset_index(drop=True, inplace=True)
    driver_analysis.to_csv(os.path.join(output_dir, 'tableau_driver_analysis.csv'), index=False)

    print("All data files have been prepared for Tableau visualization.")

##############################################################
# 8. Call prepare_tableau_data
##############################################################
# We'll assume doordash & model_input are loaded and have the same index alignment.
prepare_tableau_data(
    doordash=doordash,
    model_input=model_input,
    pred_df=None,
    feature_importances=feature_importances,
    Y_test=Y_test,
    predictions_dict=pred_dict,
    output_dir="./tableau_output"
)

print("\nDone! Check ./tableau_output for your CSVs.")
