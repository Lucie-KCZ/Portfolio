#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 20:39:02 2025
@author: lucie

Overall Goal:
This script loads and cleans historical DoorDash data by performing the following:
- Computing delivery durations and driver availability ratios.
- Imputing missing store categories based on the most common category per store.
- Performing dummy encoding on categorical variables.
- Preparing a cleaned, feature-engineered dataset ready for modeling.
- Visualizing pairwise relationships and correlations among features (optional).
- Computing Variance Inflation Factor (VIF) to check for multicollinearity.
"""

# Import necessary libraries
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define a custom function for PairGrid's upper triangle to display correlation as a dot
def corrdot(*args, **kwargs):
  corr_r = args[0].corr(args[1], 'pearson')
  corr_text = f"{corr_r:2.2f}".replace("0.", ".")
  ax = plt.gca()
  ax.set_axis_off()

  marker_size = abs(corr_r) * 10000
  ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
            vmin=-1, vmax=1, transform=ax.transAxes)

  font_size = abs(corr_r) * 40 + 5
  ax.annotate(corr_text, [.5, .5], xycoords="axes fraction",
              ha='center', va='center', fontsize=font_size)

  # Set pandas display option to show up to 20 columns
  pd.options.display.max_columns = 20

# ---------------------------------------------------------------------
# Load DoorDash historical data from CSV
# ---------------------------------------------------------------------
doordash = pd.read_csv('~/Documents/Work/Data_Sci/Portfolio/2024_Doordash/data/raw/historical_data.csv')

# Convert relevant time columns to datetime objects
doordash['actual_delivery_time'] = pd.to_datetime(doordash['actual_delivery_time'])
doordash['created_at'] = pd.to_datetime(doordash['created_at'])

# Calculate total delivery duration
doordash['total_duration'] = doordash['actual_delivery_time'] - doordash['created_at']

# Compute ratio of available dashers
doordash['available_dashers'] = (
  doordash['total_onshift_dashers'] - doordash['total_busy_dashers']
) / doordash['total_onshift_dashers']

# Replace non-finite values (division by zero, etc.) with 0
doordash.loc[~np.isfinite(doordash['available_dashers']), 'available_dashers'] = 0

# Calculate total non-preparation time as sum of two estimated durations
doordash['total_non_prep_time'] = (
  doordash['estimated_order_place_duration'] +
    doordash['estimated_store_to_consumer_driving_duration']
)

# Convert some fields to categorical
doordash['order_protocol'] = doordash['order_protocol'].astype('category', copy=False)
doordash['market_id'] = doordash['market_id'].astype('category', copy=False)

# ----- Imputing missing store_primary_category -----
unique_category_counts = doordash.groupby("store_id")["store_primary_category"].nunique()
stores_with_multiple_categories = unique_category_counts[unique_category_counts > 1]

print("Stores with multiple categories:\n", stores_with_multiple_categories)
print("Number of stores with more than one category:", len(stores_with_multiple_categories))

# Impute the missing categories with the most common category per store
for store_id in doordash['store_id'].unique():
  store_data = doordash.loc[doordash['store_id'] == store_id, :]
  if store_data['store_primary_category'].isna().any() and not store_data['store_primary_category'].isna().all():
    category_series = store_data['store_primary_category']
    category_counts = category_series.groupby(category_series).size()
    max_count = category_counts.max()
    common_categories = category_counts[category_counts == max_count].index.to_list()
    most_common_category = common_categories[0]

    nan_positions = store_data['store_primary_category'].isna()
    store_data.loc[nan_positions, 'store_primary_category'] = most_common_category
    doordash.loc[doordash['store_id'] == store_id, :] = store_data

# ----- Preparing dataset for modeling -----
# Drop columns not needed
model_input = doordash.drop(columns=[
  'num_distinct_items',
  'min_item_price',
  'max_item_price',
  'total_onshift_dashers',
  'total_busy_dashers',
  'store_id'
])

# Check numeric columns for non-finite values
for col in model_input.select_dtypes(include=[np.number]).columns:
  if ~np.isfinite(model_input[col]).all():
    print(f"Non-finite values found in numeric column: {col}")

# Check columns with missing values
for col in model_input.columns:
  if model_input[col].isna().any():
    print(f"Missing values found in column: {col}")

# Drop rows where 'total_duration' is missing
na_total_duration = model_input['total_duration'].isna()
rows_with_na_duration = na_total_duration.index[na_total_duration]
model_input.drop(rows_with_na_duration, axis=0, inplace=True)

# Drop rows where 'available_dashers' is negative
negative_dashers = model_input['available_dashers'] < 0
rows_negative_dashers = negative_dashers.index[negative_dashers]
model_input.drop(rows_negative_dashers, axis=0, inplace=True)

# Create dummy variables
market_id_dummies = pd.get_dummies(model_input['market_id'], prefix='market_id')
store_category_dummies = pd.get_dummies(model_input['store_primary_category'], prefix='store_cat_id')
order_protocol_dummies = pd.get_dummies(model_input['order_protocol'], prefix='order_prot_id')

# Reset indices to merge properly
model_input.reset_index(drop=True, inplace=True)
market_id_dummies.reset_index(drop=True, inplace=True)
store_category_dummies.reset_index(drop=True, inplace=True)
order_protocol_dummies.reset_index(drop=True, inplace=True)

model_input = pd.merge(model_input, market_id_dummies, left_index=True, right_index=True)
model_input = pd.merge(model_input, store_category_dummies, left_index=True, right_index=True)
model_input = pd.merge(model_input, order_protocol_dummies, left_index=True, right_index=True)

# Drop original categorical columns
model_input.drop(['market_id', 'store_primary_category', 'order_protocol'], axis=1, inplace=True)

# Prepare X by dropping the target ('total_duration') only
X = model_input.drop('total_duration', axis=1)

# Make sure numeric only
X = X.select_dtypes(include=[np.number])

# Drop any remaining rows with missing values
X.dropna(inplace=True)

# ----- (Optional) Visualization with seaborn PairGrid -----
# # ----- Visualization with seaborn PairGrid -----
# # Set style and font scale for seaborn plots
# sns.set(style='white', font_scale=0.8)
# plt.rcParams.update({'axes.titlesize': 10, 'axes.labelsize': 16})
# 
# # Initialize PairGrid using the independent variables X
# g = sns.PairGrid(X, aspect=1.4, diag_sharey=False)
# 
# # Apply mappings to the PairGrid: lower triangle with regression, 
# # diagonal with histograms, and upper triangle with custom correlation dots
# g.map_lower(sns.regplot, lowess=True, ci=False, line_kws={'color': 'black'})
# g.map_diag(sns.histplot, kde_kws={'color': 'black'})
# g.map_upper(corrdot)
# 
# # Adjust tick labels and axis title properties for each subplot in the grid
# for ax in g.axes.flatten():
#     if ax is not None:
#         # Rotate x-axis tick labels and adjust font size
#         for label in ax.get_xticklabels():
#             label.set_rotation(45)
#             label.set_fontsize(8)
# 
#         # Rotate y-axis tick labels and adjust font size
#         for label in ax.get_yticklabels():
#             label.set_rotation(45)
#             label.set_fontsize(8)
# 
#         # Rotate and set font size for y-axis title
#         ax.yaxis.label.set_rotation(45)
#         ax.yaxis.label.set_fontsize(10)
# 
#         # Rotate and set font size for x-axis title
#         ax.xaxis.label.set_rotation(45)
#         ax.xaxis.label.set_fontsize(10)
# 
# # Rotate and shift top-row subplot titles manually for better readability
# for ax in g.axes[0]:  # Top row axes
#     title_obj = ax.title
#     if title_obj.get_text():
#         title_obj.set_rotation(45)
#         x0, y0 = title_obj.get_position()
#         # Shift the title to the left by adjusting its x-coordinate
#         title_obj.set_position((x0 - 0.5, y0))
# 
# # Adjust margins for the whole figure to prevent clipping of labels
# g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.25)
# 
# # Display the PairGrid plot
# plt.show()

# ----- Compute Variance Inflation Factor (VIF) -----
# Rebuild X to ensure all numeric columns from model_input (without 'total_duration')
X = model_input.drop('total_duration', axis=1)

# Convert boolean columns into int (if any exist)
bool_cols = X.select_dtypes(include='bool').columns.tolist()
X[bool_cols] = X[bool_cols].astype(int)

# Numeric only
X = X.select_dtypes(include=[np.number])
X.dropna(inplace=True)

# Example correlation checks:
correlation_matrix = X.corr()
high_corr = correlation_matrix.abs() > 0.9
print("High correlation matrix:\n", high_corr)

for i in range(high_corr.shape[0]):
  if sum(high_corr.iloc[i]) > 1:
    print("Feature with correlation > 0.9:", high_corr.columns[i])

# Low variance
low_variance_cols = X.columns[X.std() < 1e-6]
print("Low variance columns:", low_variance_cols)
X.drop(low_variance_cols, axis=1, inplace=True)

# Compute VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

multicollinearity = True
while multicollinearity:
  vif_data.replace([np.inf, -np.inf], np.nan, inplace=True)
  vif_data.dropna(inplace=True)
  vif_data.sort_values('VIF', ascending=False, inplace=True)
  highest_value = vif_data.iloc[0]
  if highest_value.VIF > 20:
    X.drop([highest_value.feature], axis=1, inplace=True)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    print(highest_value.feature, "has been removed due to high VIF.")
  else:
    multicollinearity = False

# Environment clean-up (optional)
keep_vars = {'model_input', 'X', '__builtins__'}
for var_name in list(globals().keys()):
  if var_name not in keep_vars:
    del globals()[var_name]
del globals()['var_name']
