import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import datetime

def prepare_tableau_data(model_input, pred_df, feature_importances, Y_test, pred_dict, X_test):
    """
    Prepares separate CSV files for Tableau visualizations
    
    Parameters:
    - model_input: DataFrame containing the original cleaned data
    - pred_df: DataFrame containing model prediction results
    - feature_importances: DataFrame containing feature importance scores
    - Y_test: Series containing actual test values
    - pred_dict: Dictionary containing regression results
    - X_test: DataFrame containing test features
    """
    # 1. Summary Statistics by Feature
    feature_stats = X_test.agg(['count', 'mean', 'std']).round(2)
    feature_stats.to_csv('tableau_feature_stats.csv')

    # 2. Time-based Analysis
    if 'created_at' in X_test.columns:
        time_analysis = X_test.copy()
        time_analysis['hour'] = pd.to_datetime(time_analysis['created_at']).dt.hour
        
        hourly_stats = time_analysis.groupby('hour').agg({
            col: ['count', 'mean'] for col in X_test.columns if X_test[col].dtype in ['float64', 'int64']
        }).round(2)
        hourly_stats.to_csv('tableau_hourly_stats.csv')
    
    # 3. Model Performance Data
    performance_data = pd.DataFrame({
        'actual': Y_test
    })
    
    # Extract predictions from pred_dict
    models_results = pd.DataFrame(pred_dict)
    best_models = models_results.nsmallest(3, 'RMSE')
    performance_data['best_model'] = best_models.iloc[0]['regression_model']
    performance_data['best_rmse'] = best_models.iloc[0]['RMSE']
    
    performance_data.to_csv('tableau_model_performance.csv')

    # 4. Feature Importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_importances.index,
        'importance': feature_importances['Gini_importance']
    }).sort_values('importance', ascending=False)
    
    feature_importance_df.to_csv('tableau_feature_importance.csv')

    print("All data files have been prepared for Tableau visualization.")
    
    return {
        'feature_stats': feature_stats,
        'model_performance': performance_data,
        'feature_importance': feature_importance_df
    }

def prepare_executive_dashboard_data(model_input):
    """
    Prepares data specifically for executive dashboard visualizations
    
    Parameters:
    - model_input: DataFrame containing the cleaned data
    """
    # 1. KPI Summary
    numeric_cols = model_input.select_dtypes(include=[np.number]).columns
    
    kpi_summary = {
        'total_orders': len(model_input),
        'total_features': len(model_input.columns)
    }
    
    # Add mean and std for each numeric column
    for col in numeric_cols:
        kpi_summary[f'avg_{col}'] = model_input[col].mean()
        kpi_summary[f'std_{col}'] = model_input[col].std()
    
    # 2. Time Series Data (if created_at exists)
    if 'created_at' in model_input.columns:
        time_series = model_input.set_index(pd.to_datetime(model_input['created_at']))
        time_series = time_series.resample('H').agg({
            col: ['count', 'mean'] for col in numeric_cols
        }).round(2)
        time_series.to_csv('tableau_time_series.csv')
    else:
        time_series = None
    
    # Save KPI summary
    pd.DataFrame([kpi_summary]).to_csv('tableau_kpi_summary.csv')
    
    return {
        'kpi_summary': kpi_summary,
        'time_series': time_series
    }

def prepare_model_performance_data(model_input, pred_dict, Y_test, feature_importances, X_test):
    """
    Prepares data specifically for model performance dashboard
    
    Parameters:
    - model_input: DataFrame containing the cleaned data
    - pred_dict: Dictionary containing regression results
    - Y_test: Series containing actual test values
    - feature_importances: DataFrame containing feature importance scores
    - X_test: DataFrame containing test features
    """
    # 1. Overall Model Comparison
    model_comparison = pd.DataFrame(pred_dict)
    
    # 2. Feature Importance Analysis
    feature_importance = feature_importances.sort_values('Gini_importance', 
                                                       ascending=True)
    
    # 3. Error Distribution Analysis
    error_analysis = pd.DataFrame()
    error_analysis['actual'] = Y_test
    
    # Calculate basic statistics for test set features
    feature_stats = X_test.describe()
    
    # Save all to CSV
    model_comparison.to_csv('tableau_model_comparison.csv')
    feature_importance.to_csv('tableau_feature_importance.csv')
    error_analysis.to_csv('tableau_error_analysis.csv')
    feature_stats.to_csv('tableau_feature_stats.csv')
    
    return {
        'model_comparison': model_comparison,
        'feature_importance': feature_importance,
        'error_analysis': error_analysis,
        'feature_stats': feature_stats
    }

# Example usage:
# After running your models, you can call these functions like this:
tableau_data = prepare_tableau_data(model_input, pred_df, importances, Y_test, pred_dict, X_test)
exec_dashboard = prepare_executive_dashboard_data(model_input)
model_perf = prepare_model_performance_data(model_input, pred_dict, Y_test, importances, X_test)
