"""
Template for data preprocessing pipeline for ML prediction models
"""

import pandas as pd
import numpy as np

def handle_missing_data(df):
    """Handle missing values in price data"""
    # Forward fill for price data (use last known price)
    df = df.fillna(method='ffill')
    # Drop remaining NaN rows
    df = df.dropna()
    return df

def remove_outliers(df, columns, method='iqr'):
    """Remove outliers using IQR method"""
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[columns] >= lower_bound) & (df[columns] <= upper_bound)]

def adjust_for_corporate_actions(df, splits, dividends):
    """Adjust prices for stock splits and dividends"""
    # Implementation depends on data format
    # Generally, split-adjusted prices are already provided by data sources
    pass

def normalize_features(df, columns, method='standard'):
    """Normalize features for ML models"""
    if method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

# Example usage
# df = handle_missing_data(raw_data)
# df = remove_outliers(df, ['Close', 'Volume'])
# df_normalized, scaler = normalize_features(df, feature_columns)

