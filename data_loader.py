"""
Data loading and preprocessing for the hybrid HVAC model.

This module will handle loading the PRBS, weather, and other relevant
datasets, performing time alignment, cleaning, and feature engineering
needed for the RC and NN models.
"""
import pandas as pd
import numpy as np
import re

def to_snake_case(name):
    """Converts a string from CamelCase to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def get_clean_data(url="https://raw.githubusercontent.com/fena200023-prog/project1/dataseaat/ds1-PRBS-5min.csv"):
    """
    Loads and cleans a single PRBS dataset from a URL.

    This function performs the following steps:
    1. Loads data from the specified URL.
    2. Skips the first row which contains unit definitions.
    3. Parses the 'Time' column into a timezone-aware datetime index.
    4. Renames columns to a standard snake_case format.
    5. Converts all data columns to numeric types, coercing errors.
    6. Handles missing values using forward fill.
    """
    try:
        # 1. Load data, skipping the units row (row 1)
        df = pd.read_csv(url, skiprows=[1])
        print("Successfully loaded data.")

        # 2. Handle Timestamp
        # Using pd.to_datetime with UTC=True is a robust way to handle different timezone formats
        df['Time'] = pd.to_datetime(df['Time'], utc=True)
        df.set_index('Time', inplace=True)
        df.index.name = 'timestamp'
        print("Parsed and set timestamp index.")

        # 3. Clean and standardize column names
        df.columns = [to_snake_case(col) for col in df.columns]
        print("Standardized column names.")

        # 4. Convert all columns to numeric, coercing errors to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        print("Converted columns to numeric.")

        # 5. Handle missing values
        # Forward fill is a reasonable strategy for sensor data
        df.ffill(inplace=True)
        # Backward fill to catch any NaNs at the beginning
        df.bfill(inplace=True)
        print("Handled missing values.")

        # 6. Drop columns that might be all NaN if they existed
        df.dropna(axis=1, how='all', inplace=True)

        print("Data cleaning complete.")
        return df

    except Exception as e:
        print(f"An error occurred during data cleaning: {e}")
        return None

if __name__ == "__main__":
    print("--- Running Data Cleaning and Preprocessing ---")
    cleaned_df = get_clean_data()

    if cleaned_df is not None:
        print("\n--- Cleaned DataFrame Info ---")
        cleaned_df.info()
        print("\n--- Cleaned DataFrame Head ---")
        print(cleaned_df.head())
        print("\n--- Missing Values Check ---")
        print(cleaned_df.isnull().sum())
        print("\n--- Data cleaning and preprocessing test successful ---")
    else:
        print("\n--- Data cleaning and preprocessing test failed ---")
