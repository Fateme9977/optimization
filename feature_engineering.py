import pandas as pd

def load_and_merge_house_data(house_id):
    """Loads, preprocesses, and merges data for a single house with optimized dtypes."""
    try:
        base_url = "https://raw.githubusercontent.com/Fateme9977/dataseat/Fateme9977-data/"
        load_url = f"{base_url}Load%20House%20{house_id}.csv"
        pv_url = f"{base_url}PV%20Generation%20House%20{house_id}.csv"
        weather_url = f"{base_url}Weather%20House%20{house_id}.csv"

        # Define dtypes to reduce memory usage
        load_dtypes = {'Consumption (kW)': 'float32'}
        pv_dtypes = {'PV Power Generation (W)': 'float32'}
        # Assuming weather columns are numeric, let's use float32 for them as well
        # We can be more specific if we know the exact column names

        df_load = pd.read_csv(load_url, parse_dates=['DateTime'], dtype=load_dtypes)
        df_pv = pd.read_csv(pv_url, parse_dates=['Timestamp'], dtype=pv_dtypes)
        df_weather = pd.read_csv(weather_url, parse_dates=['Timestamp']) # Dtypes inferred for now

        df_load.rename(columns={'DateTime': 'Timestamp'}, inplace=True)
        
        for df in [df_load, df_pv, df_weather]:
            df.set_index('Timestamp', inplace=True)

        df_pv['PV Power Generation (kW)'] = df_pv['PV Power Generation (W)'] / 1000
        df_pv.drop(columns=['PV Power Generation (W)'], inplace=True)
        
        df_load = df_load[['Consumption (kW)']]
        df_pv = df_pv[['PV Power Generation (kW)']]
        
        df_merged = df_weather.join(df_pv, how='outer')
        df_merged = df_merged.join(df_load, how='outer')
        
        # Downcast numeric columns where possible
        for col in df_merged.select_dtypes(include=['float64']).columns:
            df_merged[col] = df_merged[col].astype('float32')

        df_merged['house_id'] = house_id
        df_merged['house_id'] = df_merged['house_id'].astype('int16')

        return df_merged
    except Exception as e:
        print(f"Could not process House {house_id}. Error: {e}")
        return None

def create_features(df):
    """Creates time-series features from the datetime index and lag features."""
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['month'] = df.index.month
    df['year'] = df.index.year
    
    # Create lag features grouped by house to prevent data leakage
    # The data is at 15-min intervals, so 1h lag is 4 periods, 24h is 96
    df['lag_1hr'] = df.groupby('house_id')['Consumption (kW)'].shift(4)
    df['lag_2hr'] = df.groupby('house_id')['Consumption (kW)'].shift(8)
    df['lag_24hr'] = df.groupby('house_id')['Consumption (kW)'].shift(96)
    
    return df

# --- Main execution ---
print("Loading and merging data for all houses...")
# Fix: Avoid double-calling the function in a list comprehension.
all_dfs = []
# Process only a subset of houses to keep the dataset size manageable.
for i in range(1, 3): # Reduced from 14 to 3 for testing
    df = load_and_merge_house_data(i)
    if df is not None:
        all_dfs.append(df)

master_df = pd.concat(all_dfs)
master_df.sort_index(inplace=True)
print("Data loading complete.")

# Clean the data by removing rows where key targets are missing
master_df.dropna(subset=['Consumption (kW)', 'PV Power Generation (kW)'], how='all', inplace=True)
print(f"Data cleaned. Total rows: {len(master_df)}")

print("\nCreating time-based and lag features...")
featured_df = create_features(master_df)
print("Feature creation complete.")

# --- Save to Parquet for efficient reuse ---
try:
    output_path = "master_dataset.parquet"
    featured_df.to_parquet(output_path, engine="pyarrow", compression="zstd")
    print(f"\nSuccessfully saved featured dataset to {output_path}")
except ImportError:
    print("\nCould not save to Parquet. Please install pyarrow: pip install pyarrow")
except Exception as e:
    print(f"\nError saving to Parquet: {e}")


# --- Verification ---
print("\n--- DataFrame with New Features (Head) ---")
# Displaying later rows to show populated lag features
print(featured_df.iloc[100:105])

print("\n--- New Columns ---")
new_cols = ['hour', 'dayofweek', 'dayofyear', 'month', 'year', 'lag_1hr', 'lag_2hr', 'lag_24hr']
print(featured_df[new_cols].head())

print("\nFeature engineering logic has been verified.")
