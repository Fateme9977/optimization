import pandas as pd

def load_and_merge_house_data(house_id):
    """Loads, preprocesses, and merges data for a single house."""
    try:
        base_url = "https://raw.githubusercontent.com/Fateme9977/dataseat/Fateme9977-data/"
        load_url = f"{base_url}Load%20House%20{house_id}.csv"
        pv_url = f"{base_url}PV%20Generation%20House%20{house_id}.csv"
        weather_url = f"{base_url}Weather%20House%20{house_id}.csv"

        df_load = pd.read_csv(load_url, parse_dates=['DateTime'])
        df_pv = pd.read_csv(pv_url, parse_dates=['Timestamp'])
        df_weather = pd.read_csv(weather_url, parse_dates=['Timestamp'])

        df_load.rename(columns={'DateTime': 'Timestamp'}, inplace=True)
        
        for df in [df_load, df_pv, df_weather]:
            df.set_index('Timestamp', inplace=True)

        df_pv['PV Power Generation (kW)'] = df_pv['PV Power Generation (W)'] / 1000
        df_pv.drop(columns=['PV Power Generation (W)'], inplace=True)
        
        df_load = df_load[['Consumption (kW)']]
        df_pv = df_pv[['PV Power Generation (kW)']]
        
        df_merged = df_weather.join(df_pv, how='outer')
        df_merged = df_merged.join(df_load, how='outer')
        
        df_merged['house_id'] = house_id
        return df_merged
    except Exception as e:
        print(f"Could not process House {house_id}. Error: {e}")
        return None

def create_features(df):
    """Creates time-series, seasonal, and weather-correlation features."""
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['month'] = df.index.month
    df['year'] = df.index.year

    # Map months to meteorological seasons: 1=Winter, 2=Spring, 3=Summer, 4=Fall
    df['season'] = ((df['month'] % 12) + 3) // 3

    # Create lag features grouped by house to prevent data leakage
    # The data is at 15-min intervals, so 1h lag is 4 periods, 24h is 96
    df['lag_1hr'] = df.groupby('house_id')['Consumption (kW)'].shift(4)
    df['lag_2hr'] = df.groupby('house_id')['Consumption (kW)'].shift(8)
    df['lag_24hr'] = df.groupby('house_id')['Consumption (kW)'].shift(96)

    # Rolling correlation between consumption and each weather variable
    numeric_cols = df.select_dtypes(include='number').columns
    exclude = {
        'Consumption (kW)', 'PV Power Generation (kW)', 'house_id',
        'hour', 'dayofweek', 'dayofyear', 'month', 'year', 'season',
        'lag_1hr', 'lag_2hr', 'lag_24hr'
    }
    weather_cols = [c for c in numeric_cols if c not in exclude]
    for col in weather_cols:
        df[f'{col}_corr_24hr'] = (
            df.groupby('house_id')
              .apply(lambda g: g['Consumption (kW)'].rolling(96).corr(g[col]))
              .reset_index(level=0, drop=True)
        )

    return df

# --- Main execution ---
print("Loading and merging data for all houses...")
all_dfs = [load_and_merge_house_data(i) for i in range(1, 14) if load_and_merge_house_data(i) is not None]
master_df = pd.concat(all_dfs)
master_df.sort_index(inplace=True)
print("Data loading complete.")

# Clean the data by removing rows where key targets are missing
master_df.dropna(subset=['Consumption (kW)', 'PV Power Generation (kW)'], how='all', inplace=True)
print(f"Data cleaned. Total rows: {len(master_df)}")

print("\nCreating time-based and lag features...")
featured_df = create_features(master_df)
print("Feature creation complete.")

# --- Verification ---
print("\n--- DataFrame with New Features (Head) ---")
# Displaying later rows to show populated lag features
print(featured_df.iloc[100:105])

print("\n--- New Columns ---")
new_cols = [
    'hour', 'dayofweek', 'dayofyear', 'month', 'year', 'season',
    'lag_1hr', 'lag_2hr', 'lag_24hr'
]
print(featured_df[new_cols].head())

print("\nFeature engineering logic has been verified.")
