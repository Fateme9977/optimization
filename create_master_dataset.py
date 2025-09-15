import pandas as pd

def load_and_merge_house_data(house_id):
    """Loads, preprocesses, and merges data for a single house."""
    try:
        base_url = "https://raw.githubusercontent.com/Fateme9977/dataseat/Fateme9977-data/"
        load_url = f"{base_url}Load%20House%20{house_id}.csv"
        pv_url = f"{base_url}PV%20Generation%20House%20{house_id}.csv"
        weather_url = f"{base_url}Weather%20House%20{house_id}.csv"

        df_load = pd.read_csv(load_url)
        df_pv = pd.read_csv(pv_url)
        df_weather = pd.read_csv(weather_url)

        df_load.rename(columns={'DateTime': 'Timestamp'}, inplace=True)
        
        for df in [df_load, df_pv, df_weather]:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.set_index('Timestamp', inplace=True)

        df_pv['PV Power Generation (kW)'] = df_pv['PV Power Generation (W)'] / 1000
        df_pv.drop(columns=['PV Power Generation (W)'], inplace=True)
        
        df_load = df_load[['Consumption (kW)']]
        df_pv = df_pv[['PV Power Generation (kW)']]
        
        # Merge using an outer join
        df_merged = df_weather.join(df_pv, how='outer')
        df_merged = df_merged.join(df_load, how='outer')
        
        df_merged['house_id'] = house_id
        
        print(f"Successfully processed House {house_id}. Shape: {df_merged.shape}")
        return df_merged
        
    except Exception as e:
        print(f"Could not process House {house_id}. Error: {e}")
        return None

# --- Main execution ---
all_dfs = []
for i in range(1, 14):
    house_df = load_and_merge_house_data(i)
    if house_df is not None:
        all_dfs.append(house_df)

print("\nConcatenating all houses into a single DataFrame...")
master_df = pd.concat(all_dfs)
master_df.sort_index(inplace=True)
print("Concatenation complete.")

print(f"\nTotal rows before cleaning: {len(master_df)}")

# --- Data Cleaning ---
# Drop rows where both consumption and pv generation are missing, as they are not useful for modeling.
# This handles the misaligned start/end dates from the outer join.
initial_rows = len(master_df)
master_df.dropna(subset=['Consumption (kW)', 'PV Power Generation (kW)'], how='all', inplace=True)
cleaned_rows = len(master_df)
print(f"Dropped {initial_rows - cleaned_rows} rows with no consumption or PV data.")
print(f"Total rows after cleaning: {cleaned_rows}")


# --- Verification ---
print("\n--- Master DataFrame Info ---")
master_df.info()

print("\n--- Data points per house ---")
print(master_df['house_id'].value_counts().sort_index())

print("\nProcess to generate master dataset has been verified.")
