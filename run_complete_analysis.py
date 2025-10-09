import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from optimization_multiobj import optimize_day_multiobj
import warnings

# --- Configuration ---
# Suppress specific FutureWarning from pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Part 1: Core Functions ---

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

        df_merged = df_weather.join(df_pv, how='outer').join(df_load, how='outer')
        df_merged['house_id'] = house_id
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
    df['lag_1hr'] = df.groupby('house_id')['Consumption (kW)'].shift(4)
    df['lag_2hr'] = df.groupby('house_id')['Consumption (kW)'].shift(8)
    df['lag_24hr'] = df.groupby('house_id')['Consumption (kW)'].shift(96)
    return df

def train_consumption_model(featured_df):
    """Trains a RandomForestRegressor model and returns it."""
    print("Step 1.3: Preparing data and training the consumption model...")

    numerical_features = [
        'Temperature', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Dew Point', 'Fill Flag',
        'Relative Humidity', 'Sun Zenith (rad)', 'Surface Albedo', 'Pressure', 'Wind Direction',
        'Wind Speed', 'hour', 'dayofweek', 'dayofyear', 'month', 'year',
        'lag_1hr', 'lag_2hr', 'lag_24hr'
    ]
    categorical_features = ['Cloud Type']
    target = 'Consumption (kW)'

    training_df = featured_df.copy()
    training_df.dropna(subset=[target], inplace=True)

    for col in numerical_features:
        if training_df[col].isnull().any():
            training_df[col].fillna(training_df[col].median(), inplace=True)
    for col in categorical_features:
        if training_df[col].isnull().any():
            training_df[col].fillna(training_df[col].mode()[0], inplace=True)

    training_df = pd.get_dummies(training_df, columns=categorical_features, drop_first=True, dtype=float)
    final_feature_list = numerical_features + [col for col in training_df.columns if col.startswith('Cloud Type_')]

    # Use a smaller sample for faster training to avoid timeouts
    print("   Using a 20% sample of the data for training to ensure timely execution.")
    sample_df = training_df.sample(frac=0.2, random_state=42)

    X = sample_df[final_feature_list]
    y = sample_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_leaf=5)
    model.fit(X_train, y_train)

    print("Step 1.4: Evaluating the model...")
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    print(f"   Model training complete. R-squared on test set: {r2:.4f}")

    return model

def run_mpc_simulation(model, featured_df, house_id, date_str):
    """Runs the MPC simulation for a specific house and date."""
    print(f"\n--- Running Simulation for House {house_id} on {date_str} ---")

    house_df_sim = featured_df[featured_df['house_id'] == house_id].copy()
    day_to_simulate = house_df_sim[house_df_sim.index.date == pd.to_datetime(date_str).date()].copy()

    if day_to_simulate.empty:
        print(f"   No data available for House {house_id} on {date_str}. Skipping.")
        return

    # Prepare features for prediction
    numerical_features = [
        'Temperature', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Dew Point', 'Fill Flag',
        'Relative Humidity', 'Sun Zenith (rad)', 'Surface Albedo', 'Pressure', 'Wind Direction',
        'Wind Speed', 'hour', 'dayofweek', 'dayofyear', 'month', 'year',
        'lag_1hr', 'lag_2hr', 'lag_24hr'
    ]
    categorical_features = ['Cloud Type']

    for col in numerical_features:
        if day_to_simulate[col].isnull().any():
            day_to_simulate[col].fillna(house_df_sim[col].median(), inplace=True)
    for col in categorical_features:
        if day_to_simulate[col].isnull().any():
            day_to_simulate[col].fillna(house_df_sim[col].mode()[0], inplace=True)

    day_to_simulate = pd.get_dummies(day_to_simulate, columns=categorical_features, dtype=float)
    for col in model.feature_names_in_:
        if col not in day_to_simulate.columns:
            day_to_simulate[col] = 0.0

    X_pred = day_to_simulate[model.feature_names_in_]

    # Predict and run optimization
    predicted_load = model.predict(X_pred)
    optimization_input = pd.DataFrame({
        'load_kW': predicted_load,
        'pv_kW': day_to_simulate['PV Power Generation (kW)'].fillna(0)
    }, index=day_to_simulate.index)

    results = optimize_day_multiobj(optimization_input, alpha=0.3, lambda_import=1.0, lambda_peak=0.2)

    # Display results
    if results['status'] == 'Optimal':
        print("Status: Optimal solution found.\n")
        before, after = results['before'], results['after']
        print(f"{'Metric':<20} | {'Before (Predicted)':<20} | {'After (Optimized)':<20}")
        print("-" * 65)
        print(f"{'Total Load (kWh)':<20} | {before['E_load_kWh']:<20.2f} | {after['E_load_kWh']:<20.2f}")
        print(f"{'Grid Import (kWh)':<20} | {before['E_import_kWh']:<20.2f} | {after['E_import_kWh']:<20.2f}")
        print(f"{'PV Curtailment (kWh)':<20} | {before['Curtailment_kWh']:<20.2f} | {after['Curtailment_kWh']:<20.2f}")
        print(f"{'Peak Import (kW)':<20} | {before['NetLoad_peak_kW']:<20.2f} | {after['NetLoad_peak_kW']:<20.2f}")
        print(f"{'Self-Sufficiency (SSR)':<20} | {before['SSR'] * 100:<19.2f}% | {after['SSR'] * 100:<19.2f}%")
    else:
        print(f"Optimization failed with status: {results['status']}")

# --- Part 2: Main Execution ---

if __name__ == "__main__":
    print("--- Starting Complete Energy Optimization Analysis ---")

    # Step 1: Load, merge, and feature-engineer data for all houses
    print("\nStep 1.1: Loading and merging data for all houses...")
    all_dfs = [load_and_merge_house_data(i) for i in range(1, 14) if load_and_merge_house_data(i) is not None]
    master_df = pd.concat(all_dfs)
    master_df.sort_index(inplace=True)
    master_df.dropna(subset=['Consumption (kW)', 'PV Power Generation (kW)'], how='all', inplace=True)

    print("Step 1.2: Creating time-based and lag features...")
    featured_df = create_features(master_df)

    # Step 2: Train the model
    consumption_model = train_consumption_model(featured_df)

    # Step 3: Run MPC simulation for multiple houses
    run_mpc_simulation(consumption_model, featured_df, house_id=1, date_str='2021-07-15')
    run_mpc_simulation(consumption_model, featured_df, house_id=5, date_str='2021-07-15')

    print("\n--- Complete Analysis Finished ---")