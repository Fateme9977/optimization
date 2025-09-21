"""
Script to run a closed-loop simulation of the MPC controller.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_loader import get_clean_data
from rc_model import RCModel
from nn_model import ResidualNN, train_nn_model
from hybrid_model import HybridModel
from mpc_controller import MPCController
from sklearn.preprocessing import StandardScaler

def main():
    # 1. Load and prepare data
    print("Loading data...")
    df = get_clean_data()
    if df is None: return

    train_df = df.iloc[:int(len(df) * 0.7)]
    sim_df = df.iloc[int(len(df) * 0.7):]

    # 2. Train the Hybrid Model (our "virtual building" and MPC model)
    print("Training the predictive model...")
    temp_col = 's2_a'
    power_col = 'p_rad_tot'

    # --- FIX: Manually scale the power column to a more reasonable scale (kW) ---
    # This is crucial for the linear model to learn a meaningful coefficient.
    df[power_col] = df[power_col] / 1000.0
    train_df[power_col] = train_df[power_col] / 1000.0
    sim_df[power_col] = sim_df[power_col] / 1000.0
    print(f"'{power_col}' column scaled to kW.")
    # -------------------------------------------------------------------------

    rc_model = RCModel()
    rc_model.train(train_df, temp_col=temp_col)

    X_rc_train, _ = rc_model._create_features(train_df, temp_col, 'p_rad_tot', 'sol_glob', 'tout')
    y_train_actual = train_df.loc[X_rc_train.index, temp_col]
    rc_preds_train = rc_model.predict(train_df)
    residuals_train = y_train_actual - rc_preds_train

    X_nn_train = X_rc_train.copy()
    X_nn_train['hour'] = X_nn_train.index.hour
    X_nn_train['dayofweek'] = X_nn_train.index.dayofweek
    y_nn_train = residuals_train.loc[X_nn_train.index]

    scaler = StandardScaler().fit(X_nn_train)
    X_nn_train_scaled = scaler.transform(X_nn_train)
    X_nn_train_scaled = pd.DataFrame(X_nn_train_scaled, index=X_nn_train.index, columns=X_nn_train.columns)

    nn_model = ResidualNN(input_size=X_nn_train.shape[1], hidden_size1=16, hidden_size2=None)
    nn_model = train_nn_model(nn_model, X_nn_train_scaled, y_nn_train, epochs=30, learning_rate=0.0005, weight_decay=1e-4)

    hybrid_model = HybridModel(rc_model, nn_model, scaler, X_nn_train.columns.tolist())

    # 3. Initialize the MPC Controller
    print("Initializing MPC controller...")
    mpc = MPCController(hybrid_model=hybrid_model,
                        horizon=12,  # 1 hour horizon (12 * 5 min)
                        comfort_range=(21.5, 23.5),
                        p_max=df[power_col].max() * 0.8, # Use 80% of historical max
                        lambda_comfort=10000.0, # Increased penalty for comfort violation
                        lambda_power=1.0)

    # 4. Run the simulation
    print("Running MPC simulation...")

    # Create a copy of the simulation dataframe to store our controlled results
    sim_df_controlled = sim_df.copy()

    # We need at least 2 historical points, so we start the control loop at the 2nd index.
    # The first 2 points of the controlled sim will be the same as the actual data.
    for t in tqdm(range(2, len(sim_df_controlled) - mpc.N)):
        # Get the current state from the *controlled* history
        current_history_df = sim_df_controlled.iloc[t-2 : t+1]
        current_temp = current_history_df.iloc[-1][temp_col]

        # Get the weather forecast from the original data (as it's an external input)
        future_weather = sim_df.iloc[t : t + mpc.N]

        # Get the optimal power plan from the MPC
        power_plan = mpc.plan(current_temp, future_weather)
        chosen_power = power_plan[0] # Apply the first step of the plan

        # Update the controlled dataframe with the chosen power for the current step
        sim_df_controlled.loc[sim_df_controlled.index[t], power_col] = chosen_power

        # Now, predict the temperature for the *next* step (t+1) using the updated history
        # The input to the predictor needs the history from t-1 to t, with power at t updated.
        prediction_input_df = sim_df_controlled.iloc[t-1 : t+2]

        # It's possible the prediction returns empty if there's not enough data, handle this
        next_temp_pred_series = hybrid_model.predict(prediction_input_df)

        if not next_temp_pred_series.empty:
            next_temp_pred = next_temp_pred_series.iloc[-1]
            # Update the temperature for the next step in our controlled simulation
            sim_df_controlled.loc[sim_df_controlled.index[t+1], temp_col] = next_temp_pred

    # 5. Analyze and plot results
    print("Analyzing results...")
    # For clarity, let's rename the columns in our results
    sim_df_controlled.rename(columns={temp_col: 'temp_controlled', power_col: 'power_chosen'}, inplace=True)
    hist_df = sim_df_controlled.join(sim_df[[temp_col]].rename(columns={temp_col: 'temp_actual'}))

    total_energy_kwh = (hist_df['power_chosen'] / 1000).sum() * (5/60) # 5-min intervals
    comfort_violations = (
        (hist_df['temp_controlled'] < mpc.T_min).sum() +
        (hist_df['temp_controlled'] > mpc.T_max).sum()
    ) * 5 # In minutes

    print("\n--- MPC Simulation Results ---")
    print(f"Total Energy Consumed: {total_energy_kwh:.2f} kWh")
    print(f"Total Comfort Violation Time: {comfort_violations:.2f} minutes")

    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    hist_df['temp_actual'].plot(ax=axs[0], label='Actual Temp (Uncontrolled)', style='--', alpha=0.7)
    hist_df['temp_controlled'].plot(ax=axs[0], label='Controlled Temp (MPC)')
    axs[0].axhline(mpc.T_min, color='r', linestyle='--', label='Comfort Zone')
    axs[0].axhline(mpc.T_max, color='r', linestyle='--')
    axs[0].set_ylabel('Temperature (°C)')
    axs[0].legend()

    hist_df['power_chosen'].plot(ax=axs[1], label='HVAC Power (MPC)')
    axs[1].set_ylabel('Power (W)')
    axs[1].legend()

    plt.suptitle('MPC Closed-Loop Simulation')
    plt.savefig('mpc_simulation.png')
    print("\nSimulation plot saved to mpc_simulation.png")

if __name__ == "__main__":
    main()
