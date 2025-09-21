"""
Script to run the RC model identification and evaluation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_clean_data
from rc_model import RCModel

def calculate_nrmse(y_true, y_pred):
    """Calculates the Normalized Root Mean Squared Error."""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    nrmse = rmse / (y_true.max() - y_true.min())
    return nrmse

def main():
    # 1. Load Data
    print("Loading and cleaning data...")
    df = get_clean_data()
    if df is None:
        print("Data loading failed. Exiting.")
        return

    # Print descriptive statistics to understand data scales
    print("\n--- Descriptive Statistics of Cleaned Data ---")
    print(df.describe())
    print("--------------------------------------------\n")

    # For this test, we will train and predict on the same dataset.
    # A proper evaluation would use a train/test split.
    train_df = df

    # 2. Define model inputs
    temp_col = 's2_a'
    power_col = 'p_rad_tot'
    solar_col = 'sol_glob'
    outdoor_temp_col = 'tout'

    # Specifically investigate the HVAC power input column
    print(f"\n--- Analysis of '{power_col}' column ---")
    print(df[power_col].describe())
    print("\nValue Counts:")
    print(df[power_col].value_counts())
    print("-------------------------------------------\n")

    # 3. Train the RC Model
    rc_model = RCModel()
    rc_model.train(train_df,
                   temp_col=temp_col,
                   power_col=power_col,
                   solar_col=solar_col,
                   outdoor_temp_col=outdoor_temp_col)

    # 4. Make predictions on the training data
    print("Making predictions on the training data...")
    predictions = rc_model.predict(train_df,
                                   temp_col=temp_col,
                                   power_col=power_col,
                                   solar_col=solar_col,
                                   outdoor_temp_col=outdoor_temp_col)

    # Align actual values with predictions (since predictions will have fewer due to lags)
    actuals = train_df.loc[predictions.index, temp_col]

    # 5. Evaluate the model
    nrmse = calculate_nrmse(actuals, predictions)
    print(f"\nModel Evaluation (on training data):")
    print(f"Normalized RMSE: {nrmse:.4f}")

    # 6. Plot results
    print("Generating plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    actuals.plot(ax=ax, label='Actual Temperature', alpha=0.8)
    predictions.plot(ax=ax, label='RC Model Prediction', linestyle='--', alpha=0.8)

    ax.set_title('RC Model: Actual vs. Predicted Temperature (Training Data)')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Indoor Temperature (°C)')
    ax.legend()
    ax.grid(True)

    # Save the plot
    plot_filename = "rc_model_fit.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")

if __name__ == "__main__":
    main()
