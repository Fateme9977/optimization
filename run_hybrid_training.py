"""
Main script to train and evaluate the hybrid (RC + NN) model.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from data_loader import get_clean_data
from rc_model import RCModel
from nn_model import ResidualNN, train_nn_model
from hybrid_model import HybridModel
import torch

def calculate_nrmse(y_true, y_pred):
    """Calculates the Normalized Root Mean Squared Error."""
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    nrmse = rmse / (y_true.max() - y_true.min())
    return nrmse

def main():
    # 1. Load Data
    print("Loading and cleaning data...")
    df = get_clean_data()
    if df is None: return

    # 2. Train/Validation Split (70/30 split)
    split_index = int(len(df) * 0.7)
    train_df = df.iloc[:split_index]
    val_df = df.iloc[split_index:]
    print(f"Data split into training ({len(train_df)} rows) and validation ({len(val_df)} rows).")

    # 3. Train the RC Model on the training set
    temp_col = 's2_a'
    rc_model = RCModel()
    rc_model.train(train_df, temp_col=temp_col)

    # 4. Get RC predictions and residuals for both sets
    rc_preds_train = rc_model.predict(train_df)
    rc_preds_val = rc_model.predict(val_df)

    y_train_actual = train_df.loc[rc_preds_train.index, temp_col]
    y_val_actual = val_df.loc[rc_preds_val.index, temp_col]

    residuals_train = y_train_actual - rc_preds_train
    residuals_val = y_val_actual - rc_preds_val

    # Evaluate RC model on validation set for comparison
    nrmse_rc_val = calculate_nrmse(y_val_actual, rc_preds_val)
    print(f"RC Model NRMSE on Validation Set: {nrmse_rc_val:.4f}")

    # 5. Prepare data for the Neural Network
    # Use the same features as the RC model + time-based features
    X_rc_train, _ = rc_model._create_features(train_df, temp_col, 'p_rad_tot', 'sol_glob', 'tout')
    X_rc_val, _ = rc_model._create_features(val_df, temp_col, 'p_rad_tot', 'sol_glob', 'tout')

    # Add time-based features
    X_nn_train = X_rc_train.copy()
    X_nn_train['hour'] = X_nn_train.index.hour
    X_nn_train['dayofweek'] = X_nn_train.index.dayofweek

    X_nn_val = X_rc_val.copy()
    X_nn_val['hour'] = X_nn_val.index.hour
    X_nn_val['dayofweek'] = X_nn_val.index.dayofweek

    # Align residuals with the feature sets
    y_nn_train = residuals_train.loc[X_nn_train.index]
    y_nn_val = residuals_val.loc[X_nn_val.index]

    # Feature Scaling
    scaler = StandardScaler()
    X_nn_train_scaled = scaler.fit_transform(X_nn_train)
    X_nn_val_scaled = scaler.transform(X_nn_val)

    X_nn_train_scaled = pd.DataFrame(X_nn_train_scaled, index=X_nn_train.index, columns=X_nn_train.columns)
    X_nn_val_scaled = pd.DataFrame(X_nn_val_scaled, index=X_nn_val.index, columns=X_nn_val.columns)

    # 6. Train the Neural Network
    print("\nConfiguring and training the Residual NN...")
    input_size = X_nn_train_scaled.shape[1]
    # Use a simpler, single-layer network to start
    nn_model = ResidualNN(input_size=input_size, hidden_size1=16, hidden_size2=None)

    # Use more conservative hyperparameters to avoid overfitting
    nn_model = train_nn_model(nn_model, X_nn_train_scaled, y_nn_train,
                              epochs=30,
                              learning_rate=0.0005,
                              weight_decay=1e-4)

    # 7. Evaluate the Hybrid Model
    # Instantiate the HybridModel with the trained components
    hybrid_model = HybridModel(rc_model=rc_model,
                               nn_model=nn_model,
                               nn_feature_scaler=scaler,
                               nn_feature_columns=X_nn_train.columns.tolist())

    # Get final predictions from the hybrid model
    hybrid_preds_val = hybrid_model.predict(val_df)

    # Align actuals for comparison
    y_val_actual_aligned = y_val_actual.loc[hybrid_preds_val.index]

    # Final evaluation
    nrmse_hybrid_val = calculate_nrmse(y_val_actual_aligned, hybrid_preds_val)

    print("\n--- Final Model Performance on Validation Set ---")
    print(f"RC Model NRMSE:      {nrmse_rc_val:.4f}")
    print(f"Hybrid Model NRMSE:  {nrmse_hybrid_val:.4f}")
    improvement = (nrmse_rc_val - nrmse_hybrid_val) / nrmse_rc_val
    print(f"Improvement:         {improvement:.2%}")

    # 8. Plot results
    fig, ax = plt.subplots(figsize=(15, 7))
    y_val_actual.loc[hybrid_preds_val.index].plot(ax=ax, label='Actual Temperature', alpha=0.7)
    rc_preds_val.loc[hybrid_preds_val.index].plot(ax=ax, label='RC Model', linestyle=':', alpha=0.9)
    hybrid_preds_val.plot(ax=ax, label='Hybrid Model (RC+NN)', linestyle='--', alpha=0.9)
    ax.set_title('Hybrid Model vs. RC Model (Validation Set)')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Indoor Temperature (°C)')
    ax.legend()
    plot_filename = "hybrid_model_fit.png"
    plt.savefig(plot_filename)
    print(f"\nValidation plot saved to {plot_filename}")


if __name__ == "__main__":
    main()
