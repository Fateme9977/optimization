"""
This module defines the HybridModel class that combines the RC model and the
residual NN model for temperature prediction.
"""
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from rc_model import RCModel
from nn_model import ResidualNN

class HybridModel:
    """
    A hybrid model that combines a physics-based RC model with a neural
    network that corrects for the RC model's residuals.
    """
    def __init__(self, rc_model: RCModel, nn_model: ResidualNN, nn_feature_scaler: StandardScaler, nn_feature_columns: list):
        self.rc_model = rc_model
        self.nn_model = nn_model
        self.nn_feature_scaler = nn_feature_scaler
        self.nn_feature_columns = nn_feature_columns
        self.temp_col = 's2_a' # This should probably be more flexible later
        self.power_col = 'p_rad_tot'
        self.solar_col = 'sol_glob'
        self.outdoor_temp_col = 'tout'

    def _prepare_nn_features(self, df):
        """Prepares the feature matrix for the neural network."""
        # Get the base features from the RC model's internal method
        X_rc, _ = self.rc_model._create_features(df, self.temp_col, self.power_col, self.solar_col, self.outdoor_temp_col)

        # Add time-based features
        X_nn = X_rc.copy()
        X_nn['hour'] = X_nn.index.hour
        X_nn['dayofweek'] = X_nn.index.dayofweek

        # Ensure columns are in the correct order and scale them
        X_nn = X_nn[self.nn_feature_columns]
        X_nn_scaled = self.nn_feature_scaler.transform(X_nn)

        X_nn_scaled = pd.DataFrame(X_nn_scaled, index=X_nn.index, columns=self.nn_feature_columns)

        return X_nn_scaled

    def predict(self, df):
        """
        Makes a hybrid prediction on the given dataframe.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.Series: The final hybrid temperature predictions.
        """
        # 1. Get RC model prediction
        rc_preds = self.rc_model.predict(df)

        # 2. Prepare features for NN
        X_nn_scaled = self._prepare_nn_features(df)

        # Align RC predictions with the NN features (as some rows are dropped due to lags)
        rc_preds_aligned = rc_preds.loc[X_nn_scaled.index]

        # 3. Get NN prediction
        self.nn_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_nn_scaled.values, dtype=torch.float32)
            nn_preds_tensor = self.nn_model(X_tensor)
            nn_preds = pd.Series(nn_preds_tensor.numpy().flatten(), index=X_nn_scaled.index)

        # 4. Combine predictions
        hybrid_preds = rc_preds_aligned + nn_preds

        return hybrid_preds
