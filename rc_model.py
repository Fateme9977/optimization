"""
Physics-based RC model for building thermal dynamics.

This module contains the implementation of a linear system identification model
(ARX) which serves as a discrete-time equivalent of a continuous-time RC model.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class RCModel:
    """
    A simple ARX (AutoRegressive with eXogenous inputs) model to simulate
    the thermal dynamics of a building zone.

    This model predicts the next indoor temperature based on past temperatures
    and current/past exogenous inputs (like outdoor temperature, HVAC power,
    and solar radiation).
    """
    def __init__(self):
        self.model = LinearRegression()
        self.feature_columns = []

    def _create_features(self, df, temp_col, power_col, solar_col, outdoor_temp_col):
        """Creates the lagged feature matrix for the ARX model."""
        features = {
            # Autoregressive terms (past internal temperatures)
            f'{temp_col}_lag1': df[temp_col].shift(1),
            f'{temp_col}_lag2': df[temp_col].shift(2),
            # Exogenous terms (inputs)
            f'{outdoor_temp_col}_lag0': df[outdoor_temp_col].shift(0),
            f'{outdoor_temp_col}_lag1': df[outdoor_temp_col].shift(1),
            f'{power_col}_lag0': df[power_col].shift(0),
            f'{power_col}_lag1': df[power_col].shift(1),
            f'{solar_col}_lag0': df[solar_col].shift(0),
            f'{solar_col}_lag1': df[solar_col].shift(1),
        }
        feature_df = pd.DataFrame(features)

        # The target is the current temperature
        target = df[temp_col]

        # Drop rows with NaN values resulting from the shift operations
        full_df = pd.concat([target, feature_df], axis=1)
        full_df.dropna(inplace=True)

        y = full_df[temp_col]
        X = full_df.drop(columns=[temp_col])

        return X, y

    def train(self, df, temp_col='s2_a', power_col='p_rad_tot',
              solar_col='sol_glob', outdoor_temp_col='tout'):
        """
        Trains the linear regression model.

        Args:
            df (pd.DataFrame): The cleaned training dataframe.
            temp_col (str): The name of the indoor temperature column.
            power_col (str): The name of the HVAC power input column.
            solar_col (str): The name of the solar gain proxy column.
            outdoor_temp_col (str): The name of the outdoor temperature column.
        """
        print("Creating features for RC model training...")
        X, y = self._create_features(df, temp_col, power_col, solar_col, outdoor_temp_col)

        self.feature_columns = X.columns.tolist()

        print("Training RC model...")
        self.model.fit(X, y)

        # Print the learned coefficients for interpretability
        print("\n--- RC Model Coefficients ---")
        for feature, coef in zip(self.feature_columns, self.model.coef_):
            print(f"{feature}: {coef:.4f}")
        print(f"Intercept: {self.model.intercept_:.4f}")
        print("---------------------------\n")

    def predict(self, df, temp_col='s2_a', power_col='p_rad_tot',
                solar_col='sol_glob', outdoor_temp_col='tout'):
        """
        Makes predictions on a given dataframe.

        Args:
            df (pd.DataFrame): The dataframe to make predictions on.
            (Column names should match those used in training)

        Returns:
            pd.Series: A series of temperature predictions.
        """
        if not self.feature_columns:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        # We only need the feature part from this function
        X, _ = self._create_features(df, temp_col, power_col, solar_col, outdoor_temp_col)

        # Ensure order of columns is the same as in training
        X = X[self.feature_columns]

        predictions = self.model.predict(X)

        return pd.Series(predictions, index=X.index, name='temp_prediction')
