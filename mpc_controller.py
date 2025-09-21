"""
Model Predictive Control (MPC) for HVAC system optimization.

This module defines the MPC controller that uses the hybrid (RC + NN) model
to predict future thermal behavior and optimize HVAC operation. The
optimization is formulated as a linear program using `pulp`.
"""
import pulp
import pandas as pd
import numpy as np
from hybrid_model import HybridModel

class MPCController:
    """
    An MPC controller that uses a simplified linear model derived from the RC
    component to optimize HVAC power over a prediction horizon.
    """
    def __init__(self, hybrid_model: HybridModel, horizon: int,
                 comfort_range: tuple, p_max: float,
                 lambda_comfort: float, lambda_power: float):
        self.model = hybrid_model
        self.N = horizon
        self.T_min, self.T_max = comfort_range
        self.P_max = p_max
        self.lambda_comfort = lambda_comfort
        self.lambda_power = lambda_power

        # Extract linear model coefficients from the RC model
        # This is a simplification, assuming the effect of power is dominated by the lag-0 term.
        # A more complex implementation would unroll the full ARX model.
        rc_coeffs = dict(zip(hybrid_model.rc_model.feature_columns, hybrid_model.rc_model.model.coef_))
        self.C_p = rc_coeffs.get(f'{self.model.power_col}_lag0', 0.0)

    def plan(self, current_temp: float, future_weather: pd.DataFrame):
        """
        Creates and solves the optimization problem for the next N steps.

        Args:
            current_temp (float): The latest known indoor temperature.
            future_weather (pd.DataFrame): A dataframe with weather forecasts for the horizon.

        Returns:
            list: A list of optimal power settings for the horizon.
        """
        # 1. Get baseline prediction (assuming zero power)
        # Create a dummy dataframe for prediction
        # This is a simplification. A real implementation would need a more robust way
        # to handle the initial state (past temperatures, etc.)
        future_df = future_weather.copy()
        future_df[self.model.temp_col] = current_temp
        future_df[self.model.power_col] = 0 # Assume zero power for baseline
        future_df[self.model.solar_col] = future_weather['sol_glob'] # Example mapping
        future_df[self.model.outdoor_temp_col] = future_weather['tout'] # Example mapping

        # The hybrid model needs past values, so we can't just predict on the future.
        # For this example, we'll use a major simplification: the temperature evolves from the
        # current temperature based only on the linear power coefficient.
        # T_pred[t] = T_current + C_p * P_hvac[t]

        # 2. Set up the LP Problem
        prob = pulp.LpProblem("HVAC_MPC", pulp.LpMinimize)

        # 3. Define variables
        P_hvac = pulp.LpVariable.dicts("P_hvac", range(self.N), lowBound=0, upBound=self.P_max)
        T_pred = pulp.LpVariable.dicts("T_pred", range(self.N))
        slack_upper = pulp.LpVariable.dicts("slack_upper", range(self.N), lowBound=0)
        slack_lower = pulp.LpVariable.dicts("slack_lower", range(self.N), lowBound=0)

        # 4. Define Objective
        objective = (
            self.lambda_power * pulp.lpSum(P_hvac) +
            self.lambda_comfort * pulp.lpSum(slack_upper) +
            self.lambda_comfort * pulp.lpSum(slack_lower)
        )
        prob += objective

        # 5. Define Constraints
        T_current_loop = current_temp
        for t in range(self.N):
            # Simplified model: T_next = T_current + C_p * P_hvac
            # This is a very strong simplification to make the LP tractable.
            # A real system would need to account for outdoor temp, solar, and lags.
            # We add those effects as a simple offset for now.

            weather_effect = (future_weather.iloc[t]['tout'] - T_current_loop) * 0.01 # Heuristic
            solar_effect = future_weather.iloc[t]['sol_glob'] * 0.0001 # Heuristic

            # This is a basic Euler integration of a simplified model
            T_pred[t] = T_current_loop + self.C_p * P_hvac[t] + weather_effect + solar_effect

            # Comfort constraints
            prob += T_pred[t] <= self.T_max + slack_upper[t]
            prob += T_pred[t] >= self.T_min - slack_lower[t]

            T_current_loop = T_pred[t] # The next step's temp depends on this one

        # 6. Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] == 'Optimal':
            return [pulp.value(P_hvac[t]) for t in range(self.N)]
        else:
            return [0.0] * self.N # Return zero power if optimization fails
