# Home Energy Optimization with Predictive Control

This project demonstrates an end-to-end pipeline for optimizing household energy consumption. It uses machine learning to predict energy needs and applies a Model Predictive Control (MPC) strategy to minimize grid dependency and reduce costs by intelligently scheduling energy usage against solar power generation.

## Project Workflow

The entire project is encapsulated in a single, powerful script that performs the following steps:

1.  **Data Aggregation**: Fetches and merges data for 13 houses from a public repository, including energy consumption, solar PV generation, and weather data.
2.  **Feature Engineering**: Creates time-based features (e.g., hour, day of the week) and lag features, which are crucial for accurate time-series forecasting.
3.  **Predictive Modeling**:
    *   To ensure efficient execution, the system trains a **Random Forest Regressor** on a 20% random sample of the total dataset. This is a common practice to balance model performance with computational cost.
    *   The model learns to predict a household's energy consumption based on weather conditions and historical usage patterns.
4.  **MPC Simulation**:
    *   The trained model is used to forecast energy consumption for a future period (e.g., the next 24 hours).
    *   An optimization algorithm (`optimization_multiobj.py`) takes these predictions and creates an optimal schedule for shifting flexible energy loads.
    *   The simulation is run for multiple houses to demonstrate the system's effectiveness under different conditions.

## How to Run the Analysis

The project is streamlined for ease of use. Follow these steps to run the complete analysis.

### 1. Prerequisites

Ensure you have Python installed, along with the necessary libraries. You can install them with a single command:

```bash
pip install pandas scikit-learn pulp
```

### 2. Execute the Main Script

Run the `run_complete_analysis.py` script from your terminal:

```bash
python3 run_complete_analysis.py
```

The script will automatically perform all steps, from data loading to training and simulation, and print the results to the console.

## Interpreting the Output

The script will produce the following output:

1.  **Model Performance**: The R-squared (R²) value of the trained model on the test set. This metric indicates how well the model's predictions match the actual data.
2.  **Simulation Results**: For each simulated house, a table will be displayed comparing the energy metrics *before* and *after* optimization. Key metrics include:
    *   **Grid Import (kWh)**: The total energy drawn from the grid. A lower value is better.
    *   **PV Curtailment (kWh)**: The amount of unused solar energy. A lower value is better.
    *   **Peak Import (kW)**: The highest power drawn from the grid at any point. A lower value reduces stress on the grid.
    *   **Self-Sufficiency (SSR)**: The percentage of energy needs met by on-site solar generation. A higher value is better.

### Example Output

```
--- Running Simulation for House 1 on 2021-07-15 ---
Status: Optimal solution found.

Metric               | Before (Predicted)   | After (Optimized)
-----------------------------------------------------------------
Total Load (kWh)     | 5.13                 | 5.13
Grid Import (kWh)    | 2.74                 | 1.97
PV Curtailment (kWh) | 1.24                 | 0.46
Peak Import (kW)     | 0.40                 | 0.40
Self-Sufficiency (SSR) | 46.50%               | 61.53%
```
This demonstrates a significant improvement, with grid import reduced and self-sufficiency increased by over 15%.

## Core Files

-   `run_complete_analysis.py`: **(Main Script)** The single script required to run the entire project.
-   `optimization_multiobj.py`: **(Existing Module)** A pre-existing helper script containing the core optimization logic using the `pulp` library.