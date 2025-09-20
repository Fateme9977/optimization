import csv
import os
from datetime import datetime

LOG_HEADER = [
    "run_id", "timestamp", "seed", "house_id", "model", "dataset", "horizon_min",
    "batch_size", "precision", "early_exit", "quantization", "retrieval_k",
    "power_cap_W", "epochs", "wall_clock_s", "energy_Wh", "emissions_kg_co2eq",
    "energy_CPU_Wh", "energy_GPU_Wh", "mem_GB_peak", "RMSE", "MAE", "R2",
    "latency_ms_p50", "latency_ms_p95", "notes"
]

def initialize_logging(log_file):
    """Creates the log file and writes the header if it doesn't exist."""
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(LOG_HEADER)
            print(f"Initialized log file: {log_file}")

def log_experiment(log_file, log_data):
    """Appends a new row of experiment data to the log file."""
    if not isinstance(log_data, dict):
        raise TypeError("log_data must be a dictionary.")

    # Ensure all header columns are present in the log_data
    row_to_write = [log_data.get(col, "") for col in LOG_HEADER]

    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_to_write)

if __name__ == '__main__':
    # --- Example Usage ---
    log_file = "experiment_logs.csv"
    initialize_logging(log_file)

    # Example data from a hypothetical experiment run
    example_log_data = {
        "run_id": "example_run_001",
        "timestamp": datetime.now().isoformat(),
        "seed": 42,
        "house_id": 1,
        "model": "EAMTF",
        "dataset": "SHEERM",
        "horizon_min": 15,
        "batch_size": 64,
        "precision": "BF16",
        "early_exit": True,
        "quantization": "shallow_head",
        "retrieval_k": 5,
        "power_cap_W": "uncapped",
        "epochs": 10,
        "wall_clock_s": 1234.5,
        "energy_Wh": 0.5,
        "energy_CPU_Wh": 0.1,
        "energy_GPU_Wh": 0.4,
        "mem_GB_peak": 4.2,
        "RMSE": 0.15,
        "MAE": 0.1,
        "R2": 0.9,
        "latency_ms_p50": 50.1,
        "latency_ms_p95": 150.2,
        "notes": "Initial test run."
    }

    log_experiment(log_file, example_log_data)
    print(f"Successfully logged example data to {log_file}")
