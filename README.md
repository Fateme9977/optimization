# Energy-Adaptive Multi-Task Forecasting (EAMTF)

This project implements the EAMTF framework, a system for energy-efficient time-series forecasting for residential net-load. It incorporates several advanced deep learning and systems optimizations to reduce energy consumption and latency while maintaining high accuracy.

## Features

- **Multi-Task Learning:** Jointly forecasts Load, PV, and Net-Load.
- **Early-Exit Mechanism:** Routes inputs to a shallow (fast) or deep (accurate) head based on difficulty signals (e.g., clear-sky index).
- **Advanced Optimizations:** Supports:
    - Mixed-Precision Training (BF16/FP16)
    - Post-Training Quantization (PTQ)
    - Retrieval-Augmented Context (FAISS)
    - Low-Rank Adaptation (LoRA) for efficient fine-tuning.
- **Comprehensive Measurement:** Includes tools for logging energy consumption (CodeCarbon), performance metrics, and system parameters to a structured CSV file.
- **Reproducible Experimentation:** A configurable sweep-based framework using YAML for managing experiments and ablation studies.

## Codebase Structure

- `feature_engineering.py`: Loads raw data, processes it, and saves a master dataset as `master_dataset.parquet`.
- `eamtf_model.py`: Defines the main EAMTF model architecture using PyTorch.
- `baseline_models.py`: Defines baseline LSTM and GRU models for comparison.
- `training_pipeline.py`: Contains the core logic for training a model based on a configuration.
- `run_sweep.py`: The main entry point to run a sweep of experiments.
- `config.yaml`: Configuration file for experiments, models, and system parameters.
- `logging_utils.py`: Utility for writing structured logs to `experiment_logs.csv`.
- `energy_monitor.py`: A context manager for measuring energy consumption.
- `retrieval.py`: Implements the FAISS-based analog day retriever.
- `analysis.py`: Contains functions for statistical analysis of results.
- `requirements.txt`: A list of Python dependencies.
- `.gitignore`: Specifies files and directories to be ignored by version control.

## How to Run

Follow these steps to set up the environment and run the experimental pipeline.

### 1. Set Up the Environment

First, install the required Python dependencies.

```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset

Run the feature engineering script. This will download the raw data for a subset of houses, process it, and create the `master_dataset.parquet` file which is required for training.

```bash
python3 feature_engineering.py
```

### 3. Run an Experimental Sweep

Execute the main sweep script. This will run a series of experiments as defined in `run_sweep.py` based on the configuration in `config.yaml`. The results of each run will be appended to `experiment_logs.csv`.

```bash
python3 run_sweep.py
```

### 4. Analyze the Results

The `experiment_logs.csv` file contains the results of all experimental runs. You can use this file for analysis. The `analysis.py` script also provides functions that can be used to perform statistical tests on the results.

## Customizing Experiments

To run different experiments, you can modify:
- **`config.yaml`**: Change the base parameters for models, training, etc.
- **`run_sweep.py`**: Modify the `sweep_variations` list to define different ablation studies or hyperparameter searches.
