import yaml
import copy
from datetime import datetime

from training_pipeline import run_training_job, load_config
from logging_utils import log_experiment

def main():
    """
    Runs a sweep of experiments based on a base configuration and a
    defined set of experimental variations.
    """
    base_config = load_config('config.yaml')
    log_file = base_config['logging']['log_file']

    # --- Define the experimental sweep ---
    # Each item in the list is a dictionary of parameters to override in the base config.
    sweep_variations = [
        # Baseline run (from config.yaml)
        {},

        # Ablation: Turn off early exit
        {'early_exit': {'enabled': False}},

        # Ablation: Try a different LoRA rank
        {'lora': {'rank': 16}},

        # Ablation: Use FP32 precision
        {'precision': 'FP32'},
    ]

    print(f"--- Starting experimental sweep with {len(sweep_variations)} variations ---")

    for i, variation in enumerate(sweep_variations):
        # Create a deep copy of the base config for this run
        run_config = copy.deepcopy(base_config)

        # Update the config with the current variation
        # A simple dict update won't work for nested keys, so we do it manually
        if 'early_exit' in variation:
            run_config['early_exit'].update(variation['early_exit'])
        if 'lora' in variation:
            run_config['lora'].update(variation['lora'])
        if 'precision' in variation:
            run_config['precision'] = variation['precision']

        # Generate a unique run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_config['run_id'] = f"{base_config['run_id']}_sweep_{i}_{timestamp}"

        # Run the training job
        results = run_training_job(run_config)

        # Prepare data for logging
        log_data = {
            # Experiment params
            "run_id": run_config['run_id'],
            "timestamp": timestamp,
            "seed": run_config['seed'],
            "house_id": run_config['house_ids'],
            "model": run_config['model_name'],
            "dataset": "SHEERM", # Assuming for now
            "horizon_min": run_config['horizon_min'],
            "batch_size": run_config['hyperparameters']['batch_size'],
            "precision": run_config['precision'],
            "early_exit": run_config['early_exit']['enabled'],
            "quantization": run_config['quantization']['target'] if run_config['quantization']['enabled'] else 'off',
            "retrieval_k": run_config['retrieval']['k'] if run_config['retrieval']['enabled'] else 0,
            "power_cap_W": run_config['power_cap_W'],
            "epochs": run_config['hyperparameters']['epochs'],
            # Results
            **results # Add all metrics from the results dictionary
        }

        # Log the results to the CSV file
        log_experiment(log_file, log_data)
        print(f"--- Logged results for run {run_config['run_id']} ---")

    print("\n--- Experimental sweep finished ---")

if __name__ == '__main__':
    main()
