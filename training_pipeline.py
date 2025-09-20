import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from eamtf_model import EAMTF
from logging_utils import initialize_logging, log_experiment
from energy_monitor import EnergyMonitor

def load_config(config_path='config.yaml'):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_data_loaders(config):
    """Loads data and creates PyTorch DataLoaders."""
    try:
        df = pd.read_parquet(config['data_path'])
    except Exception as e:
        print(f"Could not load data from {config['data_path']}. Make sure to run feature_engineering.py first.")
        print(f"Error: {e}")
        return None, None

    # This is a placeholder for actual data preparation
    # In a real scenario, we would create sequences (X, y)
    # For now, create dummy tensors to test the training loop structure
    num_samples = 100
    seq_len = 96
    input_features = 10 # Should match feature engineering
    output_horizon = 4

    X = torch.randn(num_samples, seq_len, input_features)
    # y will have 3 tasks (Load, PV, Net-Load)
    y = torch.randn(num_samples, output_horizon, 3)

    dataset = TensorDataset(X, y)
    data_loader = DataLoader(
        dataset,
        batch_size=config['hyperparameters']['batch_size'],
        shuffle=True
    )
    return data_loader, data_loader # Returning same for train/valid for now

def train_one_epoch(model, data_loader, optimizer, criterion, scaler, device):
    """Runs a single training epoch with mixed precision."""
    model.train()
    total_loss = 0

    for x_batch, y_batch in data_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        # Use autocast for the forward pass
        with autocast():
            # Dummy difficulty signals for now
            difficulty_signals = {'csi': 0.8}
            forecast, _ = model(x_batch, difficulty_signals)
            loss = criterion(forecast, y_batch)

        # Scale the loss and call backward()
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(data_loader)

def run_training_job(config):
    """
    Runs a complete training job based on a given configuration.

    Returns:
        dict: A dictionary of results and metrics from the run.
    """
    initialize_logging(config['logging']['log_file'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Run: {config['run_id']} on device {device} ---")

    # Data
    train_loader, valid_loader = get_data_loaders(config)
    if train_loader is None:
        return {}

    # Model
    # These sizes are hardcoded for now, should come from data prep
    model = EAMTF(
        input_size=10,
        hidden_size=64,
        output_size=4,
        lora_rank=config['lora']['rank']
    ).to(device)

    # Apply optimizations based on config
    if config['quantization']['enabled'] and config['quantization']['target'] == 'shallow_head':
        # In a real scenario, we'd train first, then apply PTQ.
        # For now, we just call the method to show it's wired up.
        # model.quantize_shallow_head() # This would be done after training
        pass

    # Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=config['hyperparameters']['learning_rate'])
    criterion = nn.MSELoss()

    # GradScaler for mixed precision
    use_amp = config['precision'] in ['BF16', 'FP16']
    scaler = GradScaler(enabled=use_amp)

    print("Starting training...")
    final_results = {}
    with EnergyMonitor() as monitor:
        for epoch in range(config['hyperparameters']['epochs']):
            avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
            print(f"Epoch {epoch+1}/{config['hyperparameters']['epochs']}, Avg. Loss: {avg_loss:.4f}")

    print("Training finished.")

    # Collate results
    final_results.update(monitor.results)
    # Placeholder for actual performance metrics
    final_results.update({
        'RMSE': 0.0, 'MAE': 0.0, 'R2': 0.0,
        'latency_ms_p50': 0.0, 'latency_ms_p95': 0.0
    })

    print(f"--- Finished Run: {config['run_id']} ---")
    return final_results

if __name__ == '__main__':
    # This block can be used for testing the pipeline directly
    print("Running a single training job directly for testing...")
    test_config = load_config()
    results = run_training_job(test_config)

    print("\n--- Final Run Results ---")
    if results:
        for key, value in results.items():
            print(f"{key}: {value}")
