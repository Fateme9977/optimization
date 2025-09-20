import torch
import torch.nn as nn

class BaseTimeSeriesModel(nn.Module):
    """
    A simple baseline model for time-series forecasting using a recurrent core.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, model_type='LSTM'):
        super().__init__()
        if model_type not in ['LSTM', 'GRU']:
            raise ValueError("model_type must be 'LSTM' or 'GRU'")

        RecurrentLayer = getattr(nn, model_type)
        self.recurrent = RecurrentLayer(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input sequence, shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: The forecast, shape (batch_size, output_size).
        """
        # We only need the last hidden state to make the prediction.
        # The output of the recurrent layer is (output, (h_n, c_n)) for LSTM
        # or (output, h_n) for GRU. We'll use the final hidden state.
        _, (last_hidden, _) = self.recurrent(x) if isinstance(self.recurrent, nn.LSTM) else self.recurrent(x)

        # Use the hidden state of the last layer
        prediction = self.fc(last_hidden[-1])
        return prediction

if __name__ == '__main__':
    # --- Example Usage ---
    batch_size = 16
    seq_len = 96
    input_features = 10
    output_horizon = 4

    # --- Test LSTM Baseline ---
    print("Testing LSTM Baseline Model...")
    lstm_baseline = BaseTimeSeriesModel(
        input_size=input_features,
        hidden_size=64,
        output_size=output_horizon,
        model_type='LSTM'
    )
    dummy_input = torch.randn(batch_size, seq_len, input_features)
    output = lstm_baseline(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, output_horizon)
    print("LSTM Baseline verified.")

    # --- Test GRU Baseline ---
    print("\nTesting GRU Baseline Model...")
    gru_baseline = BaseTimeSeriesModel(
        input_size=input_features,
        hidden_size=64,
        output_size=output_horizon,
        model_type='GRU'
    )
    output = gru_baseline(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, output_horizon)
    print("GRU Baseline verified.")
