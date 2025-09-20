import torch
import torch.nn as nn
import loralib as lora

class ShallowHead(nn.Module):
    """A tiny MLP head for 'easy' time-steps. This head is a target for quantization."""
    def __init__(self, input_size, output_size):
        super().__init__()
        # Using standard nn.Linear to be compatible with quantization
        self.fc = nn.Linear(input_size, output_size * 3) # for Load, PV, Net-Load
        self.output_size = output_size

    def forward(self, x):
        # x is the hidden state from the backbone
        x = self.fc(x)
        # Reshape to (batch_size, output_size, 3) for the three tasks
        return x.view(x.size(0), self.output_size, 3)

class DeepHead(nn.Module):
    """A more complex head for 'difficult' time-steps, with LoRA capability."""
    def __init__(self, input_size, output_size, nhead=2, num_layers=2, r=4):
        super().__init__()
        # For simplicity, using a GRU here. Could be replaced with a Transformer.
        self.gru = nn.GRU(input_size, input_size, num_layers=num_layers, batch_first=True)
        self.fc = lora.Linear(input_size, output_size * 3, r=r)
        self.output_size = output_size

    def forward(self, x):
        # x is the sequence of hidden states from the backbone
        x, _ = self.gru(x)
        # We only need the last hidden state for forecasting
        x = x[:, -1, :]
        x = self.fc(x)
        return x.view(x.size(0), self.output_size, 3)

class EAMTF(nn.Module):
    """
    Energy-Adaptive Multi-Task Forecasting (EAMTF) model.

    Combines a shared backbone with an early-exit mechanism that routes inputs
    to either a shallow (fast) or deep (accurate) head based on difficulty signals.
    """
    def __init__(self, input_size, hidden_size, output_size, backbone_layers=2, lora_rank=4):
        super().__init__()
        # Note: loralib does not currently support GRU layers directly.
        # LoRA is applied to the Linear layers in the heads.
        self.backbone = nn.GRU(
            input_size, hidden_size,
            num_layers=backbone_layers,
            batch_first=True
        )
        self.shallow_head = ShallowHead(hidden_size, output_size)
        self.deep_head = DeepHead(hidden_size, output_size, r=lora_rank)

    def freeze_for_lora_tuning(self):
        """Freezes backbone and non-LoRA weights for fine-tuning."""
        print("Freezing model for LoRA-only tuning...")
        lora.mark_only_lora_as_trainable(self)
        print("Model frozen. Only LoRA parameters are trainable.")

    def quantize_shallow_head(self):
        """Applies dynamic quantization to the shallow head.

        This is typically done post-training.
        """
        print("Applying dynamic quantization to the shallow head...")
        self.shallow_head = torch.quantization.quantize_dynamic(
            self.shallow_head, {nn.Linear}, dtype=torch.qint8
        )
        print("Quantization of shallow head complete.")

    def forward(self, x, difficulty_signals):
        """
        Forward pass with early-exit logic.

        Args:
            x (torch.Tensor): Input sequence, shape (batch_size, seq_len, input_size).
            difficulty_signals (dict): A dictionary of signals like 'csi', 'ghi_slope'.
                                       Used to decide which head to use.

        Returns:
            torch.Tensor: The forecast from either the shallow or deep head.
            str: The name of the head that was used ('shallow' or 'deep').
        """
        # --- Gating Logic ---
        # This is a placeholder for the full gating logic.
        # For now, we'll use a simple CSI threshold.
        use_deep_head = difficulty_signals.get('csi', 1.0) < 0.7

        # --- Backbone ---
        # The shallow head only needs the final hidden state.
        # The deep head could potentially use the full sequence of hidden states.
        backbone_output, last_hidden = self.backbone(x)

        if use_deep_head:
            # Route to deep head
            # Note: The deep head could take `backbone_output` for more context
            forecast = self.deep_head(backbone_output)
            used_head = 'deep'
        else:
            # Route to shallow head (early exit)
            # We use the last hidden state from the backbone's final layer
            forecast = self.shallow_head(last_hidden[-1])
            used_head = 'shallow'

        # --- Night PV Skip ---
        # Placeholder for skipping PV head computation at night (GHI ~ 0)
        # This would typically be handled by masking the loss for the PV head.
        if difficulty_signals.get('is_night', False):
            # We don't change the output tensor shape, but the loss calculation
            # for the PV component should be skipped. We can signal this.
            pass # Logic to be implemented in the training loop

        return forecast, used_head

if __name__ == '__main__':
    # --- Example Usage ---
    batch_size = 16
    seq_len = 96 # 24 hours of 15-min data
    input_features = 10 # e.g., lags, time features, weather
    output_horizon = 4 # 1 hour forecast

    model = EAMTF(input_size=input_features, hidden_size=64, output_size=output_horizon)

    # Simulate a batch of data
    dummy_input = torch.randn(batch_size, seq_len, input_features)

    # --- Test Case 1: Easy (Clear sky) -> Should use shallow head ---
    print("Testing with 'easy' signal (CSI=0.8)...")
    easy_signals = {'csi': 0.8, 'is_night': False}
    forecast_easy, head_easy = model(dummy_input, easy_signals)
    print(f"Used head: {head_easy}")
    print(f"Output shape: {forecast_easy.shape}") # (batch, horizon, 3 tasks)
    assert head_easy == 'shallow'
    assert forecast_easy.shape == (batch_size, output_horizon, 3)

    # --- Test Case 2: Difficult (Cloudy) -> Should use deep head ---
    print("\nTesting with 'difficult' signal (CSI=0.4)...")
    difficult_signals = {'csi': 0.4, 'is_night': False}
    forecast_deep, head_deep = model(dummy_input, difficult_signals)
    print(f"Used head: {head_deep}")
    print(f"Output shape: {forecast_deep.shape}")
    assert head_deep == 'deep'
    assert forecast_deep.shape == (batch_size, output_horizon, 3)

    print("\nModel structure and basic early-exit logic verified.")

    # --- Test Case 3: Quantization of Shallow Head ---
    print("\n--- Testing Quantization ---")
    # Note: Quantization is done in-place
    print("Original shallow head:")
    print(model.shallow_head)

    model.quantize_shallow_head()

    print("\nQuantized shallow head:")
    print(model.shallow_head)
    # Verify that the submodule has been replaced
    assert "DynamicQuantizedLinear" in str(model.shallow_head.fc)
    print("\nShallow head quantization verified.")

    # --- Test Case 4: LoRA Integration ---
    print("\n--- Testing LoRA Integration ---")
    lora_model = EAMTF(input_size=input_features, hidden_size=64, output_size=output_horizon, lora_rank=8)

    total_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"Original trainable parameters: {total_params}")

    lora_model.freeze_for_lora_tuning()

    lora_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"Trainable parameters after LoRA freeze: {lora_params}")

    assert lora_params > 0
    assert lora_params < total_params
    print("\nLoRA integration verified.")
