"""
Neural network model for predicting the residuals of the RC model.

This module implements a feed-forward neural network using PyTorch to
learn the non-linear dynamics not captured by the physics-based RC model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

class ResidualNN(nn.Module):
    """
    A simple feed-forward neural network to model the residuals of the RC model.
    Can be configured as a one or two hidden layer network.
    """
    def __init__(self, input_size, hidden_size1=16, hidden_size2=None):
        super(ResidualNN, self).__init__()
        self.hidden_size2 = hidden_size2

        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()

        if self.hidden_size2:
            self.layer2 = nn.Linear(hidden_size1, hidden_size2)
            self.relu2 = nn.ReLU()
            self.output_layer = nn.Linear(hidden_size2, 1)
        else:
            self.output_layer = nn.Linear(hidden_size1, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu1(out)
        if self.hidden_size2:
            out = self.layer2(out)
            out = self.relu2(out)
        out = self.output_layer(out)
        return out

def train_nn_model(model, X_train, y_train, epochs=50, batch_size=32, learning_rate=0.001, weight_decay=1e-5):
    """
    Generic training loop for the neural network.

    Args:
        model (nn.Module): The PyTorch model to train.
        X_train (pd.DataFrame): The training feature data.
        y_train (pd.Series): The training target data.
        epochs (int): Number of training epochs.
        batch_size (int): Size of training batches.
        learning_rate (float): Learning rate for the optimizer.
        weight_decay (float): L2 regularization strength.

    Returns:
        nn.Module: The trained model.
    """
    # Convert data to PyTorch tensors
    X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print("\n--- Training Neural Network ---")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (inputs, targets) in enumerate(dataloader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}')

    print("--- NN Training Complete ---\n")
    return model
