"""
Model definition for MNIST classification.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTModel(nn.Module):
    """
    A simple feed-forward neural network for MNIST digit classification.

    Architecture:
    - Input layer: 784 neurons (28x28 flattened image)
    - Hidden layer: configurable size
    - Output layer: 10 neurons (one for each digit)
    """

    def __init__(self, hidden_size: int = 128, dropout_rate: float = 0.3):
        """
        Initialize the model with specified hidden layer size.

        Args:
            hidden_size: Number of neurons in the hidden layer
            dropout_rate: Dropout probability for regularization
        """
        super(MNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, 1, 28, 28]

        Returns:
            Output tensor of shape [batch_size, 10]
        """
        # Flatten the input: [batch_size, 1, 28, 28] -> [batch_size, 784]
        x = self.flatten(x)

        # First hidden layer with ReLU activation
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Second hidden layer with ReLU activation
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Output layer (logits)
        x = self.fc3(x)

        return x

    def get_loss_and_accuracy(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Calculate loss and accuracy for a given batch.

        Args:
            batch: Tuple of (inputs, targets)

        Returns:
            Dictionary containing loss and accuracy metrics
        """
        inputs, targets = batch

        # Forward pass
        outputs = self(inputs)

        # Calculate loss
        loss = F.cross_entropy(outputs, targets)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        accuracy = correct / targets.size(0)

        return {
            "loss": loss,
            "accuracy": accuracy,
            "correct": correct,
            "total": targets.size(0),
        }
