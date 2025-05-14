import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MNISTModel

logger = logging.getLogger("mnist_training")


class Trainer:
    """
    Trainer class that handles the training process for MNIST classification
    with gradient accumulation.
    """

    def __init__(
        self,
        model: MNISTModel,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        learning_rate: float,
        accumulation_steps: int,
        model_save_path: str,
        use_cuda: bool = True,
        log_interval: int = 100,
    ):
        """
        Initialize the trainer with model, data loaders, and training parameters.

        Args:
            model: The neural network model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            learning_rate: Learning rate for the optimizer
            accumulation_steps: Number of steps to accumulate gradients
            model_save_path: Directory to save model checkpoints
            use_cuda: Whether to use GPU if available
            log_interval: How often to log training progress (in batches)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.learning_rate = learning_rate
        self.accumulation_steps = accumulation_steps
        self.model_save_path = model_save_path
        self.log_interval = log_interval

        # Set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Create model save directory if it doesn't exist
        if model_save_path:
            Path(model_save_path).mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch with gradient accumulation.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        if self.train_loader is None:
            raise ValueError("Train loader is required for training")

        self.model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        # Set up progress bar
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}",
        )

        # Reset gradients at the beginning
        self.optimizer.zero_grad()

        for batch_idx, (data, target) in pbar:
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)

            # Get metrics (loss, accuracy)
            metrics = self.model.get_loss_and_accuracy((data, target))
            loss = metrics["loss"]

            # Scale the loss according to accumulation steps
            loss = loss / self.accumulation_steps

            # Backward pass
            loss.backward()

            # Update metrics
            running_loss += loss.item() * self.accumulation_steps
            running_correct += metrics["correct"]
            running_total += metrics["total"]

            # Update weights if we reached accumulation steps or at the last batch
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(
                self.train_loader
            ):
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Log progress
            if batch_idx % self.log_interval == 0:
                avg_loss = running_loss / (batch_idx + 1)
                accuracy = (
                    100.0 * running_correct / running_total if running_total > 0 else 0
                )

                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{accuracy:.2f}%"})

        # Calculate final metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = running_correct / running_total if running_total > 0 else 0

        logger.info(
            f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
        )

        return {"loss": epoch_loss, "accuracy": epoch_accuracy}

    def validate(self) -> Dict[str, float]:
        """
        Validate the model on the validation set.

        Returns:
            Dictionary with validation metrics
        """
        if self.val_loader is None:
            raise ValueError("Validation loader is required for validation")

        return self.evaluate(self.val_loader)

    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluate the model on the given data loader.

        Args:
            data_loader: DataLoader for evaluation

        Returns:
            Accuracy on the provided dataset
        """
        self.model.eval()

        correct = 0
        total = 0
        running_loss = 0.0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Get metrics
                metrics = self.model.get_loss_and_accuracy((data, target))

                # Update metrics
                running_loss += metrics["loss"].item()
                correct += metrics["correct"]
                total += metrics["total"]

        # Calculate final metrics
        avg_loss = running_loss / len(data_loader)
        accuracy = correct / total if total > 0 else 0

        return accuracy

    def train(self, epochs: int) -> List[Dict[str, Dict[str, float]]]:
        """
        Train the model for the specified number of epochs.

        Args:
            epochs: Number of training epochs

        Returns:
            List of dictionaries containing training and validation metrics for each epoch
        """
        if self.train_loader is None:
            raise ValueError("Train loader is required for training")

        logger.info(f"Starting training for {epochs} epochs with gradient accumulation")
        logger.info(f"Accumulation steps: {self.accumulation_steps}")

        history = []
        best_val_accuracy = 0.0

        for epoch in range(1, epochs + 1):
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_accuracy = self.validate() if self.val_loader is not None else 0.0

            logger.info(f"Epoch {epoch} - Validation Accuracy: {val_accuracy:.4f}")

            # Save metrics
            epoch_metrics = {"train": train_metrics, "val": {"accuracy": val_accuracy}}
            history.append(epoch_metrics)

            # Save model if it's the best so far
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model(os.path.join(self.model_save_path, "best_model.pt"))
                logger.info(
                    f"Saved best model with validation accuracy: {val_accuracy:.4f}"
                )

            # Save latest model
            self.save_model(os.path.join(self.model_save_path, "latest_model.pt"))

        logger.info("Training completed")
        return history

    def save_model(self, path: str) -> None:
        """
        Save the model to the specified path.

        Args:
            path: Path to save the model
        """
        try:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                path,
            )
            logger.debug(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise

    def load_model(self, path: str) -> None:
        """
        Load the model from the specified path.

        Args:
            path: Path to the saved model
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Only load optimizer if it exists in the checkpoint and we're training
            if "optimizer_state_dict" in checkpoint and self.train_loader is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
