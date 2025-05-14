#!/usr/bin/env python3
"""
Main entry point for MNIST training with gradient accumulation.
This module handles command line arguments and initiates the training process.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.logging import RichHandler

from data import get_mnist_dataloaders
from model import MNISTModel
from trainer import Trainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("mnist_training")

app = typer.Typer(help="Train a model on MNIST dataset with gradient accumulation")


@app.command()
def train(
    batch_size: int = typer.Option(64, help="Batch size for training"),
    virtual_batch_size: int = typer.Option(
        256, help="Virtual batch size for gradient accumulation"
    ),
    learning_rate: float = typer.Option(0.001, help="Learning rate for optimizer"),
    epochs: int = typer.Option(5, help="Number of training epochs"),
    model_save_path: str = typer.Option(
        "./saved_models", help="Path to save the trained model"
    ),
    data_path: str = typer.Option("./data", help="Path to store/load MNIST data"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    log_interval: int = typer.Option(
        100, help="How often to log training progress (in batches)"
    ),
    hidden_size: int = typer.Option(128, help="Hidden layer size for the model"),
    use_cuda: bool = typer.Option(True, help="Use CUDA if available"),
):
    """
    Train a neural network on MNIST using gradient accumulation.

    This function loads the MNIST dataset, initializes the model and trainer,
    and runs the training process with the specified parameters.
    """
    try:
        # Create directories if they don't exist
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
        Path(data_path).mkdir(parents=True, exist_ok=True)

        # Calculate accumulation steps
        if virtual_batch_size % batch_size != 0:
            logger.warning(
                f"Virtual batch size ({virtual_batch_size}) is not divisible by batch size ({batch_size}). "
                f"Adjusting virtual batch size to {(virtual_batch_size // batch_size) * batch_size}"
            )
            virtual_batch_size = (virtual_batch_size // batch_size) * batch_size

        accumulation_steps = virtual_batch_size // batch_size
        logger.info(f"Using {accumulation_steps} gradient accumulation steps")

        # Get data loaders
        train_loader, val_loader, test_loader = get_mnist_dataloaders(
            data_path=data_path, batch_size=batch_size, seed=seed
        )

        # Initialize model
        model = MNISTModel(hidden_size=hidden_size)

        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=learning_rate,
            accumulation_steps=accumulation_steps,
            model_save_path=model_save_path,
            use_cuda=use_cuda,
            log_interval=log_interval,
        )

        # Train the model
        trainer.train(epochs=epochs)

        # Evaluate on test set
        test_accuracy = trainer.evaluate(test_loader)
        logger.info(f"Final test accuracy: {test_accuracy:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


@app.command()
def evaluate(
    model_path: str = typer.Option(..., help="Path to the saved model file"),
    data_path: str = typer.Option("./data", help="Path to MNIST data"),
    batch_size: int = typer.Option(128, help="Batch size for evaluation"),
    hidden_size: int = typer.Option(128, help="Hidden layer size of the model"),
    use_cuda: bool = typer.Option(True, help="Use CUDA if available"),
):
    """
    Evaluate a trained model on the MNIST test set.
    """
    try:
        # Get data loaders (we only need the test loader)
        _, _, test_loader = get_mnist_dataloaders(
            data_path=data_path, batch_size=batch_size
        )

        # Initialize model
        model = MNISTModel(hidden_size=hidden_size)

        # Initialize trainer (with dummy train/val loaders)
        trainer = Trainer(
            model=model,
            train_loader=None,
            val_loader=None,
            test_loader=test_loader,
            learning_rate=0.001,  # Doesn't matter for evaluation
            accumulation_steps=1,  # Doesn't matter for evaluation
            model_save_path="",  # Doesn't matter for evaluation
            use_cuda=use_cuda,
        )

        # Load model weights
        trainer.load_model(model_path)

        # Evaluate on test set
        test_accuracy = trainer.evaluate(test_loader)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    app()
