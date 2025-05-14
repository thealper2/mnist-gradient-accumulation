"""
Data loading and processing module for MNIST dataset.
"""

import logging
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

logger = logging.getLogger("mnist_training")


def get_mnist_dataloaders(
    data_path: str,
    batch_size: int,
    seed: int = 42,
    val_split: float = 0.1,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load and prepare MNIST dataset, splitting into train, validation, and test sets.

    Args:
        data_path: Directory path to store the MNIST dataset
        batch_size: Batch size for the data loaders
        seed: Random seed for reproducibility
        val_split: Fraction of training data to use for validation
        num_workers: Number of workers for data loading

    Returns:
        A tuple containing (train_loader, val_loader, test_loader)
    """
    try:
        # Set the random seed for reproducibility
        torch.manual_seed(seed)

        # Create path if it doesn't exist
        Path(data_path).mkdir(parents=True, exist_ok=True)

        # Define transforms
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )

        # Download and load the training data
        logger.info(f"Loading MNIST dataset from {data_path}")
        train_dataset = datasets.MNIST(
            root=data_path, train=True, download=True, transform=transform
        )

        # Split into train and validation sets
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size

        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

        # Download and load the test data
        test_dataset = datasets.MNIST(
            root=data_path, train=False, download=True, transform=transform
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        logger.info(
            f"Dataset loaded. Train size: {len(train_dataset)}, "
            f"Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}"
        )

        return train_loader, val_loader, test_loader

    except Exception as e:
        logger.error(f"Failed to load MNIST dataset: {str(e)}")
        raise
