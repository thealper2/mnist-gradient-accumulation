# PyTorch Gradient Accumulation

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A PyTorch implementation demonstrating gradient accumulation technique on MNIST dataset using Adam optimizer.

## What is Gradient Accumulation?

Gradient accumulation is a technique that:
- Simulates larger batch sizes by accumulating gradients over multiple steps
- Reduces GPU memory usage by processing smaller batches
- Performs weight updates only after N accumulation steps

**Mathematical Equivalent**:
Effective Batch Size = Batch Size × Accumulation Steps


## Installation

1. Clone the repository:
```bash
git clone https://github.com/thealper2/mnist-gradient-accumulation.git
cd mnist-gradient-accumulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python3 main.py train
python3 main.py evaluate
```

### Custom Training

```bash
python main.py \
    --batch-size 128 \
    --epochs 20 \
    --learning-rate 0.001 \
    --accumulation-steps 8 \
    --device cuda \
    --model-save-path best_model.pt
```