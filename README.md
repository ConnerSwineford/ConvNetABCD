# Deep Learning Identification of Reward-Related Neural Substrates of Preadolescent Irritability: A Novel 3D CNN Application for fMRI
**Authors:** *Johanna Walker, Conner Swineford, Krupali Patel, Lea R. Dougherty, Jillian Lee Wiggins*


## Overview

This repository contains scripts to train, evaluate, and utilize a 3D Convolutional Neural Network (CNN) model for medical imaging tasks. The model is designed to process 3D medical images in NIfTI format and includes utilities for data handling and performance evaluation.

## Getting Started

### Installation

Clone the repository and navigate to the project directory:
```sh
git clone https://github.com/ConnerSwineford/ConvNetABCD.git
cd ConvNetABCD
```

### Prerequisites

Ensure you have Python 3.6+ installed.

Users can run this script to install all the required packages by executing:
```sh
python setup.py
```

### Example

#### Training
To train the model, run:
```sh
python train.py --data_path /path/to/dataframe.csv --batch_size 16 --epochs 10 --lr 0.001 --outdir ./output --workers 4
```

#### Evaluation
To evaluate the model, run:
```sh
python evaluate.py --data_path /path/to/dataframe.csv --weights ./output/model_weights.pth --outdir ./evaluation_output --workers 4
```

## Files and Scripts

### 1. `train.py`
This script handles the training process of the 3D CNN model.

- **Arguments**:
  - `--data_path`: Path to the CSV file containing training data information.
  - `--batch_size`: Number of samples per batch.
  - `--epochs`: Number of training epochs.
  - `--lr`: Learning rate.
  - `--outdir`: Directory to save model weights and logs.
  - `--workers`: Number of worker processes for distributed training.
  - `--weights`: Optional path to pre-trained model weights.
  - `--seed`: Optional random seed for reproducibility.

### 2. `evaluate.py`
This script is used for evaluating the trained 3D CNN model.

- **Arguments**:
  - `--data_path`: Path to the evaluation data directory.
  - `--weights`: Path to the trained model weights.
  - `--workers`: Number of worker processes for evaluation.
  - `--outdir`: Directory to save evaluation outputs.
  - `--batch_size`: Batch size for evaluation.
  - `--affine`: Path to the affine transformation file.

### 3. `model.py`
This script defines the architecture of the 3D CNN model and its components.

### 4. `utils.py`
This script contains utility functions for data processing and handling.
