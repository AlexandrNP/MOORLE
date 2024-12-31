Code for the paper "Data Imbalance in Drug Response Prediction – Multi-Objective Optimization Approach in Deep Learning Setting".
Deep learning model is a modified version of DeepTTC: https://github.com/jianglikun/DeepTTC

# MOORLE PyTorch Implementation

This repository contains the PyTorch implementation of Multi-Objective Optimization Regularized by Loss Entropy (MOORLE) loss function and associated dataset handling for drug response prediction.

## Overview

The implementation includes:
1. A custom PyTorch Dataset class (`DrugResponseDataset`) for handling drug response data
2. MOORLE loss function implementation with entropy-based regularization for handling imbalanced datasets

## Features

- Custom dataset class that handles:
  - Drug response data
  - RNA expression data
  - Drug encoding data
- MOORLE loss function with:
  - Drug-wise MSE computation
  - Entropy-based regularization
  - Configurable regularization strength

## Installation

```bash
pip install torch pandas numpy
```
Usage
1. Creating a Dataset
pythonCopyfrom drug_response_dataset import DrugResponseDataset

# Initialize dataset
```
dataset = DrugResponseDataset(
    labels=labels,
    response_df=response_df,
    rna_df=rna_df,
    drug_df=drug_df,
    device=device,
    dtype=torch.float
)
```
2. Using the MOORLE Loss

```pythonCopyfrom MOORLELoss import MOORLELoss

# Calculate loss
loss = MOORLELoss(
    predicted=model_predictions,
    drug_ids=batch_drug_ids,
    label=true_labels,
    device=device,
    alpha=1.0
)
```
Dataset Class Details
The DrugResponseDataset class handles:

*Label conversion to tensors
*Drug ID processing and integer conversion
*RNA and drug data conversion to TensorDicts
*Efficient data retrieval during training

Each sample contains:

Gene expression tensor
Drug encoding tensor
Response label
Drug ID
Gene ID

Loss Function Details
The MOORLELoss function combines:

Mean Squared Error (MSE) for each drug group
Entropy-based regularization to encourage balanced predictions

Key components:

Drug-wise loss computation
Softmax distribution of losses
Entropy calculation
Regularization based on maximum possible entropy

Parameters
DrugResponseDataset

labels: List of response labels
response_df: DataFrame with drug response data
rna_df: DataFrame with RNA expression data
drug_df: DataFrame with drug features
device: Torch device (CPU/GPU)
dtype: Tensor data type (default: torch.float)

MOORLELoss

predicted: Predicted values tensor
drug_ids: Drug identifiers tensor
label: True labels tensor
device: Torch device (CPU/GPU)
alpha: Regularization strength (default: 1)

Limitations

Requires GPU support for optimal performance
Memory usage scales with dataset size
Alpha parameter requires tuning for specific applications


# MOORLE LightGBM Implementation

This repository contains the implementation of Multi-Objective Optimization Regularized by Loss Entropy (MOORLE) loss function for LightGBM. The implementation allows training LightGBM models with group-aware loss function that helps handle imbalanced datasets.

## Overview

MOORLE is designed to handle imbalanced datasets by incorporating multi-objective optimization principles through entropy-based regularization. This implementation extends LightGBM's functionality to support group-aware training with the MOORLE loss function.

## Features

- Custom dataset class (`ExtendedDataset`) that supports group/domain information
- MOORLE loss function implementation with numerical gradient computation
- Support for both MSE and entropy-based regularization components
- Configurable regularization strength through the alpha parameter

## Installation

```bash
pip install lightgbm numpy scipy


Citations:

Narykov, Oleksandr, et al. "Data Imbalance in Drug Response Prediction-Multi-Objective Optimization Approach in Deep Learning Setting." bioRxiv (2024): 2024-03.

Jiang, Likun, et al. "DeepTTA: a transformer-based model for predicting cancer drug response." Briefings in Bioinformatics 23.3 (2022): bbac100.
