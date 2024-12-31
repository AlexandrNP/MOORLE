Code for the paper "Data Imbalance in Drug Response Prediction – Multi-Objective Optimization Approach in Deep Learning Setting".
Deep learning model is a modified version of DeepTTC: https://github.com/jianglikun/DeepTTC

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
