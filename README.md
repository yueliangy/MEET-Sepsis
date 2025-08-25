# MEET-Sepsis: Multi-Endogenous-View Enhanced Time-Series Representation Learning for Early Sepsis Prediction

This repository provides the implementation of **MEET-Sepsis**, a novel model for early sepsis prediction. The method enhances weak early-stage temporal features using a Multi-Endogenous-view Representation Enhancement (MERE) mechanism and a Cascaded Dual-convolution Time-series Attention (CDTA) module.

## Project Structure

- `train.py` — Main script to train the model.
- `TSRL.py` — Model definition and core representation learning modules.
- `eval_fun.py` — Evaluation metrics and functions.

## Requirements

- Python 3.8+
- PyTorch >= 1.10
- NumPy
- Scikit-learn
- XGBoost

## Usage

python train.py
