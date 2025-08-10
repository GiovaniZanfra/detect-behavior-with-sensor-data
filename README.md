# Project Overview

This repository contains the codebase for a gesture recognition system designed for the **CMI Competition**.  
It focuses on training and evaluating machine learning models to classify gesture sequences based on a combination of IMU (Inertial Measurement Unit) and other sensor data.  
The main objective is to maximize classification accuracy while ensuring the pipeline is modular, reproducible, and easy to adapt to new data sources or model architectures.

---

## Context

This project was created to address a competition task involving **sequence classification**.  
The input data is a combination of IMU sensor readings and possibly other time-series features.  
The goal is to **predict the correct gesture class** for each sequence.

Unlike standard gesture recognition approaches, this repository focuses on:
- Handling large datasets efficiently.
- Providing a modular approach for experimentation with various modeling strategies.
- Supporting rapid prototyping for both feature engineering and model training.

---

## Repository Structure

### `preprocessing/`
Handles **data loading, cleaning, and transformation**.  
This includes:
- Removing invalid or inconsistent samples.
- Handling missing values.
- Feature scaling and normalization.
- Generating derived features.

### `feature_selection/`
Implements **optional feature selection steps**.  
The code supports different methods like:
- Lasso-based selection.
- Sequential Feature Selection (SFS).
- Bootstrapping for feature stability analysis.  
> **Note**: For large datasets, you may choose to skip feature selection entirely.

### `model_training/`
Core training scripts for multiple algorithms, including:
- **LightGBM** for tabular features.
- Potential integration with **deep learning** for raw sequences.
- Support for **K-Fold cross-validation** (can be skipped if hyperparameter tuning is not required).

### `inference/`
Handles the generation of predictions for submission:
- Loads trained models.
- Runs inference on test data.
- Outputs results in the required submission format.

---

## Optimization Priorities

When using this repository, the main areas to optimize for performance are:

1. **Feature Engineering**
   - Improve signal representation (e.g., frequency-domain features, sensor fusion).
   - Experiment with data augmentation for sequences.

2. **Model Selection**
   - Compare boosting methods (LightGBM, CatBoost, XGBoost) for tabular data.
   - Consider RNNs, CNNs, or Transformers for sequential modeling.

3. **Ensembling**
   - Blend predictions from IMU-only models and full-feature models.
   - Stack multiple classifiers for improved generalization.

4. **Data Preprocessing**
   - Ensure train/test distributions match.
   - Normalize/scale consistently across splits.

---

## Requirements

To run this repository, ensure you have:

- **Python 3.8+**
- Installed dependencies:
  ```bash
  pip install -r requirements.txt
