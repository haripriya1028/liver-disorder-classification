# Liver Disorder Classification Using ANFIS

An implementation of an Adaptive Neuro-Fuzzy Inference System (ANFIS) for liver disorder classification using the UCI Indian Liver Patient dataset, with Logistic Regression as a baseline for performance comparison.

## Overview

This project explores the application of neuro-fuzzy learning to a binary classification problem involving liver-related biochemical measurements.

ANFIS combines fuzzy logic, which represents relationships through membership functions and fuzzy rules, with neural network-based learning to adjust model parameters during training.

The project implements ANFIS in PyTorch and evaluates its performance against a Logistic Regression model using standard classification metrics.

## Objectives

- Implement a trainable ANFIS model using PyTorch.
- Use Gaussian membership functions to represent input features.
- Address class imbalance using class-weighted binary cross-entropy loss.
- Evaluate classification performance using multiple metrics.
- Visualize training loss, learned membership functions, and confusion matrices.
- Compare ANFIS with a Logistic Regression baseline.

## Dataset

The project uses the **Indian Liver Patient dataset** from the UCI Machine Learning Repository.

The dataset contains 583 instances and 10 input features. This implementation selects four biochemical features:

| Feature | Description |
|---|---|
| TB | Total Bilirubin |
| DB | Direct Bilirubin |
| Alkphos | Alkaline Phosphotase |
| Sgpt | Alanine Aminotransferase (ALT/SGPT) |

The target variable, `Selector`, is converted into a binary classification label:

- `0`: Original Selector 1
- `1`: Original Selector 2

**Note:** These labels preserve the dataset's original encoding. They should not be interpreted as definitive clinical diagnoses without verifying the dataset's target-label definitions.

## Methodology

### 1. Data Preprocessing
- Selected four biochemical input features.
- Checked for missing and infinite values.
- Replaced infinite values with missing values and imputed missing feature values using the median.
- Converted the target into binary labels.

### 2. Train-Test Split and Scaling
- Split the dataset into 80% training data and 20% testing data.
- Used stratified sampling to preserve class proportions.
- Applied `StandardScaler`, fitting it on the training data and transforming both datasets.

### 3. ANFIS Model
The ANFIS implementation uses:
- Gaussian membership functions with learnable centers and widths.
- Fuzzy rules generated from combinations of membership functions.
- Learnable linear consequent functions for individual rules.
- Normalized rule firing strengths to combine rule outputs into a final prediction.

With four input features and two membership functions per feature, the model generates 16 fuzzy rules.

### 4. Model Training
- Implemented the model using PyTorch.
- Used the Adam optimizer with a learning rate of 0.005.
- Trained for 400 epochs.
- Used `BCEWithLogitsLoss` with a positive-class weight to account for class imbalance.

### 5. Evaluation and Comparison
The ANFIS model is compared with Logistic Regression trained using balanced class weights.

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Visualizations include the training loss curve, learned Gaussian membership functions, confusion matrices, and a comparison of evaluation metrics.

## Results

The models were evaluated on the same held-out test set of 117 samples.

| Metric | ANFIS | Logistic Regression |
|---|---:|---:|
| Accuracy | 0.6325 | 0.6239 |
| Precision | 0.4308 | 0.4306 |
| Recall | 0.8235 | 0.9118 |
| F1-score | 0.5657 | 0.5849 |
| ROC-AUC | 0.6662 | 0.7318 |

### Observations

- **Accuracy:** ANFIS achieved slightly higher accuracy than Logistic Regression.
- **Recall:** Logistic Regression identified a larger proportion of positive-class instances.
- **F1-score:** Logistic Regression achieved a slightly higher F1-score.
- **ROC-AUC:** Logistic Regression showed better discrimination between the two classes based on predicted probabilities.

Overall, ANFIS demonstrated the ability to learn a neuro-fuzzy classification model, but Logistic Regression performed better on recall, F1-score, and ROC-AUC in this experiment.

These results are based on a single train-test split and should not be treated as evidence of clinical effectiveness.

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- ucimlrepo

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib ucimlrepo torch
```

### 3. Run the notebook

Open the notebook in Jupyter Notebook or Google Colab and execute the cells in order.

The dataset is fetched from the UCI Machine Learning Repository using `ucimlrepo`, so an internet connection is required when downloading the dataset.

## Future Improvements

- Experiment with different numbers of membership functions and fuzzy rules.
- Tune the classification threshold to investigate precision-recall trade-offs.
- Evaluate performance using stratified cross-validation.
- Investigate alternative feature combinations and feature-selection methods.
- Compare ANFIS with additional classification algorithms.
- Assess model stability across different random train-test splits.

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not be used to make clinical decisions.
