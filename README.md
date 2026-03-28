# Credit Card Fraud Detection 

## Project Overview
This project aims to detect fraudulent credit card transactions using highly imbalanced anonymized data. We implemented an end-to-end Machine Learning pipeline utilizing an XGBoost classifier, addressing extreme class imbalance through SMOTE, and providing deep model explainability and fairness audits.

## Repository Structure
- `data/`: Contains the dataset (Note: `creditcard.csv` must be downloaded from Kaggle/source and placed here).
- `models/`: Serialized pre-trained models (`best_model.joblib`).
- `notebooks/`: Jupyter notebooks for exploratory data analysis, visualizations, and step-by-step experimentation.
- `src/`: Reproducible Python scripts for data preprocessing and model training.
- `requirements.txt`: Python dependencies required to run the code.

## Key Findings & Metrics
- **Model**: XGBoost Classifier
- **Primary Metric (AUPRC)**: 0.8171
- **Precision**: 71.7%
- **Recall**: 80.0%
- **Class Imbalance Handling**: Applied SMOTE strictly to the training set to prevent data leakage.

## Explainability (SHAP & LIME)
- Utilized **SHAP** to understand global feature importance. Features like `V14`, `V10`, and `V4` were identified as top predictors.
- Integrated **LIME** for local explainability to interpret individual transaction predictions, ensuring the model's logic is transparent to compliance and business teams.

## Bias & Fairness Audit
- Conducted a fairness audit using transaction amount (`Low` vs `High`) as a proxy sensitive group.
- **Baseline Disparate Impact**: 0.8929 (Slight disadvantage to 'Low' transaction group).
- **Mitigation Strategy**: Applied threshold adjustment (0.35 for Low, 0.50 for High), improving Disparate Impact to **0.9821**, achieving high parity.

## Reproducing the Results
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Place the `creditcard.csv` dataset in the `data/` folder.
4. Navigate to `src/` and run `python train_evaluate.py` to train the model and output metrics.
