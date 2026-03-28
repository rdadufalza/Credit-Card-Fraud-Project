import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, classification_report
import joblib
import os

def main():
    print("Loading data...")
    # Assuming data is placed in the data/ folder
    data_path = '../data/creditcard.csv'
    if not os.path.exists(data_path):
        print(f"Please place creditcard.csv in the data directory.")
        return

    df = pd.read_csv(data_path)
    df.drop_duplicates(inplace=True)

    print("Preprocessing...")
    rob_scaler = RobustScaler()
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("Training XGBoost model...")
    model = XGBClassifier(random_state=42)
    model.fit(X_train_smote, y_train_smote)

    print("Evaluating model...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auprc = average_precision_score(y_test, y_pred_proba)
    print(f"AUPRC: {auprc:.4f}")
    print("
Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Saving model...")
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/best_model.joblib')
    print("Done!")

if __name__ == '__main__':
    main()
