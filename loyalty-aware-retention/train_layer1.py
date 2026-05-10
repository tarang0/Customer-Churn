"""
Layer 1: Churn Prediction on Telco Customer Churn Dataset
Predicts whether a customer will churn (cancel subscription) based on 19 features.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    accuracy_score, f1_score
)
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'


def load_and_explore(path):
    print("=" * 60)
    print("LOADING TELCO CUSTOMER CHURN DATASET")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"Shape: {df.shape[0]:,} customers, {df.shape[1]} columns\n")

    print(f"Churn distribution:")
    churn_counts = df['Churn'].value_counts()
    for label, count in churn_counts.items():
        print(f"  {label}: {count:,} ({count / len(df):.1%})")

    print(f"\nColumns by type:")
    for col in df.columns:
        print(f"  {col:25s} {str(df[col].dtype):10s} unique={df[col].nunique()}")

    return df


def preprocess(df):
    print("\n" + "=" * 60)
    print("PREPROCESSING")
    print("=" * 60)

    df = df.copy()
    df.drop('customerID', axis=1, inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing = df['TotalCharges'].isna().sum()
    if missing > 0:
        print(f"TotalCharges: {missing} missing values → filled with median")
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    print(f"\nEncoding {len(cat_cols)} categorical columns: {cat_cols}")

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    feature_cols = [c for c in df.columns if c != 'Churn']
    print(f"\nFinal features ({len(feature_cols)}): {feature_cols}")
    print(f"Target: Churn (0=stayed, 1=churned)")
    print(f"Class balance: {df['Churn'].mean():.1%} churned")

    return df, feature_cols, encoders


def train_and_evaluate(df, feature_cols):
    print("\n" + "=" * 60)
    print("LAYER 1: TRAINING CHURN PREDICTION MODEL")
    print("=" * 60)

    X = df[feature_cols]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"Train churn rate: {y_train.mean():.1%}  Test churn rate: {y_test.mean():.1%}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        random_state=42,
        verbosity=0,
        eval_metric='logloss',
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "-" * 40)
    print("RESULTS")
    print("-" * 40)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n  AUC-ROC:   {auc:.4f}")
    print(f"  Accuracy:  {acc:.4f} ({acc:.1%})")
    print(f"  F1-Score:  {f1:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stayed', 'Churned']))

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"                 Predicted Stayed  Predicted Churned")
    print(f"  Actual Stayed       {cm[0][0]:>5}           {cm[0][1]:>5}")
    print(f"  Actual Churned      {cm[1][0]:>5}           {cm[1][1]:>5}")

    print(f"\nFeature Importance (top 10):")
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    for feat, imp in importance.head(10).items():
        bar = "█" * int(imp * 50)
        print(f"  {feat:25s} {bar} {imp:.4f}")

    return model, X_test, y_test


def main():
    df = load_and_explore(DATA_PATH)
    df, feature_cols, encoders = preprocess(df)
    model, X_test, y_test = train_and_evaluate(df, feature_cols)

    print("\n" + "=" * 60)
    print("LAYER 1 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
