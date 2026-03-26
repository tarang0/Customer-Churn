"""
Combined training pipeline — Layers 1 + 2 + 3
Saves all artifacts for the Streamlit dashboard.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    accuracy_score, f1_score, silhouette_score
)
from xgboost import XGBClassifier
import shap

warnings.filterwarnings('ignore')

DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
ARTIFACTS_DIR = 'artifacts'


def load_and_preprocess(path):
    print("=" * 60)
    print("LOADING & PREPROCESSING")
    print("=" * 60)

    df = pd.read_csv(path)
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col])
        encoders[col] = le

    print(f"Loaded {len(df):,} customers, {len(cat_cols)} categoricals encoded")
    print(f"Churn rate: {df['Churn'].mean():.1%}")

    return df, encoders


def train_churn_model(df, encoders):
    print("\n" + "=" * 60)
    print("LAYER 1: CHURN PREDICTION")
    print("=" * 60)

    feature_cols = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
        'gender_enc', 'Partner_enc', 'Dependents_enc', 'PhoneService_enc',
        'MultipleLines_enc', 'InternetService_enc', 'OnlineSecurity_enc',
        'OnlineBackup_enc', 'DeviceProtection_enc', 'TechSupport_enc',
        'StreamingTV_enc', 'StreamingMovies_enc', 'Contract_enc',
        'PaperlessBilling_enc', 'PaymentMethod_enc',
    ]

    feature_display_names = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod',
    ]

    X = df[feature_cols]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        random_state=42, verbosity=0, eval_metric='logloss',
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1:       {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Stayed', 'Churned']))

    return model, feature_cols, feature_display_names


def train_clustering(df):
    print("\n" + "=" * 60)
    print("LAYER 2: CUSTOMER SEGMENTATION")
    print("=" * 60)

    cluster_features = pd.DataFrame()
    cluster_features['tenure'] = df['tenure']
    cluster_features['MonthlyCharges'] = df['MonthlyCharges']
    cluster_features['TotalCharges'] = df['TotalCharges']

    service_cols = ['PhoneService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies']
    cluster_features['num_services'] = sum(
        (df[col] == 'Yes').astype(int) if col in df.columns
        else (df[col] == 1).astype(int)
        for col in service_cols
    )
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    if df['Contract'].dtype == object:
        cluster_features['contract_length'] = df['Contract'].map(contract_map)
    else:
        cluster_features['contract_length'] = df['Contract_enc']

    cluster_features['has_internet'] = (
        (df['InternetService'] != 'No').astype(int) if df['InternetService'].dtype == object
        else (df['InternetService_enc'] != 0).astype(int)
    )
    cluster_features['avg_monthly_revenue'] = np.where(
        df['tenure'] > 0, df['TotalCharges'] / df['tenure'], df['MonthlyCharges']
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_features)

    best_k, best_sil = 3, 0
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        if sil > best_sil:
            best_sil, best_k = sil, k

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['cluster'] = km_final.fit_predict(X_scaled)

    print(f"  Optimal K: {best_k} (silhouette={best_sil:.4f})")
    for c in range(best_k):
        mask = df['cluster'] == c
        print(f"  Cluster {c}: {mask.sum():,} customers, "
              f"churn={df.loc[mask, 'Churn'].mean():.1%}, "
              f"avg_total=${df.loc[mask, 'TotalCharges'].mean():,.0f}")

    return km_final, scaler, cluster_features.columns.tolist(), best_k, df


def setup_shap(model, df, feature_cols):
    print("\n" + "=" * 60)
    print("LAYER 3: SHAP EXPLAINER")
    print("=" * 60)

    explainer = shap.TreeExplainer(model)
    sample = df[feature_cols].iloc[:10]
    sv = explainer.shap_values(sample)
    print(f"  SHAP ready — shape: {np.array(sv).shape}")
    return explainer


def train_all_models(df, feature_cols, feature_display_names):
    print("\n" + "=" * 60)
    print("MULTI-MODEL COMPARISON")
    print("=" * 60)

    X = df[feature_cols]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight='balanced', random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42, verbosity=0, eval_metric='logloss',
        ),
        'SVM (RBF)': SVC(
            kernel='rbf', class_weight='balanced', probability=True, random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=11, weights='distance'
        ),
    }

    needs_scaling = {'Logistic Regression', 'SVM (RBF)', 'K-Nearest Neighbors'}

    results = []
    for name, model in models.items():
        Xtr = X_train_scaled if name in needs_scaling else X_train
        Xte = X_test_scaled if name in needs_scaling else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        f1_val = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        prec = float(report['1']['precision'])
        rec = float(report['1']['recall'])

        results.append({
            'Model': name, 'AUC-ROC': auc, 'Accuracy': acc,
            'F1': f1_val, 'Precision': prec, 'Recall': rec,
        })
        print(f"  {name:25s}  AUC={auc:.4f}  Acc={acc:.4f}  F1={f1_val:.4f}")

    results_df = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)
    best = results_df.iloc[0]
    print(f"\n  BEST MODEL: {best['Model']} (AUC={best['AUC-ROC']:.4f})")

    return results_df, scaler


def save_all(churn_model, feature_cols, feature_display_names,
             km_model, km_scaler, cluster_feature_names, best_k,
             shap_explainer, df, encoders,
             model_comparison=None, feature_scaler=None):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    artifacts = {
        'churn_model': churn_model,
        'feature_cols': feature_cols,
        'feature_display_names': feature_display_names,
        'km_model': km_model,
        'km_scaler': km_scaler,
        'cluster_feature_names': cluster_feature_names,
        'best_k': best_k,
        'shap_explainer': shap_explainer,
        'df': df,
        'encoders': encoders,
        'model_comparison': model_comparison,
        'feature_scaler': feature_scaler,
    }

    path = os.path.join(ARTIFACTS_DIR, 'all_artifacts.pkl')
    with open(path, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"\nSaved all artifacts to {path}")


def main():
    df_raw = pd.read_csv(DATA_PATH)
    df, encoders = load_and_preprocess(DATA_PATH)

    df_with_orig = df.copy()
    skip_cols = {'TotalCharges', 'Churn', 'customerID'}
    for col in df_raw.select_dtypes(include='object').columns:
        if col not in skip_cols:
            df_with_orig[col] = df_raw[col].values

    churn_model, feature_cols, feature_display_names = train_churn_model(df, encoders)
    df_with_orig['TotalCharges'] = pd.to_numeric(df_with_orig['TotalCharges'], errors='coerce').fillna(df_with_orig['TotalCharges'].median())
    km_model, km_scaler, cluster_feature_names, best_k, df_with_orig = train_clustering(df_with_orig)
    shap_explainer = setup_shap(churn_model, df, feature_cols)

    model_comparison, feature_scaler = train_all_models(df, feature_cols, feature_display_names)

    save_all(churn_model, feature_cols, feature_display_names,
             km_model, km_scaler, cluster_feature_names, best_k,
             shap_explainer, df_with_orig, encoders,
             model_comparison, feature_scaler)

    print("\n" + "=" * 60)
    print("ALL LAYERS TRAINED — Run: streamlit run app.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
