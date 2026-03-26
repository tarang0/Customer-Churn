"""
Layer 2: Customer Value Segmentation using K-Means Clustering
Finds natural customer segments based on behavioral + financial features.
Uses elbow method + silhouette score to determine optimal number of clusters.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

DATA_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
CLUSTER_RANGE = range(2, 9)


def load_and_preprocess(path):
    print("=" * 60)
    print("LOADING & PREPROCESSING")
    print("=" * 60)

    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    print(f"Loaded {len(df):,} customers")
    return df


def engineer_clustering_features(df):
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING FOR CLUSTERING")
    print("=" * 60)

    cf = pd.DataFrame()

    cf['tenure'] = df['tenure']
    cf['MonthlyCharges'] = df['MonthlyCharges']
    cf['TotalCharges'] = df['TotalCharges']

    service_cols = ['PhoneService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies']
    cf['num_services'] = sum((df[col] == 'Yes').astype(int) for col in service_cols)

    cf['contract_length'] = df['Contract'].map({
        'Month-to-month': 0, 'One year': 1, 'Two year': 2
    })

    cf['has_internet'] = (df['InternetService'] != 'No').astype(int)

    cf['avg_monthly_revenue'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges']
    )

    print(f"Clustering features ({len(cf.columns)}):")
    for col in cf.columns:
        print(f"  {col:25s}  min={cf[col].min():.1f}  max={cf[col].max():.1f}  "
              f"mean={cf[col].mean():.1f}")

    return cf


def find_optimal_k(X_scaled, feature_names):
    print("\n" + "=" * 60)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("=" * 60)

    inertias = []
    silhouettes = []

    for k in CLUSTER_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        print(f"  K={k}  |  Inertia={km.inertia_:>12,.0f}  |  Silhouette={sil:.4f}")

    best_k = list(CLUSTER_RANGE)[np.argmax(silhouettes)]
    print(f"\nBest K by silhouette score: {best_k}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(list(CLUSTER_RANGE), inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax1.set_ylabel('Inertia (within-cluster sum of squares)', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.set_xticks(list(CLUSTER_RANGE))
    ax1.grid(True, alpha=0.3)

    colors = ['#4CAF50' if k == best_k else '#2196F3' for k in CLUSTER_RANGE]
    ax2.bar(list(CLUSTER_RANGE), silhouettes, color=colors, edgecolor='white')
    ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score (higher = better)', fontsize=14, fontweight='bold')
    ax2.set_xticks(list(CLUSTER_RANGE))
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (k, s) in enumerate(zip(CLUSTER_RANGE, silhouettes)):
        ax2.text(k, s + 0.005, f'{s:.3f}', ha='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig('cluster_selection.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cluster_selection.png")

    return best_k


def run_clustering(X_scaled, best_k, cf, df):
    print("\n" + "=" * 60)
    print(f"RUNNING K-MEANS WITH K={best_k}")
    print("=" * 60)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cf = cf.copy()
    cf['cluster'] = km.fit_predict(X_scaled)
    df = df.copy()
    df['cluster'] = cf['cluster']

    return km, cf, df


def profile_clusters(cf, df, best_k):
    print("\n" + "=" * 60)
    print("CLUSTER PROFILES")
    print("=" * 60)

    feature_cols = [c for c in cf.columns if c != 'cluster']

    profiles = []
    for cluster_id in range(best_k):
        mask = cf['cluster'] == cluster_id
        cluster_cf = cf[mask]
        cluster_df = df[mask]

        profile = {
            'Cluster': cluster_id,
            'Size': len(cluster_cf),
            'Pct': f"{len(cluster_cf) / len(cf):.1%}",
            'Avg Tenure (mo)': cluster_cf['tenure'].mean(),
            'Avg Monthly ($)': cluster_cf['MonthlyCharges'].mean(),
            'Avg Total ($)': cluster_cf['TotalCharges'].mean(),
            'Avg Services': cluster_cf['num_services'].mean(),
            'Avg Contract': cluster_cf['contract_length'].mean(),
            'Churn Rate': cluster_df['Churn'].mean(),
        }
        profiles.append(profile)

    profiles_df = pd.DataFrame(profiles)

    profiles_df = profiles_df.sort_values('Avg Total ($)', ascending=False).reset_index(drop=True)
    cluster_rank = {row['Cluster']: i for i, row in profiles_df.iterrows()}

    print(f"\n{'='*100}")
    for _, row in profiles_df.iterrows():
        churn_pct = row['Churn Rate']
        risk = "LOW" if churn_pct < 0.2 else ("MEDIUM" if churn_pct < 0.35 else "HIGH")
        risk_icon = "🟢" if risk == "LOW" else ("🟡" if risk == "MEDIUM" else "🔴")

        print(f"\n  Cluster {int(row['Cluster'])}  ({row['Size']:,} customers, {row['Pct']})")
        print(f"  {'─'*50}")
        print(f"  Avg Tenure:       {row['Avg Tenure (mo)']:.0f} months")
        print(f"  Avg Monthly:      ${row['Avg Monthly ($)']:.0f}")
        print(f"  Avg Total Rev:    ${row['Avg Total ($)']:,.0f}")
        print(f"  Avg Services:     {row['Avg Services']:.1f}")
        print(f"  Avg Contract:     {row['Avg Contract']:.1f} (0=month, 1=1yr, 2=2yr)")
        print(f"  Churn Rate:       {churn_pct:.1%}  {risk_icon} {risk} RISK")

    print(f"\n{'='*100}")

    # --- Visualization ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Customer Segments (K={len(profiles_df)})', fontsize=16, fontweight='bold')

    metrics = [
        ('Avg Tenure (mo)', 'Avg Tenure (months)', '#2196F3'),
        ('Avg Monthly ($)', 'Avg Monthly Charges ($)', '#4CAF50'),
        ('Avg Total ($)', 'Avg Total Revenue ($)', '#FF9800'),
        ('Avg Services', 'Avg Number of Services', '#9C27B0'),
        ('Avg Contract', 'Avg Contract Length', '#00BCD4'),
        ('Churn Rate', 'Churn Rate', '#F44336'),
    ]

    for ax, (col, title, color) in zip(axes.flatten(), metrics):
        vals = profiles_df[col].values
        cluster_labels = [f"C{int(c)}" for c in profiles_df['Cluster']]
        bars = ax.bar(cluster_labels, vals, color=color, edgecolor='white', alpha=0.85)
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for b, v in zip(bars, vals):
            fmt = f'{v:.1%}' if col == 'Churn Rate' else (f'${v:,.0f}' if '$' in col else f'{v:.1f}')
            ax.text(b.get_x() + b.get_width() / 2, v + (max(vals) * 0.02),
                    fmt, ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig('cluster_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: cluster_profiles.png")

    return profiles_df


def main():
    df = load_and_preprocess(DATA_PATH)
    cf = engineer_clustering_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cf)

    best_k = find_optimal_k(X_scaled, cf.columns.tolist())
    km, cf, df = run_clustering(X_scaled, best_k, cf, df)
    profiles_df = profile_clusters(cf, df, best_k)

    print("\n" + "=" * 60)
    print("LAYER 2 COMPLETE")
    print("=" * 60)
    print(f"\nClusters found: {best_k}")
    print(f"Charts saved: cluster_selection.png, cluster_profiles.png")


if __name__ == '__main__':
    main()
