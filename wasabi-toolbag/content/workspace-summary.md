# Telco Customer Churn — Workspace Summary

## Project Overview

Multi-layer machine learning pipeline for telecom customer churn prediction and retention strategy, built on the IBM Telco Customer Churn dataset (7,043 customers, 19 features).

## Workspace Structure

```
telco-churn/
├── app.py                  # Streamlit web dashboard (4-tab UI)
├── train_all.py            # Combined training: all layers + multi-model comparison
├── train_layer1.py         # Layer 1: XGBoost churn prediction (standalone)
├── train_layer2.py         # Layer 2: K-Means customer segmentation (standalone)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset
├── artifacts/
│   └── all_artifacts.pkl   # Serialized trained models + data
├── cluster_profiles.png    # Cluster comparison visualization
├── cluster_selection.png   # Elbow + silhouette score plots
└── artifacts/radar_chart.png  # Multi-model radar comparison
```

## Architecture: Multi-Layer ML Pipeline

| Layer | Script(s) | Algorithm | Output |
|-------|-----------|-----------|--------|
| Layer 1 | `train_layer1.py`, `train_all.py` | XGBoost Classifier (n_estimators=300, max_depth=5) | Churn probability per customer (AUC=0.83) |
| Layer 2 | `train_layer2.py`, `train_all.py` | K-Means Clustering (K=3, silhouette-selected) | Customer segments: Budget Basics, Flight Risks, Premium Loyalists |
| Layer 3 | `train_all.py`, `app.py` | SHAP TreeExplainer | Per-customer churn explanations + personalized retention strategies |
| Dashboard | `app.py` | Streamlit | 4-tab UI: EDA, Model Comparison, Cluster Analysis, Predict & Retain |
| Multi-model | `train_all.py` | LR, RF, XGBoost, SVM, KNN | Comparison table + radar chart |

## Languages and Frameworks

- **Language:** Python (≥3.8 required, targets 3.8–3.11)
- **ML Framework:** scikit-learn + XGBoost
- **Explainability:** SHAP (TreeExplainer)
- **Web UI:** Streamlit
- **Visualization:** Matplotlib (Agg backend)
- **Data:** pandas, numpy

## Dependencies (Inferred — No Requirements File Exists)

| Package | Purpose | Estimated Version |
|---------|---------|-------------------|
| numpy | Numerical arrays | ≥ 1.21 |
| pandas | DataFrame manipulation, CSV loading | ≥ 1.3 |
| scikit-learn | ML utilities, preprocessing, clustering, metrics | ≥ 1.0 |
| xgboost | Primary churn prediction model (XGBClassifier) | ≥ 1.6 |
| shap | Model explainability (TreeExplainer) | ≥ 0.41 |
| matplotlib | Chart generation | ≥ 3.4 |
| streamlit | Web dashboard | ≥ 1.10 |

**Standard library:** `os`, `pickle`, `warnings`

## Build and Test Tools

- **Build System:** None — scripts executed directly with `python`
- **Dependency Management:** None (no requirements.txt, pyproject.toml, setup.py, or Amazon internal systems)
- **Testing Framework:** None — no test files, no pytest/unittest configuration
- **Linting/Formatting:** None — no flake8, ruff, black, pylint, or mypy configured
- **CI/CD:** None configured

### How to Run

```bash
python train_layer1.py      # Train Layer 1 (XGBoost churn)
python train_layer2.py      # Train Layer 2 (K-Means clusters)
python train_all.py         # Train all layers + multi-model comparison
streamlit run app.py        # Launch dashboard
```

## Code Style Conventions (Observed)

- 4-space indentation
- Module-level docstrings present
- f-strings for string formatting
- Section banners with `print("=" * 60)` for progress reporting
- No type annotations used
- Imports grouped: standard library → third-party (no explicit separator)

## Logging and Metrics

- **Logging:** No logging framework — all output via `print()` statements
- **Warning handling:** `warnings.filterwarnings('ignore')` applied globally in training scripts
- **Metrics tracking:** scikit-learn metrics computed and printed to stdout only; no persistent metric store
- **Streamlit notifications:** `st.error()`, `st.warning()`, `st.info()`, `st.success()` for UI alerts

### Guidance for Future Code

When adding new code to this project, use `print()` for consistency with existing patterns:
```python
print("=" * 60)
print("SECTION NAME")
print("=" * 60)
print(f"  Metric: {value:.4f}")
```

If the project matures to need proper logging:
```python
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

## Key Gaps and Recommendations

| Area | Status | Priority |
|------|--------|----------|
| Dependency pinning | No requirements.txt or pyproject.toml | **Critical** |
| Python version pin | No .python-version file | High |
| Testing | No test infrastructure | High |
| Linting/formatting | No automated enforcement | Medium |
| Experiment tracking | No MLflow/W&B | Medium |
| Structured logging | Using print() only | Low |
