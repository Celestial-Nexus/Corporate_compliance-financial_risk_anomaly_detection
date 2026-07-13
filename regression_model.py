"""
regression_model.py
────────────────────────────────────────────────────────────────────────────
Corporate Compliance – Multi-Variable Regression & Anomaly Flagging

What this does
──────────────
• Trains a Logistic Regression (+ optional RandomForest) on the scored
  metallurgical ledger to predict Is_Fraud_Ground_Truth.
• Features: numeric transaction columns + encoded categoricals.
• Prints a full classification report with real accuracy, precision,
  recall, F1 and ROC-AUC values.
• Regenerates model_metrics.png from *actual* sklearn metrics.
• Exports feature importances and calibration curve.
• Can be run standalone or imported by dashboard.py.
"""

import math
import warnings

import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FILE        = "metallurgical_ledgers_scored.xlsx"
FALLBACK_CSV      = "metallurgical_ledgers.csv"
TARGET_COL        = "Is_Fraud_Ground_Truth"
RANDOM_STATE      = 42
TEST_SIZE         = 0.20
MAHAL_THRESHOLD   = 0.975    # Chi-squared percentile for Mahalanobis outlier


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def load_and_engineer(path: str = INPUT_FILE) -> pd.DataFrame:
    """Load scored workbook (or CSV fallback) and create ML features."""
    try:
        df = pd.read_excel(path)
        print(f"  Loaded scored workbook: {path}")
    except FileNotFoundError:
        df = pd.read_csv(FALLBACK_CSV, parse_dates=["Date"])
        print(f"  Fallback to CSV: {FALLBACK_CSV}")

    # ── Make sure ground-truth column exists ─────────────────────────────────
    if TARGET_COL not in df.columns:
        raw = pd.read_csv(FALLBACK_CSV)
        df = df.merge(
            raw[["Transaction_ID", TARGET_COL]],
            on="Transaction_ID", how="left"
        )

    # ── Date-derived features ─────────────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"])
    df["month"]      = df["Date"].dt.month
    df["day_of_week"]= df["Date"].dt.dayofweek
    df["quarter"]    = df["Date"].dt.quarter

    # ── Price deviation ───────────────────────────────────────────────────────
    df["price_dev_pct"] = (
        (df["Unit_Price_USD"] - df["Market_Spot_Price"]).abs()
        / df["Market_Spot_Price"].replace(0, np.nan) * 100.0
    )

    # ── Log-transform heavy-tailed columns ───────────────────────────────────
    for col in ["Total_Value_USD", "Volume_MT"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # ── Categorical encoding ──────────────────────────────────────────────────
    for col in ["Commodity", "Payment_Method", "Vendor_Country"]:
        le = LabelEncoder()
        df[f"enc_{col}"] = le.fit_transform(df[col].fillna("Unknown"))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mahalanobis Distance (real implementation)
# ─────────────────────────────────────────────────────────────────────────────
def compute_mahalanobis(df: pd.DataFrame) -> pd.Series:
    """
    Compute per-row Mahalanobis distance using numeric transaction features.
    Returns a Series of distances (same index as df).
    """
    MAHAL_FEATURES = [
        "Volume_MT", "Unit_Price_USD", "Total_Value_USD",
        "price_dev_pct", "log_Total_Value_USD", "log_Volume_MT",
    ]
    feat_cols = [c for c in MAHAL_FEATURES if c in df.columns]
    X = df[feat_cols].fillna(df[feat_cols].median())

    # Robust covariance estimate
    mu  = X.mean().values
    cov = np.cov(X.values, rowvar=False)

    try:
        cov_inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.eye(len(feat_cols))

    dists = []
    for row in X.values:
        diff = row - mu
        d    = math.sqrt(max(0.0, float(diff @ cov_inv @ diff)))
        dists.append(d)

    return pd.Series(dists, index=df.index, name="mahalanobis_distance")


def plot_mahalanobis_real(df: pd.DataFrame, dist_series: pd.Series,
                          threshold: float, filename: str = "mahalanobis_outliers.png") -> None:
    """Scatter plot using real transaction data coloured by fraud label."""
    plt.style.use("ggplot")

    x = df["log_Total_Value_USD"].fillna(0)
    y = df["price_dev_pct"].clip(upper=200).fillna(0)
    is_outlier = dist_series > threshold
    is_fraud   = df[TARGET_COL].fillna(0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Mahalanobis Distance – Multivariate Outlier Detection\n"
                 "(Real Metallurgical Ledger Data)", fontsize=14, fontweight="bold")

    # Panel 1: Mahalanobis distance distribution
    ax0 = axes[0]
    ax0.hist(dist_series, bins=80, color="#3498db", edgecolor="white", alpha=0.75)
    ax0.axvline(threshold, color="#e74c3c", linewidth=2, linestyle="--",
                label=f"Threshold = {threshold:.1f}")
    ax0.set_xlabel("Mahalanobis Distance", fontsize=11)
    ax0.set_ylabel("Count", fontsize=11)
    ax0.set_title("Distance Distribution", fontweight="bold")
    ax0.legend()
    n_out = int(is_outlier.sum())
    ax0.annotate(f"{n_out} outliers\n({n_out/len(df)*100:.1f}%)",
                 xy=(threshold, ax0.get_ylim()[1] * 0.8),
                 xytext=(threshold * 1.05, ax0.get_ylim()[1] * 0.85),
                 fontsize=9, color="#e74c3c",
                 arrowprops=dict(arrowstyle="->", color="#e74c3c"))

    # Panel 2: Scatter – log(total value) vs price deviation
    ax1 = axes[1]
    # Normal clean transactions
    mask_clean  = (~is_outlier) & (is_fraud == 0)
    mask_out    = is_outlier  & (is_fraud == 0)
    mask_fraud  = is_fraud == 1

    ax1.scatter(x[mask_clean],  y[mask_clean],  c="#3498db", alpha=0.4, s=12,
                label="Normal Transactions", edgecolors="none")
    ax1.scatter(x[mask_out],    y[mask_out],    c="#f39c12", alpha=0.7, s=40,
                marker="D", label="Mahalanobis Outlier (unlabelled)")
    ax1.scatter(x[mask_fraud],  y[mask_fraud],  c="#e74c3c", alpha=0.9, s=55,
                marker="X", label="Ground-Truth Fraud")

    ax1.set_xlabel("log(Total Transaction Value USD)", fontsize=11)
    ax1.set_ylabel("Price Deviation from Spot (%)", fontsize=11)
    ax1.set_title("Transaction Feature Space", fontweight="bold")
    ax1.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Logistic Regression + Random Forest
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "log_Total_Value_USD", "log_Volume_MT",
    "price_dev_pct", "month", "day_of_week", "quarter",
    "enc_Commodity", "enc_Payment_Method", "enc_Vendor_Country",
]

# Append optional risk score columns if present in scored workbook
OPTIONAL_SCORE_COLS = [
    "ofac_risk_score", "price_delta_risk_score",
    "smurfing_risk_score", "benfords_law_risk_score",
    "geopolitical_risk_score", "mahalanobis_risk_score",
]


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    base = [c for c in FEATURE_COLS if c in df.columns]
    extra = [c for c in OPTIONAL_SCORE_COLS if c in df.columns]
    return df[base + extra].fillna(0)


def train_models(df: pd.DataFrame) -> dict:
    """Train Logistic Regression and Random Forest; return metrics dict."""
    X = build_feature_matrix(df)
    y = df[TARGET_COL].fillna(0).astype(int)

    print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Class balance  : fraud={y.sum():,} ({y.mean()*100:.1f}%) | "
          f"clean={len(y)-y.sum():,} ({(1-y.mean())*100:.1f}%)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results = {}

    # ── Logistic Regression ──────────────────────────────────────────────────
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        )),
    ])
    lr_pipe.fit(X_tr, y_tr)
    y_pred_lr   = lr_pipe.predict(X_te)
    y_prob_lr   = lr_pipe.predict_proba(X_te)[:, 1]

    results["LR"] = {
        "accuracy":  accuracy_score(y_te, y_pred_lr),
        "precision": precision_score(y_te, y_pred_lr, zero_division=0),
        "recall":    recall_score(y_te, y_pred_lr, zero_division=0),
        "f1":        f1_score(y_te, y_pred_lr, zero_division=0),
        "roc_auc":   roc_auc_score(y_te, y_prob_lr),
        "y_prob":    y_prob_lr,
        "y_pred":    y_pred_lr,
        "y_true":    y_te,
        "report":    classification_report(y_te, y_pred_lr, target_names=["Clean","Fraud"]),
        "model":     lr_pipe,
        "coefs":     dict(zip(X.columns, lr_pipe.named_steps["clf"].coef_[0])),
    }

    # ── Random Forest ────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    y_pred_rf = rf.predict(X_te)
    y_prob_rf = rf.predict_proba(X_te)[:, 1]

    results["RF"] = {
        "accuracy":  accuracy_score(y_te, y_pred_rf),
        "precision": precision_score(y_te, y_pred_rf, zero_division=0),
        "recall":    recall_score(y_te, y_pred_rf, zero_division=0),
        "f1":        f1_score(y_te, y_pred_rf, zero_division=0),
        "roc_auc":   roc_auc_score(y_te, y_prob_rf),
        "y_prob":    y_prob_rf,
        "y_pred":    y_pred_rf,
        "y_true":    y_te,
        "report":    classification_report(y_te, y_pred_rf, target_names=["Clean","Fraud"]),
        "model":     rf,
        "importances": dict(zip(X.columns, rf.feature_importances_)),
    }

    # ── Cross-validated F1 ───────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_f1_lr = cross_val_score(lr_pipe, X, y, cv=cv, scoring="f1").mean()
    cv_f1_rf = cross_val_score(rf,       X, y, cv=cv, scoring="f1").mean()
    results["LR"]["cv_f1"] = cv_f1_lr
    results["RF"]["cv_f1"] = cv_f1_rf

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. Visualisations (from real metrics)
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_metrics(results: dict, filename: str = "model_metrics.png") -> None:
    """Regenerate model_metrics.png using REAL sklearn metrics."""
    plt.style.use("ggplot")

    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    lr = results["LR"]
    rf = results["RF"]

    lr_vals = [lr["accuracy"], lr["precision"], lr["recall"], lr["f1"], lr["roc_auc"]]
    rf_vals = [rf["accuracy"], rf["precision"], rf["recall"], rf["f1"], rf["roc_auc"]]
    lr_vals_pct = [v * 100 for v in lr_vals]
    rf_vals_pct = [v * 100 for v in rf_vals]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Compliance Fraud Detection – Model Performance\n(Real sklearn Metrics on Held-Out Test Set)",
                 fontsize=14, fontweight="bold")

    # Bar chart: LR vs RF
    ax0 = axes[0]
    bars_lr = ax0.bar(x - width/2, lr_vals_pct, width, label="Logistic Regression",
                      color="#2c3e50", alpha=0.85)
    bars_rf = ax0.bar(x + width/2, rf_vals_pct, width, label="Random Forest",
                      color="#e74c3c", alpha=0.85)
    ax0.set_ylim(0, 115)
    ax0.set_xticks(x)
    ax0.set_xticklabels(metric_names, fontsize=10)
    ax0.set_ylabel("Score (%)", fontsize=11)
    ax0.set_title("Model Comparison (Test Set)", fontweight="bold")
    ax0.legend()
    for bar in list(bars_lr) + list(bars_rf):
        h = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width()/2, h + 1,
                 f"{h:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ROC curve: both models
    ax1 = axes[1]
    for model_name, color in [("LR", "#2c3e50"), ("RF", "#e74c3c")]:
        fpr, tpr, _ = roc_curve(results[model_name]["y_true"],
                                  results[model_name]["y_prob"])
        roc_auc     = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=color, linewidth=2,
                 label=f"{'Logistic Regression' if model_name=='LR' else 'Random Forest'}"
                       f" (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("ROC Curve (Real Data)", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {filename}")


def plot_feature_importance(results: dict, filename: str = "feature_importance.png") -> None:
    """Bar chart of Random Forest feature importances."""
    plt.style.use("ggplot")
    importances = results["RF"]["importances"]
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    labels, vals = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v > 0.10 else "#3498db" if v > 0.04 else "#95a5a6"
              for v in vals]
    bars = ax.barh(labels, vals, color=colors, edgecolor="white")
    ax.set_xlabel("Feature Importance (Gini)", fontsize=11)
    ax.set_title("Random Forest – Feature Importances\n(Fraud Detection Model)", fontweight="bold")
    ax.invert_yaxis()
    for bar, v in zip(bars, vals):
        ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {filename}")


def plot_confusion_matrix(results: dict, filename: str = "confusion_matrix.png") -> None:
    """Plot side-by-side confusion matrices for both models."""
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrices – Fraud Detection (Real Test Set)",
                 fontsize=14, fontweight="bold")
    for ax, (name, label) in zip(axes, [("LR", "Logistic Regression"), ("RF", "Random Forest")]):
        cm = confusion_matrix(results[name]["y_true"], results[name]["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["Clean", "Fraud"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(label, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> dict:
    DIVIDER = "=" * 65
    print(DIVIDER)
    print("  Multi-Variable Regression & Mahalanobis Analysis")
    print(DIVIDER)

    print("\n[STEP 1/5]  Loading & engineering features …")
    df = load_and_engineer()
    print(f"            {len(df):,} rows | {df.columns.nunique()} columns after engineering")

    print("\n[STEP 2/5]  Computing Mahalanobis distances on real data …")
    dist = compute_mahalanobis(df)
    df["mahalanobis_distance"] = dist

    # Chi-squared threshold: p = 0.975, df = 6 features
    n_features = 6
    threshold  = math.sqrt(chi2.ppf(MAHAL_THRESHOLD, df=n_features))
    df["mahalanobis_risk_score"] = (dist / threshold * 100.0).clip(upper=100.0).round(4)
    df["is_mahal_outlier"] = (dist > threshold).astype(int)

    n_out = df["is_mahal_outlier"].sum()
    print(f"            Threshold : {threshold:.4f} (χ² p={MAHAL_THRESHOLD})")
    print(f"            Outliers  : {n_out:,} ({n_out/len(df)*100:.2f}%)")

    print("\n[STEP 3/5]  Generating Mahalanobis scatter plot (real data) …")
    plot_mahalanobis_real(df, dist, threshold)

    print("\n[STEP 4/5]  Training Logistic Regression + Random Forest …")
    results = train_models(df)

    for name, label in [("LR", "Logistic Regression"), ("RF", "Random Forest")]:
        m = results[name]
        print(f"\n  ── {label} ──")
        print(f"    Accuracy  : {m['accuracy']*100:.2f}%")
        print(f"    Precision : {m['precision']*100:.2f}%")
        print(f"    Recall    : {m['recall']*100:.2f}%")
        print(f"    F1-Score  : {m['f1']*100:.2f}%")
        print(f"    ROC-AUC   : {m['roc_auc']:.4f}")
        print(f"    5-Fold CV F1: {m['cv_f1']*100:.2f}%")
        print(f"\n    Classification Report:\n{m['report']}")

    print("\n[STEP 5/5]  Generating metric charts …")
    plot_model_metrics(results)
    plot_feature_importance(results)
    plot_confusion_matrix(results)

    print(f"\n✓ Regression analysis complete.\n")
    results["df"] = df
    results["mahal_threshold"] = threshold
    return results


if __name__ == "__main__":
    main()
