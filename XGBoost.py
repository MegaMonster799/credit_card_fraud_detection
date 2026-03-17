import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve
)

from xgboost import XGBClassifier
import shap

RANDOM_STATE = 42
N_SPLITS = 5

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("credit_card_fraud_dataset.csv")

# -------------------------------
# 2. BASE FEATURE ENGINEERING
# -------------------------------
df = df.drop(columns=["TransactionID"])

df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
df = df.sort_values("TransactionDate").reset_index(drop=True)

df["Hour"] = df["TransactionDate"].dt.hour
df["DayOfWeek"] = df["TransactionDate"].dt.dayofweek
df["Month"] = df["TransactionDate"].dt.month
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
df["IsNight"] = df["Hour"].isin([0, 1, 2, 3, 4]).astype(int)

df = df.drop(columns=["TransactionDate"])

y = df["IsFraud"].astype(int).values
X_base = df.drop(columns=["IsFraud"]).copy()

# -------------------------------
# 3. LEAKAGE-SAFE TARGET ENCODING (SMOOTHED)
# -------------------------------
def add_risk_features(X_train, y_train, X_test, cat_cols, smoothing=50):
    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train_series = pd.Series(y_train, index=X_train.index, name="IsFraud")

    global_mean = y_train_series.mean()

    for col in cat_cols:
        tmp = pd.DataFrame({col: X_train[col], "IsFraud": y_train_series})
        stats = tmp.groupby(col)["IsFraud"].agg(["mean", "count"])

        # Smoothed target encoding
        stats["smoothed"] = (
            (stats["mean"] * stats["count"] + global_mean * smoothing)
            / (stats["count"] + smoothing)
        )

        mapping = stats["smoothed"]

        X_train[f"{col}_Risk"] = X_train[col].map(mapping).fillna(global_mean)
        X_test[f"{col}_Risk"] = X_test[col].map(mapping).fillna(global_mean)

        # Frequency encoding
        freq = stats["count"]
        X_train[f"{col}_Freq"] = X_train[col].map(freq).fillna(0)
        X_test[f"{col}_Freq"] = X_test[col].map(freq).fillna(0)

    return X_train, X_test

# -------------------------------
# 4. STRATIFIED 5-FOLD CV
# -------------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

metrics_list = []
confusion_matrices = []
best_thresholds = []

cat_for_risk = ["MerchantID", "Location"]

for fold, (train_idx, test_idx) in enumerate(skf.split(X_base, y), start=1):
    print(f"\n--- Fold {fold}/{N_SPLITS} ---")

    X_train_base, X_test_base = X_base.iloc[train_idx], X_base.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Add risk features
    X_train, X_test = add_risk_features(X_train_base, y_train, X_test_base, cat_for_risk)

    # Drop original string columns
    X_train = X_train.drop(columns=["MerchantID", "Location"])
    X_test = X_test.drop(columns=["MerchantID", "Location"])

    # One-hot encode TransactionType
    X_train = pd.get_dummies(X_train, columns=["TransactionType"], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=["TransactionType"], drop_first=True)

    # Align columns
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Convert to float
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    # Compute scale_pos_weight
    spw = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    # XGBoost model
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        n_estimators=600,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predict
    y_prob = model.predict_proba(X_test)[:, 1]

    # Threshold tuning (maximize F1)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])
    best_thresh = thresholds[best_idx]
    best_thresholds.append(best_thresh)

    y_pred = (y_prob >= best_thresh).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices.append(cm)

    metrics_list.append({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": best_thresh
    })

    print(
        f"Acc: {accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | "
        f"F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | "
        f"Thresh: {best_thresh:.4f}"
    )

# -------------------------------
# 5. AVERAGED CONFUSION MATRIX
# -------------------------------
avg_cm = np.mean(confusion_matrices, axis=0)
avg_cm_rounded = np.rint(avg_cm).astype(int)

plt.figure(figsize=(6, 5))
sns.heatmap(
    avg_cm_rounded,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legit", "Fraud"],
    yticklabels=["Legit", "Fraud"]
)
plt.title("XGBoost Confusion Matrix (Averaged Across Folds)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# 6. AVERAGE METRICS
# -------------------------------
metrics_df = pd.DataFrame(metrics_list)
print("\nAverage metrics across folds:")
print(metrics_df.mean(numeric_only=True))

print("\nAverage tuned threshold:", np.mean(best_thresholds))

# -------------------------------
# 7. FINAL MODEL + SHAP
# -------------------------------
X_full = X_base.copy()
X_full, _ = add_risk_features(X_full, y, X_full.copy(), cat_for_risk)

X_full = X_full.drop(columns=["MerchantID", "Location"])
X_full = pd.get_dummies(X_full, columns=["TransactionType"], drop_first=True)
X_full = X_full.astype(float)

final_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    n_estimators=600,
    scale_pos_weight=(y == 0).sum() / max(1, (y == 1).sum()),
    random_state=RANDOM_STATE,
    n_jobs=-1
)

final_model.fit(X_full, y)

explainer = shap.TreeExplainer(final_model)
shap_vals = explainer.shap_values(X_full)

shap.summary_plot(shap_vals, X_full, feature_names=X_full.columns, plot_size=(10, 8))
