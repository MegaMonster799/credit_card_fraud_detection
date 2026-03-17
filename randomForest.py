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
from sklearn.ensemble import RandomForestClassifier
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
# 3. LEAKAGE-SAFE RISK ENCODING
# -------------------------------
def add_risk_features(X_train, y_train, X_test, cat_cols):
    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train_series = pd.Series(y_train, index=X_train.index, name="IsFraud")

    for col in cat_cols:
        tmp = pd.DataFrame({col: X_train[col], "IsFraud": y_train_series})
        risk = tmp.groupby(col)["IsFraud"].mean()
        global_mean = y_train_series.mean()

        X_train[f"{col}_Risk"] = X_train[col].map(risk).fillna(global_mean)
        X_test[f"{col}_Risk"] = X_test[col].map(risk).fillna(global_mean)

        freq = X_train[col].value_counts()
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

    # DROP ORIGINAL STRING COLUMNS (FIX)
    X_train = X_train.drop(columns=["MerchantID", "Location"])
    X_test = X_test.drop(columns=["MerchantID", "Location"])

    # One-hot encode TransactionType
    X_train = pd.get_dummies(X_train, columns=["TransactionType"], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=["TransactionType"], drop_first=True)

    # Align columns
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # Convert to float (NOW SAFE)
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=18,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Predict
    y_prob = rf.predict_proba(X_test)[:, 1]

    # Threshold tuning
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
plt.title("Random Forest Confusion Matrix (Averaged Across Folds)")
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

final_rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=18,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
final_rf.fit(X_full, y)

explainer = shap.TreeExplainer(final_rf)
shap_vals = explainer.shap_values(X_full)

shap.summary_plot(shap_vals[1], X_full, feature_names=X_full.columns, plot_size=(10, 8))
