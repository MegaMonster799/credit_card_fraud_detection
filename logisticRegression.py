import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import shap

RANDOM_STATE = 42

# ---------------------------------------------------
# 1) LOAD + PREPARE DATA
# ---------------------------------------------------
df = pd.read_csv("credit_card_fraud_10k.csv").dropna(subset=["is_fraud"])

X = df.drop(columns=["transaction_id", "is_fraud"]).copy()
y = df["is_fraud"].astype(int).values

# Encode categorical
X["merchant_category"] = X["merchant_category"].astype("category").cat.codes

# Ensure numeric types
numeric_casts = {
    "amount": float,
    "transaction_hour": int,
    "foreign_transaction": int,
    "location_mismatch": int,
    "device_trust_score": float,
    "velocity_last_24h": int,
    "cardholder_age": int
}
for col, dtype in numeric_casts.items():
    X[col] = X[col].astype(dtype)

feature_names = X.columns.tolist()

# ---------------------------------------------------
# 2) HELPER FUNCTIONS
# ---------------------------------------------------
def best_f1_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    return float(thresholds[np.argmax(f1_scores)])

def evaluate_with_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    return acc, prec, rec, f1, auc, cm

# ---------------------------------------------------
# 3) NESTED CV SETTINGS
# ---------------------------------------------------
outer_k = 5
inner_k = 3

outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=RANDOM_STATE)

param_grid = list(ParameterGrid({
    "penalty": ["l1", "l2"],
    "C": [0.01, 0.1, 1.0, 10.0],
    "solver": ["liblinear"],
    "class_weight": [None, "balanced"]
}))

fold_metrics = []
fold_conf_mats = []

# ---------------------------------------------------
# 4) NESTED CROSS-VALIDATION
# ---------------------------------------------------
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
    print(f"\n========== OUTER FOLD {fold}/{outer_k} ==========")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=RANDOM_STATE)

    best_score = -np.inf
    best_params = None
    best_threshold = 0.5

    # ----- INNER LOOP: hyperparameter + threshold tuning -----
    for params in param_grid:
        inner_scores = []
        inner_thresholds = []

        for tr_idx, val_idx in inner_cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=5000,
                    **params
                ))
            ])

            pipe.fit(X_tr, y_tr)
            val_prob = pipe.predict_proba(X_val)[:, 1]

            t_star = best_f1_threshold(y_val, val_prob)
            inner_thresholds.append(t_star)

            _, _, _, f1, _, _ = evaluate_with_threshold(y_val, val_prob, t_star)
            inner_scores.append(f1)

        mean_f1 = np.mean(inner_scores)
        median_t = np.median(inner_thresholds)

        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = params
            best_threshold = median_t

    print("Best inner params:", best_params)
    print(f"Best inner F1: {best_score:.4f}")
    print(f"Chosen threshold: {best_threshold:.3f}")

    # ----- OUTER EVALUATION -----
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=5000,
            **best_params
        ))
    ])

    final_pipe.fit(X_train, y_train)
    test_prob = final_pipe.predict_proba(X_test)[:, 1]

    acc, prec, rec, f1, auc, cm = evaluate_with_threshold(y_test, test_prob, best_threshold)

    fold_metrics.append({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "threshold": best_threshold
    })
    fold_conf_mats.append(cm)

    print(f"Fold {fold} | Acc {acc:.4f} | Prec {prec:.4f} | Rec {rec:.4f} | F1 {f1:.4f} | AUC {auc:.4f}")

# -------------------------------
# 5) AVERAGED CONFUSION MATRIX (WHOLE NUMBERS)
# -------------------------------

# Average confusion matrix across folds
avg_cm = np.mean(fold_conf_mats, axis=0)

# Convert to whole numbers
avg_cm_rounded = np.rint(avg_cm).astype(int)

plt.figure(figsize=(6, 5))
sns.heatmap(
    avg_cm_rounded,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Fraud", "Fraud"],
    yticklabels=["No Fraud", "Fraud"]
)
plt.title("LogReg Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

metrics_df = pd.DataFrame(fold_metrics)
print("\nAverage metrics across folds:")
print(metrics_df.mean(numeric_only=True))

# ---------------------------------------------------
# 6) FINAL MODEL + SHAP
# ---------------------------------------------------
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=5000,
        **best_params
    ))
])

final_model.fit(X, y)

X_scaled = final_model.named_steps["scaler"].transform(X)
lr_model = final_model.named_steps["lr"]

explainer = shap.LinearExplainer(lr_model, X_scaled, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_scaled)

impact_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("\nGlobal SHAP Feature Importance:")
print(impact_df)

shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, plot_size=(10, 8))
