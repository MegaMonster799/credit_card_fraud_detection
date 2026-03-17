import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import shap

# -------------------------------
# 0. REPRODUCIBILITY
# -------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("credit_card_fraud_10k.csv")

# -------------------------------
# 2. EDA
# -------------------------------
print("Dataset shape:", df.shape)
print("\nFraud class distribution:\n", df["is_fraud"].value_counts())

sns.countplot(x="is_fraud", data=df)
plt.title("Fraud Class Distribution")
plt.show()

# -------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------
X = df.drop(columns=["transaction_id", "is_fraud"]).copy()
y = df["is_fraud"].values.astype(int)

X["merchant_category"] = X["merchant_category"].astype("category").cat.codes

numeric_cols = [
    "amount", "transaction_hour", "device_trust_score",
    "velocity_last_24h", "cardholder_age"
]
for col in numeric_cols:
    X[col] = X[col].astype(float)

binary_cols = ["foreign_transaction", "location_mismatch"]
for col in binary_cols:
    X[col] = X[col].astype(int)

feature_names = X.columns.tolist()
X_np = X.values.astype(np.float32)

# -------------------------------
# 4. ANN ARCHITECTURE
# -------------------------------
def make_ann(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.10),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.10),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")]
    )
    return model

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=8,
    restore_best_weights=True,
    verbose=0
)

# -------------------------------
# 5. STRATIFIED 5-FOLD CV + SMOTE + THRESHOLD TUNING
# -------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

fold_metrics = []
fold_cms = []
fold = 1

for train_idx, test_idx in skf.split(X_np, y):
    print(f"\n--- Fold {fold} ---")

    X_train, X_test = X_np[train_idx], X_np[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    model = make_ann(input_dim=X_train_res.shape[1])
    _ = model.fit(
        X_train_res, y_train_res,
        validation_split=0.2,
        epochs=80,
        batch_size=256,
        callbacks=[early_stop],
        verbose=0
    )

    y_prob = model.predict(X_test_scaled, verbose=0).ravel()

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    best_idx = np.argmax(f1_scores[:-1])
    best_thresh = thresholds[best_idx]
    print(f"Best threshold: {best_thresh:.3f}")

    y_pred = (y_prob >= best_thresh).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_prob)

    fold_metrics.append({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "threshold": float(best_thresh)
    })

    print(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}"
    )

    fold_cms.append(confusion_matrix(y_test, y_pred))
    fold += 1

# -------------------------------
# 6. AVERAGED CONFUSION MATRIX (WHOLE NUMBERS)
# -------------------------------
avg_cm = np.mean(fold_cms, axis=0)
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
plt.title("ANN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------
# 7. AVERAGE METRICS
# -------------------------------
metrics_df = pd.DataFrame(fold_metrics)
print("\nPer-fold metrics:")
print(metrics_df)
print("\nAverage metrics:")
print(metrics_df.mean(numeric_only=True))

# -------------------------------
# 8. FINAL MODEL + SHAP
# -------------------------------
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X_np)

smote = SMOTE(random_state=SEED)
X_res, y_res = smote.fit_resample(X_scaled_full, y)

final_model = make_ann(input_dim=X_res.shape[1])
_ = final_model.fit(
    X_res, y_res,
    validation_split=0.2,
    epochs=80,
    batch_size=256,
    callbacks=[early_stop],
    verbose=0
)

bg_size = min(100, X_res.shape[0])
explain_size = min(300, X_scaled_full.shape[0])

background = shap.sample(X_res, bg_size, random_state=SEED)
to_explain = shap.sample(X_scaled_full, explain_size, random_state=SEED)

def predict_fn(x):
    return final_model.predict(x, verbose=0).ravel()

explainer = shap.KernelExplainer(predict_fn, background)
shap_values = explainer.shap_values(to_explain, nsamples=200)

shap.summary_plot(
    shap_values,
    to_explain,
    feature_names=feature_names,
    show=True
)
