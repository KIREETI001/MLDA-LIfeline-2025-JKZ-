# ctg_pipeline_excel.py
# Robust, leak-free CTG pipeline for Excel .xls with multi-sheets

# =========================
# 1) Imports & settings
# =========================
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import shap


# -----------------------------------
# Helper: ensure output folder exists
# -----------------------------------
def ensure_dir(path="plots"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# -----------------------------------
# Helper: safe classification report
# -----------------------------------
def safe_classification_report(y_true, y_pred, label_enc=None, title=""):
    """
    Prints classification_report with correct alignment between labels and names.
    Works whether you used LabelEncoder or not.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    labels_present = np.unique(np.concatenate([y_true, y_pred]))
    if label_enc is not None and hasattr(label_enc, "classes_"):
        # Full class list from the encoder (keeps consistent ordering 0..K-1)
        full_labels = np.arange(len(label_enc.classes_))
        # Names for full set
        full_names = [str(c) for c in label_enc.classes_]
        # Subset to labels present (keeps order matching 'labels' argument)
        names_for_present = [full_names[i] for i in labels_present]
        print(f"\n{title}")
        print(
            classification_report(
                y_true, y_pred, labels=labels_present, target_names=names_for_present
            )
        )
    else:
        names = [str(i) for i in labels_present]
        print(f"\n{title}")
        print(
            classification_report(
                y_true, y_pred, labels=labels_present, target_names=names
            )
        )


# =========================
# 2) Load Excel data
# =========================
# Adjust filename/sheet index as needed
xls = pd.ExcelFile("CTG.xls")
print("Sheets in file:", xls.sheet_names)

# Pick the correct sheet (you used index 2 earlier)
df = pd.read_excel(xls, sheet_name=xls.sheet_names[2], header=0)
print("Dataset shape:", df.shape)
print(df.head(3))
print(df.info())

# =========================
# 3) Cleaning & selection
# =========================
# Drop obvious meta columns if present
drop_cols = ["FileName", "SegFile", "Date"]
drop_cols = [c for c in drop_cols if c in df.columns]
if drop_cols:
    df = df.drop(columns=drop_cols)

# Identify target. You said 'NSP' is the label.
target_col = "NSP"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")

# Coerce numerics for all NON-target columns (turns numeric-looking strings → numbers; others → NaN)
for c in df.columns:
    if c != target_col:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Fill NaNs (simple forward/backward fill; you can change to median if you prefer)
df = df.fillna(method="ffill").fillna(method="bfill")

# Build X, y
y_raw = df[target_col].copy()
X = df.drop(columns=[target_col]).copy()

# Keep only numeric columns in X (should already be numeric after coercion)
X = X.select_dtypes(include=[np.number])
print("Final feature count:", X.shape[1])

# Encode labels (keep the encoder for names later)
label_enc = LabelEncoder()
y = label_enc.fit_transform(y_raw)
print("Label mapping (encoder classes_):", dict(enumerate(label_enc.classes_)))

# =========================
# 4) EDA (optional but useful)
# =========================
plots_dir = ensure_dir("plots")

# Correlation heatmap (numeric features only)
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(numeric_only=True), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "corr_heatmap.png"), dpi=180)
plt.close()

# Class distribution (original)
plt.figure(figsize=(6, 4))
pd.Series(y, name="class").value_counts().sort_index().plot(kind="bar")
plt.title("Class Distribution (Encoded)")
plt.xlabel("Class index")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "class_distribution_raw.png"), dpi=180)
plt.close()

# Selected CTG features (plot only if they exist)
features_to_plot = ["LB", "ASTV", "AC", "DS"]
for feat in features_to_plot:
    if feat in X.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=y, y=X[feat])
        plt.title(f"{feat} by Class (encoded)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"box_{feat}.png"), dpi=180)
        plt.close()

print(f"EDA plots saved to: ./{plots_dir}")

# =========================
# 5) Train/Test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# =========================
# 6) Pipelines (no leakage)
# =========================
# We’ll scale numeric features; no categorical features expected now.
num_cols = X.columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), num_cols)],
    remainder="drop",
)

# ---- Logistic Regression pipeline ----
logreg_pipe = ImbPipeline(
    steps=[
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),          # apply SMOTE on TRAIN fold inside pipeline
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None)),
    ]
)

# ---- Random Forest pipeline ----
rf_pipe = ImbPipeline(
    steps=[
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
        )),
    ]
)

# ---- XGBoost pipeline ----
xgb_pipe = ImbPipeline(
    steps=[
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", xgb.XGBClassifier(
            tree_method="hist", eval_metric="mlogloss", random_state=42
        )),
    ]
)

# =========================
# 7) Train & evaluate
# =========================
def evaluate_model(name, pipe, X_train, y_train, X_test, y_test, enc):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    safe_classification_report(y_test, y_pred, label_enc=enc, title=f"=== {name} ===")
    print(f"{name} - Balanced Accuracy: {bal_acc:.4f}")
    print(f"{name} - Macro F1:        {macro_f1:.4f}")

    return y_pred, pipe

# Logistic Regression
y_pred_lr, logreg_pipe = evaluate_model("Logistic Regression", logreg_pipe, X_train, y_train, X_test, y_test, label_enc)

# Random Forest
y_pred_rf, rf_pipe = evaluate_model("Random Forest", rf_pipe, X_train, y_train, X_test, y_test, label_enc)

# XGBoost
y_pred_xgb, xgb_pipe = evaluate_model("XGBoost", xgb_pipe, X_train, y_train, X_test, y_test, label_enc)

# Confusion matrix for RF (as example)
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "cm_rf.png"), dpi=180)
plt.close()

# =========================
# 8) SHAP explainability (RF)
# =========================
# We need the trained RF estimator and the scaled numeric test set
# Extract the trained RF model and preprocessor from the pipeline
rf_est = rf_pipe.named_steps["clf"]
pre = rf_pipe.named_steps["pre"]

# Transform X_test with the preprocessor to match the model input
X_test_trans = pre.transform(X_test)  # numpy array
feature_names = pre.get_feature_names_out()  # names after ColumnTransformer

# TreeExplainer for RF
explainer = shap.TreeExplainer(rf_est)

# Depending on SHAP version, shap_values may be a list (per class) or array
shap_values = explainer.shap_values(X_test_trans)

# Summary plot (handles list/array internally)
plt.figure()
shap.summary_plot(shap_values, X_test_trans, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "shap_summary.png"), dpi=180, bbox_inches="tight")
plt.close()

# Bar plot
plt.figure()
shap.summary_plot(shap_values, X_test_trans, feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "shap_bar.png"), dpi=180, bbox_inches="tight")
plt.close()

print(f"All figures saved to: ./{plots_dir}")
