import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

plt.rcParams["figure.figsize"] = (7, 5)

# Load test data
TARGET_COL = "NSP"
DATA_PATH = "./CTG_cleaned.csv"
df = pd.read_csv(DATA_PATH)
df = df[df[TARGET_COL].notna()]
y_str = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL])
y_encoded = y_str - 1  # Convert [1,2,3] to [0,1,2]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded
)

# Class weights for sample weighting (used only for XGBoost)
classes_ = np.array(sorted(np.unique(y_train)))
class_weights = compute_class_weight(class_weight="balanced", classes=classes_, y=y_train)
CW = {int(cls): float(w) for cls, w in zip(classes_, class_weights)}

# Evaluation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)
scoring = {
    "bal_acc": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "acc": "accuracy",
}

def eval_model(name, pipe, Xtr, ytr, Xte, yte, use_sw=False):
    try:
        cv_out = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=-1)
        cv_bal_acc = float(np.nanmean(cv_out["test_bal_acc"]))
        cv_f1_macro = float(np.nanmean(cv_out["test_f1_macro"]))
        cv_acc = float(np.nanmean(cv_out["test_acc"]))
    except Exception:
        cv_bal_acc = np.nan; cv_f1_macro = np.nan; cv_acc = np.nan

    if use_sw:
        sw = pd.Series(ytr).map(CW).values
        pipe.fit(Xtr, ytr, **{"clf__sample_weight": sw})
    else:
        pipe.fit(Xtr, ytr)

    yhat = pipe.predict(Xte)
    acc = accuracy_score(yte, yhat)
    bal = balanced_accuracy_score(yte, yhat)
    f1m = f1_score(yte, yhat, average="macro")

    y_true_orig = yte + 1
    y_pred_orig = yhat + 1
    report = classification_report(y_true_orig, y_pred_orig, labels=[1, 2, 3], digits=4, zero_division=0)

    return {
        "model": name,
        "cv_bal_acc_mean": cv_bal_acc,
        "cv_f1_macro_mean": cv_f1_macro,
        "cv_acc_mean": cv_acc,
        "test_acc": acc,
        "test_bal_acc": bal,
        "test_f1_macro": f1m,
        "y_pred": yhat,
        "y_pred_str": y_pred_orig,
        "report": report,
    }, pipe

# Load and evaluate models
model_names = ["RandomForest", "LogisticRegression", "SVM", "XGBoost"]
results = []
models_fitted = {}

for name in model_names:
    try:
        pipe = joblib.load(f"ctg_pipeline_{name}.joblib")
        print(f"\n=== Evaluating {name} ===")
        use_sw = name == "XGBoost"
        res, fitted = eval_model(name, pipe, X_train, y_train, X_test, y_test, use_sw=use_sw)
        results.append(res)
        models_fitted[name] = fitted
    except Exception as e:
        print(f"Failed to load or evaluate {name}: {e}")

# Results table
res_tbl = pd.DataFrame([{
    "model": r["model"],
    "cv_bal_acc_mean": np.round(r["cv_bal_acc_mean"], 4),
    "cv_f1_macro_mean": np.round(r["cv_f1_macro_mean"], 4),
    "cv_acc_mean": np.round(r["cv_acc_mean"], 4),
    "test_balanced_accuracy": np.round(r["test_bal_acc"], 4),
    "test_macro_f1": np.round(r["test_f1_macro"], 4),
    "test_accuracy": np.round(r["test_acc"], 4),
} for r in results])

res_tbl = res_tbl.sort_values(["cv_bal_acc_mean", "test_balanced_accuracy", "test_macro_f1"], ascending=[False, False, False])
print("\n=== Results (CV-first ranking) ===")
print(res_tbl.to_string(index=False))

# Prediction comparison
preds_df = pd.DataFrame({"y_true": y_test + 1})
for r in results:
    preds_df[r["model"]] = r["y_pred_str"]
print("\n=== Head of prediction comparison ===")
print(preds_df.head())

# Save results
res_tbl.to_csv("ctg_all_models_results.csv", index=False)
preds_df.to_csv("ctg_all_models_predictions.csv", index=False)
print("\nSaved: ctg_all_models_results.csv, ctg_all_models_predictions.csv")

# Confusion matrix plots
CLASS_ORDER = ["Normal", "Suspect", "Pathologic"]
def plot_confusion_matrix(y_true, y_pred, model_name):
    """Draw a normalized confusion matrix heatmap with percentages."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
    plt.title(f"{model_name} - Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()

for r in results:
    plot_confusion_matrix(y_test, r["y_pred"], r["model"])

# Best model summary
best_row = res_tbl.iloc[0]
best_name = best_row["model"]
best_pipe = models_fitted[best_name]
best = next(r for r in results if r["model"] == best_name)

print(f"\n=== Best model (CV-first): {best_name} ===")
print(
    f"CV_bal_acc={best['cv_bal_acc_mean']:.4f} | CV_f1_macro={best['cv_f1_macro_mean']:.4f} | "
    f"Test_bal_acc={best['test_bal_acc']:.4f} | Test_macro_f1={best['test_f1_macro']:.4f} | Test_acc={best['test_acc']:.4f}"
)
print("Classification report (best):")
print(best["report"])

# Normalized confusion matrix
cm = confusion_matrix(y_test + 1, best["y_pred_str"], labels=[1, 2, 3])
cm_norm = cm / cm.sum(axis=1, keepdims=True)
plt.figure()
plt.imshow(cm_norm, interpolation='nearest', cmap='Blues')
plt.title(f"CTG — {best_name} (Normalized Confusion Matrix)")
plt.colorbar()
tick_marks = np.arange(len(CLASS_ORDER))
plt.xticks(tick_marks, CLASS_ORDER, rotation=45, ha='right')
plt.yticks(tick_marks, CLASS_ORDER)
plt.xlabel("Predicted")
plt.ylabel("True")

# Annotate each cell with its value
thresh = cm_norm.max() / 2.
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                 ha="center", va="center",
                 color="white" if cm_norm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

# Save best pipeline again (optional)
joblib.dump(best_pipe, f"ctg_best_pipeline_{best_name}.joblib")
print(f"\nSaved best model pipeline -> ctg_best_pipeline_{best_name}.joblib")