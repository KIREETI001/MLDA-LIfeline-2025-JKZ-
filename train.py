import warnings, math, time
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, balanced_accuracy_score
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

from xgboost import XGBClassifier
plt.rcParams["figure.figsize"] = (7, 5)

# Import and availability check for XGBoost
try:
    from xgboost import XGBClassifier 
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# Randomness control
RANDOM_STATE = 40
np.random.seed(RANDOM_STATE)

# Making the data more aesthetic & readable
hr = lambda s: print("\n" + "="*18 + f" {s} " + "="*18)

# Data loading & Validation 
TARGET_COL = "NSP"          
DATA_PATH  = "./CTG_cleaned.csv" 
df = pd.read_csv(DATA_PATH)

df.columns

# Setting up for training
df = df[df['NSP'].notna()]
y_str = df[TARGET_COL].astype(int)
X = df.drop(columns=[TARGET_COL])

# Replace NSP numbers to words for better readability
label_map = {1: "Normal", 2: "Suspect", 3: "Pathologic"}
y_enc = y_str.map(label_map)

# Print NSP counts
hr("Class distribution")
print(y_enc.value_counts().sort_index())

# Convert NSP from [1,2,3] to [0,1,2] for ML algorithms
y_encoded = y_str - 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)
print(f"Train: {len(y_train)} | Test: {len(y_test)}")

# Pipeline setup
# Categorical features are ignored as they do not contribute to model

numeric_features = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler()),
])

preprocess = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features)], remainder="drop")

print("Numeric features:", len(numeric_features))

# Balancing class weights for imbalanced classes
classes_ = np.array(sorted(np.unique(y_train))) 
class_weights = compute_class_weight(class_weight="balanced", classes=classes_, y=y_train)
CW = {int(cls): float(w) for cls, w in zip(classes_, class_weights)}
print(CW)

# RandomForest
rf = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=CW  
    ))
])

# Multinomial Logistic Regression
logreg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        class_weight=CW,
        random_state=RANDOM_STATE
    ))
])

# SVM (RBF) with probabilities
svm = Pipeline([
    ("prep", preprocess),
    ("clf", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE))
])

# XGBoost
if XGB_AVAILABLE:
    xgb = Pipeline([
        ("prep", preprocess),
        ("clf", XGBClassifier(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=len(classes_),
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "bal_acc": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "acc": "accuracy",
}

def eval_model(name, pipe, Xtr, ytr, Xte, yte, use_sw=False):
    # Cross‑validation first
    try:
        cv_out = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=-1)
        cv_bal_acc = float(np.nanmean(cv_out["test_bal_acc"]))
        cv_f1_macro = float(np.nanmean(cv_out["test_f1_macro"]))
        cv_acc = float(np.nanmean(cv_out["test_acc"]))
    except Exception as e:
        cv_bal_acc = np.nan; cv_f1_macro = np.nan; cv_acc = np.nan

    # Fit (with optional sample weights)
    if use_sw:
        sw = pd.Series(ytr).map(CW).values
        pipe.fit(Xtr, ytr, **{"clf__sample_weight": sw})
    else:
        pipe.fit(Xtr, ytr)

    # Test metrics
    yhat = pipe.predict(Xte)
    acc = accuracy_score(yte, yhat)
    bal = balanced_accuracy_score(yte, yhat)
    f1m = f1_score(yte, yhat, average="macro")

    # Convert back to original labels for reporting
    y_true_orig = yte + 1  # Convert [0,1,2] back to [1,2,3]
    y_pred_orig = yhat + 1  # Convert [0,1,2] back to [1,2,3]
    
    LABEL_ORDER = [1, 2, 3]  # Use numeric labels for classification_report
    report = classification_report(y_true_orig, y_pred_orig, labels=LABEL_ORDER, digits=4, zero_division=0)

    return {
        "model": name,
        "cv_bal_acc_mean": cv_bal_acc,
        "cv_f1_macro_mean": cv_f1_macro,
        "cv_acc_mean": cv_acc,
        "test_acc": acc,
        "test_bal_acc": bal,
        "test_f1_macro": f1m,
        "y_pred": yhat,
        "y_pred_str": y_pred_orig,  # Original scale for display
        "report": report,
    }, pipe

results = []
models_fitted = {}

# Model training

hr("Training RandomForest")
res_rf, fitted_rf = eval_model("RandomForest", rf, X_train, y_train, X_test, y_test, use_sw=False)
results.append(res_rf); models_fitted["RandomForest"] = fitted_rf

hr("Training Logistic Regression")
res_lr, fitted_lr = eval_model("LogReg", logreg, X_train, y_train, X_test, y_test, use_sw=False)
results.append(res_lr); models_fitted["LogReg"] = fitted_lr

hr("Training SVM (RBF)")
res_svm, fitted_svm = eval_model("SVM-RBF", svm, X_train, y_train, X_test, y_test, use_sw=False)
results.append(res_svm); models_fitted["SVM-RBF"] = fitted_svm

hr("Training XGBoost")
res_xgb, fitted_xgb = eval_model("XGB", xgb, X_train, y_train, X_test, y_test, use_sw=True)
results.append(res_xgb); models_fitted["XGB"] = fitted_xgb


# Results table (sorted by CV balanced accuracy, then test balanced accuracy)
res_tbl = pd.DataFrame([{
    "model": r["model"],
    "cv_bal_acc_mean": np.round(r["cv_bal_acc_mean"], 4),
    "cv_f1_macro_mean": np.round(r["cv_f1_macro_mean"], 4),
    "cv_acc_mean": np.round(r["cv_acc_mean"], 4),
    "test_balanced_accuracy": np.round(r["test_bal_acc"], 4),
    "test_macro_f1": np.round(r["test_f1_macro"], 4),
    "test_accuracy": np.round(r["test_acc"], 4),
} for r in results])

# Primary key: CV balanced accuracy; tiebreakers: test balanced acc, then macro‑F1
res_tbl = res_tbl.sort_values([
    "cv_bal_acc_mean", "test_balanced_accuracy", "test_macro_f1"
], ascending=[False, False, False])

hr("Results (CV-first ranking)")
print(res_tbl.to_string(index=False))

# Prediction comparison (human‑readable)
preds_df = pd.DataFrame({"y_true": y_test+1}) # Convert [0,1,2] back to [1,2,3]
for r in results:
    preds_df[r["model"]] = r["y_pred_str"]
hr("Head of prediction comparison")
print(preds_df.head())

# Save CSVs
res_tbl.to_csv("ctg_all_models_results.csv", index=False)
preds_df.to_csv("ctg_all_models_predictions.csv", index=False)
print("Saved: ctg_all_models_results.csv, ctg_all_models_predictions.csv")

CLASS_ORDER = ["Normal", "Suspect", "Pathologic"]

# Define a helper function to draw each confusion matrix neatly
def plot_confusion_matrix(y_true, y_pred, model_name):
    """Draw a confusion matrix heatmap for a given model."""
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])  # assuming y_test is encoded as 0/1/2
    plt.figure(figsize=(5,4))                              # figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
    plt.title(f"{model_name} - Confusion Matrix")           # title per model
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.show()

# Loop through each trained model result in 'results'
for r in results:
    model_name = r["model"]            # get model name (e.g. "RandomForest")
    y_true = y_test                    # true labels from test set
    y_pred = r["y_pred"]               # predicted labels from model
    plot_confusion_matrix(y_true, y_pred, model_name)  # plot each

best_row = res_tbl.iloc[0]
best_name = best_row["model"]
best_pipe = models_fitted[best_name]

# Find the corresponding result dict for report/preds
best = next(r for r in results if r["model"] == best_name)

hr(f"Best model (CV-first): {best_name}")
print(
    f"CV_bal_acc={best['cv_bal_acc_mean']:.4f} | CV_f1_macro={best['cv_f1_macro_mean']:.4f} | "
    f"Test_bal_acc={best['test_bal_acc']:.4f} | Test_macro_f1={best['test_f1_macro']:.4f} | Test_acc={best['test_acc']:.4f}"
)
print("Classification report (best):")
print(best["report"])  

cm = confusion_matrix(y_test + 1, best["y_pred_str"], labels=[1, 2, 3])
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure()
plt.imshow(cm_norm, interpolation='nearest')
plt.title(f"CTG — {best_name} (Normalized Confusion Matrix)")
plt.colorbar()
plt.xticks(range(3), ["Normal", "Suspect", "Pathologic"], rotation=45, ha='right')
plt.yticks(range(3), ["Normal", "Suspect", "Pathologic"])
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.show()

# Save the best pipeline
out_path = f"ctg_best_pipeline_{best_name}.joblib"
joblib.dump(best_pipe, out_path)
print(f"Saved best model pipeline -> {out_path}")

