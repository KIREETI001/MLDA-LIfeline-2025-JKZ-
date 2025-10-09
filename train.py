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

models = {
    "RandomForest": rf,
    "LogisticRegression": logreg,
    "SVM": svm,
}

if XGB_AVAILABLE:
    models["XGBoost"] = xgb

for name, pipe in models.items():
    out_path = f"ctg_pipeline_{name}.joblib"
    joblib.dump(pipe, out_path)
    print(f"Saved model pipeline -> {out_path}")
