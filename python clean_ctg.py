# -----------------------------
# CTG Cleaning & Preprocessing
# -----------------------------

# --- Imports (data wrangling & ML utilities) ---
import os                                # Work with file paths (detect .xls vs .xlsx)
import re                                # Clean/normalize column names
import joblib                            # Save preprocessing artifacts for reuse
import numpy as np                       # Numerical ops, NaN handling
import pandas as pd                      # DataFrame operations (read Excel, clean, save)

from typing import List                  # Type hints for clarity/readability

from sklearn.model_selection import train_test_split        # Proper train/test split
from sklearn.compose import ColumnTransformer               # Build preprocessing pipelines
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer                    # Robust imputation


# --- CONFIG (edit these to match your file/setup) ---
INPUT_FILE = "CTG.xlsx"   # Your converted file; use "CTG (1).xls" if you prefer .xls
SHEET_NAME = 0            # Sheet index or name if your Excel has multiple sheets
SAVE_PREFIX = "ctg_clean" # All outputs share this prefix (CSV, Parquet, pipeline)
RANDOM_STATE = 42         # Reproducibility for split
TEST_SIZE = 0.2           # 80/20 split is a solid baseline

# Target column candidates (common names in CTG datasets)
TARGET_CANDIDATES = ["nsp", "fetal_health", "class", "label", "target"]

# Whether to cap outliers using IQR winsorization (reduces model sensitivity to extremes)
USE_IQR_WINSORIZE = True

# Optional plausible physiological bounds (domain-guardrails).
# If a value is outside a plausible range, we mark it NaN and let imputation handle it.
PLAUSIBLE_BOUNDS = {
    "lb": (80, 200),      # baseline fetal heart rate (bpm)
    "ac": (0, 10),        # accelerations (rate/count; keep broad)
    "fm": (0, 50),        # fetal movements
    "uc": (0, 20),        # uterine contractions
    "dl": (0, 10),        # light decelerations
    "ds": (0, 10),        # severe decelerations
    "dp": (0, 10),        # prolonged decelerations
    "astv": (0, 100),     # % abnormal short-term variability
    "mstv": (0, 10),      # mean short-term variability
    "mltv": (0, 10),      # mean long-term variability
    "width": (0, 200),    # histogram width (UCI CTG features)
    "min": (50, 200),
    "max": (80, 240),
    "nmax": (0, 100),
    "nzeros": (0, 100),
    "mode": (50, 200),
    "mean": (50, 200),
    "median": (50, 200),
    "variance": (0, 1000),
    "tendency": (-1, 1),
}


# --- Helper: read Excel robustly for .xls and .xlsx ---
def read_ctg(path: str, sheet=0) -> pd.DataFrame:
    """
    Reads both .xls (via xlrd) and .xlsx (via openpyxl default).
    Justification: users often receive older .xls files from hospital systems.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xls":
        # xlrd is required for legacy Excel
        return pd.read_excel(path, sheet_name=sheet, engine="xlrd")
    # for .xlsx pandas will auto-use openpyxl if available
    return pd.read_excel(path, sheet_name=sheet)


# --- Helper: normalize column names to snake_case ---
def clean_col(c: str) -> str:
    """
    Standardize headers: trim, replace spaces/dashes with underscores,
    drop non-alphanumeric chars, lowercase.
    Justification: avoids future KeyErrors and makes code portable across versions.
    """
    c = c.strip()
    c = re.sub(r"[\s\-/]+", "_", c)
    c = re.sub(r"[^0-9a-zA-Z_]", "", c)
    return c.lower()


# --- Helper: find the target column among common names ---
def find_target_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Searches typical CTG target names. Fails loudly if nothing sensible is found.
    Justification: many public CTG datasets use 'NSP' (1=Normal,2=Suspect,3=Pathologic),
    but some rename to 'fetal_health' or 'Class'.
    """
    cols = df.columns.tolist()
    # exact matches first
    for cand in candidates:
        if cand in cols:
            return cand
    # case-insensitive fallback
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    # last-chance heuristic: if last col has tiny cardinality, treat it as label
    last = cols[-1]
    if df[last].nunique(dropna=True) <= 5:
        print(f"[WARN] Target not found by name; guessing last column '{last}' as label.")
        return last
    raise ValueError(f"Target column not found. Tried {candidates}. Available: {cols}")


# --- Helper: IQR winsorization (outlier capping) ---
def winsorize_iqr(s: pd.Series, k: float = 1.5) -> pd.Series:
    """
    Caps values outside [Q1 - k*IQR, Q3 + k*IQR].
    Justification: In clinical data, extreme values can be data-entry artifacts or rare events.
    Capping reduces undue influence while keeping the record.
    """
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return s.clip(lower=lo, upper=hi)


# ========= 1) LOAD & INSPECT =========
df_raw = read_ctg(INPUT_FILE, SHEET_NAME)           # Read the Excel sheet into a DataFrame
print("Raw shape:", df_raw.shape)                   # Basic overview of rows/cols before cleaning
print("Raw columns:", list(df_raw.columns))         # See raw header names to catch typos/spaces


# ========= 2) CLEAN HEADERS & DROP EMPTY COLUMNS =========
df = df_raw.copy()                                  # Work on a copy; keep raw for auditability
df.columns = [clean_col(c) for c in df.columns]     # Normalize headers to snake_case

# Drop columns that are entirely empty (all NaN or all blank strings)
empty_cols = [c for c in df.columns
              if df[c].isna().all() or (df[c].astype(str).str.strip() == "").all()]
if empty_cols:
    print("Dropping empty columns:", empty_cols)
    df = df.drop(columns=empty_cols)


# ========= 3) STANDARDIZE MISSING TOKENS & COERCE NUMERICS =========
# Replace common "missing" tokens with np.nan so pandas treats them uniformly
df = df.replace(
    to_replace=["", " ", "na", "n/a", "null", "none", "?", "-", "--"],
    value=np.nan
)

# Try to convert text columns that are mostly numbers into numeric dtype
for c in df.columns:
    if df[c].dtype == object:
        # detect numeric-looking strings (e.g., "123", "3.14", "-7", "1e-3")
        numlike = df[c].astype(str).str.match(r"^[-+]?\d*\.?\d+(e[-+]?\d+)?$", case=False)
        if numlike.mean() > 0.6:                    # if majority look numeric, coerce
            df[c] = pd.to_numeric(df[c], errors="coerce")

print("Dtypes after coercion:\n", df.dtypes)        # Sanity-check result of coercion


# ========= 4) IDENTIFY TARGET & NORMALIZE ITS LABELS =========
target_col = find_target_column(df, TARGET_CANDIDATES)   # Find the label column
print("Target column ->", target_col)

# If target is numeric with values {1,2,3}, map to clinical labels; else, tidy strings.
if pd.api.types.is_numeric_dtype(df[target_col]):
    # typical NSP encoding: 1=Normal, 2=Suspect, 3=Pathologic
    mapping = {1: "Normal", 2: "Suspect", 3: "Pathologic", 0: "Unknown"}
    df[target_col] = df[target_col].map(mapping).fillna("Unknown")
else:
    # normalize variants like "normal"/"Normal"/"N" => "Normal"
    s = df[target_col].astype(str).str.strip().str.lower()
    s = s.replace({"1": "normal", "2": "suspect", "3": "pathologic",
                   "n": "normal", "s": "suspect", "p": "pathologic"})
    df[target_col] = s.str.capitalize()

# Justification:
# - Clinical stakeholders understand "Normal/Suspect/Pathologic".
# - Consistent labels also help stratified splitting and metrics per class.


# ========= 5) DROP DUPLICATE ROWS =========
before = len(df)
df = df.drop_duplicates()
print(f"Removed duplicates: {before - len(df)}")
# Justification:
# - Duplicates can bias training and inflate test performance via leakage.


# ========= 6) PHYSIOLOGICAL BOUNDS CHECK (OPTIONAL BUT USEFUL) =========
# Any value outside domain-plausible range -> set to NaN for later imputation
for col_key, (lo, hi) in PLAUSIBLE_BOUNDS.items():
    if col_key in df.columns:
        df[col_key] = pd.to_numeric(df[col_key], errors="coerce")
        df.loc[(df[col_key] < lo) | (df[col_key] > hi), col_key] = np.nan
# Justification:
# - Impossible values are likely errors; better to impute than to mislead the model.


# ========= 7) OUTLIER CAPPING (IQR WINSORIZATION) =========
if USE_IQR_WINSORIZE:
    for c in df.columns:
        if c == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() >= 20:
            df[c] = winsorize_iqr(df[c], k=1.5)
# Justification:
# - Reduces undue influence of extreme values (measurement spikes, digitization artifacts),
#   while preserving each row (important for clinical datasets with limited size).


# ========= 8) SPLIT FEATURES BY TYPE =========
# Identify categorical features as low-cardinality non-numeric (safe heuristic)
categorical_cols = [c for c in df.columns
                    if c != target_col and (df[c].dtype == object or df[c].dtype.name == "category")
                    and df[c].nunique(dropna=True) <= 20]

# All remaining numerics (excluding target) are numeric features
numeric_cols = [c for c in df.columns
                if c != target_col and pd.api.types.is_numeric_dtype(df[c])]

print("Categorical columns:", categorical_cols)
print("Numeric columns (first 10):", numeric_cols[:10])


# ========= 9) CREATE AN 'EXPORT VIEW' WITH SIMPLE IMPUTATION =========
# We keep two views:
# - df_export: fills NaNs for a clean CSV/Parquet you can hand off or EDA safely
# - df_model: keeps NaNs, to be imputed only on TRAIN (prevents leakage)

df_export = df.copy()

# Fill numeric NaNs with median (robust to skew)
for c in numeric_cols:
    df_export[c] = df_export[c].fillna(df_export[c].median())

# Fill categorical NaNs with mode (most frequent)
for c in categorical_cols:
    mode = df_export[c].mode(dropna=True)
    df_export[c] = df_export[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")

# Quick sanity prints
print("Top 10 columns by remaining NaNs (export view):")
print(df_export.isna().sum().sort_values(ascending=False).head(10))

print("Class distribution:")
print(df_export[target_col].value_counts(dropna=False))


# ========= 10) SAVE CLEANED DATA (CSV + PARQUET) =========
csv_path = f"{SAVE_PREFIX}.csv"
parquet_path = f"{SAVE_PREFIX}.parquet"
df_export.to_csv(csv_path, index=False)
df_export.to_parquet(parquet_path, index=False)
print(f"Saved: {csv_path} and {parquet_path}")
# Justification:
# - CSV for universal readability, Parquet for compact, fast I/O in Python/Scala/R.


# ========= 11) BUILD A LEAKAGE-SAFE PREPROCESSOR FOR MODELING =========
# Model-ready view: keep NaNs to impute on TRAIN ONLY via pipeline
X = df.drop(columns=[target_col])
y = df[target_col].astype("category")  # explicit category dtype helps with ordered outputs

# Split with stratification to preserve class ratios
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Build column-wise transformers:
# - Numeric: SimpleImputer(median) then StandardScaler (common for many ML algos)
# - Categorical: SimpleImputer(mode) then OneHotEncoder (ignore unknowns at test time)
num_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_features = [c for c in X.columns if c not in num_features]

numeric_imputer = SimpleImputer(strategy="median")
scaler = StandardScaler(with_mean=True, with_std=True)

categorical_imputer = SimpleImputer(strategy="most_frequent")
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# ColumnTransformer wires per-type preprocessing into a single object
preprocess = ColumnTransformer(
    transformers=[
        ("num_impute", numeric_imputer, num_features),
        ("num_scale", scaler, num_features),           # scaling runs after imputation
        ("cat_impute", categorical_imputer, cat_features),
        ("cat_ohe", ohe, cat_features),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# Fit the preprocessor on TRAIN ONLY (prevents information leakage)
_ = preprocess.fit(X_train)

# Save the fitted preprocessor + useful metadata for your training script
joblib.dump({
    "preprocess": preprocess,
    "numeric_features": num_features,
    "categorical_features": cat_features,
    "target_col": target_col,
    "classes_": y.cat.categories.tolist(),
}, f"{SAVE_PREFIX}_preprocess.joblib")

print(f"Saved preprocessing bundle: {SAVE_PREFIX}_preprocess.joblib")

# Done — you now have:
# - ctg_clean.csv / ctg_clean.parquet: clean, imputed export
# - ctg_clean_preprocess.joblib: train-fitted pipeline to plug into models
