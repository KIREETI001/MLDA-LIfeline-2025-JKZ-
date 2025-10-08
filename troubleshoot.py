import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from pathlib import Path

MODEL = "ctg_best_pipeline_XGB.joblib"

pipe = joblib.load(MODEL)
print("Pipeline steps:", list(getattr(pipe, "named_steps", {}).keys()))

# try to locate a ColumnTransformer used as the preprocessor
ct = None
if hasattr(pipe, "named_steps"):
    ct = pipe.named_steps.get("prep") or pipe.named_steps.get("preprocess") or pipe.named_steps.get("preprocessor")
if ct is None and hasattr(pipe, "named_steps"):
    for step in pipe.named_steps.values():
        if isinstance(step, ColumnTransformer):
            ct = step
            break

# Print column names the ColumnTransformer was fitted with (if available)
if ct is not None:
    trs = getattr(ct, "transformers_", None)
    cols = []
    if trs:
        for name, transformer, spec in trs:
            if isinstance(spec, (list, tuple)):
                cols.extend(list(spec))
    if cols:
        print(f"Detected input columns used at fit (count={len(cols)}):")
        print(cols)
    else:
        print("ColumnTransformer found but no explicit column list extracted.")

    # try to show number of output features after preprocessing (if available)
    try:
        out = ct.get_feature_names_out()
        print("Preprocessor output feature count:", len(out))
    except Exception:
        pass
else:
    print("No ColumnTransformer found in pipeline metadata.")

# Print final estimator expected feature count (n_features_in_)
final_est = None
if hasattr(pipe, "named_steps"):
    # search for a step that has n_features_in_
    for name, step in pipe.named_steps.items():
        if hasattr(step, "n_features_in_"):
            final_est = (name, step)
            break

if final_est:
    name, step = final_est
    print(f"Final estimator step '{name}' expects n_features_in_ = {getattr(step, 'n_features_in_', None)}")
else:
    # fallback: try transforming a sample to infer transformed width
    if Path(SAMPLE_CSV).exists():
        df = pd.read_csv(SAMPLE_CSV)
        if "NSP" in df.columns:
            Xs = df.drop(columns=["NSP"])
        else:
            Xs = df
        try:
            # try pipeline without final estimator
            pre = pipe[:-1] if hasattr(pipe, "steps") else pipe
            Xt = pre.transform(Xs.iloc[:5])
            print("Transformed shape (5 rows) ->", Xt.shape, "=> transformed feature count =", Xt.shape[1])
        except Exception as e:
            print("Couldn't transform sample to infer feature count:", e)
    else:
        print("No fallback sample CSV found to infer shape.")