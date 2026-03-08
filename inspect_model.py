"""
Run this in your project to inspect what features the preprocessor/model needs.
Place this file next to app.py and run: python inspect_model.py
"""
import joblib
from pathlib import Path
import numpy as np

MODEL_DIR = Path("models")

print("=" * 60)
print("  Model Inspector")
print("=" * 60)

# Load preprocessor
pp = joblib.load(MODEL_DIR / "preprocessor.pkl")
print(f"\nPreprocessor type: {type(pp)}")

if hasattr(pp, "feature_names_in_"):
    cols = list(pp.feature_names_in_)
    print(f"feature_names_in_: {len(cols)} features")
    print("First 20:", cols[:20])
    print("Last 20:", cols[-20:])
elif hasattr(pp, "get_feature_names_out"):
    cols = list(pp.get_feature_names_out())
    print(f"get_feature_names_out: {len(cols)} features")
    print("First 20:", cols[:20])
elif hasattr(pp, "transformers_"):
    print("ColumnTransformer detected")
    for name, trans, features in pp.transformers_:
        print(f"  [{name}] {type(trans).__name__}: {features[:5]}... ({len(features)} cols)")
elif hasattr(pp, "steps"):
    print("Pipeline steps:", [s[0] for s in pp.steps])
    for name, step in pp.steps:
        if hasattr(step, "feature_names_in_"):
            print(f"  {name} feature_names_in_: {len(step.feature_names_in_)}")
            print("  First 20:", list(step.feature_names_in_)[:20])

# Load XGB
xgb = joblib.load(MODEL_DIR / "xgb_model.pkl")
print(f"\nXGBoost type: {type(xgb)}")
if hasattr(xgb, "feature_names_in_"):
    print(f"feature_names_in_: {len(xgb.feature_names_in_)} features")
    print("First 20:", list(xgb.feature_names_in_)[:20])
if hasattr(xgb, "n_features_in_"):
    print(f"n_features_in_: {xgb.n_features_in_}")
if hasattr(xgb, "get_booster"):
    try:
        fn = xgb.get_booster().feature_names
        if fn:
            print(f"Booster feature_names ({len(fn)}): {fn[:20]}")
            # Save all feature names to file
            with open("model_feature_names.txt", "w") as f:
                f.write("\n".join(fn))
            print(f"\n✅ All {len(fn)} feature names saved to model_feature_names.txt")
    except Exception as e:
        print(f"Booster features error: {e}")

print("\n" + "="*60)