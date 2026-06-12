import sys
import os

print("=== Python Environment ===")
print("Python executable:", sys.executable)
print("Python version:", sys.version)

print("\n=== CatBoost GPU Check ===")
try:
    import catboost
    gpu_count = catboost.utils.get_gpu_device_count()
    print("CatBoost GPU count:", gpu_count)
except Exception as e:
    print("CatBoost check failed:", e)

print("\n=== XGBoost GPU Check ===")
try:
    import xgboost as xgb
    print("XGBoost version:", xgb.__version__)
    # Try a simple GPU training
    import numpy as np
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    dtrain = xgb.DMatrix(X, label=y)
    params = {'objective': 'binary:logistic', 'device': 'cuda'}
    bst = xgb.train(params, dtrain, num_boost_round=1)
    print("XGBoost CUDA check: SUCCESS")
except Exception as e:
    print("XGBoost CUDA check: FAILED -", e)

print("\n=== Environment Variables ===")
for k in ["CUDA_PATH", "CUDA_HOME", "PATH"]:
    v = os.environ.get(k, "")
    if k == "PATH":
        cuda_paths = [p for p in v.split(os.pathsep) if "cuda" in p.lower() or "nvidia" in p.lower()]
        print("CUDA/NVIDIA in PATH:", cuda_paths)
    else:
        print(f"{k}: {v}")
