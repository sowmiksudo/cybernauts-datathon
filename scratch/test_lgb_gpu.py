import lightgbm as lgb
import numpy as np

try:
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    ds = lgb.Dataset(X, label=y)
    params = {"objective": "binary", "device": "gpu", "verbose": -1}
    lgb.train(params, ds, num_boost_round=1)
    print("LightGBM GPU: SUCCESS")
except Exception as e:
    print("LightGBM GPU: FAILED -", e)
