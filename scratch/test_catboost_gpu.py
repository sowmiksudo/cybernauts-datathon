try:
    from catboost.utils import get_gpu_device_count
    print("Direct import count:", get_gpu_device_count())
except Exception as e:
    print("Direct import failed:", e)

try:
    from _catboost import _get_gpu_device_count
    print("_catboost count:", _get_gpu_device_count())
except Exception as e:
    print("_catboost failed:", e)
