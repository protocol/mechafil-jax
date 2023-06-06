import numpy as np

def datetime64_delta_to_days(delta: np.timedelta64) -> int:
    return int(delta.astype('timedelta64[D]') / np.timedelta64(1, 'D'))