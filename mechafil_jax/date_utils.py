from datetime import date, timedelta
import numpy as np

def datetime64_delta_to_days(delta: np.timedelta64) -> int:
    return int(delta.astype('timedelta64[D]') / np.timedelta64(1, 'D'))

def get_t(start_date: date, forecast_length: int = None, end_date: date = None):
    if forecast_length is None and end_date is None:
        raise ValueError("Must specify either forecast_length or end_date")
    if forecast_length is not None and end_date is not None:
        raise ValueError("Must specify either forecast_length or end_date, not both")
    if end_date is not None:
        forecast_length = (end_date - start_date).days
    t = [start_date + timedelta(days=i) for i in range(forecast_length)]
    return t
