"""
Provides dataframe *like* utilities that operate on native dictionaries.
"""
from typing import Dict
import numpy as np

def filter_by_date(dict_in, 
                   start_date: np.datetime64 = None, 
                   end_date: np.datetime64 = None, 
                   date_key: str = 'date') -> Dict:
    date_vector = dict_in.pop(date_key)
    mask = np.ones(len(date_vector), dtype=bool)
    if start_date is not None:
        mask = (date_vector >= start_date)
    if end_date is not None:
        mask &= (date_vector <= end_date)
    date_vector_filtered = date_vector[mask]

    filtered_dict = dict(map(lambda k_v: (k_v[0], k_v[1][mask]), dict_in.items()))
    filtered_dict[date_key] = date_vector_filtered
    return filtered_dict
