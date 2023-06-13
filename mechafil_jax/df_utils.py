"""
Provides dataframe *like* utilities that operate on native dictionaries.
"""
from typing import Dict
import numpy as np

def __filter_by_date(dict_in, 
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

def __check_dates(dict1, dict2, date_key='date'):
    datevec1 = dict1[date_key]
    datevec2 = dict2[date_key]
    return np.all(datevec1 == datevec2)


def filter_by_day(dict_in,
                  start_day: int = None,
                  end_day: int = None,
                  day_key: str = 'days') -> Dict:
    day_vector = dict_in.pop(day_key)
    mask = np.ones(len(day_vector), dtype=bool)
    if start_day is not None:
        mask = (day_vector >= start_day)
    if end_day is not None:
        mask &= (day_vector <= end_day)
    day_vector_filtered = day_vector[mask]
    
    filtered_dict = dict(map(lambda k_v: (k_v[0], k_v[1][mask]), dict_in.items()))
    filtered_dict[day_key] = day_vector_filtered
    return filtered_dict

def check_days(dict1, dict2, day_key='days'):
    dayvec1 = dict1[day_key]
    dayvec2 = dict2[day_key]
    return np.all(dayvec1 == dayvec2)

def inner_join(dict1, dict2, key='days', coerce=True):
    """
    Inner join two dictionaries on a key.

    If coerce is set to true, the dictionaries are filtered to the intersection of their day ranges.
    """
    if coerce:
        dict1 = filter_by_day(dict1, start_date=dict2[key][0], end_date=dict2[key][-1], day_key=key)
        dict2 = filter_by_day(dict2, start_date=dict1[key][0], end_date=dict1[key][-1], day_key=key)
    else:
        if not check_days(dict1, dict2, day_key=key):
            raise ValueError("Dates do not match, implement partial join or set coerce=True")
    assert np.all(dict1[key] == dict2[key])
    dict2.pop(key)
    return {**dict1, **dict2}