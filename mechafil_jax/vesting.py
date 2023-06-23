from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from functools import partial

from .constants import NETWORK_START
from . import df_utils
from . import date_utils

FIL_BASE = 2_000_000_000.0
PL_AMOUNT = 0.15 * FIL_BASE
FOUNDATION_AMOUNT = 0.05 * FIL_BASE
STORAGE_MINING = 0.55 * FIL_BASE
MINING_RESERVE = 0.15 * FIL_BASE

"""
A port of the vesting module in mechafil to JAX.
"""

@partial(jax.jit, static_argnums=(0,1,2))
def compute_vesting_trajectory(
    start_date: np.datetime64, end_date: np.datetime64, start_vested_amt: int
) -> Dict:
    """
    15% to PL -> 6-year linear vesting
    5% to FIlecoin foundation -> 6-year linear vesting
    10% to Investors -> Linear vesting with different durations (taken from lotus):
        - 0 days: 10_632_000
        - 6 months: 19_015_887 + 32_787_700
        - 1 yrs: 22_421_712 + 9_400_000
        - 2 yrs: 7_223_364
        - 3 yrs: 87_637_883 + 898_958
        - 6 yrs: 9_805_053
        (total of 199_822_557)

    Info taken from:
        - https://coinlist.co/assets/index/filecoin_2017_index/Filecoin-Sale-Economics-e3f703f8cd5f644aecd7ae3860ce932064ce014dd60de115d67ff1e9047ffa8e.pdf
        - https://spec.filecoin.io/#section-systems.filecoin_token.token_allocation
        - https://filecoin.io/blog/filecoin-circulating-supply/
        - https://github.com/filecoin-project/lotus/blob/e65fae28de2a44947dd24af8c7dafcade29af1a4/chain/stmgr/supply.go#L148
    """
    # we assume vesting started at main net launch, in 2020-10-15
    launch_date = NETWORK_START
    start_day = date_utils.datetime64_delta_to_days(start_date - launch_date)
    end_day = date_utils.datetime64_delta_to_days(end_date - launch_date)

    # Get entire daily vesting trajectory
    full_vest_dict = {
        "days": np.arange(0, end_day),
        "six_month_vest_saft": vest(19_015_887 + 32_787_700, 183, end_day),
        "one_year_vest_saft": vest(22_421_712 + 9_400_000, 365 * 1, end_day),
        "two_year_vest_saft": vest(7_223_364, 365 * 2, end_day),
        "three_year_vest_saft": vest(87_637_883 + 898_958, 365 * 3, end_day),
        "six_year_vest_saft": vest(9_805_053, 365 * 6, end_day),
        "six_year_vest_pl": vest(PL_AMOUNT, 365 * 6, end_day),
        "six_year_vest_foundation": vest(FOUNDATION_AMOUNT, 365 * 6, end_day),
    }

    # Filter vesting trajectory for desired dates
    vest_dict = df_utils.filter_by_day(full_vest_dict, start_day)
    day_vector = vest_dict.pop("days")
    # Compute total cumulative vesting
    total_day_vest = jnp.sum(jnp.asarray(list(vest_dict.values())), axis=0)
    # start_vested_amt = get_vested_amount(start_date)  # NOTE: this needs to be passed in
                                                        # b/c we don't want to diff through
                                                        # the data download, so we do that a-priori
                                                        # and pass in necessary info
    total_vest = jnp.cumsum(total_day_vest) + start_vested_amt
    
    vest_dict_to_return = {
        "days": day_vector,
        'total_day_vest': total_day_vest,
        'total_vest': total_vest,  # for debugging. can remove later
        **vest_dict
    }
    return vest_dict_to_return

@partial(jax.jit, static_argnums=(0,1,2))
def vest(amount: float, time: int, end_day: int) -> jnp.ndarray:
    """
    amount -- total amount e.g 300M FIL for SAFT
    time -- vesting time in days
    end_day -- end day for the vesting trajectory
    """
    ones_ = jnp.ones(int(time))[:end_day]
    extra_to_pad_ = max(0, end_day - int(time))
    ones_padded_ = jnp.pad(ones_, (0, extra_to_pad_))
    vest_ = ones_padded_ / time
    return amount * vest_
