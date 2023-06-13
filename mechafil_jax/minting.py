from typing import Union, Dict

import numpy as np
from numpy.typing import NDArray

import jax
import jax.numpy as jnp

from functools import partial

from .constants import EXA, EXBI, PIB, NETWORK_START
from .date_utils import datetime64_delta_to_days

LAMBDA = np.log(2) / (
    6.0 * 365
)  # minting exponential reward decay rate (6yrs half-life)
FIL_BASE = 2_000_000_000.0
STORAGE_MINING = 0.55 * FIL_BASE
SIMPLE_ALLOC = 0.3 * STORAGE_MINING  # total simple minting allocation
BASELINE_ALLOC = 0.7 * STORAGE_MINING  # total baseline minting allocation
GROWTH_RATE = float(
    np.log(2) / 365.0
)  # daily baseline growth rate (the "g" from https://spec.filecoin.io/#section-systems.filecoin_token)

# NOTE: the baseline storage value is the baseline storage power at the genesis
# The spec notes that this value is 2.888888888, but the actual data from starboard
# shows that the value is 2.766213637444971.  We use the actual data here.
#
# Query:
# 3189227188947035000 from https://observable-api.starboard.ventures/api/v1/observable/network-storage-capacity/new_baseline_power
BASELINE_STORAGE = 2.766213637444971 * EXA / EXBI  # b_0 from https://spec.filecoin.io/#section-systems.filecoin_token

@partial(jax.jit, static_argnums=(0, 1, 6, 7, 8))
def compute_minting_trajectory_df(
    start_date: np.datetime64,
    end_date: np.datetime64,
    rb_total_power_eib: Union[jnp.ndarray, NDArray],
    qa_total_power_eib: Union[jnp.ndarray, NDArray],
    qa_day_onboarded_power_pib: Union[jnp.ndarray, NDArray],
    qa_day_renewed_power_pib: Union[jnp.ndarray, NDArray],
    zero_cum_capped_power_eib: float,
    init_baseline_eib: float,
    minting_base: str = 'RBP'
) -> Dict:
    # do things in EIB to prevent overflow
    # using float64 seems to be not good in JAX at the moment

    # we assume minting started at main net launch, in 2020-10-15
    start_day = datetime64_delta_to_days(start_date - NETWORK_START)
    end_day = datetime64_delta_to_days(end_date - NETWORK_START)

    minting_base = minting_base.lower()
    capped_power_reference = 'network_RBP_EIB' if minting_base == 'rbp' else 'network_QAP_EIB'

    minting_dict = {
        "days": jnp.arange(start_day, end_day),
        "network_RBP_EIB": rb_total_power_eib,
        "network_QAP_EIB": qa_total_power_eib,
        "day_onboarded_power_QAP_PIB": qa_day_onboarded_power_pib,
        "day_renewed_power_QAP_PIB": qa_day_renewed_power_pib,
    }

    # Compute cumulative rewards due to simple minting
    minting_dict["cum_simple_reward"] = cum_simple_minting(minting_dict["days"])
    # Compute cumulative rewards due to baseline minting
    minting_dict["network_baseline_EIB"] = compute_baseline_power_array(start_date, end_date, init_baseline_eib)

    minting_dict["capped_power_EIB"] = jnp.minimum(minting_dict["network_baseline_EIB"], minting_dict[capped_power_reference])
    # zero_cum_capped_power = get_cum_capped_rb_power(start_date)
    minting_dict["cum_capped_power_EIB"] = minting_dict["capped_power_EIB"].cumsum() + zero_cum_capped_power_eib
    minting_dict["network_time"] = network_time(minting_dict["cum_capped_power_EIB"])
    minting_dict["cum_baseline_reward"] = cum_baseline_reward(minting_dict["network_time"])
    # Add cumulative rewards and get daily rewards minted
    minting_dict["cum_network_reward"] = minting_dict["cum_baseline_reward"] + minting_dict["cum_simple_reward"]
    cum_network_reward_zero = minting_dict["cum_network_reward"][0]
    
    day_network_reward = jnp.zeros(len(minting_dict["cum_network_reward"])+1).astype(jnp.float32)
    day_network_reward = day_network_reward.at[1:].set(minting_dict["cum_network_reward"])
    day_network_reward = day_network_reward.at[0].set(cum_network_reward_zero) # equiv. to prepend in NP
    day_network_reward = jnp.diff(day_network_reward)
    day_network_reward = day_network_reward.at[0].set(day_network_reward[1]) # to match mechaFIL
    minting_dict["day_network_reward"] = day_network_reward

    return minting_dict

@jax.jit
def cum_simple_minting(day: int) -> float:
    """
    Simple minting - the total number of tokens that should have been emitted
    by simple minting up until date provided.
    """
    return SIMPLE_ALLOC * (1 - jnp.exp(-LAMBDA * day))


@partial(jax.jit, static_argnums=(0,1,2))
def compute_baseline_power_array(
    start_date: np.datetime64, end_date: np.datetime64, init_baseline: float,
) -> Union[jnp.ndarray, NDArray, float]:
    arr_len = datetime64_delta_to_days(end_date - start_date)
    exponents = jnp.arange(0, arr_len)
    baseline_power_arr = init_baseline * jnp.exp(GROWTH_RATE * exponents)
    return baseline_power_arr


@jax.jit
def network_time(cum_capped_power: Union[jnp.ndarray, NDArray, float]) -> Union[jnp.ndarray, NDArray, float]:
    b0 = BASELINE_STORAGE
    g = GROWTH_RATE
    return (1 / g) * jnp.log(((g * (cum_capped_power)) / b0) + 1)


@jax.jit
def cum_baseline_reward(network_time: Union[jnp.ndarray, NDArray, float]) -> Union[jnp.ndarray, NDArray, float]:
    return BASELINE_ALLOC * (1 - jnp.exp(-LAMBDA * network_time))
