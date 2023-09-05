from typing import Union, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

import jax
import jax.lax as lax
import jax.numpy as jnp
from functools import partial

import mechafil_jax.imitate_lax as imitate_lax

from .locking import (
    get_day_schedule_pledge_release,
    compute_day_reward_release,
    compute_day_delta_pledge,
    compute_day_locked_rewards,
    compute_day_locked_pledge,
)
from .constants import NETWORK_START
from .date_utils import datetime64_delta_to_days
from .df_utils import inner_join

"""
A port of the supply module in mechafil to JAX.
"""

@jax.jit
def accum_gas_burn(arggs):
    cur_day_gas_burn, prev_day_gas_burn, daily_burnt_fil = arggs
    return prev_day_gas_burn + daily_burnt_fil

@jax.jit
def passthrough(arggs):
    cur_day_gas_burn, prev_day_gas_burn, daily_burnt_fil = arggs
    return cur_day_gas_burn

@jax.jit
def update_cs_day(carry, x):
    # Compute daily change in initial pledge collateral
    day_idx, current_day_idx, cs_dict, known_scheduled_pledge_release_vec, circ_supply, daily_burnt_fil, len_burnt_fil_vec, renewal_rate_vec, duration, lock_target, gamma_vec, gamma_weight_type_vec = carry

    day_pledge_locked_vec = cs_dict["day_locked_pledge"]
    scheduled_pledge_release = get_day_schedule_pledge_release(
        day_idx,
        current_day_idx,
        day_pledge_locked_vec,
        known_scheduled_pledge_release_vec,
        duration,
    )
    pledge_delta = compute_day_delta_pledge(
        cs_dict["day_network_reward"][day_idx],
        circ_supply,
        cs_dict["day_onboarded_power_QAP_PIB"][day_idx],
        cs_dict["day_renewed_power_QAP_PIB"][day_idx],
        cs_dict["network_QAP_EIB"][day_idx],
        cs_dict["network_baseline_EIB"][day_idx],
        renewal_rate_vec[day_idx],
        scheduled_pledge_release,
        lock_target[day_idx],
        gamma_vec[day_idx],
        gamma_weight_type_vec[day_idx],
    )
    # Get total locked pledge (needed for future day_locked_pledge)
    day_locked_pledge, day_renewed_pledge = compute_day_locked_pledge(
        cs_dict["day_network_reward"][day_idx],
        circ_supply,
        cs_dict["day_onboarded_power_QAP_PIB"][day_idx],
        cs_dict["day_renewed_power_QAP_PIB"][day_idx],
        cs_dict["network_QAP_EIB"][day_idx],
        cs_dict["network_baseline_EIB"][day_idx],
        renewal_rate_vec[day_idx],
        scheduled_pledge_release,
        lock_target[day_idx],
        gamma_vec[day_idx],
        gamma_weight_type_vec[day_idx],
    )

    # Compute daily change in block rewards collateral
    day_locked_rewards = compute_day_locked_rewards(
        cs_dict["day_network_reward"][day_idx]
    )
    day_reward_release = compute_day_reward_release(
        cs_dict["network_locked_reward"][day_idx - 1]
    )
    reward_delta = day_locked_rewards - day_reward_release
    # Update dataframe
    cs_dict["day_locked_pledge"] = cs_dict["day_locked_pledge"].at[day_idx].set(day_locked_pledge)
    cs_dict["day_renewed_pledge"] = cs_dict["day_renewed_pledge"].at[day_idx].set(day_renewed_pledge)
    cs_dict["network_locked_pledge"] = cs_dict["network_locked_pledge"].at[day_idx].set(
        cs_dict["network_locked_pledge"][day_idx - 1] + pledge_delta
    )
    cs_dict["network_locked_reward"] = cs_dict["network_locked_reward"].at[day_idx].set(
        cs_dict["network_locked_reward"][day_idx - 1] + reward_delta
    )
    cs_dict["network_locked"] = cs_dict["network_locked"].at[day_idx].set(
        cs_dict["network_locked"][day_idx - 1] + pledge_delta + reward_delta
    )
    
    # Update gas burnt
    pred = (day_idx >= len_burnt_fil_vec)
    gas_burn_val = lax.cond(
        pred,
        accum_gas_burn,
        passthrough,
        (cs_dict["network_gas_burn"][day_idx], cs_dict["network_gas_burn"][day_idx-1], daily_burnt_fil)
    )
    cs_dict["network_gas_burn"] = cs_dict["network_gas_burn"].at[day_idx].set(gas_burn_val)

    # Find circulating supply balance and update
    circ_supply = (
        cs_dict["disbursed_reserve"][day_idx]  # from initialise_circulating_supply_df
        + cs_dict["cum_network_reward"][day_idx]  # from the minting_model
        + cs_dict["total_vest"][day_idx]  # from vesting_model
        - cs_dict["network_locked"][day_idx]  # from simulation loop
        - cs_dict["network_gas_burn"][day_idx]  # comes from user inputs
    )
    circ_supply = jnp.maximum(circ_supply, 0)
    cs_dict["circ_supply"] = cs_dict["circ_supply"].at[day_idx].set(circ_supply)

    return_carry = (
        day_idx + 1,
        current_day_idx,
        cs_dict,
        known_scheduled_pledge_release_vec,
        circ_supply,
        daily_burnt_fil,
        len_burnt_fil_vec,
        renewal_rate_vec,
        duration,
        lock_target,
        gamma_vec,
        gamma_weight_type_vec,
    )
    return (return_carry, None)

@partial(jax.jit, static_argnums=(0,1,2,3,4,5,6))
def forecast_circulating_supply(
    start_date: np.datetime64,
    current_date: np.datetime64,
    end_date: np.datetime64,
    circ_supply_zero: float,
    locked_fil_zero: float,
    daily_burnt_fil: float,
    duration: int,
    renewal_rate_vec: Union[jnp.array, NDArray],
    burnt_fil_vec: jnp.array,
    vest_dict: Dict,
    mint_dict: Dict,
    known_scheduled_pledge_release_vec: Union[jnp.array, NDArray],
    lock_target: Union[jnp.array, NDArray],
    gamma: Union[jnp.array, NDArray],
    gamma_weight_type: Union[jnp.array, NDArray],
) -> Dict:
    # we assume all stats started at main net launch, in 2020-10-15
    start_day = datetime64_delta_to_days(start_date - NETWORK_START)
    current_day = datetime64_delta_to_days(current_date - NETWORK_START)
    end_day = datetime64_delta_to_days(end_date - NETWORK_START)
    # initialise dataframe and auxilialy variables
    cs_dict = initialise_circulating_supply_dict(
        start_date,
        end_date,
        circ_supply_zero,
        locked_fil_zero,
        burnt_fil_vec,
        vest_dict,
        mint_dict,
    )
    circ_supply = circ_supply_zero
    sim_len = end_day - start_day
    assert len(renewal_rate_vec) == sim_len, "renewal_rate must be of length sim_len = len(historical_data) + forecast_length = {sim_len}"
    assert len(lock_target) == sim_len, "lock_target vec must be of length sim_len = len(historical_data) + forecast_length = {sim_len}"

    day_idx_start = 1
    current_day_idx = current_day - start_day
    init_in = (day_idx_start, current_day_idx, cs_dict, known_scheduled_pledge_release_vec, 
               circ_supply, daily_burnt_fil, len(burnt_fil_vec), renewal_rate_vec, duration, 
               lock_target, gamma, gamma_weight_type)
    ret, _ = lax.scan(update_cs_day, init_in, None, length=sim_len)
    # ret, _ = imitate_lax.scan(update_cs_day, init_in, None, length=sim_len-1)  # for debugging and seeing print statements
    cs_dict = ret[2]

    return cs_dict


@partial(jax.jit, static_argnums=(0,1,2,3))
def initialise_circulating_supply_dict(
    start_date: np.datetime64,
    end_date: np.datetime64,
    circ_supply_zero: float,
    locked_fil_zero: float,
    burnt_fil_vec: Union[jnp.array, NDArray],
    vest_dict: Dict,
    mint_dict: Dict,
) -> Dict:
    # we assume days start at main net launch, in 2020-10-15
    start_day = datetime64_delta_to_days(start_date - NETWORK_START)
    end_day = datetime64_delta_to_days(end_date - NETWORK_START)
    len_sim = end_day - start_day

    network_locked_pledge = jnp.zeros(len_sim)
    network_locked_pledge = network_locked_pledge.at[0].set(locked_fil_zero / 2.0)
    network_locked_reward = jnp.zeros(len_sim)
    network_locked_reward = network_locked_reward.at[0].set(locked_fil_zero / 2.0)
    network_locked = jnp.zeros(len_sim)
    network_locked = network_locked.at[0].set(locked_fil_zero)
    circ_supply = jnp.zeros(len_sim)
    circ_supply = circ_supply.at[0].set(circ_supply_zero)

    network_gas_burn = jnp.zeros(len_sim)
    network_gas_burn = network_gas_burn.at[:len(burnt_fil_vec)].set(burnt_fil_vec)

    cs_dict = {
        "days": np.arange(start_day, end_day),
        "circ_supply": circ_supply,
        "network_gas_burn": network_gas_burn,
        "day_locked_pledge": jnp.zeros(len_sim),
        "day_renewed_pledge": jnp.zeros(len_sim),
        "network_locked_pledge": network_locked_pledge,
        "network_locked": network_locked,
        "network_locked_reward": network_locked_reward,
        "disbursed_reserve": jnp.ones(len_sim)
        * (17066618961773411890063046 * 10**-18),
    }
    
    # cs_dict = inner_join(cs_dict, vest_dict, key='days', coerce=False)
    # cs_dict = inner_join(cs_dict, mint_dict, key='days', coerce=False)
    # return cs_dict
    vest_dict.pop('days')
    mint_dict.pop('days')
    # TODO: you can try to do some date alignment checking here
    # it is coded such that the dates should already be aligned, however
    return {**cs_dict, **vest_dict, **mint_dict}
