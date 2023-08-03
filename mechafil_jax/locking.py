from typing import Union

import numpy as np
from numpy.typing import NDArray

import jax
import jax.lax as lax
import jax.numpy as jnp

from mechafil_jax import constants

"""
A port of the locking module in mechafil to JAX.
"""

# Block reward collateral
@jax.jit
def compute_day_locked_rewards(day_network_reward: float) -> float:
    return 0.75 * day_network_reward

@jax.jit
def compute_day_reward_release(prev_network_locked_reward: float) -> float:
    return prev_network_locked_reward / 180.0


# Initial pledge collateral
@jax.jit
def compute_day_delta_pledge(
    day_network_reward: float,
    prev_circ_supply: float,
    day_onboarded_qa_power_pib: float,
    day_renewed_qa_power_pib: float,
    total_qa_power_eib: float,
    baseline_power_eib: float,
    renewal_rate: float,
    scheduled_pledge_release: float,
    lock_target: float,
    gamma: float,
    gamma_weight_type: int, 
) -> float:
    # convert powers to PIB
    total_qa_power_pib = total_qa_power_eib * 1024.0
    baseline_power_pib = baseline_power_eib * 1024.0

    onboards_delta = compute_new_pledge_for_added_power(
        day_network_reward,
        prev_circ_supply,
        day_onboarded_qa_power_pib,
        total_qa_power_pib,
        baseline_power_pib,
        lock_target,
        gamma,
        gamma_weight_type, 
    )
    renews_delta = compute_renewals_delta_pledge(
        day_network_reward,
        prev_circ_supply,
        day_renewed_qa_power_pib,
        total_qa_power_pib,
        baseline_power_pib,
        renewal_rate,
        scheduled_pledge_release,
        lock_target,
        gamma, 
        gamma_weight_type, 
    )
    return onboards_delta + renews_delta

@jax.jit
def compute_day_locked_pledge(
    day_network_reward: float,
    prev_circ_supply: float,
    day_onboarded_qa_power_pib: float,
    day_renewed_qa_power_pib: float,
    total_qa_power_eib: float,
    baseline_power_eib: float,
    renewal_rate: float,
    scheduled_pledge_release: float,
    lock_target: float,
    gamma: float,
    gamma_weight_type: int, 
) -> float:
    # convert powers to PIB
    total_qa_power_pib = total_qa_power_eib * 1024.0
    baseline_power_pib = baseline_power_eib * 1024.0

    print('jax', day_network_reward, prev_circ_supply, day_onboarded_qa_power_pib, total_qa_power_pib, baseline_power_pib, lock_target)
    # Total locked from new onboards
    onboards_locked = compute_new_pledge_for_added_power(
        day_network_reward,
        prev_circ_supply,
        day_onboarded_qa_power_pib,
        total_qa_power_pib,
        baseline_power_pib,
        lock_target,
        gamma, 
        gamma_weight_type, 
    )
    # print('jax', onboards_locked)
    # Total locked from renewals
    original_pledge = renewal_rate * scheduled_pledge_release
    new_pledge = compute_new_pledge_for_added_power(
        day_network_reward,
        prev_circ_supply,
        day_renewed_qa_power_pib,
        total_qa_power_pib,
        baseline_power_pib,
        lock_target,
        gamma, 
        gamma_weight_type, 
    )
    renews_locked = jnp.maximum(original_pledge, new_pledge)
    # Total locked pledge
    locked = onboards_locked + renews_locked
    # print('jax', locked, renews_locked)
    return locked, renews_locked

@jax.jit
def compute_renewals_delta_pledge(
    day_network_reward: float,
    prev_circ_supply: float,
    day_renewed_qa_power: float,
    total_qa_power: float,
    baseline_power: float,
    renewal_rate: float,
    scheduled_pledge_release: float,
    lock_target: float,
    gamma: float, 
    gamma_weight_type: int, 
) -> float:
    # Delta from sectors expiring
    expire_delta = -(1 - renewal_rate) * scheduled_pledge_release
    # Delta from sector renewing
    original_pledge = renewal_rate * scheduled_pledge_release
    new_pledge = compute_new_pledge_for_added_power(
        day_network_reward,
        prev_circ_supply,
        day_renewed_qa_power,
        total_qa_power,
        baseline_power,
        lock_target,
        gamma, 
        gamma_weight_type, 
    )
    renew_delta = jnp.maximum(0.0, new_pledge - original_pledge)
    # Delta for all scheduled sectors
    delta = expire_delta + renew_delta
    return delta

@jax.jit
def compute_new_pledge_for_added_power(
    day_network_reward: float,
    prev_circ_supply: float,
    day_added_qa_power_pib: float,
    total_qa_power_pib: float,
    baseline_power_pib: float,
    lock_target: float,
    gamma: float,
    gamma_weight_type: int,  
) -> float:
    
    # storage pledge collateral component
    storage_pledge = 20.0 * day_network_reward * (day_added_qa_power_pib / total_qa_power_pib)

    # Compute Simple and Baseline Pledge 
    simple_consensus_pledge = lock_target * prev_circ_supply * (day_added_qa_power_pib/total_qa_power_pib)
    simple_consensus_pledge = jnp.maximum(simple_consensus_pledge, 0)

    normalized_qap_growth = day_added_qa_power_pib / jnp.maximum(total_qa_power_pib, baseline_power_pib) # Fraction for Consensus Pledge Component
    baseline_consensus_pledge = lock_target * prev_circ_supply * normalized_qap_growth
    baseline_consensus_pledge = jnp.maximum(baseline_consensus_pledge, 0)

    # Compute Simple and Baseline Pledge Contributions to Consensus Pledge for Arithmetic, Geometric, and Harmonic Weightings 
   
    #ngamma parameter for weighting 
    weighting_method = gamma_weight_type #weighting schema (must be arithmetic=0, geometric=1, or harmonic=2)

    consensus_pledge = weighted_consensus(weighting_method, gamma, simple_consensus_pledge, baseline_consensus_pledge)
    # total added pledge
    added_pledge = storage_pledge + consensus_pledge

    pledge_cap = day_added_qa_power_pib * (constants.PIB / constants.GIB)  # The # of bytes in a GiB (Gibibyte)
    return jnp.minimum(pledge_cap, added_pledge)

@jax.jit
def arithmetic_fn(w, simple_consensus_pledge, baseline_consensus_pledge):
    return (1 - w) * simple_consensus_pledge + w * baseline_consensus_pledge

@jax.jit
def geometric_fn(w, simple_consensus_pledge, baseline_consensus_pledge):
    return simple_consensus_pledge ** (1 - w) * baseline_consensus_pledge ** w

@jax.jit
def harmonic_fn(w, simple_consensus_pledge, baseline_consensus_pledge):
    return 1 / ((1 - w) * simple_consensus_pledge**-1 + w * baseline_consensus_pledge**-1)

@jax.jit
def raise_exception(w, simple_consensus_pledge, baseline_consensus_pledge):
    raise ValueError('must be of proper weighting type')

@jax.jit
def weighted_consensus(weighting_method, w, simple_consensus_pledge, baseline_consensus_pledge):
    return lax.cond(
        weighting_method == 0,
        lambda _: arithmetic_fn(w, simple_consensus_pledge, baseline_consensus_pledge),
        lambda _: lax.cond(
            weighting_method == 1, 
            lambda _: geometric_fn(w, simple_consensus_pledge, baseline_consensus_pledge),
            lambda _: lax.cond(
                weighting_method == 2, 
                lambda _: harmonic_fn(w, simple_consensus_pledge, baseline_consensus_pledge),
                lambda _: jnp.zeros_like(simple_consensus_pledge) * jnp.inf,  # Default branch, raise an error or handle it accordingly
                operand=None
            ),
            operand=None
        ),
        operand=None
    )
@jax.jit
def have_known_pledge_info(arggs):
    known_scheduled_pledge_release_vec, day_i = arggs
    return known_scheduled_pledge_release_vec[day_i]

@jax.jit
def dont_have_known_pledge_info(arggs):
    return 0.0

@jax.jit
def have_modeled_pledge_info(arggs):
    day_pledge_locked_vec, day_i, duration = arggs
    return day_pledge_locked_vec[day_i - duration]

@jax.jit
def dont_have_modeled_pledge_info(arggs):
    return 0.0

@jax.jit
def get_day_schedule_pledge_release(
    day_i,
    current_day_i,
    day_pledge_locked_vec: Union[jnp.array, NDArray],
    known_scheduled_pledge_release_vec: np.array,
    duration: int,
) -> float:
    
    # scheduled releases coming from known active sectors
    pred = day_i > len(known_scheduled_pledge_release_vec) - 1
    known_day_release = lax.cond(
        pred, 
        dont_have_known_pledge_info,
        have_known_pledge_info, 
        (known_scheduled_pledge_release_vec, day_i)
     )
    
    pred = day_i - duration >= current_day_i
    model_day_release = lax.cond(
        pred,
        have_modeled_pledge_info,
        dont_have_modeled_pledge_info,
        (day_pledge_locked_vec, day_i, duration)
    )
    
    # Total pledge schedule releases
    day_pledge_schedules_release = known_day_release + model_day_release
    return day_pledge_schedules_release
