from typing import Callable, Union
import numbers
from functools import partial

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from numpy.typing import NDArray

"""
A port of the power module in mechafil to JAX.
NOTE:
    - This module does not support tunable QAP mode, only basic.
"""
    
# --------------------------------------------------------------------------------------
#  QA Multiplier functions
# --------------------------------------------------------------------------------------
# NOTE: I tried adding in jax.jit here, but it seemed to make it slower?
#  we need a more systematic way to benchmark all of this
def compute_qa_factor(
    fil_plus_rate: Union[jnp.array, NDArray],
    fil_plus_m: Union[jnp.array, NDArray],
    duration_m: Callable = None,
    duration: int = None,
) -> Union[jnp.array, NDArray, float]:
    fil_plus_multipler = 1.0 + (fil_plus_m - 1) * fil_plus_rate
    if duration_m is None:
        return fil_plus_multipler
    else:
        return duration_m(duration) * fil_plus_multipler
    
# --------------------------------------------------------------------------------------
#  Onboardings
# --------------------------------------------------------------------------------------
@jax.jit
def forecast_qa_daily_onboardings(
    rb_onboard_power: Union[jnp.array, NDArray],
    fil_plus_rate: Union[jnp.array, NDArray],
    fil_plus_m: Union[jnp.array, NDArray],
    duration_m: Callable = None,
    duration: int = None,
) -> jnp.array:
    # If duration_m is not provided, qa_factor = 1.0 + 9.0 * fil_plus_rate
    qa_factor = compute_qa_factor(fil_plus_rate, fil_plus_m, duration_m, duration)
    qa_onboard_power_vec = qa_factor * rb_onboard_power
    return qa_onboard_power_vec

# --------------------------------------------------------------------------------------
#  Renewals
# --------------------------------------------------------------------------------------
@jax.jit
def basic_scalar_renewed_power(day_sched_expire_power, renewal_rate):
    return day_sched_expire_power * renewal_rate

# --------------------------------------------------------------------------------------
#  Scheduled expirations
# --------------------------------------------------------------------------------------
@jax.jit
def have_known_se_sector_info(arggs):
    known_scheduled_expire_vec, day_i = arggs
    return known_scheduled_expire_vec[day_i]

@jax.jit
def dont_have_known_se_sector_info(arggs):
    return 0.0

@jax.jit
def have_modeled_sector_expiration_info(arggs):
    day_onboard_vec, day_renewed_vec, day_i, duration = arggs
    return day_onboard_vec[day_i - duration] + day_renewed_vec[day_i - duration]

@jax.jit
def dont_have_modeled_sector_expiration_info(arggs):
    return 0.0

@jax.jit
def compute_se_and_rr(carry, x):
    # NOTE: this function has a pretty big carry, revisit to try to optimize
    # This discussion is relevant: https://github.com/google/jax/discussions/10233
    day_renewed_power_vec, known_sched_expire, day_onboarded_power, renewal_rate_vec, day_i, duration, filp_m_vec = carry
    
    # compute the components of SE power
    # known SE
    pred = day_i > len(known_sched_expire) - 1
    known_day_se_power = lax.cond(
        pred, 
        dont_have_known_se_sector_info,
        have_known_se_sector_info, 
        (known_sched_expire, day_i)
     )
    
    pred = day_i - duration >= 0
    model_day_se_power = lax.cond(
        pred,
        have_modeled_sector_expiration_info,
        dont_have_modeled_sector_expiration_info,
        (day_onboarded_power, day_renewed_power_vec, day_i, duration)
    )
    day_se_power = known_day_se_power + model_day_se_power
    
    # compute new renewed power
    day_i_renewed_power = day_se_power * renewal_rate_vec[day_i] * filp_m_vec[day_i]
    # need to set arrays in this manner -> when using jit, this is guaranteed to become inplace
    day_renewed_power_vec = day_renewed_power_vec.at[day_i].set(day_i_renewed_power)
    
    return (day_renewed_power_vec, known_sched_expire, day_onboarded_power, renewal_rate_vec, day_i+1, duration, filp_m_vec), day_se_power


@partial(jax.jit, static_argnums=(7,8,))
def forecast_power_stats(
    rb_power_zero: float,
    qa_power_zero: float,
    day_rb_onboarded_power: Union[jnp.array, NDArray],
    rb_known_scheduled_expire_vec: Union[jnp.array, NDArray],
    qa_known_scheduled_expire_vec: Union[jnp.array, NDArray],
    renewal_rate_vec: Union[jnp.array, NDArray],
    fil_plus_rate_vec: Union[jnp.array, NDArray],
    duration: int,
    forecast_length: int,
    fil_plus_m: Union[float, jnp.array, NDArray] = 10.0,
    qa_renew_relative_multiplier_vec: Union[jnp.array, NDArray] = 1.0,
):  
    """
    Forecast the power stats for a given day.

    Args:
        rb_power_zero: The initial power of the RB sector.
        qa_power_zero: The initial power of the QA sector.
        day_rb_onboarded_power: The daily onboarded power of the RB sector.
        rb_known_scheduled_expire_vec: The known scheduled expire power of the RB sector.
        qa_known_scheduled_expire_vec: The known scheduled expire power of the QA sector.
        renewal_rate_vec: The renewal rate of the RB sector.
        fil_plus_rate_vec: The FIL+ rate of the RB sector.
        duration: The duration of the RB sector.
        forecast_length: The length of the forecast.
        fil_plus_m: The FIL+ multiplier, can be a vector or scalar.
        qa_renew_relative_multiplier_vec: A vector indicating the relative multiplier applied to QA renewals.
            The default is 1.0, which means that QA power, when renewed, will renew w/ the same QA level.
            If the value is < 1.0, it will renew with less power.  This is useful for simulating scenarios where
            we explore changes to the QA multiplier.
    """
    # convert the FIL+ input into a vector
    # if already a vector, still works
    fil_plus_m_vec = jnp.ones(forecast_length) * fil_plus_m
    # NOTE: this should be moved to the top-level function I think
    assert len(fil_plus_m_vec) == forecast_length, "fil_plus_m must be of length forecast_length"

    total_rb_onboarded_power = day_rb_onboarded_power.cumsum()

    day_qa_onboarded_power = forecast_qa_daily_onboardings(
        day_rb_onboarded_power,
        fil_plus_rate_vec,
        fil_plus_m_vec,
        duration_m=None,
        duration=duration,
    )
    total_qa_onboarded_power = day_qa_onboarded_power.cumsum()

    # compute SE & RR for both RBP & QAP
    day_rb_renewed_power_vec = jnp.zeros(forecast_length)
    day_qa_renewed_power_vec = jnp.zeros(forecast_length)

    rb_relative_filp_m_vec = jnp.ones(forecast_length)
    init_in = (day_rb_renewed_power_vec, rb_known_scheduled_expire_vec, day_rb_onboarded_power, renewal_rate_vec, 0, duration, rb_relative_filp_m_vec)
    ret, day_rb_scheduled_expire_power = lax.scan(compute_se_and_rr, init_in, None, length=forecast_length)
    day_rb_renewed_power = ret[0]

    qa_relative_filp_m_vec = jnp.ones(forecast_length) * qa_renew_relative_multiplier_vec  # works if this is a scalar or vector
    init_in = (day_qa_renewed_power_vec, qa_known_scheduled_expire_vec, day_qa_onboarded_power, renewal_rate_vec, 0, duration, qa_relative_filp_m_vec)
    ret, day_qa_scheduled_expire_power = lax.scan(compute_se_and_rr, init_in, None, length=forecast_length)
    day_qa_renewed_power = ret[0]

    # Compute total scheduled expirations and renewals
    total_rb_scheduled_expire_power = day_rb_scheduled_expire_power.cumsum()
    total_rb_renewed_power = day_rb_renewed_power.cumsum()
    total_qa_scheduled_expire_power = day_qa_scheduled_expire_power.cumsum()
    total_qa_renewed_power = day_qa_renewed_power.cumsum()

    # Total RB power
    rb_power_zero_vec = jnp.ones(forecast_length) * rb_power_zero
    rb_total_power = (
        rb_power_zero_vec
        + total_rb_onboarded_power
        - total_rb_scheduled_expire_power
        + total_rb_renewed_power
    )
    # Total QA power
    qa_power_zero_vec = jnp.ones(forecast_length) * qa_power_zero
    qa_total_power = (
        qa_power_zero_vec
        + total_qa_onboarded_power
        - total_qa_scheduled_expire_power
        + total_qa_renewed_power
    )

    # put everything into a dictionary and return
    rb_dict = {
        'forecasting_step': jnp.arange(forecast_length),
        'onboarded_power': day_rb_onboarded_power,
        'cum_onboarded_power': total_rb_onboarded_power,
        'expire_scheduled_power': day_rb_scheduled_expire_power,
        'cum_expire_scheduled_power': total_rb_scheduled_expire_power,
        'renewed_power': day_rb_renewed_power,
        'cum_renewed_power': total_rb_renewed_power,
        'total_power': rb_total_power,
    }
    qa_dict = {
        'forecasting_step': jnp.arange(forecast_length),
        'onboarded_power': day_qa_onboarded_power,
        'cum_onboarded_power': total_qa_onboarded_power,
        'expire_scheduled_power': day_qa_scheduled_expire_power,
        'cum_expire_scheduled_power': total_qa_scheduled_expire_power,
        'renewed_power': day_qa_renewed_power,
        'cum_renewed_power': total_qa_renewed_power,
        'total_power': qa_total_power,
    }
    return rb_dict, qa_dict