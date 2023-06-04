from typing import Callable, Union
import numbers

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
#  Utility functions
# --------------------------------------------------------------------------------------
def scalar_or_vector_to_vector(
    input_x: Union[jnp.ndarray, NDArray, float], expected_len: int, err_msg: str = None
) -> jnp.ndarray:
    if isinstance(input_x, numbers.Number):
        return jnp.ones(expected_len) * input_x
    elif isinstance(input_x, np.ndarray):
        return jnp.array(input_x)
    else:
        err_msg_out = (
            "vector input does not match expected length!"
            if err_msg is None
            else err_msg
        )
        assert len(input_x) == expected_len, err_msg_out
        return input_x

    
# --------------------------------------------------------------------------------------
#  QA Multiplier functions
# --------------------------------------------------------------------------------------
def compute_qa_factor(
    fil_plus_rate: Union[jnp.array, NDArray, float],
    fil_plus_m: float = 10.0,
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
def forecast_rb_daily_onboardings(
    rb_onboard_power: Union[jnp.array, NDArray, float], forecast_lenght: int
) -> jnp.array:
    rb_onboarded_power_vec = scalar_or_vector_to_vector(
        rb_onboard_power,
        forecast_lenght,
        err_msg="If rb_onboard_power is provided as a vector, it must be the same length as the forecast_length",
    )
    return rb_onboarded_power_vec


def forecast_qa_daily_onboardings(
    rb_onboard_power: Union[jnp.array, NDArray, float],
    fil_plus_rate: Union[jnp.array, NDArray, float],
    forecast_lenght: int,
    fil_plus_m: float = 10.0,
    duration_m: Callable = None,
    duration: int = None,
) -> jnp.array:
    # If duration_m is not provided, qa_factor = 1.0 + 9.0 * fil_plus_rate
    qa_factor = compute_qa_factor(fil_plus_rate, fil_plus_m, duration_m, duration)
    qa_onboard_power = qa_factor * rb_onboard_power
    qa_onboard_power_vec = scalar_or_vector_to_vector(
        qa_onboard_power,
        forecast_lenght,
        err_msg="If qa_onboard_power is provided as a vector, it must be the same length as the forecast_length",
    )
    return qa_onboard_power_vec

# --------------------------------------------------------------------------------------
#  Renewals
# --------------------------------------------------------------------------------------
def basic_scalar_renewed_power(day_sched_expire_power, renewal_rate):
    return day_sched_expire_power * renewal_rate

# --------------------------------------------------------------------------------------
#  Scheduled expirations
# --------------------------------------------------------------------------------------
def have_known_se_sector_info(arggs):
    known_scheduled_expire_vec, day_i = arggs
    return known_scheduled_expire_vec[day_i]

def dont_have_known_se_sector_info(arggs):
    return 0.0

def have_modeled_sector_expiration_info(arggs):
    day_onboard_vec, day_renewed_vec, day_i, duration = arggs
    return day_onboard_vec[day_i - duration] + day_renewed_vec[day_i - duration]

def dont_have_modeled_sector_expiration_info(arggs):
    return 0.0

@jax.jit
def compute_se_and_rr(carry, x):
    # NOTE: this function has a pretty big carry.
    # This discussion is relevant: https://github.com/google/jax/discussions/10233
    # Ultimately - the goal of this is not necessarily for speed, but for differentiability, so it may be OBE.
    day_rb_renewed_power_vec, rb_known_sched_expire, day_rb_onboarded_power, renewal_rate_vec, day_i, duration = carry
    
    # compute the components of SE power
    # known SE
    pred = day_i > len(rb_known_sched_expire) - 1
    known_day_se_power = lax.cond(
        pred, 
        dont_have_known_se_sector_info,
        have_known_se_sector_info, 
        (rb_known_sched_expire, day_i)
     )
    
    pred = day_i - duration >= 0
    model_day_se_power = lax.cond(
        pred,
        have_modeled_sector_expiration_info,
        dont_have_modeled_sector_expiration_info,
        (day_rb_onboarded_power, day_rb_renewed_power_vec, day_i, duration)
    )
    day_se_power = known_day_se_power + model_day_se_power
    
    # compute new renewed power
    day_i_rb_renewed_power = day_se_power * renewal_rate_vec[day_i]
    # need to set arrays in this manner -> when using jit, this is guaranteed to become inplace
    day_rb_renewed_power_vec = day_rb_renewed_power_vec.at[day_i].set(day_i_rb_renewed_power)
    
    return (day_rb_renewed_power_vec, rb_known_sched_expire, day_rb_onboarded_power, renewal_rate_vec, day_i+1, duration), day_se_power

def forecast_power_stats(
    rb_power_zero: float,
    qa_power_zero: float,
    rb_onboard_power: Union[jnp.array, NDArray, float],
    rb_known_scheduled_expire_vec: Union[jnp.array, NDArray],
    qa_known_scheduled_expire_vec: Union[jnp.array, NDArray],
    renewal_rate: Union[jnp.array, NDArray, float],
    fil_plus_rate: Union[jnp.array, NDArray, float],
    duration: int,
    forecast_length: int,
    fil_plus_m: float = 10,
    **kwargs  # a noop in this port, but here for backwards compatibility
):
    # force duration to be an integer
    duration = int(duration)
    
    renewal_rate_vec = scalar_or_vector_to_vector(
        renewal_rate,
        forecast_length,
        err_msg="If renewal_rate is provided as a vector, it must be the same length as the forecast_length",
    )

    day_rb_onboarded_power = forecast_rb_daily_onboardings(
        rb_onboard_power, forecast_length
    )
    total_rb_onboarded_power = day_rb_onboarded_power.cumsum()

    day_qa_onboarded_power = forecast_qa_daily_onboardings(
        rb_onboard_power,
        fil_plus_rate,
        forecast_length,
        fil_plus_m,
        duration_m=None,
        duration=duration,
    )
    total_qa_onboarded_power = day_qa_onboarded_power.cumsum()

    # compute SE & RR for both RBP & QAP
    day_rb_renewed_power_vec = jnp.zeros(len(day_rb_onboarded_power))
    day_qa_renewed_power_vec = jnp.zeros(len(day_qa_onboarded_power))

    init_in = (day_rb_renewed_power_vec, rb_known_scheduled_expire_vec, day_rb_onboarded_power, renewal_rate_vec, 0, duration)
    ret, day_rb_scheduled_expire_power = lax.scan(compute_se_and_rr, init_in, None, length=forecast_length)
    day_rb_renewed_power = ret[0]

    init_in = (day_qa_renewed_power_vec, qa_known_scheduled_expire_vec, day_qa_onboarded_power, renewal_rate_vec, 0, duration)
    ret, day_qa_scheduled_expire_power = lax.scan(compute_se_and_rr, init_in, None, length=forecast_length)
    day_qa_renewed_power = ret[0]

    # Compute total scheduled expirations and renewals
    total_rb_scheduled_expire_power = day_rb_scheduled_expire_power.cumsum()
    total_rb_renewed_power = day_rb_renewed_power.cumsum()
    total_qa_scheduled_expire_power = day_qa_scheduled_expire_power.cumsum()
    total_qa_renewed_power = day_qa_renewed_power.cumsum()

    # Total RB power
    rb_power_zero_vec = np.ones(forecast_length) * rb_power_zero
    rb_total_power = (
        rb_power_zero_vec
        + total_rb_onboarded_power
        - total_rb_scheduled_expire_power
        + total_rb_renewed_power
    )
    # Total QA power
    qa_power_zero_vec = np.ones(forecast_length) * qa_power_zero
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