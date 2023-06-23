from typing import Union, Dict, Tuple
import datetime
from functools import partial

import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
import jax

import mechafil_jax.power as power
import mechafil_jax.vesting as vesting
import mechafil_jax.minting as minting
import mechafil_jax.supply as supply

# @partial(jax.jit, static_argnums=(4,5,6,7,8))
def run_sim(
    rb_onboard_power: jnp.array,
    renewal_rate: jnp.array,
    fil_plus_rate: jnp.array,
    lock_target: float,
    
    start_date: datetime.date,
    current_date: datetime.date,
    forecast_length: int,
    duration: int,
    data: Tuple,
):
    end_date = current_date + datetime.timedelta(days=forecast_length)

    # extract data
    # TODO: move into its own function to clean up
    rb_power_zero = data["rb_power_zero"]
    qa_power_zero = data["qa_power_zero"]
    historical_raw_power_eib = data["historical_raw_power_eib"]
    historical_qa_power_eib = data["historical_qa_power_eib"]
    historical_onboarded_qa_power_pib = data["historical_onboarded_qa_power_pib"]
    historical_renewed_qa_power_pib = data["historical_renewed_qa_power_pib"]
    
    rb_known_scheduled_expire_vec = data["rb_known_scheduled_expire_vec"]
    qa_known_scheduled_expire_vec = data["qa_known_scheduled_expire_vec"]
    known_scheduled_pledge_release_full_vec = data["known_scheduled_pledge_release_full_vec"]
    start_vested_amt = data["start_vested_amt"]
    zero_cum_capped_power_eib = data["zero_cum_capped_power_eib"]
    init_baseline_eib = data["init_baseline_eib"]
    circ_supply_zero = data["circ_supply_zero"]
    locked_fil_zero = data["locked_fil_zero"]
    daily_burnt_fil = data["daily_burnt_fil"]
    burnt_fil_vec = data["burnt_fil_vec"]
    historical_renewal_rate = data["historical_renewal_rate"]

    rb_power_forecast, qa_power_forecast = power.forecast_power_stats(
        rb_power_zero, qa_power_zero, 
        rb_onboard_power, rb_known_scheduled_expire_vec, qa_known_scheduled_expire_vec,
        renewal_rate, fil_plus_rate, duration, forecast_length
    )
    # TODO: move the code block below into its own function
    #################################################
    # need to concatenate historical power (from start_date to current_date-1) to this
    rb_total_power_eib = jnp.concatenate((historical_raw_power_eib, rb_power_forecast["total_power"] / 1024.0))
    qa_total_power_eib = jnp.concatenate((historical_qa_power_eib, qa_power_forecast["total_power"] / 1024.0))
    # rb_day_onboarded_power_pib = rb_power_forecast["onboarded_power"]
    # rb_day_renewed_power_pib = rb_power_forecast["renewed_power"]
    qa_day_onboarded_power_pib = jnp.concatenate([historical_onboarded_qa_power_pib, qa_power_forecast["onboarded_power"]])
    qa_day_renewed_power_pib = jnp.concatenate([historical_renewed_qa_power_pib, qa_power_forecast["renewed_power"]])

    #################################################

    vesting_forecast = vesting.compute_vesting_trajectory(
        np.datetime64(start_date), 
        np.datetime64(end_date), 
        start_vested_amt
    )

    minting_forecast = minting.compute_minting_trajectory_df(
        np.datetime64(start_date),
        np.datetime64(end_date),
        rb_total_power_eib,
        qa_total_power_eib,
        qa_day_onboarded_power_pib,
        qa_day_renewed_power_pib,
        zero_cum_capped_power_eib,
        init_baseline_eib
    )

    full_renewal_rate_vec = np.concatenate(
        [historical_renewal_rate, renewal_rate]
    )
    supply_forecast = supply.forecast_circulating_supply(
        np.datetime64(start_date),
        np.datetime64(current_date),
        np.datetime64(end_date),
        circ_supply_zero,
        locked_fil_zero,
        daily_burnt_fil,
        duration,
        full_renewal_rate_vec,
        burnt_fil_vec,
        vesting_forecast,
        minting_forecast,
        known_scheduled_pledge_release_full_vec,
        lock_target=lock_target,
    )

    # collate results
    results = {
        "rb_total_power_eib": rb_total_power_eib,
        "qa_total_power_eib": qa_total_power_eib,
        # "rb_day_onboarded_power_pib": rb_day_onboarded_power_pib,
        # "rb_day_renewed_power_pib": rb_day_renewed_power_pib,
        "qa_day_onboarded_power_pib": qa_day_onboarded_power_pib,
        "qa_day_renewed_power_pib": qa_day_renewed_power_pib,
        **vesting_forecast,
        **minting_forecast,
        **supply_forecast,
    }
    return results