from typing import Dict, Union
import datetime
from functools import partial

# remove comments if you want to run in 64 bit
# from jax import config
# config.update("jax_enable_x64", True)

import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
import jax

import mechafil_jax.power as power
import mechafil_jax.vesting as vesting
import mechafil_jax.minting as minting
import mechafil_jax.supply as supply
import mechafil_jax.constants as C

@partial(jax.jit, static_argnums=(4,5,6,7))
def run_sim(
    rb_onboard_power: jnp.array,
    renewal_rate: jnp.array,
    fil_plus_rate: jnp.array,
    lock_target: Union[float, jnp.array],
    start_date: datetime.date,
    current_date: datetime.date,
    forecast_length: int,
    duration: int,
    data: Dict,
    baseline_function_EIB: jnp.array = None,
    fil_plus_m: Union[float, jnp.array] = 10.0,
    qa_renew_relative_multiplier_vec: jnp.array = 1.0,
    gamma: Union[float, jnp.array] = 1.0,
    gamma_weight_type: Union[int, jnp.array] = 0,
):
    """
    Run a simulation of the Filecoin network.

    Parameters:
    -----------
    rb_onboard_power: jnp.array
        The raw power onboarded to the network, in EiB.
    renewal_rate: jnp.array
        The renewal rate of the network, in EiB/day.
    fil_plus_rate: jnp.array
        The FIL+ rate of the network, in EiB/day.
    lock_target: Union[float, jnp.array]
        The target lock ratio of the network. If a float, then the lock target is constant across the simulation. If it is
        a jnp.array, then it must be of length `forecast_length` and the lock target can be time-varying.  This applies to the
        forecasts, but not historical data.
    start_date: datetime.date
        The start date of the simulation.
    current_date: datetime.date
        The current date of the simulation.
    forecast_length: int
        The length of the forecast, in days.
    duration: int
        Average sector duration.
    data: Dict
        A dictionary of historical data. See `mechafil_jax.data` for more details.
    fil_plus_m: Union[float, jnp.array]
        The FIL+ multiplier. If a float, then the multiplier is constant across the simulation. If it is a vector, then it must
        be of length `forecast_length` and the multiplier can be time-varying. This applies to the forecasts, but not historical data.
    qa_renew_relative_multiplier_vec: jnp.array
        The QA renewal relative multiplier. If a float, then the multiplier is constant across the simulation. If it is a vector, then it must
        be of length `forecast_length` and the multiplier can be time-varying. This applies to the forecasts, but not historical data.
    gamma: Union[float, jnp.array]
        The gamma parameter. If a float, then the multiplier is constant across the simulation. If it is a vector, then it must
        be of length `forecast_length` and the multiplier can be time-varying. This applies to the forecasts, but not historical data.
    gamma_weight_type: Union[int, jnp.array]
        The gamma weight type. If a float, then the multiplier is constant across the simulation. If it is a vector, then it must
        be of length `forecast_length` and the multiplier can be time-varying. This applies to the forecasts, but not historical data.
    """

    end_date = current_date + datetime.timedelta(days=forecast_length)

    # extract data
    rb_power_zero = data["rb_power_zero"]
    qa_power_zero = data["qa_power_zero"]
    historical_raw_power_eib = data["historical_raw_power_eib"]
    historical_qa_power_eib = data["historical_qa_power_eib"]
    historical_onboarded_rb_power_pib = data["historical_onboarded_rb_power_pib"]
    historical_onboarded_qa_power_pib = data["historical_onboarded_qa_power_pib"]
    historical_renewed_qa_power_pib = data["historical_renewed_qa_power_pib"]
    historical_renewed_rb_power_pib = data["historical_renewed_rb_power_pib"]
    
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
        renewal_rate, fil_plus_rate, duration, forecast_length, 
        fil_plus_m=fil_plus_m, qa_renew_relative_multiplier_vec=qa_renew_relative_multiplier_vec
    )
    # TODO: move the code block below into its own function
    #################################################
    # need to concatenate historical power (from start_date to current_date-1) to this
    rb_total_power_eib = jnp.concatenate((historical_raw_power_eib, (rb_power_forecast["total_power"][:-1] / 1024.0)))
    qa_total_power_eib = jnp.concatenate((historical_qa_power_eib, (qa_power_forecast["total_power"][:-1] / 1024.0)))
    rb_day_onboarded_power_pib = jnp.concatenate((historical_onboarded_rb_power_pib, (rb_power_forecast["onboarded_power"][:-1])))
    rb_day_renewed_power_pib = jnp.concatenate((historical_renewed_rb_power_pib, (rb_power_forecast["renewed_power"][:-1])))
    qa_day_onboarded_power_pib = jnp.concatenate([historical_onboarded_qa_power_pib, qa_power_forecast["onboarded_power"][:-1]])
    qa_day_renewed_power_pib = jnp.concatenate([historical_renewed_qa_power_pib, qa_power_forecast["renewed_power"][:-1]])

    #################################################

    vesting_forecast = vesting.compute_vesting_trajectory(
        np.datetime64(start_date), 
        np.datetime64(end_date), 
        start_vested_amt
    )

    if baseline_function_EIB is None:
        baseline_function_EIB = minting.compute_baseline_power_array(
            np.datetime64(start_date), 
            np.datetime64(end_date), 
            init_baseline_eib
        )

    minting_forecast = minting.compute_minting_trajectory_df(
        np.datetime64(start_date),
        np.datetime64(end_date),
        rb_total_power_eib,
        qa_total_power_eib,
        qa_day_onboarded_power_pib,
        qa_day_renewed_power_pib,
        zero_cum_capped_power_eib,
        baseline_function_EIB
    )

    full_renewal_rate_vec = jnp.concatenate(
        [historical_renewal_rate, renewal_rate]
    )
    historical_target_lock = jnp.ones(len(historical_renewal_rate)) * 0.3
    # will throw an error if lock_target is a vector of length != forecast_length, but potentially
    # the error will be cryptic, so consider improving this
    lock_target = jnp.ones(forecast_length) * lock_target
    full_lock_target_vec = jnp.concatenate(
        [historical_target_lock, lock_target]
    )
    historical_gamma = jnp.ones(len(historical_renewal_rate)) * 1.0
    gamma = jnp.ones(forecast_length) * gamma
    historical_gamma_weight_type = jnp.zeros(len(historical_renewal_rate))
    gamma_weight_type = jnp.ones(forecast_length) * gamma_weight_type
    full_gamma_vec = jnp.concatenate(
        [historical_gamma, gamma]
    )
    full_gamma_weight_type_vec = jnp.concatenate(
        [historical_gamma_weight_type, gamma_weight_type]
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
        lock_target=full_lock_target_vec,
        gamma=full_gamma_vec,
        gamma_weight_type=full_gamma_weight_type_vec,
    )

    # collate results
    results = {
        "rb_total_power_eib": rb_total_power_eib,
        "qa_total_power_eib": qa_total_power_eib,
        "rb_day_onboarded_power_pib": rb_day_onboarded_power_pib,
        "rb_day_renewed_power_pib": rb_day_renewed_power_pib,
        "qa_day_onboarded_power_pib": qa_day_onboarded_power_pib,
        "qa_day_renewed_power_pib": qa_day_renewed_power_pib,
        "full_renewal_rate": full_renewal_rate_vec,
        **vesting_forecast,
        **minting_forecast,
        **supply_forecast,
    }

    ############################
    # add generated quantities
    dppq = C.PIB_PER_SECTOR * (results['day_locked_pledge']-results['day_renewed_pledge'])/results['day_onboarded_power_QAP_PIB']
    dppq = dppq.at[0].set(dppq[1]) # div/by/zero fix for ROI
    results['day_pledge_per_QAP'] = dppq
    results['day_rewards_per_sector'] = C.EIB_PER_SECTOR * results['day_network_reward'] / results['network_QAP_EIB']
    days_1y = 365
    rps = jnp.convolve(results['day_rewards_per_sector'], jnp.ones(days_1y), mode='full')
    results['1y_return_per_sector'] = rps[days_1y-1:1-days_1y]
    results['1y_sector_roi'] = results['1y_return_per_sector'] / results['day_pledge_per_QAP'][:1-days_1y]
    ############################

    return results
