import unittest
from datetime import date, timedelta

from jax import config
config.update("jax_enable_x64", True)

import pystarboard.data as data
import mechafil.data as mecha_data  # remove this and associated code once we remove this from mechafil

import mechafil.power as np_power
import mechafil.minting as np_minting
import mechafil_jax.minting as jax_minting
import mechafil_jax.constants as C

import numpy as np
from tqdm.auto import tqdm

class TestMinting(unittest.TestCase):
    def test_forecast_minting_stats(self):
        # setup data access
        # TODO: better way to do this?
        data.setup_spacescope('/Users/kiran/code/filecoin-mecha-twin/kiran_spacescope_auth.json')
        mecha_data.setup_spacescope('/Users/kiran/code/filecoin-mecha-twin/kiran_spacescope_auth.json')

        forecast_length = 360*2
        start_date = date(2021, 3, 16)
        current_date = date.today() - timedelta(days=2)
        end_date = current_date + timedelta(days=forecast_length)

        # Get sector scheduled expirations
        res = data.get_sector_expiration_stats(start_date, current_date, end_date)
        rb_known_scheduled_expire_vec = res[0]
        qa_known_scheduled_expire_vec = res[1]
        known_scheduled_pledge_release_full_vec = res[2]
        # Get daily stats
        fil_stats_df = data.get_historical_network_stats(start_date, current_date, end_date)
        current_day_stats = fil_stats_df[fil_stats_df["date"] >= current_date].iloc[0]
        rb_power_zero = current_day_stats["total_raw_power_eib"] * 1024.0
        qa_power_zero = current_day_stats["total_qa_power_eib"] * 1024.0

        # sample values, doesn't matter for minting except consider sweeping RBP if you are paranoid
        rr = 0.3
        fpr = 0.3
        duration = 360
        rbp = 3

        mechafil_rb_df, mechafil_qa_df = np_power.forecast_power_stats(
            rb_power_zero,
            qa_power_zero,
            rbp,
            rb_known_scheduled_expire_vec,
            qa_known_scheduled_expire_vec,
            rr,
            fpr,
            duration,
            forecast_length,
            qap_method='basic'
        )

        # get data needed to run minting w/ JAX
        mechafil_rb_df["total_raw_power_eib"] = mechafil_rb_df["total_power"] / 1024.0
        mechafil_qa_df["total_qa_power_eib"] = mechafil_qa_df["total_power"] / 1024.0
        power_df = np_power.build_full_power_stats_df(
            fil_stats_df,
            mechafil_rb_df,
            mechafil_qa_df,
            start_date,
            current_date,
            end_date,
        )

        rb_total_power_eib = power_df["total_raw_power_eib"].values
        qa_total_power_eib = power_df["total_qa_power_eib"].values
        qa_day_onboarded_power_pib = power_df["day_onboarded_qa_power_pib"].values
        qa_day_renewed_power_pib = power_df["day_renewed_qa_power_pib"].values

        np_minting_df = np_minting.compute_minting_trajectory_df(
            start_date,
            end_date,
            rb_total_power_eib,
            qa_total_power_eib,
            qa_day_onboarded_power_pib,
            qa_day_renewed_power_pib,
            minting_base = 'RBP'
        )
        zero_cum_capped_power_eib = data.get_cum_capped_rb_power(start_date) / C.EXBI
        init_baseline_eib = data.get_storage_baseline_value(start_date) / C.EXBI
        baseline_function_EIB = jax_minting.compute_baseline_power_array(
            np.datetime64(start_date), 
            np.datetime64(end_date), 
            init_baseline_eib
        )

        jax_minting_dict = jax_minting.compute_minting_trajectory_df(
            np.datetime64(start_date),
            np.datetime64(end_date),
            rb_total_power_eib,
            qa_total_power_eib,
            qa_day_onboarded_power_pib,
            qa_day_renewed_power_pib,
            zero_cum_capped_power_eib,
            baseline_function_EIB
        )

        # check that the two minting trajectories are the same
        mechafil_keys = ['network_baseline', 'cum_capped_power', 'cum_simple_reward', 'cum_baseline_reward', 'cum_network_reward', 'day_network_reward']
        jax_keys = ['network_baseline_EIB', 'cum_capped_power_EIB', 'cum_simple_reward', 'cum_baseline_reward', 'cum_network_reward', 'day_network_reward']
        for k in zip(mechafil_keys, jax_keys):
            mechafil_key = k[0]
            jax_key = k[1]

            y_mechafil = np_minting_df[mechafil_key].values
            y_jax = jax_minting_dict[jax_key]

            if 'EIB' in jax_key:
                y_mechafil = y_mechafil / C.EIB
            is_close = np.allclose(y_mechafil, y_jax)
            self.assertTrue(is_close)
        
if __name__ == '__main__':
    unittest.main()