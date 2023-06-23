import unittest
from datetime import date, timedelta

import mechafil.power as np_power
import mechafil.data as data
import mechafil.minting as np_minting
import mechafil.vesting as np_vesting

import mechafil.supply as np_supply
import mechafil_jax.supply as jax_supply
import mechafil_jax.constants as C

import numpy as np
import jax.numpy as jnp
import tqdm.auto as tqdm

class TestSupply(unittest.TestCase):
    def test_forecast_supply(self):
        # setup data access
        # TODO: better way to do this?
        data.setup_spacescope('/Users/kiran/code/filecoin-mecha-twin/kiran_spacescope_auth.json')

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

        # consider sweeping these to build confidence in JAX implementation
        rr = 0.3
        fpr = 0.3
        duration = 360
        rbp = 3
        lock_target = 0.3

        rb_power_df, qa_power_df = np_power.forecast_power_stats(
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
        rb_power_df["total_raw_power_eib"] = rb_power_df["total_power"] / 1024.0
        qa_power_df["total_qa_power_eib"] = qa_power_df["total_power"] / 1024.0
        power_df = np_power.build_full_power_stats_df(
            fil_stats_df,
            rb_power_df,
            qa_power_df,
            start_date,
            current_date,
            end_date,
        )
        # Forecast Vesting
        vest_df = np_vesting.compute_vesting_trajectory_df(start_date, end_date)
        # Forecast minting stats and baseline
        rb_total_power_eib = power_df["total_raw_power_eib"].values
        qa_total_power_eib = power_df["total_qa_power_eib"].values
        qa_day_onboarded_power_pib = power_df["day_onboarded_qa_power_pib"].values
        qa_day_renewed_power_pib = power_df["day_renewed_qa_power_pib"].values
        mint_df = np_minting.compute_minting_trajectory_df(
            start_date,
            end_date,
            rb_total_power_eib,
            qa_total_power_eib,
            qa_day_onboarded_power_pib,
            qa_day_renewed_power_pib,
        )
        # Forecast circulating supply
        start_day_stats = fil_stats_df.iloc[0]
        circ_supply_zero = start_day_stats["circulating_fil"]
        locked_fil_zero = start_day_stats["locked_fil"]
        daily_burnt_fil = fil_stats_df["burnt_fil"].diff().mean()
        burnt_fil_vec = fil_stats_df["burnt_fil"].values
        forecast_renewal_rate_vec = np.ones(forecast_length)*rr
        past_renewal_rate_vec = fil_stats_df["rb_renewal_rate"].values[:-1]
        renewal_rate_vec = np.concatenate(
            [past_renewal_rate_vec, forecast_renewal_rate_vec]
        )
        # print(len(renewal_rate_vec), forecast_length)
        cil_df = np_supply.forecast_circulating_supply_df(
            start_date,
            current_date,
            end_date,
            circ_supply_zero,
            locked_fil_zero,
            daily_burnt_fil,
            duration,
            renewal_rate_vec,
            burnt_fil_vec,
            vest_df,
            mint_df,
            known_scheduled_pledge_release_full_vec,
            lock_target=lock_target
        )

        # convert mint + vest into dictionaries
        vest_dict = {
            "days": vest_df['days'].values,
            'total_vest': np.asarray(vest_df['total_vest'].values),
        }
        mint_dict = {
            'days': mint_df['days'].values,
            'network_RBP_EIB': np.asarray(mint_df['network_RBP'].values) / C.EIB,
            'network_QAP_EIB': np.asarray(mint_df['network_QAP'].values) / C.EIB,
            'day_onboarded_power_QAP_PIB': np.asarray(mint_df['day_onboarded_power_QAP'].values / C.PIB),
            'day_renewed_power_QAP_PIB': np.asarray(mint_df['day_renewed_power_QAP'].values / C.PIB),
            'cum_simple_reward': np.asarray(mint_df['cum_simple_reward'].values),
            'network_baseline_EIB': np.asarray(mint_df['network_baseline'].values) / C.EIB,
            'capped_power_EIB': np.asarray(mint_df['capped_power'].values) / C.EIB,
            'cum_capped_power_EIB': np.asarray(mint_df['cum_capped_power'].values) / C.EIB,
            'network_time': np.asarray(mint_df['network_time'].values),
            'cum_baseline_reward': np.asarray(mint_df['cum_baseline_reward'].values),
            'cum_network_reward': np.asarray(mint_df['cum_network_reward'].values),
            'day_network_reward': np.asarray(mint_df['day_network_reward'].values),
        }
        cil_jax = jax_supply.forecast_circulating_supply(
            np.datetime64(start_date),
            np.datetime64(current_date),
            np.datetime64(end_date),
            circ_supply_zero,
            locked_fil_zero,
            daily_burnt_fil,
            duration,
            renewal_rate_vec,
            jnp.asarray(burnt_fil_vec),
            vest_dict,
            mint_dict,
            jnp.asarray(known_scheduled_pledge_release_full_vec),
            lock_target=lock_target,
        )
        keys = ['circ_supply', 'network_gas_burn', 'day_locked_pledge', 'day_renewed_pledge',
                'network_locked_pledge', 'network_locked', 'network_locked_reward', 'disbursed_reserve']
        for k in keys:
            self.assertTrue(np.allclose(cil_df[k].values, np.asarray(cil_jax[k]), rtol=1e-3, atol=1e-3), k)

if __name__ == '__main__':
    unittest.main()