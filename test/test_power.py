import unittest
from datetime import date, timedelta

import mechafil_jax.power as jax_power
import mechafil.power as np_power
import mechafil.data as data

import numpy as np
import jax.numpy as jnp
from tqdm.auto import tqdm

class TestPower(unittest.TestCase):
    def test_forecast_power_stats(self):
        # setup data access
        # TODO: better way to do this?
        data.setup_spacescope('/Users/kiran/code/filecoin-mecha-twin/kiran_spacescope_auth.json')

        forecast_length = 5*365
        start_date = date(2021, 3, 16)
        current_date = date.today() - timedelta(days=2)
        end_date = current_date + timedelta(days=forecast_length)

        ############## download necessary data for forecasting
        print('Downloading historical data from Spacescope...')
        # Get sector scheduled expirations
        res = data.get_sector_expiration_stats(start_date, current_date, end_date)
        rb_known_scheduled_expire_vec = res[0]
        qa_known_scheduled_expire_vec = res[1]
        known_scheduled_pledge_release_full_vec = res[2]
        # Get daily stats
        fil_stats_df = data.get_historical_network_stats(start_date, current_date, end_date)
        current_day_stats = fil_stats_df[fil_stats_df["date"] >= current_date].iloc[0]
        # Forecast power stats
        rb_power_zero = current_day_stats["total_raw_power_eib"] * 1024.0
        qa_power_zero = current_day_stats["total_qa_power_eib"] * 1024.0
        ##############

        # test various onboarding, renewal, and fil-plus rates between jax & mechafil
        rbp_vec = [3, 6, 15]
        rr_vec = [0.3, 0.6, 0.9]
        fpr_vec = [0.3, 0.6, 0.9]
        duration_vec = [360, 540]
        
        n = len(rbp_vec) * len(rr_vec) * len(fpr_vec) * len(duration_vec)
        pbar = tqdm(total=n)
        for rbp in rbp_vec:
            for rr in rr_vec:
                for fpr in fpr_vec:
                    for duration in duration_vec:
                        # mechafil
                        rb_df_mechafil, qa_df_mechafil = np_power.forecast_power_stats(
                            rb_power_zero, qa_power_zero, rbp, rb_known_scheduled_expire_vec, qa_known_scheduled_expire_vec,
                            rr, fpr, duration, forecast_length, qap_method='basic'
                        )

                        # jax
                        rbp_vec_in = jnp.ones(forecast_length)*rbp
                        rr_vec_in = jnp.ones(forecast_length)*rr
                        fpr_vec_in = jnp.ones(forecast_length)*fpr
                        duration_int = int(duration)
                        rb_dict_jax, qa_dict_jax = jax_power.forecast_power_stats(
                            rb_power_zero, qa_power_zero, 
                            rbp_vec_in, rb_known_scheduled_expire_vec, qa_known_scheduled_expire_vec,
                            rr_vec_in, fpr_vec_in, duration_int, forecast_length
                        )

                        # compare
                        keys = ['cum_onboarded_power', 'cum_renewed_power', 'cum_expire_scheduled_power', 'total_power']
                        for k in keys:
                            try:
                                self.assertTrue(np.allclose(rb_df_mechafil[k], np.asarray(rb_dict_jax[k]), rtol=1e-3, atol=1e-3))
                                self.assertTrue(np.allclose(qa_df_mechafil[k], np.asarray(qa_dict_jax[k]), rtol=1e-3, atol=1e-3))
                            except AssertionError as e:
                                print('Failed Configuration! rbp=', rbp, 'rr=', rr, 'fpr=', fpr, 'duration=', duration)
                                raise e
                        pbar.update(1)

if __name__ == '__main__':
    unittest.main()