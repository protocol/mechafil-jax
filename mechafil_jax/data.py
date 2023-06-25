from datetime import date
import mechafil.data as data

import mechafil_jax.constants as C

def get_simulation_data(bearer_token_or_auth_file:str, 
                        start_date:date, current_date:date, end_date:date):
    # setup data access
    data.setup_spacescope(bearer_token_or_auth_file)

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

    start_vested_amt = int(data.get_vested_amount(start_date))

    zero_cum_capped_power_eib = data.get_cum_capped_rb_power(start_date) / C.EXBI
    init_baseline_eib = data.get_storage_baseline_value(start_date) / C.EXBI

    start_day_stats = fil_stats_df.iloc[0]
    circ_supply_zero = start_day_stats["circulating_fil"]
    locked_fil_zero = start_day_stats["locked_fil"]
    daily_burnt_fil = fil_stats_df["burnt_fil"].diff().mean()
    burnt_fil_vec = fil_stats_df["burnt_fil"].values
    historical_renewal_rate = fil_stats_df["rb_renewal_rate"].values[:-1]

    data_dict = {
        "rb_power_zero": rb_power_zero,
        "qa_power_zero": qa_power_zero,
        "historical_raw_power_eib": fil_stats_df["total_raw_power_eib"].values,
        "historical_qa_power_eib": fil_stats_df["total_qa_power_eib"].values,
        "historical_onboarded_qa_power_pib": fil_stats_df["day_onboarded_qa_power_pib"].values,
        "historical_renewed_qa_power_pib": fil_stats_df["day_renewed_qa_power_pib"].values,

        "rb_known_scheduled_expire_vec": rb_known_scheduled_expire_vec,
        "qa_known_scheduled_expire_vec": qa_known_scheduled_expire_vec,
        "known_scheduled_pledge_release_full_vec": known_scheduled_pledge_release_full_vec,

        "start_vested_amt": start_vested_amt,

        "zero_cum_capped_power_eib": zero_cum_capped_power_eib,
        "init_baseline_eib": init_baseline_eib,

        "circ_supply_zero": circ_supply_zero,
        "locked_fil_zero": locked_fil_zero,
        "daily_burnt_fil": daily_burnt_fil,
        "burnt_fil_vec": burnt_fil_vec,
        "historical_renewal_rate": historical_renewal_rate,
    }

    return data_dict