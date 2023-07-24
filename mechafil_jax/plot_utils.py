from datetime import date, timedelta

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

import mechafil_jax.minting as minting
import mechafil_jax.date_utils as du

def plot_x(ax, results, key, t, labels, colors, scale_by=1.0, start_idx=0):
    sweep_keys = list(results.keys())
    for ii, scenario in enumerate(sweep_keys):
        res = results[scenario]
        ax.plot(t[start_idx:], res[key][start_idx:]/scale_by, label=labels[ii], color=colors[ii])
        
    for tick in ax.get_xticklabels():
        tick.set_rotation(60)

def plot_kpi_panel(start_date:date, end_date: date, 
                   results_dict, 
                   init_baseline_eib: float = None,
                   figsize=(8,10), color_vec=None, suptitle_str=None, 
                   labels_vec=None, log_scale_power=False):
    
    t = du.get_t(start_date, end_date=end_date)
    print(start_date, len(t))
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=figsize)

    kz = list(results_dict.keys())
    if color_vec is None:
        color_vec = [None]*len(kz)
    if labels_vec is None:
        labels_vec = kz

    axx = ax[0,0]
    plot_x(axx, results_dict, 'rb_total_power_eib', t, labels_vec, color_vec)
    axx.set_ylabel('EiB')
    axx.set_title('RBP')
    axx.legend(fontsize=8)
    if log_scale_power:
        axx.set_yscale('log')
        axx.set_ylim(0)

    axx = ax[0,1]
    plot_x(axx, results_dict, 'qa_total_power_eib', t, labels_vec, color_vec)
    axx.set_ylabel('EiB')
    axx.set_title('QAP')
    axx.set_ylim(0)
    if init_baseline_eib is not None:
        baseline = minting.compute_baseline_power_array(
            np.datetime64(start_date), np.datetime64(end_date), init_baseline_eib,
        )
        axx.plot(t, baseline, linestyle=':', label='Baseline')
    if log_scale_power:
        axx.set_yscale('log')
        axx.set_ylim(0)

    axx = ax[1,0]
    plot_x(axx, results_dict, 'day_network_reward', t, labels_vec, color_vec)
    axx.set_ylabel('FIL/day')
    axx.set_title('Minting Rate')
    axx.set_ylim(0)

    axx = ax[1,1]
    plot_x(axx, results_dict, 'network_locked', t, labels_vec, color_vec, scale_by=1e6)
    axx.set_ylabel('M-FIL')
    axx.set_title('Network Locked')
    axx.axhline(0, linestyle=':', color='k')
    axx.set_ylim(0)

    axx = ax[2,0]
    plot_x(axx, results_dict, 'circ_supply', t, labels_vec, color_vec, scale_by=1e6)
    axx.set_ylabel('M-FIL')
    axx.set_title('Circulating Supply')
    axx.set_ylim(0)

    axx = ax[2,1]
    for ii, sc in enumerate(kz):
        res = results_dict[sc]
        axx.plot(t, res['network_locked']/res['circ_supply'], label=labels_vec[ii], color=color_vec[ii])
    for tick in axx.get_xticklabels():
        tick.set_rotation(60)
    axx.set_title('L/CS')
    axx.set_ylim(0)
    
    axx = ax[3,0]
    plot_x(axx, results_dict, 'day_pledge_per_QAP', t, labels_vec, color_vec, start_idx=1)
    axx.set_ylabel('FIL')
    axx.set_title('Pledge/32GiB QA Sector')
    axx.set_ylim(0)
    
    axx = ax[3,1]
    for ii, sc in enumerate(kz):
        res = results_dict[sc]
        axx.plot(t[:-364], res['1y_sector_roi']*100, label=labels_vec[ii], color=color_vec[ii])
    for tick in axx.get_xticklabels():
        tick.set_rotation(60)
    axx.set_ylabel('%')
    axx.set_title('1Y Realized FoFR')
    axx.set_ylim(0)
   
    if suptitle_str is not None: 
        plt.suptitle(suptitle_str)

    plt.tight_layout()
    # if output_fp is not None:
    #     plt.savefig(output_fp)

    return fig, ax
