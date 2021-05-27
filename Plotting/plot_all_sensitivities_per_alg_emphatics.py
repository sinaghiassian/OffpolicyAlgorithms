import json
import os

import matplotlib.pyplot as plt
import numpy as np

from Plotting.plot_params import EXP_ATTRS, AUC_AND_FINAL
from Plotting.plot_utils import replace_large_nan_inf, make_res_path, make_exp_path, make_params, make_current_params
from utils import create_name_for_save_load

plot_alpha = 1.0


def load_performance_over_alpha(alg, exp, params, auc_or_final, exp_attrs):
    res_path = make_res_path(alg, exp)
    load_file_name = os.path.join(res_path, create_name_for_save_load(
        params, excluded_params=['alpha']) + f"_mean_{auc_or_final}_over_alpha.npy")
    performance_over_alpha = np.load(load_file_name)
    performance_over_alpha = replace_large_nan_inf(
        performance_over_alpha, large=exp_attrs.learning_starting_point,
        replace_with=exp_attrs.over_limit_replacement)
    stderr_load_file_name = os.path.join(
        res_path, create_name_for_save_load(params, excluded_params=['alpha']) +
        f'_stderr_{auc_or_final}_over_alpha.npy')
    std_err_of_best_perf_over_alpha = np.load(stderr_load_file_name)
    std_err_of_best_perf_over_alpha = replace_large_nan_inf(
        std_err_of_best_perf_over_alpha, large=exp_attrs.learning_starting_point, replace_with=0.0)
    return performance_over_alpha, std_err_of_best_perf_over_alpha


def plot_sensitivity(ax, alg, exp, alphas, sp, tp, performance, stderr, exp_attrs):
    global plot_alpha
    lbl = f'{alg}_{tp}'
    ax.set_xscale('log', basex=2)
    if alg == 'ETD':
        color = 'red'
    elif alg == 'ETDLB':
        color = 'grey'
        plot_alpha -= 0.1
    else:
        color = 'black'
    ax.plot(alphas, performance, label=lbl, linestyle='-', marker='o',
            linewidth=2, markersize=5, color=color, alpha=plot_alpha)
    ax.errorbar(alphas, performance, yerr=stderr, linestyle='', elinewidth=2, markersize=5,
                color=color, alpha=plot_alpha)
    # ax.legend()
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(exp_attrs.y_lim)
    ax.yaxis.set_ticks(exp_attrs.y_axis_ticks)
    ax.tick_params(axis='y', which='major', labelsize=exp_attrs.size_of_labels)
    ax.xaxis.set_ticks(exp_attrs.x_axis_ticks_log)
    # ax.set_xticklabels(exp_attrs.x_axis_tick_labels_log, fontsize=25)
    # plt.xticks(fontsize=25)
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def get_alphas(alg, exp):
    exp_path = make_exp_path(alg, exp)
    exp_path = os.path.join(exp_path, f"{alg}.json")
    with open(exp_path) as f:
        jsn_content = json.load(f)
        return jsn_content['meta_parameters']['alpha']


def plot_all_sensitivities_per_alg_emphatics(**kwargs):
    global plot_alpha
    for exp in kwargs['exps']:
        exp_attrs = EXP_ATTRS[exp](exp)
        for auc_or_final in kwargs['auc_or_final']:
            for sp in kwargs['sp_list']:
                plot_alpha = 1.0
                alg = 'ETD'
                save_dir = os.path.join('pdf_plots', 'AllThirds', exp, f'Lmbda{sp}_{auc_or_final}')
                fig, ax = plt.subplots(figsize=kwargs['fig_size'])
                current_params = make_current_params(alg, sp, 0, 0)
                alphas = get_alphas(alg, exp)
                performance, stderr = load_performance_over_alpha(
                    alg, exp, current_params, auc_or_final, exp_attrs)
                plot_sensitivity(ax, alg, exp, alphas, sp, 0, performance, stderr, exp_attrs)
                alg = 'ETDLB'
                fp_list, sp_list, tp_list, fop_list, _ = make_params(alg, exp)
                for tp in tp_list:
                    for fop in fop_list:
                        current_params = make_current_params(alg, sp, tp, fop)
                        alphas = get_alphas(alg, exp)
                        performance, stderr = load_performance_over_alpha(
                            alg, exp, current_params, auc_or_final, exp_attrs)
                        plot_sensitivity(ax, alg, exp, alphas, sp, tp, performance, stderr, exp_attrs)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, f"sensitivity_{alg}_{exp}.pdf"),
                            format='pdf', dpi=1000, bbox_inches='tight')
                plt.show()
                print(exp, alg, auc_or_final, sp)
