import os
import numpy as np
import json
import matplotlib.pyplot as plt

from Plotting.plot_params import EXPS, EXP_ATTRS, AUC_AND_FINAL, LMBDA_AND_ZETA, ALG_COLORS
from Plotting.plot_utils import replace_large_nan_inf, make_res_path, make_exp_path, make_params, make_current_params
from utils import create_name_for_save_load

new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#17becf', 'orange', '#8c564b', '#e377c2', '#2ca02c','#bcbd22',
              '#d62728', 'black', 'cyan']
color_counter = 1


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
    global color_counter
    lbl = f'{alg}_{tp}'
    ax.set_xscale('log', basex=2)
    color = new_colors[color_counter]
    linestyle = '-'
    alpha = 1.0
    # if alg == 'PGTD2':
    #     linestyle = '--'
    #     alpha = 0.5
    ax.plot(alphas, performance, label=lbl, linestyle=linestyle, marker='o',
            linewidth=2, markersize=5, color=color, alpha=alpha)
    ax.errorbar(alphas, performance, yerr=stderr, linestyle='', elinewidth=2, markersize=5,
                color=color, alpha=alpha)
    color_counter = color_counter + 1
    # ax.legend()
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(exp_attrs.y_lim)
    ax.set_ylim([0.1, 0.8])
    ax.yaxis.set_ticks(exp_attrs.y_axis_ticks)
    ax.tick_params(axis='y', which='major', labelsize=exp_attrs.size_of_labels)
    ax.xaxis.set_ticks(exp_attrs.x_axis_ticks_log)
    ax.set_xticklabels(exp_attrs.x_axis_tick_labels_log, fontsize=25)
    plt.xticks(fontsize=25)
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def get_alphas(alg, exp):
    exp_path = make_exp_path(alg, exp)
    exp_path = os.path.join(exp_path, f"{alg}.json")
    with open(exp_path) as f:
        jsn_content = json.load(f)
        return jsn_content['meta_parameters']['alpha']


COUNTER = 0


def plot_all_sensitivities_per_alg_gradients_all_eta(**kwargs):
    global color_counter, COUNTER
    for exp in kwargs['exps']:
        exp_attrs = EXP_ATTRS[exp](exp)
        for auc_or_final in kwargs['auc_or_final']:
            for sp in kwargs['sp_list']:
                for alg in kwargs['algs']:
                    color_counter = 4
                    save_dir = os.path.join('pdf_plots', 'AllThirds', exp, f'Lmbda{sp}_{auc_or_final}')
                    fig, ax = plt.subplots(figsize=kwargs['fig_size'])
                    fp_list, sp_list, tp_list, fop_list, _ = make_params(alg, exp)
                    if alg == 'TDRC':
                        _, _, tp_list, _, _ = make_params('GTD', exp)
                        fop_list = kwargs['tdrc_beta']
                    for tp in tp_list:
                        COUNTER += 1
                        for fop in fop_list:
                            current_params = make_current_params(alg, sp, tp, fop)
                            alphas = get_alphas(alg, exp)
                            performance, stderr = load_performance_over_alpha(
                                alg, exp, current_params, auc_or_final, exp_attrs)
                            plot_sensitivity(ax, alg, exp, alphas, sp, tp, performance, stderr, exp_attrs)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    if alg == 'TDRC':
                        fig.savefig(
                            os.path.join(save_dir, f"sensitivity_{alg}_{exp}_all_eta_beta_{kwargs['tdrc_beta']}.pdf"),
                            format='pdf', dpi=1000, bbox_inches='tight')
                    else:
                        fig.savefig(os.path.join(save_dir, f"sensitivity_{alg}_{exp}_all_eta.pdf"),
                                    format='pdf', dpi=1000, bbox_inches='tight')
                    plt.show()
                    print(exp, alg, auc_or_final, sp)
