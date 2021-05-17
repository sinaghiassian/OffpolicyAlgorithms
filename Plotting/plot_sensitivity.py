import os
import matplotlib.pyplot as plt
import numpy as np

from Plotting.plot_params import EXPS, ALG_GROUPS, ALG_COLORS, EXP_ATTRS, AUC_AND_FINAL, LMBDA_AND_ZETA, PLOT_RERUN, \
    PLOT_RERUN_AND_ORIG, RERUN_POSTFIX
from Plotting.plot_utils import replace_large_nan_inf, make_res_path, load_best_rerun_params_dict, get_alphas
from utils import create_name_for_save_load


def load_best_performance_over_alpha(alg, exp, auc_or_final, best_params, exp_attrs, postfix=''):
    res_path = make_res_path(alg, exp)
    load_file_name = os.path.join(res_path, create_name_for_save_load(
        best_params, excluded_params=['alpha']) + f'_mean_{auc_or_final}_over_alpha{postfix}.npy')
    performance_over_alpha = np.load(load_file_name)
    performance_over_alpha = replace_large_nan_inf(
        performance_over_alpha, large=exp_attrs.learning_starting_point,
        replace_with=exp_attrs.over_limit_replacement)
    stderr_load_file_name = os.path.join(
        res_path, create_name_for_save_load(best_params, excluded_params=['alpha']) +
        f'_stderr_{auc_or_final}_over_alpha{postfix}.npy')
    std_err_of_best_perf_over_alpha = np.load(stderr_load_file_name)
    std_err_of_best_perf_over_alpha = replace_large_nan_inf(
        std_err_of_best_perf_over_alpha, large=exp_attrs.learning_starting_point, replace_with=0.0)
    return performance_over_alpha, std_err_of_best_perf_over_alpha


# noinspection DuplicatedCode
def plot_sensitivity(ax, alg, alphas, best_performance, stderr, exp_attrs, second_time=False):
    alpha = 1.0
    if PLOT_RERUN_AND_ORIG:
        alpha = 1.0 if second_time else 0.5
    lbl = f'{alg}'
    ax.set_xscale('log', basex=2)
    color = ALG_COLORS[alg]
    if alg == 'TD':
        color = 'grey'
        alpha=0.7
    ax.plot(alphas, best_performance, label=lbl, linestyle='-', marker='o', color=color,
            linewidth=2, markersize=5, alpha=alpha)
    ax.errorbar(alphas, best_performance, yerr=stderr, ecolor=color, mfc=color,
                mec=color, linestyle='', elinewidth=2, markersize=5, alpha=alpha)
    # ax.legend()
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(exp_attrs.y_lim)
    ax.yaxis.set_ticks(exp_attrs.y_axis_ticks)
    ax.tick_params(axis='y', which='major', labelsize=exp_attrs.size_of_labels)
    ax.xaxis.set_ticks(exp_attrs.x_axis_ticks_log)
    ax.set_xticklabels(exp_attrs.x_axis_tick_labels_log, fontsize=25)
    plt.xticks(fontsize=25)
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def plot_sensitivity_curve(**kwargs):
    for exp in kwargs['exps']:
        exp_attrs = EXP_ATTRS[exp](exp)
        for auc_or_final in kwargs['auc_or_final']:
            for sp in kwargs['sp_list']:
                save_dir = os.path.join('pdf_plots', 'sensitivity_curves', auc_or_final)
                for alg_names in kwargs['alg_groups'].values():
                    fig, ax = plt.subplots(figsize=kwargs['fig_size'])
                    for alg in alg_names:
                        if alg in ['LSTD', 'LSETD']:
                            continue
                        postfix = RERUN_POSTFIX if PLOT_RERUN else ''
                        best_params = load_best_rerun_params_dict(alg, exp, auc_or_final, sp)
                        alphas = get_alphas(alg, exp)
                        best_performance, stderr = load_best_performance_over_alpha(
                            alg, exp, auc_or_final, best_params, exp_attrs, postfix)
                        plot_sensitivity(ax, alg, alphas, best_performance, stderr, exp_attrs)
                        if PLOT_RERUN_AND_ORIG:
                            postfix = RERUN_POSTFIX
                            best_performance, stderr = load_best_performance_over_alpha(
                                alg, exp, auc_or_final, best_params, exp_attrs, postfix)
                            plot_sensitivity(ax, alg, alphas, best_performance, stderr, exp_attrs, True)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    if PLOT_RERUN_AND_ORIG:
                        prefix = '_rerun_and_original'
                    elif PLOT_RERUN:
                        prefix = RERUN_POSTFIX
                    else:
                        prefix = ''
                    fig.savefig(os.path.join(save_dir,
                                             f"{prefix}_sensitivity_curve_{'_'.join(alg_names)}{exp}Lmbda{sp}.pdf"),
                                format='pdf', dpi=1000, bbox_inches='tight')
                    plt.show()
                    print(exp, alg_names, auc_or_final, sp)
