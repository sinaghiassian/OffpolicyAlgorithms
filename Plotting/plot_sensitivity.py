import os
import numpy as np
import json
import matplotlib.pyplot as plt

from Plotting.plot_params import EXPS, ALG_GROUPS, ALG_COLORS, EXP_ATTRS, AUC_AND_FINAL, LMBDA_AND_ZETA, \
    PLOT_RERUN_AND_ORIG, RERUN, RERUN_POSTFIX
from Plotting.plot_utils import replace_large_nan_inf, make_res_path, make_exp_path, load_best_rerun_params_dict
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
    ax.plot(alphas, best_performance, label=lbl, linestyle='-', marker='o', color=ALG_COLORS[alg],
            linewidth=2, markersize=5, alpha=alpha)
    ax.errorbar(alphas, best_performance, yerr=stderr, ecolor=ALG_COLORS[alg], mfc=ALG_COLORS[alg],
                mec=ALG_COLORS[alg], linestyle='', elinewidth=2, markersize=5, alpha=alpha)
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


def get_alphas(alg, exp):
    exp_path = make_exp_path(alg, exp)
    exp_path = os.path.join(exp_path, f"{alg}.json")
    with open(exp_path) as f:
        jsn_content = json.load(f)
        return jsn_content['meta_parameters']['alpha']


def plot_sensitivity_curve():
    for exp in EXPS:
        exp_attrs = EXP_ATTRS[exp](exp)
        for auc_or_final in AUC_AND_FINAL:
            for sp in LMBDA_AND_ZETA:
                save_dir = os.path.join('pdf_plots', exp, f'Lmbda{sp}_{auc_or_final}')
                for alg_names in ALG_GROUPS.values():
                    fig, ax = plt.subplots()
                    for alg in alg_names:
                        postfix = RERUN_POSTFIX if RERUN else ''
                        best_params = load_best_rerun_params_dict(alg, exp, auc_or_final, sp)
                        alphas = get_alphas(alg, exp)
                        best_performance, stderr = load_best_performance_over_alpha(
                            alg, exp, auc_or_final, best_params, exp_attrs, postfix)
                        plot_sensitivity(ax, alg, alphas, best_performance, stderr, exp_attrs)
                        if PLOT_RERUN_AND_ORIG:
                            best_performance, stderr = load_best_performance_over_alpha(
                                alg, exp, auc_or_final, best_params, exp_attrs, postfix)
                            plot_sensitivity(ax, alg, alphas, best_performance, stderr, exp_attrs, True)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    if PLOT_RERUN_AND_ORIG:
                        postfix = '_rerun_and_original'
                    elif RERUN:
                        postfix = RERUN_POSTFIX
                    else:
                        postfix = ''
                    fig.savefig(os.path.join(save_dir, f"sensitivity_{'_'.join(alg_names)}{postfix}.pdf"),
                                format='pdf', dpi=1000, bbox_inches='tight')
                    plt.show()
                    print(exp, alg_names, auc_or_final, sp)
