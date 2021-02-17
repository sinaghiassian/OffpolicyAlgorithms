import matplotlib.pyplot as plt
import numpy as np
import os

from Plotting.plot_params import exp_names, algs_groups, color_dict
from Plotting.plot_utils import make_params, make_current_params, replace_large_nan_inf, attr_dict
from utils import create_name_for_save_load

auc_or_final = 'auc'  # 'final' or 'auc'
lmbda_or_zeta = 0.0  # 0 or 0.9


def find_best_performance_over_alpha(alg_name):
    fp_list, sp_list, tp_list, fop_list, res_path = make_params(alg_name, exp)
    if alg_name == 'TDRC':
        tp_list = [1.0]
        fop_list = [1.0]
    if alg_name == 'GTD' or alg_name == 'PGTD2' or alg_name == 'GTD2' or alg_name == 'HTD':
        tp_list = [1.0]
    sp_list = [lmbda_or_zeta]
    minimum_value = float(np.inf)
    best_performance_over_alpha = np.zeros(len(fp_list))
    std_err_of_best_perf_over_alpha = np.zeros(len(fp_list))
    best_sp, best_tp, best_fop = float(-np.inf), float(-np.inf), float(-np.inf)
    for fop in fop_list:
        for tp in tp_list:
            for sp in sp_list:
                current_params = make_current_params(alg_name, sp, tp, fop)
                load_file_name = os.path.join(res_path, create_name_for_save_load(
                    current_params, excluded_params=['alpha']) + f'_mean_{auc_or_final}_over_alpha.npy')
                performance_over_alpha = np.load(load_file_name)
                performance_over_alpha = replace_large_nan_inf(
                    performance_over_alpha, large=attrs.learning_starting_point,
                    replace_with=attrs.over_limit_replacement)
                if min(performance_over_alpha) < minimum_value:
                    best_performance_over_alpha = performance_over_alpha
                    minimum_value = min(performance_over_alpha)
                    best_tp = tp
                    best_fop = fop
                    stderr_load_file_name = os.path.join(
                        res_path, create_name_for_save_load(current_params, excluded_params=['alpha']) +
                        f'_stderr_{auc_or_final}_over_alpha.npy')
                    std_err_of_best_perf_over_alpha = np.load(stderr_load_file_name)
                    std_err_of_best_perf_over_alpha = replace_large_nan_inf(
                        std_err_of_best_perf_over_alpha, large=attrs.learning_starting_point, replace_with=0.0)
    return best_performance_over_alpha, std_err_of_best_perf_over_alpha, best_tp, best_fop, np.array(fp_list)


def plot_sensitivity(best_performance, stderr, alg_name, best_tp, best_fop, fp_list):
    lbl = f'{alg_name}_{best_tp}_{best_fop}'
    ax.set_xscale('log', basex=2)
    ax.plot(fp_list, best_performance, label=lbl, linestyle='-', marker='o', color=color_dict[alg_name],
            linewidth=2, markersize=5)
    ax.errorbar(fp_list, best_performance, yerr=stderr, ecolor=color_dict[alg_name], mfc=color_dict[alg_name],
                mec=color_dict[alg_name], linestyle='', elinewidth=2, markersize=5)
    ax.legend()
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(attrs.y_lim)
    ax.yaxis.set_ticks(attrs.y_axis_ticks)
    ax.tick_params(axis='y', which='major', labelsize=attrs.size_of_labels)
    ax.xaxis.set_ticks(attrs.x_axis_ticks_log)
    ax.set_xticklabels(attrs.x_axis_tick_labels_log, fontsize=25)
    plt.xticks(fontsize=25)


for exp in exp_names:
    attrs = attr_dict[exp](exp)
    save_dir = os.path.join('pdf_plots', exp, f'Lmbda{lmbda_or_zeta}_{auc_or_final}')
    for alg_names in algs_groups.values():
        fig, ax = plt.subplots()
        ax.set_ylim([0, 0.8])
        best_performance, stderr, best_tp, best_fop, fp_list = None, None, None, None, None
        for alg_name in alg_names:
            best_performance, stderr, best_tp, best_fop, fp_list = find_best_performance_over_alpha(alg_name)
            print(alg_name, best_tp, best_fop)
            plot_sensitivity(best_performance, stderr, alg_name, best_tp, best_fop, fp_list)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_dir + '/sensitivity_' + '_'.join(alg_names) + '.pdf',
                    format='pdf', dpi=1000, bbox_inches='tight')
        plt.show()
