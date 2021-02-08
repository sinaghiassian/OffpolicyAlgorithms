import matplotlib.pyplot as plt
import numpy as np
import os
from Plotting.plot_utils import make_params, make_current_params, make_args, make_fig, color_dict, get_alg_names
from utils import create_name_for_save_load

args = make_args()
fig, ax = make_fig()
auc_or_final = 'auc'  # 'final' or 'auc'
lmbda_or_zeta = 0  # 0 or 0.9
alg_names = get_alg_names(args.exp_name)


def find_best_performance_over_alpha(alg_name):
    fp_list, sp_list, tp_list, fop_list, res_path = make_params(alg_name, args.exp_name)
    sp_list = [lmbda_or_zeta]
    minimum_value = float(np.inf)
    best_performance_over_alpha = np.zeros(len(fp_list))
    best_sp, best_tp, best_fop = float(-np.inf), float(-np.inf), float(-np.inf)
    stderr_load_file_name = ''
    for fop in fop_list:
        for tp in tp_list:
            for sp in sp_list:
                current_params = make_current_params(alg_name, sp, tp, fop)
                load_file_name = os.path.join(res_path, create_name_for_save_load(
                    current_params, excluded_params=['alpha']) + f'_mean_{auc_or_final}_over_alpha.npy')
                performance_over_alpha = np.load(load_file_name)
                if min(performance_over_alpha) < minimum_value:
                    best_performance_over_alpha = performance_over_alpha
                    best_tp = tp
                    best_fop = fop
                    stderr_load_file_name = os.path.join(
                        res_path, create_name_for_save_load(current_params, excluded_params=['alpha']) +
                        f'_stderr_{auc_or_final}_over_alpha.npy')
    return best_performance_over_alpha, np.load(stderr_load_file_name),  best_tp, best_fop, np.array(fp_list)


def plot_sensitivity(best_performance, stderr, alg_name, best_tp, best_fop, fp_list):
    lbl = f'{alg_name}_{best_tp}_{best_fop}'
    plt.semilogx(fp_list, best_performance, label=lbl, linestyle='-', marker='o', color=color_dict[alg_name],
                 linewidth=2, markersize=5)
    plt.errorbar(fp_list, best_performance, yerr=stderr, ecolor=color_dict[alg_name], mfc=color_dict[alg_name],
                 mec=color_dict[alg_name], linestyle='', elinewidth=2, markersize=5)


for alg_name in alg_names:
    best_performance, stderr, best_tp, best_fop, fp_list = find_best_performance_over_alpha(alg_name)
    plot_sensitivity(best_performance, stderr, alg_name, best_tp, best_fop, fp_list)
plt.show()
