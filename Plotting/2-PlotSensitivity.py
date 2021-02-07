import matplotlib.pyplot as plt
import numpy as np
import os
from Plotting.plot_utils import make_params, make_current_params, make_args, make_fig
from utils import create_name_for_save_load

args = make_args()
fig, ax = make_fig()
alg_names = ['GTD']
auc_or_final = 'auc'  # 'final' or 'auc'
lmbda = 0  # 0 or 0.9


def find_best_performance_over_alpha(alg_name):
    fp_list, sp_list, tp_list, fop_list, res_path = make_params(alg_name, args.exp_name)
    minimum_value = float(np.inf)
    best_performance_over_alpha = np.zeros(len(fp_list))
    best_sp, best_tp, best_fop = float(-np.inf), float(-np.inf), float(-np.inf)
    stderr_load_file_name = ''
    for fop in fop_list:
        for tp in tp_list:
            for sp in sp_list:
                current_params = make_current_params(alg_name, sp, tp, fop)
                load_file_name = os.path.join(
                    res_path, create_name_for_save_load(current_params) + f'_mean_{auc_or_final}_over_alpha.npy')
                performance_over_alpha = np.load(load_file_name)
                if min(performance_over_alpha) < minimum_value:
                    best_performance_over_alpha = performance_over_alpha
                    best_tp = tp
                    best_fop = fop
                    stderr_load_file_name = os.path.join(
                        res_path, create_name_for_save_load(current_params) + f'_stderr_{auc_or_final}_over_alpha.npy')
    return best_performance_over_alpha, np.load(stderr_load_file_name),  best_tp, best_fop, np.array(fp_list)


def plot_sensitivity(best_performance, stderr, alg_name, best_tp, best_fop, fp_list):
    lbl = f'{alg_name}_{best_tp}_{best_fop}'
    plt.semilogx(
        fp_list, best_performance, label=lbl, linestyle='-', marker='o', color='blue', linewidth=2, markersize=5)
    plt.errorbar(fp_list, best_performance, yerr=stderr, ecolor='blue', mfc='blue',
                 mec='blue', linestyle='', elinewidth=2, markersize=5)


for alg_name in alg_names:
    best_performance, stderr, best_tp, best_fop, fp_list = find_best_performance_over_alpha(alg_name)
    plot_sensitivity(best_performance, stderr, alg_name, best_tp, best_fop, fp_list)
plt.show()
