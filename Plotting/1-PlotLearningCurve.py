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


def find_best_mean_performance(alg_name):
    fp_list, sp_list, tp_list, fop_list, res_path = make_params(alg_name, args.exp_name)
    sp_list = [lmbda_or_zeta]
    best_perf, best_fp, best_sp, best_tp, best_fop = np.inf, np.inf, np.inf, np.inf, np.inf
    best_params = dict()
    for fop in fop_list:
        for tp in tp_list:
            for sp in sp_list:
                current_params = make_current_params(alg_name, sp, tp, fop)
                load_file_name = os.path.join(res_path, create_name_for_save_load(current_params, excluded_params=[
                    'alpha']) + f'_mean_{auc_or_final}_over_alpha.npy')
                if not os.path.exists(load_file_name):
                    continue
                current_perf = np.load(load_file_name)
                min_perf = min(current_perf)
                if min_perf < best_perf:
                    best_perf = min_perf
                    best_perf_idx = int(np.nanargmin(current_perf))
                    best_fp = fp_list[best_perf_idx]
                    best_sp = sp
                    best_tp = tp
                    best_fop = fop
                    best_params = current_params
                    best_params['alpha'] = best_fp
    return best_fp, best_sp, best_tp, best_fop, best_params


def load_data(alg_name, best_params):
    res_path = os.path.join(os.getcwd(), '../Results', args.exp_name, alg_name)
    load_file_name = os.path.join(res_path, create_name_for_save_load(best_params) + '_RMSVE_mean_over_runs.npy')
    mean_lc = np.load(load_file_name)
    load_file_name = os.path.join(res_path, create_name_for_save_load(best_params) + '_RMSVE_stderr_over_runs.npy')
    stderr_lc = np.load(load_file_name)
    return mean_lc, stderr_lc


def plot_data(alg_name, mean_lc, mean_stderr, best_params):
    lbl = (alg_name + r'$\alpha=$' + str(best_params['alpha']))
    ax.plot(np.arange(mean_lc.shape[0]), mean_lc, label=lbl, linewidth=1.0, color=color_dict[alg_name])
    ax.fill_between(np.arange(
        mean_lc.shape[0]), mean_lc - mean_stderr / 2, mean_lc + mean_stderr / 2, alpha=0.3, color=color_dict[alg_name])


for alg in alg_names:
    print(alg)
    fp, sp, tp, fop, current_params = find_best_mean_performance(alg)
    if fp == np.inf:
        continue
    mean_lc, mean_stderr = load_data(alg, current_params)
    plot_data(alg, mean_lc, mean_stderr, current_params)
plt.show()
