import matplotlib.pyplot as plt
import numpy as np
import os
import pylab
from Plotting.plot_params import ALG_GROUPS, ALG_COLORS, EXP_ATTRS, EXPS, AUC_AND_FINAL, LMBDA_AND_ZETA
from Plotting.plot_utils import make_params, make_current_params, replace_large_nan_inf
from utils import create_name_for_save_load


def find_best_mean_performance(alg, exp, auc_or_final, sp, exp_attrs):
    fp_list, sp_list, tp_list, fop_list, res_path = make_params(alg, exp)
    best_perf, best_fp, best_sp, best_tp, best_fop = np.inf, np.inf, np.inf, np.inf, np.inf
    best_params = dict()
    for fop in fop_list:
        for tp in tp_list:
            current_params = make_current_params(alg, sp, tp, fop)
            load_file_name = os.path.join(res_path, create_name_for_save_load(current_params, excluded_params=[
                'alpha']) + f'_mean_{auc_or_final}_over_alpha.npy')
            if not os.path.exists(load_file_name):
                continue
            current_perf = np.load(load_file_name)
            current_perf = replace_large_nan_inf(
                current_perf, large=exp_attrs.learning_starting_point, replace_with=exp_attrs.over_limit_replacement
            )
            min_perf = min(current_perf)
            if min_perf < best_perf:
                best_perf = min_perf
                best_perf_idx = int(np.nanargmin(current_perf))
                best_fp = fp_list[best_perf_idx]
                best_params = current_params
                best_params['alpha'] = best_fp
    return best_params


def load_data(alg, exp, best_params):
    res_path = os.path.join(os.getcwd(), 'Results', exp, alg)
    load_file_name = os.path.join(res_path, create_name_for_save_load(best_params) + '_RMSVE_mean_over_runs.npy')
    mean_lc = np.load(load_file_name)
    load_file_name = os.path.join(res_path, create_name_for_save_load(best_params) + '_RMSVE_stderr_over_runs.npy')
    stderr_lc = np.load(load_file_name)
    return mean_lc, stderr_lc


def plot_data(ax, alg, mean_lc, mean_stderr, best_params, exp_attrs):
    lbl = (alg + r'$\alpha=$ ' + str(best_params['alpha']))
    ax.plot(np.arange(mean_lc.shape[0]), mean_lc, label=lbl, linewidth=1.0, color=ALG_COLORS[alg])
    ax.fill_between(np.arange(mean_lc.shape[0]), mean_lc - mean_stderr / 2, mean_lc + mean_stderr / 2,
                    alpha=0.1, color=ALG_COLORS[alg])
    ax.legend()
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(exp_attrs.x_lim)
    ax.set_ylim(exp_attrs.y_lim)
    ax.xaxis.set_ticks(exp_attrs.x_axis_ticks)
    ax.set_xticklabels(exp_attrs.x_tick_labels, fontsize=25)
    ax.yaxis.set_ticks(exp_attrs.y_axis_ticks)
    ax.tick_params(axis='y', which='major', labelsize=exp_attrs.size_of_labels)


def plot_learning_curve():
    for exp in EXPS:
        exp_attrs = EXP_ATTRS[exp](exp)
        for auc_or_final in AUC_AND_FINAL:
            for sp in LMBDA_AND_ZETA:
                save_dir = os.path.join('pdf_plots', exp, f'Lmbda{sp}_{auc_or_final}')
                for alg_names in ALG_GROUPS.values():
                    fig, ax = plt.subplots()
                    for alg in alg_names:
                        current_params = find_best_mean_performance(alg, exp, auc_or_final, sp, exp_attrs)
                        print(alg, current_params)
                        mean_lc, mean_stderr = load_data(alg, exp, current_params)
                        plot_data(ax, alg, mean_lc, mean_stderr, current_params, exp_attrs)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                        pylab.gca().set_rasterized(True)
                        fig.savefig(save_dir + '/learning_curve_' + '_'.join(alg_names) + '.pdf',
                                    format='pdf', dpi=200, bbox_inches='tight')
                    plt.show()
                    # TODO: Add rerun.
                    # TODO: change to loading the best parameters rather than finding them again.
