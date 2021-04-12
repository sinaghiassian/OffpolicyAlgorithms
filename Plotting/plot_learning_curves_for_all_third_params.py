import os

import matplotlib.pyplot as plt
import numpy as np
import pylab

from Plotting.plot_params import ALG_COLORS, EXP_ATTRS, AUC_AND_FINAL, PLOT_RERUN_AND_ORIG
from Plotting.plot_utils import make_params, get_alphas, make_current_params
from utils import create_name_for_save_load


def load_data(alg, exp, best_params, postfix=''):
    res_path = os.path.join(os.getcwd(), 'Results', exp, alg)
    generic_name = create_name_for_save_load(best_params)
    load_file_name = os.path.join(res_path, f"{generic_name}_RMSVE_mean_over_runs{postfix}.npy")
    mean_lc = np.load(load_file_name)
    load_file_name = os.path.join(res_path, f"{generic_name}_RMSVE_stderr_over_runs{postfix}.npy")
    stderr_lc = np.load(load_file_name)
    return mean_lc, stderr_lc


def plot_data(ax, alg, mean_lc, mean_stderr, best_params, exp_attrs, second_time=False):
    alpha = 1.0
    if PLOT_RERUN_AND_ORIG:
        alpha = 1.0 if second_time else 0.5
    lbl = (alg + r'$\alpha=$ ' + str(best_params['alpha']))
    color = ALG_COLORS[alg]
    ax.plot(np.arange(mean_lc.shape[0]), mean_lc, label=lbl, linewidth=1.0, color=color, alpha=alpha)
    ax.fill_between(np.arange(mean_lc.shape[0]), mean_lc - mean_stderr / 2, mean_lc + mean_stderr / 2,
                    color=color, alpha=0.1*alpha)
    # ax.legend()
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


def get_ls_rmsve(alg, exp, sp):
    res_path = os.path.join(os.getcwd(), 'Results', exp, alg)
    params = {'alpha': 0.01, 'lmbda': sp}
    if alg == 'LSETD':
        params['beta'] = 0.9
    generic_name = create_name_for_save_load(params)
    load_file_name = os.path.join(res_path, f"{generic_name}_RMSVE_mean_over_runs.npy")
    return np.load(load_file_name)


def plot_ls_solution(ax, ls_rmsve, alg, sp):
    lbl = f"{alg} $\\lambda=$ {sp}"
    x = np.arange(ls_rmsve.shape[0])
    y = ls_rmsve[-1] * np.ones(ls_rmsve.shape[0])
    ax.plot(x, y, label=lbl, linewidth=1.0, color=ALG_COLORS[alg], linestyle='--')
    # ax.legend()


def load_specific_params_dict(alg, exp, sp, tp):
    if alg == 'TD':
        return {'alpha': 0.25, 'lmbda': sp}
    if alg == 'ETD':
        return {'alpha': 0.00390625, 'lmbda': sp}
    if alg == 'ETDLB':
        return {'alpha': 0.000488281, 'lmbda': sp, 'beta': 0.2}
    if alg == 'TDRC':
        return {'alpha': 0.0625, 'lmbda': sp, 'eta': 1.0, 'tdrc_beta': 1.0}
    if alg == 'GTD':
        return {'alpha': 0.0078125, 'lmbda': sp, 'eta': tp}
    if alg == 'PGTD2':
        return {'alpha': 0.0078125, 'lmbda': sp, 'eta': tp}


def load_sample_params_dict(alg, exp, sp):
    fp_list, sp_list, tp_list, fop_list, res_path = make_params(alg, exp)
    if alg in ['TD', 'ETD', 'TB', 'Vtrace']:
        return {'alpha': fp_list[np.random.randint(0, len(fp_list))], 'lmbda': sp}
    if alg == 'ABTD':
        return {'alpha': fp_list[np.random.randint(0, len(fp_list))], 'zeta': sp}
    if alg in ['GTD', 'GTD2', 'PGTD2', 'HTD']:
        return {'alpha': fp_list[np.random.randint(0, len(fp_list))], 'lmbda': sp,
                'eta': tp_list[np.random.randint(0, len(tp_list))]}
    if alg == 'ETDLB':
        return {'alpha': fp_list[np.random.randint(0, len(fp_list))], 'lmbda': sp,
                'beta': tp_list[np.random.randint(0, len(tp_list))]}
    if alg == 'TDRC':
        return {'alpha': fp_list[np.random.randint(0, len(fp_list))], 'lmbda': sp,
                'eta': tp_list[np.random.randint(0, len(tp_list))],
                'tdrc_beta': fop_list[np.random.randint(0, len(fop_list))]}


def plot_all_learning_curves_for_third(**kwargs):
    for exp in kwargs['exps']:
        prefix = ''
        exp_attrs = EXP_ATTRS[exp](exp)
        for auc_or_final in kwargs['auc_or_final']:
            for sp in kwargs['sp_list']:
                save_dir = os.path.join('pdf_plots', 'all_third_learning_curves', auc_or_final)
                fig, ax = plt.subplots(figsize=kwargs['fig_size'])
                for alg in kwargs['algs']:
                    if alg in ['LSTD', 'LSETD']:
                        ls_rmsve = get_ls_rmsve(alg, exp, sp)
                        plot_ls_solution(ax, ls_rmsve, alg, sp)
                        continue
                    for tp in kwargs['tp_list']:
                        for fp in get_alphas(alg, exp):
                            for fop in [1.0]:
                                current_params = make_current_params(alg, sp, tp, fop, fp)
                                mean_lc, mean_stderr = load_data(alg, exp, current_params, prefix)
                                plot_data(ax, alg, mean_lc, mean_stderr, current_params, exp_attrs)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                pylab.gca().set_rasterized(True)
                fig.savefig(os.path.join(save_dir,
                            f"{prefix}_learning_curve_{'_'.join(kwargs['algs'])}{exp}Lmbda{sp}.pdf"),
                            format='pdf', dpi=200, bbox_inches='tight')
                plt.show()
                plt.close(fig)
