import matplotlib.pyplot as plt
import numpy as np
import os
import pylab
from Plotting.plot_params import ALG_GROUPS, ALG_COLORS, EXP_ATTRS, EXPS, AUC_AND_FINAL, LMBDA_AND_ZETA, \
    PLOT_RERUN_AND_ORIG, PLOT_RERUN, RERUN_POSTFIX
from Plotting.plot_utils import load_best_rerun_params_dict
from utils import create_name_for_save_load


def load_data(alg, exp, best_params, postfix=''):
    res_path = os.path.join(os.getcwd(), 'Results', exp, alg)
    generic_name = create_name_for_save_load(best_params)
    load_file_name = os.path.join(res_path, f"{generic_name}_RMSVE_mean_over_runs{postfix}.npy")
    mean_lc = np.load(load_file_name)
    load_file_name = os.path.join(res_path, f"{generic_name}_RMSVE_stderr_over_runs{postfix}.npy")
    stderr_lc = np.load(load_file_name)
    return mean_lc, stderr_lc


def plot_data(ax, alg, mean_lc, mean_stderr, best_params, exp_attrs, second_time=False, is_smoothed=False,
              smoothing_window=1):
    zoomed_in = True if is_smoothed else False
    alpha = 1.0
    if PLOT_RERUN_AND_ORIG:
        alpha = 1.0 if second_time else 0.5
    lbl = (alg + r'$\alpha=$ ' + str(best_params['alpha']))
    color = ALG_COLORS[alg]
    # if alg == 'TD':
    #     color = 'grey'
    #     alpha = 0.7
    if is_smoothed:
        mean_lc = np.convolve(mean_lc, np.ones(smoothing_window)/smoothing_window, mode='valid')
        mean_stderr = np.convolve(mean_stderr, np.ones(smoothing_window)/smoothing_window, mode='valid')
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
    if zoomed_in:
        ax.set_ylim([0.0, 0.4])
    else:
        ax.yaxis.set_ticks(exp_attrs.y_axis_ticks)
    ax.xaxis.set_ticks(exp_attrs.x_axis_ticks)
    ax.set_xticklabels(exp_attrs.x_tick_labels, fontsize=25)
    ax.tick_params(axis='y', which='major', labelsize=exp_attrs.size_of_labels)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

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
    ax.plot(x, y, label=lbl, linewidth=1.0, color=ALG_COLORS[alg], linestyle=':')
    # ax.legend()


def plot_learning_curve(**kwargs):
    is_smoothed = True if 'is_smoothed' in kwargs else False
    smoothing_window = kwargs['smoothing_window'] if 'smoothing_window' in kwargs else 1
    for exp in kwargs['exps']:
        exp_attrs = EXP_ATTRS[exp](exp)
        for auc_or_final in kwargs['auc_or_final']:
            for sp in kwargs['sp_list']:
                save_dir = os.path.join('pdf_plots', 'learning_curves', auc_or_final)
                for alg_names in kwargs['alg_groups'].values():
                    fig, ax = plt.subplots(figsize=kwargs['fig_size'])
                    for alg in alg_names:
                        if alg in ['LSTD', 'LSETD']:
                            ls_rmsve = get_ls_rmsve(alg, exp, sp)
                            plot_ls_solution(ax, ls_rmsve, alg, sp)
                            continue
                        prefix = RERUN_POSTFIX if PLOT_RERUN else ''
                        current_params = load_best_rerun_params_dict(alg, exp, auc_or_final, sp)
                        mean_lc, mean_stderr = load_data(alg, exp, current_params, prefix)
                        plot_data(ax, alg, mean_lc, mean_stderr, current_params, exp_attrs, second_time=False,
                                  is_smoothed=is_smoothed, smoothing_window=smoothing_window)
                        if PLOT_RERUN_AND_ORIG:
                            prefix = RERUN_POSTFIX
                            mean_lc, mean_stderr = load_data(alg, exp, current_params, prefix)
                            plot_data(ax, alg, mean_lc, mean_stderr, current_params, exp_attrs, second_time=True,
                                      is_smoothed=is_smoothed, smoothing_window=smoothing_window)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                        pylab.gca().set_rasterized(True)
                    if PLOT_RERUN_AND_ORIG:
                        prefix = '_rerun_and_original'
                    elif PLOT_RERUN:
                        prefix = RERUN_POSTFIX
                    else:
                        prefix = ''
                    fig.savefig(os.path.join(save_dir,
                                f"{prefix}_learning_curve_{'_'.join(alg_names)}{exp}Lmbda{sp}.pdf"),
                                format='pdf', dpi=200, bbox_inches='tight')
                    plt.show()
                    plt.close(fig)
