import matplotlib.pyplot as plt
import numpy as np
import os
import pylab
from Plotting.plot_params import ALG_GROUPS, ALG_COLORS, EXP_ATTRS, EXPS, AUC_AND_FINAL, LMBDA_AND_ZETA, \
    PLOT_RERUN_AND_ORIG, RERUN, RERUN_POSTFIX
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


def plot_data(ax, alg, mean_lc, mean_stderr, best_params, exp_attrs, second_time=False):
    alpha = 1.0
    if PLOT_RERUN_AND_ORIG:
        alpha = 1.0 if second_time else 0.5
    lbl = (alg + r'$\alpha=$ ' + str(best_params['alpha']))
    ax.plot(np.arange(mean_lc.shape[0]), mean_lc, label=lbl, linewidth=1.0, color=ALG_COLORS[alg], alpha=alpha)
    ax.fill_between(np.arange(mean_lc.shape[0]), mean_lc - mean_stderr / 2, mean_lc + mean_stderr / 2,
                    color=ALG_COLORS[alg], alpha=0.1*alpha)
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
                        postfix = RERUN_POSTFIX if RERUN else ''
                        current_params = load_best_rerun_params_dict(alg, exp, auc_or_final, sp)
                        print(alg, current_params)
                        mean_lc, mean_stderr = load_data(alg, exp, current_params, postfix)
                        plot_data(ax, alg, mean_lc, mean_stderr, current_params, exp_attrs)
                        if PLOT_RERUN_AND_ORIG:
                            postfix = RERUN_POSTFIX
                            mean_lc, mean_stderr = load_data(alg, exp, current_params, postfix)
                            plot_data(ax, alg, mean_lc, mean_stderr, current_params, exp_attrs, True)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                        pylab.gca().set_rasterized(True)
                        if PLOT_RERUN_AND_ORIG:
                            postfix = '_rerun_and_original'
                        elif RERUN:
                            postfix = RERUN_POSTFIX
                        else:
                            postfix = ''
                        fig.savefig(os.path.join(save_dir, f"learning_curve_{'_'.join(alg_names)}{postfix}.pdf"),
                                    format='pdf', dpi=200, bbox_inches='tight')
                    plt.show()
