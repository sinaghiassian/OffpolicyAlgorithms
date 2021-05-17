import os
import matplotlib.pyplot as plt
import numpy as np

from Plotting.plot_params import EXPS, ALG_GROUPS, ALG_COLORS, EXP_ATTRS, AUC_AND_FINAL, LMBDA_AND_ZETA, PLOT_RERUN, \
    RERUN_POSTFIX
from Plotting.plot_utils import make_current_params, replace_large_nan_inf, make_params
from utils import create_name_for_save_load

np.random.seed(0)
def load_all_performances(alg, exp, auc_or_final, sp, exp_attrs):
    fp_list, sp_list, tp_list, fop_list, res_path = make_params(alg, exp)
    all_performance = np.zeros((len(fp_list), len(tp_list), len(fop_list)))
    for i, fop in enumerate(fop_list):
        for j, tp in enumerate(tp_list):
            current_params = make_current_params(alg, sp, tp, fop)
            load_file_name = os.path.join(res_path, create_name_for_save_load(
                current_params, excluded_params=['alpha']) + f'_mean_{auc_or_final}_over_alpha.npy')

            if PLOT_RERUN and auc_or_final == 'auc':
                load_file_name_rerun = load_file_name.replace('.npy', f"{RERUN_POSTFIX}.npy")
                if os.path.isfile(load_file_name_rerun):
                    load_file_name = load_file_name_rerun

            performance = np.load(load_file_name)
            performance = replace_large_nan_inf(performance, large=exp_attrs.learning_starting_point,
                                                replace_with=exp_attrs.over_limit_waterfall)
            all_performance[:, j, i] = performance
    return all_performance


def plot_waterfall(ax, alg, all_performance, alg_names, exp_attrs):
    global ticker, x_axis_names, x_axis_ticks
    performance_to_plot = np.array(all_performance.flatten())
    percentage_overflowed = round((performance_to_plot > exp_attrs.learning_starting_point).sum() /
                                  performance_to_plot.size, 2)
    ok_percentage = round((performance_to_plot < exp_attrs.ok_error).sum() /
                          performance_to_plot.size, 2)
    print(alg, 'percentage_overflowed', percentage_overflowed)
    # print(alg, 'OK_percentage', ok_percentage)
    color = ALG_COLORS[alg]
    ax.scatter([(ticker + 1)] * performance_to_plot.shape[0] + np.random.uniform(
        -0.25, 0.25, performance_to_plot.shape[0]), performance_to_plot, marker='o',
                facecolors='none', color=color)
    x_axis_ticks.append(ticker + 1)
    ticker = (ticker + 1) % len(alg_names)
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=True)  # labels along the bottom edge are off
    x_axis_names.append(f'{alg}_{percentage_overflowed}')
    ax.xaxis.set_ticks(x_axis_ticks)
    ax.set_xticklabels(x_axis_names)
    ax.get_yaxis().tick_left()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', which='major', labelsize=exp_attrs.size_of_labels)
    ax.set_ylim(exp_attrs.y_lim)
    ax.yaxis.set_ticks(exp_attrs.y_axis_ticks)
    ax.set_yticklabels([])
    ax.set_xticklabels([])


ticker, x_axis_names, x_axis_ticks = 0.0, [''], [0]


def plot_waterfall_scatter(**kwargs):
    for exp in kwargs['exps']:
        exp_attrs = EXP_ATTRS[exp](exp)
        for auc_or_final in kwargs['auc_or_final']:
            for sp in kwargs['sp_list']:
                save_dir = os.path.join('pdf_plots', 'waterfalls', auc_or_final)
                for alg_names in kwargs['alg_groups'].values():
                    global ticker, x_axis_names, x_axis_ticks
                    ticker, x_axis_names, x_axis_ticks = -0.5, [''], [0]
                    fig, ax = plt.subplots(kwargs['fig_size'])
                    for alg in alg_names:
                        if alg in ['LSTD', 'LSETD']:
                            continue
                        all_performance = load_all_performances(alg, exp, auc_or_final, sp, exp_attrs)
                        plot_waterfall(ax, alg, all_performance, alg_names, exp_attrs)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    prefix = RERUN_POSTFIX if PLOT_RERUN else ''
                    fig.savefig(os.path.join(save_dir,
                                             f"{prefix}_waterfall_curve_{'_'.join(alg_names)}{exp}Lmbda{sp}.pdf"),
                                format='pdf', dpi=1000, bbox_inches='tight')
                    plt.show()
                    print(exp, alg_names, auc_or_final, sp)
