import matplotlib.pyplot as plt
import numpy as np
import os

from Plotting.plot_params import exp_names, algs_groups
from Plotting.plot_utils import make_params, make_current_params, replace_large_nan_inf, attr_dict
from utils import create_name_for_save_load

auc_or_final = 'auc'  # 'final' or 'auc'
lmbda_or_zeta = 0.0  # 0 or 0.9


def find_all_performance_over_alpha(alg_name):
    fp_list, sp_list, tp_list, fop_list, res_path = make_params(alg_name, exp)
    if alg_name == 'TDRC':
        tp_list = [1.0]
        fop_list = [1.0]
    if alg_name == 'GTD' or alg_name == 'PGTD2' or alg_name == 'GTD2' or alg_name == 'HTD':
        tp_list = [1.0]
    sp_list = [lmbda_or_zeta]
    all_performance = np.zeros((len(fp_list), len(sp_list), len(tp_list), len(fop_list)))
    for m, fop in enumerate(fop_list):
        for k, tp in enumerate(tp_list):
            for j, sp in enumerate(sp_list):
                current_params = make_current_params(alg_name, sp, tp, fop)
                load_file_name = os.path.join(
                    res_path, create_name_for_save_load(
                        current_params, excluded_params=['alpha']) + f'_mean_{auc_or_final}_over_alpha.npy')
                performance = np.load(load_file_name)
                performance = replace_large_nan_inf(performance, large=attrs.learning_starting_point,
                                                    replace_with=attrs.over_limit_waterfall)
                all_performance[:, j, k, m] = performance
    return all_performance


# noinspection PyUnresolvedReferences
def plot_waterfall(all_performance, alg_name):
    global ticker, xAxisNames, xAxisTicks
    performance_to_plot = all_performance.flatten()
    percentage_overflowed = round((performance_to_plot > attrs.learning_starting_point).sum() /
                                  performance_to_plot.size, 2)
    xAxisTicks.append(ticker + 1)
    plt.scatter([(ticker + 1)] * performance_to_plot.shape[0] + np.random.uniform(
        -0.25, 0.25, performance_to_plot.shape[0]), performance_to_plot, marker='o',
                facecolors='none', color=color_dict[alg_name])
    ticker = (ticker + 1) % len(alg_names)
    xAxisNames.append(f'{alg_name}_{percentage_overflowed}')


for exp in exp_names:
    attrs = attr_dict[exp](exp)
    save_dir = os.path.join('pdf_plots', exp, f'Lmbda{lmbda_or_zeta}_{auc_or_final}')
    for alg_names in algs_groups.values():
        fig, ax = plt.subplots()
        ticker = -0.5
        xAxisNames = ['']
        xAxisTicks = [0]
        for alg_name in alg_names:
            all_performance = find_all_performance_over_alpha(alg_name)
            plot_waterfall(all_performance, alg_name)
        ax.xaxis.set_ticks(xAxisTicks)
        ax.set_xticklabels(xAxisNames)
        ax.get_yaxis().tick_left()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', which='major', labelsize=attrs.size_of_labels)
        ax.set_ylim(attrs.y_lim)
        ax.yaxis.set_ticks(attrs.y_axis_ticks)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_dir + '/waterfall_' + '_'.join(alg_names) + '.pdf',
                    format='pdf', dpi=1000, bbox_inches='tight')
        plt.ylim(0.0, 0.8)
        plt.show()
