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


def find_all_performance_over_alpha(alg_name):
    fp_list, sp_list, tp_list, fop_list, res_path = make_params(alg_name, args.exp_name)
    sp_list = [lmbda_or_zeta]
    all_performance = np.zeros((len(fp_list), len(sp_list), len(tp_list), len(fop_list)))
    for m, fop in enumerate(fop_list):
        for k, tp in enumerate(tp_list):
            for j, sp in enumerate(sp_list):
                current_params = make_current_params(alg_name, sp, tp, fop)
                load_file_name = os.path.join(
                    res_path, create_name_for_save_load(
                        current_params, excluded_params=['alpha']) + f'_mean_{auc_or_final}_over_alpha.npy')
                all_performance[:, j, k, m] = np.load(load_file_name)
    return all_performance


def plot_waterfall(all_performance, alg_name):
    global ticker, xAxisNames, xAxisTicks
    performance_to_plot = all_performance.flatten()
    xAxisTicks.append(ticker + 1)
    plt.scatter([(ticker + 1)] * performance_to_plot.shape[0] + np.random.uniform(
        -0.25, 0.25, performance_to_plot.shape[0]), performance_to_plot, marker='o',
                facecolors='none', color=color_dict[alg_name])
    ticker = (ticker + 1) % len(alg_names)
    xAxisNames.append(alg_name)


ticker = -0.5
xAxisNames = ['']
xAxisTicks = [0]
for alg_name in alg_names:
    all_performance = find_all_performance_over_alpha(alg_name)
    plot_waterfall(all_performance, alg_name)
ax.xaxis.set_ticks(xAxisTicks)
ax.set_xticklabels(xAxisNames)
ax.get_xaxis().tick_bottom()
fig.savefig('_'.join(alg_names) + '_waterfall.pdf', format='pdf', dpi=1000, bbox_inches='tight')
plt.ylim(0.0, 0.7)
plt.show()
