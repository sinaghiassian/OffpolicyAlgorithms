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


def find_all_performance_over_alpha(alg_name):
    fp_list, sp_list, tp_list, res_path = make_params(alg_name, args.exp_name)
    all_performance = np.zeros((len(fp_list), len(sp_list), len(tp_list)))
    for j, sp in enumerate(sp_list):
        for k, tp in enumerate(tp_list):
            current_params = make_current_params(alg_name, sp, tp)
            load_file_name = os.path.join(
                res_path, create_name_for_save_load(current_params) + f'_mean_{auc_or_final}_over_alpha.npy')
            all_performance[:, j, k] = np.load(load_file_name)
    return all_performance


def plot_waterfall(all_performance, alg_name):
    global l, xAxisNames, xAxisTicks
    lbl = alg_name
    performance_to_plot = all_performance.flatten()
    plt.scatter([(l + 1)] * performance_to_plot.shape[0] + np.random.uniform(-0.25, 0.25, performance_to_plot.shape[0]),
                performance_to_plot, marker='o', facecolors='none', color='blue')
    l = (l + 1) % methodNames.shape[0]
    xAxisNames.append(methodName)

l = -0.5
xAxisNames = ['']
xAxisTicks = [0]
for alg_name in alg_names:
    all_performance = find_all_performance_over_alpha(alg_name)
    plot_waterfall(all_performance, alg_name)
plt.show()
