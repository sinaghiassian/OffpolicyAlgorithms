import argparse
from pylab import *
import json
import os
from Registry.AlgRegistry import alg_dict
from Job.JobBuilder import default_params
from utils import create_name_for_save_load

fig = plt.figure()
ax = fig.add_subplot(111)
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-n', type=str, default='FirstChain')
args = parser.parse_args()
alg_names = ['GTD']
auc_or_final = 'auc'  # 'final' or 'auc'
lmbda = 0  # 0 or 0.9


def find_all_performance_over_alpha(alg_name):
    params = dict()
    alg_param_names = alg_dict[alg_name].related_parameters()
    exp_path = os.path.join(os.getcwd(), 'Experiments', args.exp_name, alg_name, f'{alg_name}.json')
    with open(exp_path) as f:
        json_exp_params = json.load(f).get('meta_parameters')
    for param in alg_param_names:
        params[param] = json_exp_params.get(param, default_params['meta_parameters'][param])
        if not isinstance(params[param], list):
            params[param] = list([params[param]])
    res_path = os.path.join(os.getcwd(), 'Results', args.exp_name, alg_name)
    fp_list = params.get('alpha', params['alpha'])
    tp_list = [0.0]
    if 'eta' in params:
        tp_list = params['eta']
    elif 'beta' in params:
        tp_list = params['beta']
    if 'lmbda' in params:
        sp_list = params['lmbda']
    else:
        sp_list = params['zeta']
    all_performance = np.zeros((len(fp_list), len(sp_list), len(tp_list)))
    for j, sp in enumerate(sp_list):
        for k, tp in enumerate(tp_list):
            current_params = {}
            if 'lmbda' in alg_param_names:
                current_params['lmbda'] = sp
            else:
                current_params['zeta'] = sp
            if 'eta' in alg_param_names:
                current_params['eta'] = tp
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


fig = plt.figure()
ax = fig.add_subplot(111)
l = -0.5
xAxisNames = ['']
xAxisTicks = [0]
for alg_name in alg_names:
    all_performance = find_all_performance_over_alpha(alg_name)
    plot_waterfall(all_performance, alg_name)
plt.show()
