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


def find_best_performance_over_alpha(alg_name):
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
    minimum_value = float(inf)
    best_performance_over_alpha = np.zeros(len(fp_list))
    best_sp, best_tp = float(-inf), float(-inf)
    stderr_load_file_name = ''
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
            performance_over_alpha = np.load(load_file_name)
            if min(performance_over_alpha) < minimum_value:
                best_performance_over_alpha = performance_over_alpha
                best_tp = tp
                stderr_load_file_name = os.path.join(
                    res_path, create_name_for_save_load(current_params) + f'_stderr_{auc_or_final}_over_alpha.npy')
    return best_performance_over_alpha, np.load(stderr_load_file_name),  best_tp, np.array(fp_list)


def plot_sensitivity(best_performance, stderr, alg_name, best_tp, fp_list):
    lbl = f'{alg_name}_{best_tp}'
    plt.semilogx(
        fp_list, best_performance, label=lbl, linestyle='-', marker='o', color='blue', linewidth=2, markersize=5)
    plt.errorbar(fp_list, best_performance, yerr=stderr, ecolor='blue', mfc='blue',
                 mec='blue', linestyle='', elinewidth=2, markersize=5)


for alg_name in alg_names:
    best_performance, stderr, best_tp, fp_list = find_best_performance_over_alpha(alg_name)
    plot_sensitivity(best_performance, stderr, alg_name, best_tp, fp_list)
plt.show()
