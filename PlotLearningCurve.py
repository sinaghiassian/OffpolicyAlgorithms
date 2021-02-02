import argparse
from Registry import TaskRegistry
import matplotlib.pyplot as plt
from pylab import *
import json
from Job.JobBuilder import default_params
import matplotlib.patches as mpatches
import numpy
import os
from Registry.AlgRegistry import alg_dict
from Job.JobBuilder import default_params
from utils import create_name_for_save_load

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-n', type=str, default='FirstChain')
args = parser.parse_args()
alg_names = ['GTD', 'TD']
auc_or_final = 'auc'  # 'final' or 'auc'

fig = plt.figure()
ax = fig.add_subplot(111)


def find_best_mean_performance(alg_name):
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
    best_perf, best_fp, best_sp, best_tp = 1000, 1000, 1000, 1000
    best_params = dict()
    for sp in sp_list:
        for tp in tp_list:
            current_params = {'alpha': 0}
            if 'lmbda' in alg_param_names:
                current_params['lmbda'] = sp
            else:
                current_params['zeta'] = sp
            if 'eta' in alg_param_names:
                current_params['eta'] = tp
            load_file_name = os.path.join(res_path, create_name_for_save_load(current_params, excluded_params=[
                'alpha']) + f'_mean_{auc_or_final}_over_alpha.npy')
            current_perf = np.load(load_file_name)
            min_perf = min(current_perf)
            if min_perf < best_perf:
                best_perf = min_perf
                best_perf_idx = int(np.argmin(current_perf))
                best_fp = fp_list[best_perf_idx]
                best_sp = sp
                best_tp = tp
                best_params = current_params
                best_params['alpha'] = best_fp
    return best_fp, best_sp, best_tp, best_params


def load_data(alg_name, best_params):
    res_path = os.path.join(os.getcwd(), 'Results', args.exp_name, alg_name)
    load_file_name = os.path.join(res_path, create_name_for_save_load(best_params) + '_RMSVE_mean_over_runs.npy')
    mean_lc = np.load(load_file_name)
    load_file_name = os.path.join(res_path, create_name_for_save_load(best_params) + '_RMSVE_stderr_over_runs.npy')
    stderr_lc = np.load(load_file_name)
    return mean_lc, stderr_lc


def plot_data(alg_name, mean_lc, mean_stderr, best_params):
    lbl = (alg_name + r'$\alpha=$' + str(best_params['alpha']))
    ax.plot(np.arange(15000), mean_lc, label=lbl, linewidth=1.0, color='blue')
    ax.fill_between(np.arange(15000), mean_lc - mean_stderr / 2, mean_lc + mean_stderr / 2, alpha=0.3, color='blue')


for alg in alg_names:
    fp, sp, tp, current_params = find_best_mean_performance(alg)
    mean_lc, mean_stderr = load_data(alg, current_params)
    plot_data(alg, mean_lc, mean_stderr, current_params)
    plt.show()
