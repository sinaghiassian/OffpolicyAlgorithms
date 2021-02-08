import matplotlib.pyplot as plt
import argparse
import json
import os
from Job.JobBuilder import default_params
from Registry.AlgRegistry import alg_dict

colors = ['black', 'orchid', 'orange', 'blue', 'grey', 'red', 'green', 'darkred', 'darkkhaki', 'skyblue', 'aqua',
          'lime', 'cadetblue']
color_dict = {alg_name: color for alg_name, color in zip(alg_dict.keys(), colors)}


def make_params(alg_name, exp_name):
    params = dict()
    alg_param_names = alg_dict[alg_name].related_parameters()
    exp_path = os.path.join(os.getcwd(), '../Experiments', exp_name, alg_name, f'{alg_name}.json')
    res_path = os.path.join(os.getcwd(), '../Results', exp_name, alg_name)
    if not os.path.exists(exp_path):
        return [], [], [], res_path
    with open(exp_path) as f:
        json_exp_params = json.load(f).get('meta_parameters')
    for param in alg_param_names:
        params[param] = json_exp_params.get(param, default_params['meta_parameters'][param])
        if not isinstance(params[param], list):
            params[param] = list([params[param]])
    fp_list = params.get('alpha', params['alpha'])
    tp_list = [0.0]
    fop_list = [0.0]
    if 'lmbda' in params:
        sp_list = params['lmbda']
    else:
        sp_list = params['zeta']
    if 'eta' in params:
        tp_list = params['eta']
    elif 'beta' in params:
        tp_list = params['beta']
    if 'tdrc_beta' in params:
        fop_list = params['tdrc_beta']
    return fp_list, sp_list, tp_list, fop_list, res_path


def make_current_params(alg_name, sp, tp, fop):
    current_params = {'alpha': 0}
    alg_param_names = alg_dict[alg_name].related_parameters()
    if 'lmbda' in alg_param_names:
        current_params['lmbda'] = sp
    else:
        current_params['zeta'] = sp
    if 'eta' in alg_param_names:
        current_params['eta'] = tp
    elif 'beta' in alg_param_names:
        current_params['beta'] = tp
    if 'tdrc_beta' in alg_param_names:
        current_params['tdrc_beta'] = fop
    return current_params


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-n', type=str, default='FirstChain')
    return parser.parse_args()


def make_fig():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return fig, ax


def get_alg_names(exp_name):
    exp_path = os.path.join(os.getcwd(), '../Experiments', exp_name)
    alg_names = [name for name in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, name))]
    alg_names.remove('TDRC')
    return alg_names
