import argparse
import json
import numpy as np
import os
from Job.JobBuilder import default_params
from Registry.AlgRegistry import alg_dict

# noinspection SpellCheckingInspection
colors = ['black', "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
          "#17becf"]
color_dict = {alg_name: color for alg_name, color in zip(alg_dict.keys(), colors)}
algs_groups = {'main_algs': ['TD', 'GTD', 'ETD'], 'gradiets': ['GTD', 'GTD2', 'HTD', 'PGTD2', 'TDRC'],
               'emphatics': ['ETD', 'ETDLB'], 'fast_algs': ['TD', 'TB', 'Vtrace', 'ABTD']}


def make_params(alg_name, exp_name):
    params = dict()
    alg_param_names = alg_dict[alg_name].related_parameters()
    res_path = os.path.join(os.getcwd(), '../Results', exp_name, alg_name)
    exp_path = os.path.join(os.getcwd(), '../Experiments', exp_name, alg_name, f'{alg_name}.json')
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


def get_alg_names(exp_name):
    exp_path = os.path.join(os.getcwd(), '../Experiments', exp_name)
    alg_names = [name for name in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, name))]
    return alg_names


def load_sample_json_for_exp(exp_name):
    alg_name = get_alg_names(exp_name)[0]
    exp_path = os.path.join(os.getcwd(), '../Experiments', exp_name, alg_name, f'{alg_name}.json')
    if not os.path.exists(exp_path):
        print('No algorithms exist in the experiment directory...')
        raise FileExistsError
    with open(exp_path) as f:
        json_exp_params = json.load(f)
    return json_exp_params


# def get_attr(exp_name):
#     attr_dict = dict()
#     json_exp_params = load_sample_json_for_exp(exp_name)
#     if exp_name == 'FirstChain':
#         attr_dict = {'y_lim': [0.0, 0.8], 'x_lim': [0.0, json_exp_params['number_of_steps']],
#                      'y_axis_ticks': [0.1, 0.3, 0.5, 0.7], 'x_axis_ticks': [0.0, 10000, 20000],
#                      'x_tick_labels': [0, '10K', '20K'], 'over_limit_replacement': 2.0, 'over_limit_waterfall': 0.79,
#                      'learning_starting_point': 0.6891}
#     elif exp_name == 'FirstFourRoom':
#         attr_dict = {'y_lim': [0.0, 0.8], 'x_lim': [0.0, json_exp_params['number_of_steps']],
#                      'y_axis_ticks': [0.1, 0.3, 0.5, 0.7],
#                      'x_axis_ticks': [0.0, 10000, 20000, 30000, 40000, 50000],
#                      'x_tick_labels': [0, '10K', '20K', '30K', '40K', '50K'], 'over_limit_replacement': 2.0,
#                      'over_limit_waterfall': 0.79, 'learning_starting_point': 0.7268}
#     return attr_dict


class FirstChainAttr:
    def __init__(self, exp_name):
        json_exp_params = load_sample_json_for_exp(exp_name)
        self.size_of_labels = 25
        self.y_lim = [0.0, 0.8]
        self.x_lim = [0.0, json_exp_params['number_of_steps']]
        self.y_axis_ticks = [0.1, 0.3, 0.5, 0.7]
        self.x_axis_ticks = [0.0, 5000, 10000, 15000, 20000]
        self.x_tick_labels = [0, '5K', '10', '15K', '20']
        self.over_limit_replacement = 2.0
        self.over_limit_waterfall = 0.79
        self.learning_starting_point = 0.68910


class FirstFourRoomAttr:
    def __init__(self, exp_name):
        json_exp_params = load_sample_json_for_exp(exp_name)
        self.size_of_labels = 25
        self.y_lim = [0.0, 0.8]
        self.x_lim = [0.0, json_exp_params['number_of_steps']]
        self.y_axis_ticks = [0.1, 0.3, 0.5, 0.7]
        self.x_axis_ticks = [0.0, 10000, 20000, 30000, 40000, 50000]
        self.x_tick_labels = [0, '10', '20', '30', '40', '50']
        self.over_limit_replacement = 2.0
        self.over_limit_waterfall = 0.79
        self.learning_starting_point = 0.72672


class HVFirstFourRoomAttr(FirstChainAttr):
    def __init__(self, exp_name):
        super(HVFirstFourRoomAttr, self).__init__(exp_name)


attr_dict = {'FirstChain': FirstChainAttr, 'FirstFourRoom': FirstFourRoomAttr, '1HVFourRoom': HVFirstFourRoomAttr}


def replace_large_nan_inf(arr, large=1.0, replace_with=2.0):
    arr[np.isnan(arr)], arr[np.isinf(arr)], arr[arr > large] = replace_with, replace_with, replace_with
    return arr
