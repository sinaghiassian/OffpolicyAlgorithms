import argparse
import json
import numpy as np
import os
from Job.JobBuilder import default_params
from Registry.AlgRegistry import alg_dict
from utils import create_name_for_save_load


def make_res_path(alg, exp):
    return os.path.join(os.getcwd(), 'Results', exp, alg)


def make_exp_path(alg, exp):
    return os.path.join(os.getcwd(), 'Experiments', exp, alg)


def load_best_rerun_params_dict(alg, exp, auc_or_final, sp):
    res_path = make_res_path(alg, exp)
    with open(os.path.join(res_path, f"{auc_or_final}_{sp}.json")) as f:
        return json.load(f)['meta_parameters']


def get_alphas(alg, exp):
    exp_path = make_exp_path(alg, exp)
    exp_path = os.path.join(exp_path, f"{alg}.json")
    with open(exp_path) as f:
        jsn_content = json.load(f)
        return jsn_content['meta_parameters']['alpha']


def load_best_rerun_params(alg, exp, auc_or_final, sp):
    best_res_dict = load_best_rerun_params_dict(alg, exp, auc_or_final, sp)
    best_fp = best_res_dict.get('alpha', 0)
    best_tp = best_res_dict.get('eta', best_res_dict.get('beta', 0))
    best_fop = best_res_dict.get('tdrc_beta', 0)
    return best_fp, best_tp, best_fop


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-n', type=str, default='1HVFourRoom')
    # 1HVFourRoom or FirstFourRoom or FirstChain
    return parser.parse_args()


def rename_best_old_result(res_path, params_dict, file_name):
    name_to_save = create_name_for_save_load(param_dict=params_dict)
    path_and_name = os.path.join(res_path, name_to_save)
    file_name = path_and_name + file_name
    os.rename(file_name + '.npy', file_name + '_old.npy')


def load_best_perf_json(alg, exp, sp, auc_or_final):
    res_path = make_res_path(alg, exp)
    res_path = os.path.join(res_path, f"{auc_or_final}_{sp}.json")
    with open(res_path, 'r') as f:
        return json.load(f)


def load_exp_json_file(alg, exp):
    res_path = make_res_path(alg, exp)
    exp_path = make_exp_path(alg, exp)
    exp_path = os.path.join(exp_path, f'{alg}.json')
    with open(exp_path) as f:
        return json.load(f), res_path


def make_params(alg_name, exp_name):
    params = dict()
    alg_param_names = alg_dict[alg_name].related_parameters()
    json_content, res_path = load_exp_json_file(alg_name, exp_name)
    json_exp_params = json_content.get('meta_parameters')
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
    if alg_name == 'TDRC':
        tp_list, fop_list = [1.0], [1.0]
    return fp_list, sp_list, tp_list, fop_list, res_path


def make_current_params(alg_name, sp, tp, fop, fp=0):
    current_params = {'alpha': fp}
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


def get_alg_names(exp_name):
    path = os.path.join(os.getcwd(), 'Experiments', exp_name)
    alg_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    return alg_names


def load_sample_json_for_exp(exp):
    alg = get_alg_names(exp)[0]
    exp_path = make_exp_path(alg, exp)
    exp_path = os.path.join(exp_path, f'{alg}.json')
    if not os.path.exists(exp_path):
        print('No algorithms exist in the experiment directory...')
        raise FileExistsError
    with open(exp_path) as f:
        json_exp_params = json.load(f)
    return json_exp_params


def load_and_replace_large_nan_inf(load_file_name, large, replace_with):
    current_perf = np.load(load_file_name)
    return replace_large_nan_inf(current_perf, large=large, replace_with=replace_with)


class FirstChainAttr:
    def __init__(self, exp_name):
        json_exp_params = load_sample_json_for_exp(exp_name)
        self.size_of_labels = 25
        self.y_lim = [0.0, 0.8]
        self.x_lim = [0.0, json_exp_params['number_of_steps']]
        self.y_axis_ticks = [0.1, 0.3, 0.5, 0.7]
        self.x_axis_ticks = [0.0, 5000, 10000, 15000, 20000]
        self.x_tick_labels = [0, '5', '10', '15', '20']
        self.x_axis_ticks_log = [pow(2, -18), pow(2, -14), pow(2, -10), pow(2, -6), pow(2, -2)]
        self.x_axis_tick_labels_log = [-16, -13, -10, -7, -4, -1]
        self.over_limit_replacement = 2.0
        self.over_limit_waterfall = 0.79
        self.learning_starting_point = 0.68910
        self.ok_error = 0.4


class FirstFourRoomAttr:
    def __init__(self, exp_name):
        json_exp_params = load_sample_json_for_exp(exp_name)
        self.size_of_labels = 25
        self.y_lim = [0.0, 0.8]
        self.x_lim = [0.0, json_exp_params['number_of_steps']]
        self.y_axis_ticks = [0.1, 0.3, 0.5, 0.7]
        self.x_axis_ticks = [0.0, 10000, 20000, 30000, 40000, 50000]
        self.x_tick_labels = [0, '10', '20', '30', '40', '50']
        self.x_axis_ticks_log = [pow(2, -16), pow(2, -13), pow(2, -10), pow(2, -7), pow(2, -4), pow(2, -1)]
        self.x_axis_tick_labels_log = [-16, -13, -10, -7, -4, -1]
        self.over_limit_replacement = 2.0
        self.over_limit_waterfall = 0.79
        self.learning_starting_point = 0.72672
        self.ok_error = 0.4


class HVFirstFourRoomAttr(FirstFourRoomAttr):
    def __init__(self, exp_name):
        super(HVFirstFourRoomAttr, self).__init__(exp_name)


def replace_large_nan_inf(arr, large=1.0, replace_with=2.0):
    arr[np.isnan(arr)], arr[np.isinf(arr)], arr[arr > large] = replace_with, replace_with, replace_with
    return arr
