import os
import numpy as np
import json
import argparse
from Registry.AlgRegistry import alg_dict
from Job.JobBuilder import default_params


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-n', type=str, default='FirstChain')
args = parser.parse_args()

three_param_algoriothms = ['GTD', 'ETDLB']

exp_path = os.path.join(os.getcwd(), 'Experiments', args.exp_name)
alg_dir_list = [name for name in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, name))]
for alg_name, alg in alg_dict.items():
    if alg_name in alg_dir_list:
        params = dict()
        alg_param_names = alg_dict[alg_name].related_parameters()
        alg_num_params = len(alg_param_names)
        exp_path = os.path.join(os.getcwd(), 'Experiments', args.exp_name, alg_name, f'{alg_name}.json')
        with open(exp_path) as f:
            json_exp_params = json.load(f).get('meta_parameters')
        for param in alg_param_names:
            params[param] = json_exp_params.get(param, default_params['meta_parameters'][param])
            # if not isinstance(params[param], list):
            #     params[param] = list([params[param]])
        # param_combinations = it.product(*(params[Name] for Name in alg_param_names))
        res_path = os.path.join(os.getcwd(), 'Results', args.exp_name, alg_name)
        tp_list = [0.0]
        if 'eta' in params:
            tp_list = params['eta']
        elif 'beta' in params:
            tp_list = params['beta']
        for j, sp in enumerate(params.get('lmbda', params['zeta'])):
            for k, tp in enumerate(tp_list):




        # mean_over_all_alpha_auc = defaultdict(list)
        # # stderr_over_all_alpha_auc = dict()
        # # mean_over_all_alpha_final = dict()
        # # stderr_over_all_alpha_final = dict()
        # for element in param_combinations:
        #     element = dict(zip(alg_param_names, element))
        #     other_elements = dict(element)
        #     del other_elements['alpha']
        #     file_name = os.path.join(res_path, create_name_for_save_load(element) + '_mean_stderr_auc.npy')
        #     if not os.path.exists(file_name):
        #         continue
        #     result_auc = np.load(file_name)
        #     mean_over_all_alpha_auc[tuple(other_elements.values())].append(result_auc[0])
        #     # stderr_over_all_alpha_auc.append(result_auc[1])
        #     # file_name = res_path + create_name_for_save_load(element) + '_mean_stderr_final.npy'
        #     # result_final = np.load(file_name)
        #     # mean_over_all_alpha_final.append(result_final[0])
        #     # stderr_over_all_alpha_final.append(result_final[1])
        # print(mean_over_all_alpha_auc)
