import os
import numpy as np
import json
import argparse
from Registry.AlgRegistry import alg_dict
from Job.JobBuilder import default_params
from utils import create_name_for_save_load

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-n', type=str, default='FirstChain')
args = parser.parse_args()

exp_path = os.path.join(os.getcwd(), 'Experiments', args.exp_name)
alg_dir_list = [name for name in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, name))]
for alg_name, alg in alg_dict.items():
    if alg_name in alg_dir_list:
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
        auc_mean_over_alpha = np.zeros(len(fp_list))
        auc_stderr_over_alpha = np.zeros(len(fp_list))
        final_mean_over_alpha = np.zeros(len(fp_list))
        final_stderr_over_alpha = np.zeros(len(fp_list))

        for tp in tp_list:
            for sp in sp_list:
                current_params = {'alpha': 0}
                if 'lmbda' in alg_param_names:
                    current_params['lmbda'] = sp
                else:
                    current_params['zeta'] = sp
                if 'eta' in alg_param_names:
                    current_params['eta'] = tp

                for i, fp in enumerate(fp_list):

                    current_params['alpha'] = fp
                    load_file_name = os.path.join(res_path, create_name_for_save_load(current_params) +
                                                  '_mean_stderr_auc.npy')
                    auc_perf = np.load(load_file_name)
                    auc_mean_over_alpha[i] = auc_perf[0]
                    auc_stderr_over_alpha[i] = auc_perf[1]
                    load_file_name = os.path.join(res_path, create_name_for_save_load(current_params) +
                                                  '_mean_stderr_final.npy')
                    final_perf = np.load(load_file_name)
                    final_mean_over_alpha[i] = final_perf[0]
                    final_stderr_over_alpha[i] = final_perf[1]

                save_file_name = os.path.join(res_path, create_name_for_save_load(current_params, excluded_params=[
                    'alpha']) + '_mean_auc_over_alpha')
                np.save(save_file_name, auc_mean_over_alpha)
                save_file_name = os.path.join(res_path, create_name_for_save_load(current_params, excluded_params=[
                    'alpha']) + '_stderr_auc_over_alpha')
                np.save(save_file_name, auc_stderr_over_alpha)
                save_file_name = os.path.join(res_path, create_name_for_save_load(current_params, excluded_params=[
                    'alpha']) + '_mean_final_over_alpha')
                np.save(save_file_name, final_mean_over_alpha)
                save_file_name = os.path.join(res_path, create_name_for_save_load(current_params, excluded_params=[
                    'alpha']) + '_stderr_final_over_alpha')
                np.save(save_file_name, final_stderr_over_alpha)
