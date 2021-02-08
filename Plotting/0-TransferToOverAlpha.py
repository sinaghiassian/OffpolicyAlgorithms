import os
import numpy as np
from Registry.AlgRegistry import alg_dict
from utils import create_name_for_save_load
from Plotting.plot_utils import make_params, make_current_params, make_args

args = make_args()

exp_path = os.path.join(os.getcwd(), '../Experiments', args.exp_name)
alg_dir_list = [name for name in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, name))]
alg_dir_list.remove('TDRC')
for alg_name, alg in alg_dict.items():
    if alg_name in alg_dir_list:
        print(alg_name)
        fp_list, sp_list, tp_list, fop_list, res_path = make_params(alg_name, args.exp_name)
        auc_mean_over_alpha = np.zeros(len(fp_list))
        auc_stderr_over_alpha = np.zeros(len(fp_list))
        final_mean_over_alpha = np.zeros(len(fp_list))
        final_stderr_over_alpha = np.zeros(len(fp_list))
        for fop in fop_list:
            for tp in tp_list:
                for sp in sp_list:
                    current_params = make_current_params(alg_name, sp, tp, fop)
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
