import json
import os

import numpy as np

from Learning import learn
from Plotting.plot_params import EXP_ATTRS, PLOT_RERUN, PLOT_RERUN_AND_ORIG
from Plotting.plot_utils import make_params, make_current_params, load_and_replace_large_nan_inf, \
    load_best_perf_json, load_best_rerun_params, make_res_path
from utils import create_name_for_save_load, Configuration


def save_perf_over_alpha(alg, exp, auc_or_final, sp, rerun=False):
    fp_list, sp_list, tp_list, fop_list, _ = make_params(alg, exp)
    res_path = make_res_path(alg, exp)
    mean_over_alpha, stderr_over_alpha = np.zeros(len(fp_list)), np.zeros(len(fp_list))
    best_fp, best_tp, best_fop = load_best_rerun_params(alg, exp, auc_or_final, sp) if rerun else (0, 0, 0)
    for tp in tp_list:
        for fop in fop_list:
            current_params = make_current_params(alg, sp, tp, fop)
            for i, fp in enumerate(fp_list):
                current_params['alpha'] = fp
                load_name = os.path.join(res_path, create_name_for_save_load(current_params))
                perf = np.load(f"{load_name}_mean_stderr_{auc_or_final}.npy")
                if rerun and fp == best_fp and tp == best_tp and fop == best_fop:
                    perf = np.load(f"{load_name}_mean_stderr_{auc_or_final}_rerun.npy")

                mean_over_alpha[i], stderr_over_alpha[i] = perf[0], perf[1]

            save_name = os.path.join(res_path, create_name_for_save_load(current_params, excluded_params=['alpha']))
            postfix = ''
            if rerun and tp == best_tp and fop == best_fop:
                postfix = '_rerun'
            np.save(f"{save_name}_mean_{auc_or_final}_over_alpha{postfix}", mean_over_alpha)
            np.save(f"{save_name}_stderr_{auc_or_final}_over_alpha{postfix}", stderr_over_alpha)


def find_best_perf(alg, exp, auc_or_final, sp):
    exp_attrs = EXP_ATTRS[exp](exp)
    fp_list, _, tp_list, fop_list, res_path = make_params(alg, exp)
    best_params = {}
    best_perf, best_fp, best_sp, best_tp, best_fop = np.inf, np.inf, np.inf, np.inf, np.inf
    for fop in fop_list:
        for tp in tp_list:
            current_params = make_current_params(alg, sp, tp, fop)
            load_name = os.path.join(res_path, create_name_for_save_load(current_params, excluded_params=[
                'alpha']) + f'_mean_{auc_or_final}_over_alpha.npy')
            current_perf = load_and_replace_large_nan_inf(
                load_name, large=exp_attrs.learning_starting_point, replace_with=exp_attrs.over_limit_replacement)
            min_perf = min(current_perf)
            if min_perf < best_perf:
                best_perf = min_perf
                best_perf_idx = int(np.nanargmin(current_perf))
                best_fp = fp_list[best_perf_idx]
                best_params = current_params
                best_params['alpha'] = best_fp
    return best_params


def save_best_perf_in_json(alg, exp, best_params, auc_or_final, sp):
    fp_list, _, tp_list, fop_list, res_path = make_params(alg, exp)
    exp_path = res_path.replace('Results', 'Experiments')
    json_exp = os.path.join(exp_path, f"{alg}.json")
    with open(json_exp, 'r') as f:
        json_exp = json.load(f)
    json_exp['meta_parameters'] = best_params
    save_name = os.path.join(res_path, f"{auc_or_final}_{sp}.json")
    with open(save_name, 'wt') as f:
        json.dump(json_exp, f, indent=4)


def run_learning_with_best_perf(alg, exp, auc_or_final, sp):
    res_path = os.path.join(os.getcwd(), 'Results', exp, alg)
    best_perf_jsn = load_best_perf_json(alg, exp, sp, auc_or_final)
    param_dict = best_perf_jsn['meta_parameters']
    param_dict['algorithm'] = alg
    param_dict['task'] = best_perf_jsn['task']
    param_dict['environment'] = best_perf_jsn['environment']
    param_dict['num_steps'] = best_perf_jsn['number_of_steps']
    param_dict['num_of_runs'] = best_perf_jsn['number_of_runs']
    param_dict['sub_sample'] = best_perf_jsn['sub_sample']
    param_dict['save_path'] = res_path
    param_dict['save_value_function'] = False
    param_dict['rerun'] = True
    param_dict['render'] = False
    config = Configuration(param_dict)
    learn(config)


def process_data(**kwargs):
    for exp in kwargs['exps']:
        for alg in kwargs['algs']:
            for auc_or_final in kwargs['auc_or_final']:
                for sp in kwargs['sp_list']:
                    print(f"\nStarted re-running {exp}, {alg} lmbda_or_zeta: {sp}, {auc_or_final} ...")
                    save_perf_over_alpha(alg, exp, auc_or_final, sp)
                    best_params = find_best_perf(alg, exp, auc_or_final, sp)
                    if not PLOT_RERUN or not PLOT_RERUN_AND_ORIG:
                        save_best_perf_in_json(alg, exp, best_params, auc_or_final, sp)
                        run_learning_with_best_perf(alg, exp, auc_or_final, sp)
                        save_perf_over_alpha(alg, exp, auc_or_final, sp, rerun=True)
                        print(f"Finished re-running {exp}, {alg} {best_params}")
                    print(f"Finished processing {exp}, {alg} {best_params}")
