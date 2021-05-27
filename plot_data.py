from Plotting.plot_all_sensitivities_per_alg_gradients import plot_all_sensitivities_per_alg_gradients
from Plotting.plot_all_sensitivities_per_alg_gradients_all_eta import plot_all_sensitivities_per_alg_gradients_all_eta
from Plotting.plot_dist import plot_distribution, plot_dist_for_two_four_room_tasks
from Plotting.plot_all_sensitivities_per_alg_emphatics import plot_all_sensitivities_per_alg_emphatics
from Plotting.plot_learning_curve import plot_learning_curve
from Plotting.plot_learning_curves_for_all_third_params import plot_all_learning_curves_for_third
from Plotting.plot_learning_for_two_lambdas import plot_learning_curve_for_lambdas
from Plotting.plot_sensitivity import plot_sensitivity_curve
from Plotting.plot_sensitivity_for_two_lambdas import plot_sensitivity_for_lambdas
from Plotting.plot_specific_learning_curves import plot_specific_learning_curves
from Plotting.plot_waterfall import plot_waterfall_scatter
from Plotting.process_state_value_function import plot_all_final_value_functions, plot_value_functions
from process_data import process_data


func_to_run = 'collision_gradients_sensitivity_full_bootstrap'
exps = ['FirstChain']

# region process data
if func_to_run == 'process_data':
    exps = ['1HVFourRoom']
    algs = ['TD', 'GTD', 'GTD2', 'PGTD2', 'HTD', 'TDRC', 'ETD', 'ETDLB', 'TB', 'Vtrace', 'ABTD']
    auc_or_final = ['auc', 'final']
    sp_list = [0.1, 0.2, 0.3, 0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375]
    # sp_list = [0.1, 0.2, 0.3]
    process_data(exps=exps, algs=algs, auc_or_final=auc_or_final, sp_list=sp_list)
# endregion

# region learning curves
if func_to_run == 'collision_learning_curves_for_all_extra_params_full_bootstrapping':
    algs = ['PGTD2', 'GTD', 'LSTD']
    sp_list = [0.0]
    fig_size = (10, 4)
    auc_or_final = ['auc']
    # tp_list = [0.015625, 0.0625, 0.25, 1.0, 4.0, 16.0, 64.0, 256.0]
    tp_list = [0.25]
    plot_all_learning_curves_for_third(exps=exps, algs=algs, sp_list=sp_list, fig_size=fig_size,
                                       auc_or_final=auc_or_final, tp_list=tp_list)
if func_to_run == 'collision_learning_curve_for_two_lambdas':
    sp_list = [0.0, 0.9]
    fig_size = (6, 4)
    alg_groups = {'all_algs': ['TD', 'GTD', 'GTD2', 'PGTD2', 'HTD', 'TDRC', 'ETD', 'ETDLB', 'TB', 'Vtrace', 'ABTD']}
    auc_or_final = ['auc']
    plot_learning_curve_for_lambdas(exps=exps, alg_groups=alg_groups, sp_list=sp_list, fig_size=fig_size,
                                    auc_or_final=auc_or_final)
if func_to_run == 'collision_best_learning_curves_full_bootstrap':
    sp_list = [0.0]
    fig_size = (10, 4)
    alg_groups = {'main_algs': ['TD', 'GTD', 'ETD', 'LSTD', 'LSETD'],
                  'gradients': ['GTD', 'GTD2', 'HTD', 'PGTD2', 'TDRC', 'LSTD'],
                  'emphatics': ['ETD', 'ETDLB', 'LSETD'],
                  'fast_algs': ['TD', 'TB', 'Vtrace', 'ABTD', 'LSTD'],
                  'all_algs': ['TD', 'GTD', 'GTD2', 'PGTD2', 'HTD', 'TDRC', 'ETD', 'ETDLB', 'TB', 'Vtrace', 'ABTD',
                               'LSTD', 'LSETD']}
    auc_or_final = ['auc']
    plot_learning_curve(exps=exps, alg_groups=alg_groups, sp_list=sp_list, fig_size=fig_size, auc_or_final=auc_or_final)
if func_to_run == 'collision_best_learning_curves_some_algs_full_bootstrap':
    sp_list = [0.0]
    fig_size = (6, 4)
    alg_groups = {'all_algs': ['TD', 'PGTD2', 'HTD', 'ETD', 'TB', 'Vtrace', 'ABTD']}
    auc_or_final = ['auc']
    plot_learning_curve(exps=exps, alg_groups=alg_groups, sp_list=sp_list, fig_size=fig_size, auc_or_final=auc_or_final,
                        is_smoothed=True, smoothing_window=1)
if func_to_run == 'collision_best_learning_curves_some_algs_medium_bootstrap':
    sp_list = [0.5]
    fig_size = (6, 4)
    alg_groups = {'all_algs': ['TD', 'PGTD2', 'HTD', 'ETD', 'TB', 'Vtrace', 'ABTD']}
    auc_or_final = ['auc']
    plot_learning_curve(exps=exps, alg_groups=alg_groups, sp_list=sp_list, fig_size=fig_size, auc_or_final=auc_or_final,
                        is_smoothed=True, smoothing_window=500)
if func_to_run == 'collision_best_learning_curves_some_algs_minimal_bootstrap':
    sp_list = [0.9]
    fig_size = (6, 4)
    alg_groups = {'all_algs': ['TD', 'PGTD2', 'HTD', 'ETD', 'TB', 'Vtrace', 'ABTD']}
    auc_or_final = ['auc']
    plot_learning_curve(exps=exps, alg_groups=alg_groups, sp_list=sp_list, fig_size=fig_size, auc_or_final=auc_or_final,
                        is_smoothed=True, smoothing_window=500)
if func_to_run == 'collision_best_learning_curves_some_algs_no_bootstrap':
    sp_list = [1.0]
    fig_size = (6, 4)
    alg_groups = {'all_algs': ['TD', 'PGTD2', 'HTD', 'ETD', 'TB', 'Vtrace', 'ABTD']}
    auc_or_final = ['auc']
    plot_learning_curve(exps=exps, alg_groups=alg_groups, sp_list=sp_list, fig_size=fig_size, auc_or_final=auc_or_final,
                        is_smoothed=True, smoothing_window=500)
if func_to_run == 'collision_best_learning_curves_full_bootstrap_rerun_and_original':  # also need to set PLOT_RERUN = False
    # and PLOT_RERUN_AND_ORIG = True in plot_params. Also some changes are necessary in the plot_learning_curve function
    # like setting the colors and stuff for the re-run and original plots.
    sp_list = [0.0]
    fig_size = (10, 4)
    alg_groups = {'all_algs': ['GTD']}
    auc_or_final = ['final']
    plot_learning_curve(exps=exps, alg_groups=alg_groups, sp_list=sp_list, fig_size=fig_size, auc_or_final=auc_or_final)
if func_to_run == 'specific_learning_curves_full_bootstrap':
    auc_or_final = ['auc']
    fig_size = (10, 4)
    sp = 0.0
    if 'FirstChain' in exps:
        exp = 'FirstChain'
        algs = ['ETD', 'TD', 'GTD', 'TDRC', 'PGTD2']
        specific_params = {
            'TD': {'alpha': 0.03125, 'lmbda': sp},
            'ETD': {'alpha': 0.00390625, 'lmbda': sp},
            'TDRC': {'alpha': 0.0625, 'lmbda': sp, 'eta': 4.0, 'tdrc_beta': 0.01},
            'GTD': {'alpha': 0.000976562, 'lmbda': sp, 'eta': 16.0},
            'PGTD2': {'alpha': 0.0078125, 'lmbda': sp, 'eta': 16.0}
        }
        plot_specific_learning_curves(exp=exp, algs=algs, sp=sp, fig_size=fig_size, auc_or_final=auc_or_final,
                                      specific_params=specific_params)
    if 'FirstFourRoom' in exps:
        exp = 'FirstFourRoom'
        algs = ['LSTD', 'LSETD', 'ETD', 'TD', 'GTD2', 'TDRC', 'PGTD2']
        specific_params = {
            'TD': {'alpha': 0.25, 'lmbda': sp},
            'ETD': {'alpha': 0.00390625, 'lmbda': sp},
            'ETDLB': {'alpha': 0.000488281, 'lmbda': sp, 'beta': 0.2},
            'TDRC': {'alpha': 0.0625, 'lmbda': sp, 'eta': 1.0, 'tdrc_beta': 1.0},
            'GTD2': {'alpha': 0.0078125, 'lmbda': sp, 'eta': 16.0},
            'PGTD2': {'alpha': 0.0078125, 'lmbda': sp, 'eta': 16.0}
        }
        plot_specific_learning_curves(exp=exp, algs=algs, sp=sp, fig_size=fig_size, auc_or_final=auc_or_final,
                                      specific_params=specific_params)

    if '1HVFourRoom' in exps:
        exp = '1HVFourRoom'
        algs = ['LSTD', 'LSETD', 'ETDLB', 'TD', 'GTD', 'TDRC', 'PGTD2']
        specific_params = {
            'TD': {'alpha': 0.25, 'lmbda': sp},
            'ETDLB': {'alpha': 0.000488281, 'lmbda': sp, 'beta': 0.2},
            'TDRC': {'alpha': 0.0625, 'lmbda': sp, 'eta': 1.0, 'tdrc_beta': 1.0},
            'GTD': {'alpha': 0.0078125, 'lmbda': sp, 'eta': 16.0},
            'PGTD2': {'alpha': 0.0078125, 'lmbda': sp, 'eta': 16.0}
        }
        plot_specific_learning_curves(exp=exp, algs=algs, sp=sp, fig_size=fig_size, auc_or_final=auc_or_final,
                                      specific_params=specific_params)
# endregion

# region sensitivity curves
if func_to_run == 'collision_TDRC_all_eta_one_beta':
    sp_list = [0.0]
    tdrc_beta = [0.01]  # possible values are 0.1, 0.01, 1.0. Set them separately to plot.
    fig_size = (10, 6)
    algs = ['TDRC']
    auc_or_final = ['auc']
    plot_all_sensitivities_per_alg_gradients_all_eta(exps=exps, algs=algs, sp_list=sp_list, fig_size=fig_size,
                                                     auc_or_final=auc_or_final, tdrc_beta=tdrc_beta)
if func_to_run == 'collision_sensitivity_curves_for_two_lambdas':
    sp_list = [0.0, 0.9]
    fig_size = (6, 4)
    algs = ['TD', 'GTD', 'GTD2', 'PGTD2', 'HTD', 'TDRC', 'ETD', 'ETDLB', 'TB', 'Vtrace', 'ABTD']
    auc_or_final = ['auc']
    plot_sensitivity_for_lambdas(exps=exps, algs=algs, sp_list=sp_list, fig_size=fig_size,
                                 auc_or_final=auc_or_final)
if func_to_run == 'collision_sensitivity_curves_for_many_lambdas':
    sp_list = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 1.0]
    fig_size = (10, 4)
    algs = ['TD', 'GTD', 'GTD2', 'PGTD2', 'HTD', 'TDRC', 'ETD', 'ETDLB', 'TB', 'Vtrace', 'ABTD']
    # algs = ['TB', 'Vtrace', 'ABTD']
    auc_or_final = ['auc']
    plot_sensitivity_for_lambdas(exps=exps, algs=algs, sp_list=sp_list, fig_size=fig_size,
                                 auc_or_final=auc_or_final)
if func_to_run == 'collision_best_sensitivity_curves_full_bootstrapping' or 'collision_waterfall_full_bootstrap':
    sp_list = [0.0]
    fig_size = (10, 4)
    alg_groups = {'main_algs': ['TD', 'GTD', 'ETD'],
                  'gradients': ['GTD', 'GTD2', 'HTD', 'PGTD2', 'TDRC'],
                  'emphatics': ['ETD', 'ETDLB'],
                  'fast_algs': ['TD', 'TB', 'Vtrace', 'ABTD'],
                  'all_algs': ['TD', 'GTD', 'GTD2', 'PGTD2', 'HTD', 'TDRC', 'ETD', 'ETDLB', 'TB', 'Vtrace', 'ABTD']}
    auc_or_final = ['auc']
    if func_to_run == 'collision_best_sensitivity_curves_full_bootstrapping':
        plot_sensitivity_curve(exps=exps, alg_groups=alg_groups, sp_list=sp_list, fig_size=fig_size,
                               auc_or_final=auc_or_final)
    elif func_to_run == 'collision_waterfall_full_bootstrap':
        plot_waterfall_scatter(exps=exps, alg_groups=alg_groups, sp_list=sp_list, fig_size=fig_size,
                               auc_or_final=auc_or_final)
if func_to_run == 'collision_gradients_sensitivity_full_bootstrap':
    sp_list = [0.0]
    fig_size = (11, 4)
    algs = ['GTD', 'GTD2', 'PGTD2', 'HTD']
    auc_or_final = ['auc']
    plot_all_sensitivities_per_alg_gradients(exps=exps, algs=algs, sp_list=sp_list, fig_size=fig_size,
                                             auc_or_final=auc_or_final)
if func_to_run == 'collision_gradients_sensitivity_full_bootstrap_all_eta':
    sp_list = [0.0]
    fig_size = (10, 6)
    algs = ['GTD', 'GTD2', 'PGTD2', 'HTD']
    auc_or_final = ['auc']
    plot_all_sensitivities_per_alg_gradients_all_eta(exps=exps, algs=algs, sp_list=sp_list, fig_size=fig_size,
                                                     auc_or_final=auc_or_final)
if func_to_run == 'collision_emphatics_sensitivity_full_bootstrap':
    sp_list = [0.0]
    fig_size = (11, 5)
    auc_or_final = ['auc']
    plot_all_sensitivities_per_alg_emphatics(exps=exps, sp_list=sp_list, fig_size=fig_size, auc_or_final=auc_or_final)
if func_to_run == 'collision_emphatics_sensitivity_minimal_bootstrap':
    sp_list = [0.9]
    fig_size = (6, 4)
    auc_or_final = ['auc']
    plot_all_sensitivities_per_alg_emphatics(exps=exps, sp_list=sp_list, fig_size=fig_size, auc_or_final=auc_or_final)
# endregion

# region Misc
if func_to_run == 'plot_value_functions':
    plot_value_functions()
if func_to_run == 'plot_all_final_value_functions':
    plot_all_final_value_functions()
if func_to_run == 'state_dist':
    fig_size = (6, 4)
    tasks = ['EightStateCollision', 'LearnEightPoliciesTileCodingFeat',
             'HighVarianceLearnEightPoliciesTileCodingFeat']
    for task in tasks:
        plot_distribution(task=task, fig_size=fig_size)
if func_to_run == 'high_variance_and_normal_dist_comparison':
    fig_size = (6, 4)
    plot_dist_for_two_four_room_tasks(fig_size=fig_size)
# endregion

# from Plotting.process_state_value_function import plot_value_functions, plot_all_final_value_functions
# from Tasks.HighVarianceLearnEightPoliciesTileCodingFeat import HighVarianceLearnEightPoliciesTileCodingFeat
# from Tasks.LearnEightPoliciesTileCodingFeat import LearnEightPoliciesTileCodingFeat
# For building d_mu
# obj = HighVarianceLearnEightPoliciesTileCodingFeat()
# d_mu = (obj.generate_behavior_dist(20_000_000))
# numpy.save(os.path.join(os.getcwd(), 'Resources', 'HighVarianceLearnEightPoliciesTileCodingFeat', 'd_mu.npy'), d_mu)
