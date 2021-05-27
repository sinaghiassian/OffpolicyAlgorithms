import os
import numpy as np
import matplotlib.pyplot as plt


class ValueFunctionProcessor:
    def __init__(self, exp, alg):
        result_dir = os.path.join(os.getcwd(), 'Results', exp, alg, 'Sample_value_function')
        self.all_value_functions = dict()
        self.all_value_functions_of_last_step = dict()
        for value_function_name in os.listdir(result_dir):
            value_function = np.load(os.path.join(result_dir, value_function_name))
            step, run_num = (int(i) for i in value_function_name.replace('.npy', '').split('_'))
            self.all_value_functions[(step, run_num)] = value_function
            if (step == 19999 and exp == 'FirstChain') or (step == 49999 and exp == 'FirstFourRoom') or (
                    step == 49999 and exp == '1HVFourRoom'):
                self.all_value_functions_of_last_step[run_num] = value_function

    def get_value_function_by_step_and_run(self, step, run):
        return self.all_value_functions[(step, run)]

    def get_value_function_for_last_step(self, run):
        return self.all_value_functions_of_last_step[run]


# STEPS = [199, 999, 1999, 3999, 9999, 19999]
STEPS = [199, 1999, 19999]
# STEPS = [19999]
RUNS = [0, 10, 15, 20, 30, 45]
# RUNS = list(range(50))
EXPS = ['FirstChain']  # FirstChain or FirstFourRoom or 1HVFourRoom
ALGS = ['TD']
TASK = 'EightStateCollision'


def plot_value_function(ax, value_function, step=0, run=0, is_last_step=False):
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1.0)
    label = f"{step}_{run}"
    line_style = '-'
    line_width = 4
    alpha = 1.0
    color = 'blue'
    if not step:
        line_style = '--'
    if not step and is_last_step:
        line_style = '-'
    if is_last_step:
        line_width = 2
        alpha = 0.2
        color = 'red'
        ax.plot(value_function, label=label, linewidth=line_width, linestyle=line_style, alpha=alpha, color=color)
    else:
        ax.plot(value_function, label=label, linewidth=line_width, linestyle=line_style, alpha=alpha)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    # ax.legend()


def plot_value_functions():
    for exp in EXPS:
        save_dir = os.path.join('pdf_plots', 'value_functions')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        true_value_function = np.load(os.path.join(os.getcwd(), 'Resources', TASK, 'state_values.npy'))
        for alg in ALGS:
            value_processor = ValueFunctionProcessor(exp, alg)
            for run in RUNS:
                fig, ax = plt.subplots(figsize=(8, 3))
                for step in STEPS:
                    value_function = value_processor.get_value_function_by_step_and_run(step, run)
                    plot_value_function(ax, value_function, step, run)
                plot_value_function(ax, true_value_function)
                fig.savefig(os.path.join(save_dir, f"{run}_value_function_{alg}_{exp}.pdf"),
                            format='pdf', dpi=200, bbox_inches='tight')
            plt.show()


def plot_all_final_value_functions():
    for exp in EXPS:
        save_dir = os.path.join('pdf_plots', 'value_functions', 'asymptotic_value_functions')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        true_value_function = np.load(os.path.join(os.getcwd(), 'Resources', TASK, 'state_values.npy'))
        for alg in ALGS:
            value_processor = ValueFunctionProcessor(exp, alg)
            fig, ax = plt.subplots(figsize=(8, 3))
            for run in range(50):
                value_function = value_processor.get_value_function_for_last_step(run)
                plot_value_function(ax, value_function, is_last_step=True)
            plot_value_function(ax, true_value_function)
            fig.savefig(os.path.join(save_dir, f"value_function_{alg}_{exp}.pdf"),
                        format='pdf', dpi=200, bbox_inches='tight')
            plt.show()
