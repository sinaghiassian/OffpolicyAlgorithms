import os
import numpy as np
import matplotlib.pyplot as plt


class ValueFunctionProcessor:
    def __init__(self, exp, alg):
        result_dir = os.path.join(os.getcwd(), 'Results', exp, alg, 'Sample_value_function')
        self.all_value_functions = dict()
        for value_function_name in os.listdir(result_dir):
            value_function = np.load(os.path.join(result_dir, value_function_name))
            step, run_num = (int(i) for i in value_function_name.replace('.npy', '').split('_'))
            self.all_value_functions[(step, run_num)] = value_function

    def get_value_function_by_step_and_run(self, step, run):
        return self.all_value_functions[(step, run)]


STEPS = [199, 999, 1999, 3999, 9999, 19999]
RUNS = [0, 10, 15, 20]
EXPS = ['FirstChain']  # FirstChain or FirstFourRoom or 1HVFourRoom
ALGS = ['TD']
TASK = 'EightStateOffPolicyRandomFeat'


def plot_value_function(ax, value_function, step=0, run=0):
    label = f"{step}_{run}"
    ax.plot(value_function, label=label)
    ax.legend()


def plot_value_functions():
    for exp in EXPS:
        true_value_function = np.load(os.path.join(os.getcwd(), 'Resources', TASK, 'state_values.npy'))
        for alg in ALGS:
            value_processor = ValueFunctionProcessor(exp, alg)
            for run in RUNS:
                fig, ax = plt.subplots()
                plot_value_function(ax, true_value_function)
                for step in STEPS:
                    value_function = value_processor.get_value_function_by_step_and_run(step, run)
                    plot_value_function(ax, value_function, step, run)
            plt.show()



