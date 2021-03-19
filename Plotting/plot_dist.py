import os
import numpy as np
import matplotlib.pyplot as plt


def load_d_mu(task):
    return np.load(os.path.join(os.getcwd(), 'Resources', task, 'd_mu.npy'))


def load_state_values(task):
    return np.load(os.path.join(os.getcwd(), 'Resources', task, 'state_values.npy'))


def plot_d_mu(ax, d_mu, active_states):
    ax.plot(d_mu)
    x_labels = list(active_states)
    x_ticks = [x for x in range(len(x_labels))]
    ax.xaxis.set_ticks(x_ticks)
    ax.set_xticklabels(x_labels)


def find_active_states(task, d_mu, state_values, policy_no=0):
    if task == 'EightStateOffPolicyRandomFeat':
        return [x for x in range(d_mu.shape[0])]
    return np.where(state_values[policy_no] > 0)[0]


def get_active_d_mu(task, d_mu, active_states, policy_no=0):
    if task == 'EightStateOffPolicyRandomFeat':
        return d_mu
    return d_mu[active_states, policy_no].squeeze()


def plot_distribution():
    task = 'HighVarianceLearnEightPoliciesTileCodingFeat'
    d_mu = load_d_mu(task)
    state_values = load_state_values(task)
    for policy_no in range(state_values.shape[0]):
        fig, ax = plt.subplots()
        active_states = find_active_states(task, d_mu, state_values, policy_no)
        active_d_mu = get_active_d_mu(task, d_mu, active_states, policy_no)
        plot_d_mu(ax, active_d_mu, active_states)
        plt.show()
        if task == 'EightStateOffPolicyRandomFeat':
            break
