import os
import numpy as np
import matplotlib.pyplot as plt


def load_d_mu(task):
    return np.load(os.path.join(os.getcwd(), 'Resources', task, 'd_mu.npy'))


def load_state_values(task):
    return np.load(os.path.join(os.getcwd(), 'Resources', task, 'state_values.npy'))


def plot_d_mu(ax, d_mu, active_states):
    ax.plot(d_mu, linewidth=3)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    x_labels = list(active_states)
    x_ticks = [x for x in range(len(x_labels))]
    ax.xaxis.set_ticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.yaxis.set_ticks([0, 0.005, 0.01, 0.015, 0.02, 0.025])
    ax.set_ylim([0.00, 0.025])
    ax.set_yticklabels([])
    # ax.set_xticklabels([])


def find_active_states(task, d_mu, state_values, policy_no=0):
    if task == 'EightStateCollision':
        return [x for x in range(d_mu.shape[0])]
    return np.where(state_values[policy_no] > 0)[0]


def get_active_d_mu(task, d_mu, active_states, policy_no=0):
    if task == 'EightStateCollision':
        return d_mu
    return d_mu[active_states, policy_no].squeeze()


def plot_distribution(**kwargs):
    task = kwargs['task']
    d_mu = load_d_mu(task)
    state_values = load_state_values(task)
    for policy_no in range(state_values.shape[0]):
        fig, ax = plt.subplots(figsize=kwargs['fig_size'])
        active_states = find_active_states(task, d_mu, state_values, policy_no)
        active_d_mu = get_active_d_mu(task, d_mu, active_states, policy_no)
        plot_d_mu(ax, active_d_mu, active_states)
        plt.show()
        if task == 'EightStateCollision':
            break


def plot_dist_for_two_four_room_tasks(**kwargs):
    task1 = 'LearnEightPoliciesTileCodingFeat'
    task2 = 'HighVarianceLearnEightPoliciesTileCodingFeat'
    save_dir = os.path.join('pdf_plots', 'Misc', 'CompareDistsFR')
    d_mu1 = load_d_mu(task1)
    d_mu2 = load_d_mu(task2)
    state_values1 = load_state_values(task1)
    state_values2 = load_state_values(task2)
    for policy_no in range(state_values1.shape[0]):
        fig, ax = plt.subplots(figsize=kwargs['fig_size'])
        active_states = find_active_states(task1, d_mu1, state_values1, policy_no)
        active_d_mu = get_active_d_mu(task1, d_mu1, active_states, policy_no)
        plot_d_mu(ax, active_d_mu, active_states)
        active_states = find_active_states(task2, d_mu2, state_values2, policy_no)
        active_d_mu = get_active_d_mu(task2, d_mu2, active_states, policy_no)
        plot_d_mu(ax, active_d_mu, active_states)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"dist_policy_{policy_no}.pdf"),
                    format='pdf', dpi=1000, bbox_inches='tight')
        plt.show()


