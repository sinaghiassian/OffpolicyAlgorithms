import os
import numpy as np
import argparse
import random
from utils import save_result
from Registry.AlgRegistry import alg_dict
from Registry.EnvRegistry import environment_dict
from Registry.TaskRegistry import task_dict
from Job.JobBuilder import default_params
from Environments.rendering import ErrorRender

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', '-a', type=float, default=default_params['meta_parameters']['alpha'])
    parser.add_argument('--lmbda', '-l', type=float, default=default_params['meta_parameters']['lmbda'])
    parser.add_argument('--eta', '-et', type=float, default=default_params['meta_parameters']['eta'])
    parser.add_argument('--beta', '-b', type=float, default=default_params['meta_parameters']['beta'])
    parser.add_argument('--zeta', '-z', type=float, default=default_params['meta_parameters']['zeta'])
    parser.add_argument('--algorithm', '-alg', type=str, default=default_params['agent'])
    parser.add_argument('--task', '-t', type=str, default=default_params['task'])
    parser.add_argument('--num_of_runs', '-nr', type=int, default=default_params['num_of_runs'])
    parser.add_argument('--num_steps', '-ns', type=int, default=default_params['num_steps'])
    parser.add_argument('--sub_sample', '-ss', type=int, default=default_params['sub_sample'])
    parser.add_argument('--environment', '-e', type=str, default=default_params['environment'])
    parser.add_argument('--save_path', '-sp', type=str, default='Results')
    parser.add_argument('--render', '-render', type=bool, default=False)
    args = parser.parse_args()

    params = {'alpha': args.alpha, 'lmbda': args.lmbda, 'eta': args.eta, 'beta': args.beta, 'zeta': args.zeta}
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    RMSVE = np.zeros((task_dict[args.task].num_of_policies(), args.num_steps, args.num_of_runs))
    RMSVE_of_runs = np.zeros((args.num_of_runs, args.num_steps))
    for run in range(args.num_of_runs):
        np.random.seed(run)
        random.seed(a=run)
        env = environment_dict[args.environment]()
        task = task_dict[args.task](run_number=run, num_steps=args.num_steps)
        agent = alg_dict[args.algorithm](task, **params)

        RMSVE_of_run = np.zeros((task.num_policies, task.num_steps))
        agent.state = env.reset()
        is_terminal = False
        error_render = ErrorRender(task.num_policies, task.num_steps)
        for step in range(task.num_steps):
            RMSVE_of_run[:, step], error = agent.compute_rmsve()
            if args.render:
                error_render.add_error(error)
            agent.action = agent.choose_behavior_action()
            agent.next_state, r, is_terminal, info = env.step(agent.action)
            agent.learn(agent.state, agent.next_state, r, is_terminal)
            if is_terminal:
                agent.state = env.reset()
                is_terminal = False
                agent.reset()
                continue
            agent.state = agent.next_state
            if args.render:
                env.render(mode='screen', render_cls=error_render)
        # print(np.mean(RMSVE_of_run, axis=0))
        RMSVE[:, :, run] = RMSVE_of_run
    RMSVE_of_runs = np.transpose(np.mean(RMSVE, axis=0))  # Average over all policies.
    save_result(args.save_path, '_RMSVE_mean_over_runs', np.mean(RMSVE_of_runs, axis=0), params)
    print('1', np.mean(RMSVE_of_runs, axis=0))
    save_result(args.save_path, '_RMSVE_stderr_over_runs',
                np.std(RMSVE_of_runs, axis=0, ddof=1) / np.sqrt(args.num_of_runs), params)
    print('2', np.std(RMSVE_of_runs, axis=0, ddof=1) / np.sqrt(args.num_of_runs))
    final_errors_mean_over_steps = np.mean(RMSVE_of_runs[:, args.num_steps - int(0.01 * args.num_steps) - 1:args.num_steps -1], axis=1)
    save_result(args.save_path, '_mean_stderr_final',
                np.array([np.mean(final_errors_mean_over_steps),
                          np.std(final_errors_mean_over_steps, ddof=1) / np.sqrt(args.num_of_runs)]), params)
    print('3', np.array([np.mean(final_errors_mean_over_steps),
                         np.std(final_errors_mean_over_steps, ddof=1) / np.sqrt(args.num_of_runs)]))
    auc_mean_over_steps = np.mean(RMSVE_of_runs, axis=1)
    print(auc_mean_over_steps)
    save_result(args.save_path, '_mean_stderr_auc',
                np.array([np.mean(auc_mean_over_steps),
                          np.std(auc_mean_over_steps, ddof=1) / np.sqrt(args.num_of_runs)]), params)
    print('4', np.array([np.mean(auc_mean_over_steps),
                         np.std(auc_mean_over_steps, ddof=1) / np.sqrt(args.num_of_runs)]))
# TODO: Change int(0.01 * args.num_steps) - 1:args.num_steps -1] to int(0.01 * args.num_steps) - 1:]
