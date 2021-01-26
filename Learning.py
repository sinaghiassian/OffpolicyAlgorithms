import os
import numpy as np
import argparse
import random
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
    parser.add_argument('--run_number', '-r', type=int, default=default_params['meta_parameters']['run'])
    parser.add_argument('--environment', '-e', type=str, default=default_params['environment'])
    parser.add_argument('--save_path', '-sp', type=str, default='Results')
    parser.add_argument('--render', '-render', type=bool, default=False)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    np.random.seed(args.run_number)
    random.seed(a=args.run_number)

    env = environment_dict[args.environment]()
    task = task_dict[args.task](run_number=args.run_number)
    params = {'alpha': args.alpha, 'lmbda': args.lmbda, 'eta': args.eta, 'beta': args.beta, 'zeta': args.zeta}
    agent = alg_dict[args.algorithm](task, **params)

    RMSVE = np.zeros((task.num_policies, task.num_steps))
    agent.state = env.reset()
    is_terminal = False
    error_render = ErrorRender(task.num_policies, task.num_steps)
    for step in range(task.num_steps):
        RMSVE[:, step], error = agent.compute_rmsve()
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
    print(np.mean(RMSVE, axis=0))

# TODO: Collector: Save and load the data.
# TODO: Add Plotting code.
