import numpy as np
import argparse
import random
from Registry.AlgRegistry import TD, GTD
from Registry.EnvRegistry import Chain, FourRoomGridWorld
from Registry.TaskRegistry import EightStateOffPolicyRandomFeat, LearnEightPoliciesTileCodingFeat
from Job.JobBuilder import default_params
# from Environments.rendering import ErrorRender

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', '-a', type=float, default=default_params['meta_parameters']['alpha'])
    parser.add_argument('--alphav', '-av', type=float, default=default_params['meta_parameters']['alpha_v'])
    parser.add_argument('-lmbda', '-l', type=float, default=default_params['meta_parameters']['lmbda'])
    parser.add_argument('--algorithm', '-alg', type=str, default=default_params['agent'])
    parser.add_argument('--task', '-t', type=str, default=default_params['task'])
    parser.add_argument('--run_number', '-r', type=int, default=default_params['meta_parameters']['run'])
    parser.add_argument('--environment', '-e', type=str, default=default_params['environment'])
    parser.add_argument('--save_path', '-sp', type=str, default='Experiments/')
    parser.add_argument('--render', '-render', type=bool, default=False)

    args = parser.parse_args()
    np.random.seed(args.run_number)
    random.seed(a=args.run_number)

    environment_dict = {'FourRoomGridWorld': FourRoomGridWorld, 'Chain': Chain}
    env = environment_dict[args.environment]()

    task_dict = {'EightStateOffPolicyRandomFeat': EightStateOffPolicyRandomFeat,
                 'LearnEightPoliciesTileCodingFeat': LearnEightPoliciesTileCodingFeat}
    task = task_dict[args.task](run_number=args.run_number)

    alg_dict = {'TD': TD, 'GTD': GTD}
    alg_params = {
        'TD': {
            'alpha': args.alpha, 'lmbda': args.lmbda, 'run': args.run_number,
            'num_features': task.num_features, 'GAMMA': task.GAMMA
        },
        'TDMultiplePolicy': {
            'alpha': args.alpha, 'lmbda': args.lmbda, 'run': args.run_number,
            'num_features': task.num_features, 'GAMMA': task.GAMMA
        },
        'GTD': {
            'alpha': args.alpha, 'alpha_v': args.alphav, 'lmbda': args.lmbda, 'run': args.run_number,
            'num_features': task.num_features, 'GAMMA': task.GAMMA
        },
        'GTDMultiplePolicy': {
            'alpha': args.alpha, 'alpha_v': args.alphav, 'lmbda': args.lmbda, 'run': args.run_number,
            'num_features': task.num_features, 'GAMMA': task.GAMMA
        }
    }
    agent = alg_dict[args.algorithm](task, **alg_params[args.algorithm])

    RMSVE = np.zeros((task.num_policies, task.num_steps))
    agent.state = env.reset()
    is_terminal = False
    # error_render = ErrorRender(task.num_policies, task.num_steps)
    for step in range(task.num_steps):
        RMSVE[:, step], error = agent.compute_rmsve()
        # error_render.add_error(error)
        agent.action = agent.choose_behavior_action()
        agent.next_state, r, is_terminal, info = env.step(agent.action)
        agent.learn(agent.state, agent.next_state, r, is_terminal)
        if is_terminal:
            agent.state = env.reset()
            is_terminal = False
            continue
        agent.state = agent.next_state
        # if args.render:
        #     env.render(mode='screen', render_cls=error_render)
    print(np.mean(RMSVE[:, :], axis=0))

# TODO: Collector: Save and load the data.
# TODO: Implement new algorithms and check against the old code.
# TODO: Add Plotting code.
# TODO: JOB submission. Add Cedar compatibility and AWS/Google compute compatibility.
