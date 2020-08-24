import numpy as np
import os
import argparse

from Registry.AlgRegistry import TD
from Registry.EnvRegistry import Chain
from Registry.ProbRegistry import EightStateOffPolicyRandomFeat

from Job.JobBuilder import default_params


def compute_rmsve(w):
    est_value = np.dot(feature_rep, w)
    error = (est_value - state_values) * (est_value - state_values)
    RMSVE[step] = np.sqrt(np.sum(d_mu * error))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', '-a', type=float, default=default_params['meta_parameters']['alpha'])
    parser.add_argument('-lmbda', '-l', type=float, default=default_params['lmbda'])
    parser.add_argument('--algorithm', '-alg', type=str, default=default_params['agent'])
    parser.add_argument('--problem', '-p', type=str, default=default_params['problem'])
    parser.add_argument('--run_number', '-r', type=int, default=default_params['meta_parameters']['run'])
    parser.add_argument('--environment', '-e', type=str, default=default_params['environment'])
    parser.add_argument('--save_path', '-sp', type=str, default='Experiments/')
    args = parser.parse_args()
    np.random.seed(args.run_number)

    environment_dict = {'Chain': Chain}
    env = environment_dict[args.environment]()

    prob_dict = {'EightStateOffPolicyRandomFeat': EightStateOffPolicyRandomFeat}
    prob = prob_dict[args.problem]()

    alg_dict = {'TD': TD}
    alg_params = {
        'TD': {
            'alpha': args.alpha, 'lmbda': args.lmbda, 'run': args.run_number,
            'num_features': prob.num_features, 'GAMMA': prob.GAMMA
        }
    }
    agent = alg_dict[args.algorithm]()

    RMSVE = np.zeros(prob.num_steps)
    feature_rep = prob.get_feat_rep(args.run_number)
    s = env.reset()
    for step in range(prob.num_steps):
        x = feature_rep[s, :]
        compute_rmsve(agent.w)

