import numpy as np
import os
import argparse

from Registry.AlgRegistry import TD
from Registry.EnvRegistry import Chain
from Registry.ProbRegistry import EightStateOffPolicyRandomFeat

from Job.JobBuilder import default_params


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
            'alpha': args.alpha, 'lmbda': args.lmbda, 'num_features': prob.num_features, 'GAMMA': prob.GAMMA
        }
    }
    agent = alg_dict[args.algorithm]()

