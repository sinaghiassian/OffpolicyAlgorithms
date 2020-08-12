import argparse
import numpy as np
from Job.JobBuilder import *
import Environments
import importlib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', '-a', type=float, default=default_params['meta_parameters']['alpha'])
    parser.add_argument('--algorithm', '-alg', type=str, default=default_params['agent'])
    parser.add_argument('--problem', '-p', type=str, default=default_params['problem'])
    parser.add_argument('--run_number', '-r', type=int, default=default_params['meta_parameters']['run'])
    parser.add_argument('--feature_kind', '-f', type=str, default=default_params['feature_kind'])
    parser.add_argument('--environment', '-e', type=str, default=default_params['environment'])
    parser.add_argument('--save_path', '-sp', type=str, default='Experiments/')

    args = parser.parse_args()

    np.random.seed(args.run_number)

    module_env = importlib.import_module(f'Environments.{args.environment}')
    env = getattr(module_env, args.environment)()

    module_alg = importlib.import_module(f'Algorithms.{args.algorithm}.{args.algorithm}')
    #algorithm = getattr(module_alg, args.algorithm)({'alpha': args.alpha, 'GAMMA': prob.get_gamma(), 'feature_size': feature_size})

    #print(algorithm)
