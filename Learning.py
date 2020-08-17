import numpy as np
import os
import argparse


from Problems.ChainProb import ChainProb



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


