import os
from Job.JobBuilder import JobBuilder
import argparse


def find_all_experiment_configuration(experiments_path: str):
    if experiments_path.endswith('.json'):
        yield experiments_path
    for root, _, files in os.walk(experiments_path):
        for file in files:
            if file.endswith('.json'):
                yield os.path.join(root, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory_or_file', '-f', type=str, help='Json file path or Json files directory', required=True)
    parser.add_argument('--server', '-s', type=str, help='Input server name, Cedar or Niagara', required=True)
    args = parser.parse_args()
    for path in find_all_experiment_configuration(args.directory_or_file):
        builder = JobBuilder(json_path=os.path.join(os.getcwd(), path), server_name=args.server)
        builder()
