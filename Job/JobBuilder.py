import os
import json
import numpy as np

default_params = {
    'agent': 'TD',
    'problem': 'GridWorld1D',
    'feature_kind': 'dependent',
    'environment': 'GridWorld1D',
    'meta_parameters': {
        'alpha': [.5 ** i for i in range(4, 16)],
        "run": 30
    }
}


def find_all_experiment_configuration(experiments_path: str):
    if experiments_path.endswith('.json'):
        yield experiments_path
    for root, _, files in os.walk(experiments_path):
        for file in files:
            if file.endswith('.json'):
                yield os.path.join(root, file)


class JobBuilder:
    def __init__(self, json_path: str):
        self._path = json_path
        with open(self._path) as f:
            self._params = json.load(f)

        self._batch_params = {
            'ALPHA': ' '.join([f'{num:.10f}' for num in self.alpha]),
            'RUN': f'0..{self.run}',
            'ALGORITHM': self.agent,
            'ENVIRONMENT': self.environment,
            'FEATUREKIND': self.feature_kind,
            'PROBLEM': self.problem,
            'SAVEPATH': self.save_path,

        }

    @property
    def agent(self):
        return self._params.get('agent', default_params['agent'])

    @property
    def problem(self):
        return self._params.get('problem', default_params['problem'])

    @property
    def environment(self):
        return self._params.get('environment', default_params['environment'])

    @property
    def feature_kind(self):
        return self._params.get('feature_kind', default_params['feature_kind'])

    @property
    def save_path(self):
        return os.path.dirname(self._path).replace("/Experiments/", "/Results/")

    @property
    def alpha(self):
        parameters = self._params.get('meta_parameters', {})
        return np.asarray(parameters.get('alpha', default_params['meta_parameters']['alpha']))

    @property
    def run(self):
        parameters = self._params.get('meta_parameters', {})
        return np.asarray(parameters.get('run', default_params['meta_parameters']['run']))

    def to_shell(self):
        with open('Job/SubmitJobsTemplates.SL', 'r') as f:
            text = f.read()
            for k, v in self._batch_params.items():
                text = text.replace(f'__{k}__', v)
        return text

    def run_batch(self):
        print(self.to_shell())
        with open('RunningSubmitJobs.SL', 'wt') as f:
            f.write(self.to_shell())
        os.system('sbatch RunningSubmitJobs.SL')
        os.remove('RunningSubmitJobs.SL')

    def __call__(self, *args, **kwargs):
        return self.run_batch()
