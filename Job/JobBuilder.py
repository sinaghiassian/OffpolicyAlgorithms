import os
import json
import numpy as np
from utils import ImmutableDict
import time

default_params = ImmutableDict(
    {
        # 'agent': 'GTD',
        # 'task': 'EightStateOffPolicyRandomFeat',
        # 'environment': 'Chain',
        'agent': 'GTD',
        'task': 'LearnEightPoliciesTileCodingFeat',
        'environment': 'FourRoomGridWorld',

        'meta_parameters': {
            'alpha': 0.01,
            'eta': 0.0,
            'beta': 0.0,
            'zeta': 0.0,
            'lmbda': 0.0,
            "run": 0
        }
    }
)


class JobBuilder:
    def __init__(self, json_path, server_name):
        self.possible_server_names = ['NIAGARA', 'Niagara', 'niagara', 'CEDAR', 'Cedar', 'cedar']
        self._path = json_path
        self.server_name = server_name
        with open(self._path) as f:
            self._params = json.load(f)

        self._batch_params = ImmutableDict(
            {
                'ALPHA': ' '.join([f'{num:.10f}' for num in self.alpha]),
                'LMBDA': ' '.join([f'{num:.5f}' for num in self.lmbda]),
                'ETA': ' '.join([f'{num:.5f}' for num in self.eta]),
                'BETA': ' '.join([f'{num:.5f}' for num in self.beta]),
                'ZETA': ' '.join([f'{num:.5f}' for num in self.zeta]),
                'ALGORITHM': self.agent,
                'TASK': self.task,
                'RUN': f'0..{self.run}',
                'ENVIRONMENT': self.environment,
                'SAVEPATH': self.save_path,
            })

    @property
    def alpha(self):
        parameters = self._params.get('meta_parameters')
        return np.asarray(parameters.get('alpha', [default_params['meta_parameters']['alpha']]))

    @property
    def lmbda(self):
        parameters = self._params.get('meta_parameters')
        return np.asarray(parameters.get('lmbda', [default_params['meta_parameters']['lmbda']]))

    @property
    def eta(self):
        parameters = self._params.get('meta_parameters')
        return np.asarray(parameters.get('eta', [default_params['meta_parameters']['eta']]))

    @property
    def beta(self):
        parameters = self._params.get('meta_parameters')
        return np.asarray(parameters.get('beta', [default_params['meta_parameters']['beta']]))

    @property
    def zeta(self):
        parameters = self._params.get('meta_parameters')
        return np.asarray(parameters.get('zeta', [default_params['meta_parameters']['zeta']]))

    @property
    def agent(self):
        return self._params.get('agent', default_params['agent'])

    @property
    def task(self):
        return self._params.get('task', default_params['task'])

    @property
    def run(self):
        parameters = self._params.get('meta_parameters')
        return np.asarray(parameters.get('run', default_params['meta_parameters']['run']))

    @property
    def environment(self):
        return self._params.get('environment', default_params['environment'])

    @property
    def save_path(self):
        return os.path.dirname(self._path).replace("/Experiments/", "/Results/")

    def create_dat_file(self):
        with open('Job/Cedar_Create_Config_Template.sh', 'r') as f:
            text = f.read()
            for k, v in self._batch_params.items():
                text = text.replace(f'__{k}__', v)
        return text

    def to_shell(self):
        if self.server_name == 'Niagara' or self.server_name == 'niagara' or self.server_name == 'NIAGARA':
            with open('Job/SubmitJobsTemplates.SL', 'r') as f:
                text = f.read()
                for k, v in self._batch_params.items():
                    text = text.replace(f'__{k}__', v)
            return text
        elif self.server_name == 'Cedar' or self.server_name == 'cedar' or self.server_name == 'CEDAR':
            with open('Job/SubmitJobsTemplatesCedar.SL', 'r') as f:
                text = f.read()
                num_of_jobs = sum(1 for line in open('exports.dat'))
                text = text.replace('__NUM_OF_JOBS__', str(num_of_jobs))
            return text

    def run_batch(self):
        if self.server_name not in self.possible_server_names:
            print('Code for running on this server does not exist. Please use either Cedar or Niagara.')
            raise NotImplementedError
        elif self.server_name == 'Niagara' or self.server_name == 'NIAGARA' or self.server_name == 'niagara':
            print('Submitted the ' + self.agent + 'algorithm jobs on Niagara...')
        elif self.server_name == 'Cedar' or self.server_name == 'CEDAR' or self.server_name == 'cedar':
            print('Running the ' + self.agent + 'algorithm jobs on Cedar...')
            with open('Create_Configs.sh', 'wt') as f:
                f.write(self.create_dat_file())
            time.sleep(1)
            os.system('bash Create_Configs.sh')
        with open('Submit_Jobs.SL', 'wt') as f:
            f.write(self.to_shell())
        time.sleep(1)
        os.system('sbatch Submit_Jobs.SL')
        time.sleep(1)
        os.remove('Submit_Jobs.SL')
        if self.server_name == 'Cedar' or self.server_name == 'CEDAR' or self.server_name == 'cedar':
            os.remove('exports.dat')
            os.remove('Create_Configs.sh')

    def __call__(self):
        return self.run_batch()
