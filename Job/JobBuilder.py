import os
import json
import numpy as np
from utils import ImmutableDict

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
            'eta': 0.01,
            'beta': 0.1,
            'zeta': 0.1,
            'lmbda': 0.1,
            "run": 0
        }
    }
)


class JobBuilder:
    def __init__(self, json_path, server_name):
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

    def to_shell(self):
        with open('Job/SubmitJobsTemplates.SL', 'r') as f:
            text = f.read()
            for k, v in self._batch_params.items():
                text = text.replace(f'__{k}__', v)
        return text

    def create_dat_file(self):
        with open('Cedar_Create_Config_Template.sh', 'wt') as f:
            text = f.read()
            for k, v in self._batch_params.items():
                text = text.replace(f'__{k}__', v)
        return text

    def run_batch(self):
        if self.server_name == 'Niagara':
            print('Running on Niagara...')
            # print(self.to_shell())
            with open('SubmitJobs.SL', 'wt') as f:
                f.write(self.to_shell())
            os.system('sbatch SubmitJobs.SL')
            os.remove('SubmitJobs.SL')
        elif self.server_name == 'Cedar':
            print('Running on Cedar...')
            with open('CREATE_CONFIG.sh', 'wt') as f:
                f.write(self.create_dat_file())
            os.system('CREATE_CONFIG.sh')
            # os.remove('CREATE_CONFIG.sh')
        else:
            print('Error! Please input the server name as follows: Learning.py -s "<Niagara> or <Cedar> ')
            exit()

    def __call__(self):
        return self.run_batch()
