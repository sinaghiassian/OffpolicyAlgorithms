import os
import json
import numpy as np
from utils import ImmutableDict
import time

default_params = ImmutableDict(
    {
        # 'agent': 'ABTD',
        # 'task': 'EightStateOffPolicyRandomFeat',
        # 'environment': 'Chain',
        'agent': 'GTD',
        'task': 'LearnEightPoliciesTileCodingFeat',
        'environment': 'FourRoomGridWorld',
        # 'agent': 'TD',
        # 'task': 'HighVarianceLearnEightPoliciesTileCodingFeat',
        # 'environment': 'FourRoomGridWorld',

        'sub_sample': 1,
        'num_of_runs': 1,
        'num_steps': 4000,
        'meta_parameters': {
            'alpha': 0.0001,
            'eta': 0.01,
            'beta': 0.1,
            'zeta': 0.1,
            'lmbda': 0.9,
            'tdrc_beta': 1.0
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
                'ETA': ' '.join([f'{num:.10f}' for num in self.eta]),
                'BETA': ' '.join([f'{num:.5f}' for num in self.beta]),
                'ZETA': ' '.join([f'{num:.5f}' for num in self.zeta]),
                'TDRCBETA': ' '.join([f'{num:.5f}' for num in self.tdrc_beta]),
                'NUMOFRUNS': f'{self.num_of_runs}',
                'NUMSTEPS': f'{self.num_steps}',
                'SUBSAMPLE': f'{self.sub_sample}',
                'ALGORITHM': self.agent,
                'TASK': self.task,
                'ENVIRONMENT': self.environment,
                'SAVEPATH': self.save_path
            })

    @property
    def tdrc_beta(self):
        parameters = self._params.get('meta_parameters')
        return np.asarray(parameters.get('tdrc_beta', [default_params['meta_parameters']['tdrc_beta']]))

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
    def num_of_runs(self):
        return np.asarray(self._params.get('number_of_runs', default_params['num_of_runs']))

    @property
    def num_steps(self):
        return np.asarray(self._params.get('number_of_steps', default_params['num_steps']))

    @property
    def sub_sample(self):
        return np.asarray(self._params.get('sub_sample', default_params['sub_sample']))

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
        if self.server_name.upper() == 'NIAGARA':
            with open('Job/SubmitJobsTemplates.SL', 'r') as f:
                text = f.read()
                for k, v in self._batch_params.items():
                    text = text.replace(f'__{k}__', v)
            return text
        elif self.server_name.upper() == 'CEDAR':
            with open('Job/SubmitJobsTemplatesCedar.SL', 'r') as f:
                text = f.read()
                alg = self._batch_params['ALGORITHM']
                num_of_jobs = sum(1 for _ in open(f'exports_{alg}.dat'))
                text = text.replace('__ALG__', self._batch_params['ALGORITHM'])
                text = text.replace('__NUM_OF_JOBS__', str(num_of_jobs))
            return text

    def run_batch(self):
        if self.server_name not in self.possible_server_names:
            print('Code for running on this server does not exist. Please use either Cedar or Niagara.')
            raise NotImplementedError
        elif self.server_name.upper() == 'NIAGARA':
            print('Submitting the ' + self.agent + ' algorithm jobs on Niagara...')
        elif self.server_name.upper() == 'CEDAR':
            print('Submitting the ' + self.agent + ' algorithm jobs on Cedar...')
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
            os.remove('Create_Configs.sh')
            # alg = self._batch_params['ALGORITHM']
            # os.remove(f'exports_{alg}.dat')

    def __call__(self):
        return self.run_batch()
