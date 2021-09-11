import itertools
import json
import os
from collections import defaultdict
from itertools import zip_longest
from typing import List, Optional, Dict

import numpy as np

from Job.JobBuilder import default_params
from Registry.AlgRegistry import alg_dict
from utils import Configuration


def split_dict_of_list_to_dicts(dict_of_list: Dict[str, list]) -> List[Dict[str, float]]:
    """split a given dictionary of lists into list of dictionaries.

    >>> split_dict_of_list_to_dicts({'alpha': [1, 2, 3], 'lambda': [4, 5], 'gamma': [6]})
    [{'alpha': 1, 'lambda': 4, 'gamma': 6}, {'alpha': 1, 'lambda': 5, 'gamma': 6}, {'alpha': 2, 'lambda': 4, 'gamma': 6}, {'alpha': 2, 'lambda': 5, 'gamma': 6}, {'alpha': 3, 'lambda': 4, 'gamma': 6}, {'alpha': 3, 'lambda': 5, 'gamma': 6}]

    Args:
        dict_of_list (Dict[str, list]): a dictionary of lists.

    Returns:
        List[Dict[str, float]]: list of dictionaries.


    """
    keys = dict_of_list.keys()
    values = [[e for e in result if e is not None] for result in itertools.product(*dict_of_list.values())]
    result = [dict(zip(keys, v)) for v in values]
    return result


def group_dicts_by_first_key(list_of_dicts: List[Dict[str, float]]) -> Dict[str, List[Dict[str, float]]]:
    """
    >>> group_dicts_by_first_key([{'alpha': 1, 'lambda': 4, 'gamma': 6}, {'alpha': 1, 'lambda': 5, 'gamma': 6}, {'alpha': 2, 'lambda': 4, 'gamma': 6}, {'alpha': 2, 'lambda': 5, 'gamma': 6}, {'alpha': 3, 'lambda': 4, 'gamma': 6}, {'alpha': 3, 'lambda': 5, 'gamma': 6}])
    {1: [{'alpha': 1, 'lambda': 4, 'gamma': 6}, {'alpha': 1, 'lambda': 5, 'gamma': 6}], 2: [{'alpha': 2, 'lambda': 4, 'gamma': 6}, {'alpha': 2, 'lambda': 5, 'gamma': 6}], 3: [{'alpha': 3, 'lambda': 4, 'gamma': 6}, {'alpha': 3, 'lambda': 5, 'gamma': 6}]}

    """
    first_key = get_first_key_of_dictionary(list_of_dicts[0])
    final_grouped = defaultdict(list)
    for inner_dict in list_of_dicts:
        final_grouped[inner_dict[first_key]].append(inner_dict)

    return dict(final_grouped)


def group_dicts_over_first_key(list_of_dicts: List[Dict[str, float]]) -> Dict[Dict[str, float], List[float]]:
    """
    >>> group_dicts_over_first_key([{'alpha': 1, 'lambda': 4, 'gamma': 6}, {'alpha': 1, 'lambda': 5, 'gamma': 6}, {'alpha': 2, 'lambda': 4, 'gamma': 6}, {'alpha': 2, 'lambda': 5, 'gamma': 6}, {'alpha': 3, 'lambda': 4, 'gamma': 6}, {'alpha': 3, 'lambda': 5, 'gamma': 6}])
    {(('lambda', 4), ('gamma', 6)): [1, 2, 3], (('lambda', 5), ('gamma', 6)): [1, 2, 3]}

    :param list_of_dicts:
    :return:
    """
    first_key = get_first_key_of_dictionary(list_of_dicts[0])
    final_grouped = defaultdict(list)
    for inner_dict in list_of_dicts:
        first_value = inner_dict[first_key]
        del inner_dict[first_key]
        final_grouped[tuple(inner_dict.items())].append(first_value)

    return dict(final_grouped)


def get_first_key_of_dictionary(d: dict) -> str:
    return list(d.keys())[0]


class ParameterBuilder:
    def __init__(self):
        self.final_params_dict = dict()

    def add_algorithm_params(self, configuration: Configuration):
        for k in alg_dict[configuration.algorithm].related_parameters():
            self.final_params_dict[k] = configuration[k]
        return self

    def build(self):
        return self.final_params_dict


class JsonParameterBuilder:
    def __init__(self):
        self.final_params_dict = dict()
        self.exp_name = None
        self.alg_name = None
        self.alg_related_params = None

    def add_experiment(self, exp_name):
        self.exp_name = exp_name
        return self

    def add_algorithm(self, alg_name):
        self.alg_name = alg_name
        self.alg_related_params = alg_dict[alg_name].related_parameters()
        return self

    def build(self) -> Dict[str, list]:
        json_path = PathFactory.make_experiment_path(self.exp_name, self.alg_name)

        with open(json_path) as f:
            json_config = json.load(f)

        for param_name in self.alg_related_params:
            self.final_params_dict[param_name] = list(json_config['meta_parameters'].get(param_name, [default_params['meta_parameters'][param_name]]))

        return self.final_params_dict


class PathFactory:
    @staticmethod
    def make_experiment_path(exp_name, alg_name):
        return os.path.join(os.getcwd(), 'Experiments', exp_name, alg_name, f'{alg_name}.json')

    @staticmethod
    def make_result_path(exp_name, alg_name):
        return os.path.join(os.getcwd(), 'Results', exp_name, alg_name, f'{alg_name}.json')


class DataPersister:

    @staticmethod
    def save_result(result_arr: np.ndarray, result_name: str, configuration: Configuration):
        full_path_file_to_save = DataPersister.create_full_path_file_name(result_name, configuration)
        np.save(full_path_file_to_save, result_arr)

    @staticmethod
    def save_best_pref_over_first_param(exp_name, alg_name, auc_or_final):
        all_configuration = JsonParameterBuilder().add_experiment(exp_name).add_algorithm(alg_name).build()
        list_of_configuration = split_dict_of_list_to_dicts(all_configuration)
        first_param_key = get_first_key_of_dictionary(all_configuration)
        first_param_length = len(all_configuration[first_param_key])
        mean_over_alpha, stderr_over_alpha = np.zeros(first_param_length), np.zeros(first_param_length)

        grouped_over_first = group_dicts_over_first_key(list_of_configuration)

        for grouped, first_values in grouped_over_first.items():
            grouped_params = dict(grouped)
            print('------------------------------------------------')
            print(grouped_params)
            for index, first_value in enumerate(first_values):
                grouped_params[first_param_key] = first_value
                current_params = Configuration(grouped_params)
                current_params.algorithm = alg_name
                current_params.save_path = PathFactory.make_result_path(exp_name, alg_name)
                current_params.rerun = False

                full_path_file_to_save = DataPersister.create_full_path_file_name(f'_mean_stderr_{auc_or_final}', current_params)
                # perf = np.load(full_path_file_to_save)
                # mean_over_alpha[index], stderr_over_alpha[index] = perf[0], perf[1]
                print(full_path_file_to_save)

    @staticmethod
    def create_full_path_file_name(result_name: str, configuration: Configuration) -> str:
        params = ParameterBuilder().add_algorithm_params(configuration).build()
        file_name_to_save = DataPersister.create_file_name(params, excluded_params=None)
        full_path_file_to_save = os.path.join(configuration.save_path, file_name_to_save)
        full_path_file_to_save = f'{full_path_file_to_save}{result_name}'
        if configuration.rerun:
            full_path_file_to_save = f'{full_path_file_to_save}_rerun'
        return f'{full_path_file_to_save}.npy'

    @staticmethod
    def create_file_name(param: dict, excluded_params: Optional[list]) -> str:
        if excluded_params is None:
            excluded_params = []
        final_str = ''
        for k, v in param.items():
            if k in excluded_params:
                continue
            if k == 'alpha' or k == 'eta':
                split_str = str.split(f'{v:.10f}', '.')
            else:
                split_str = str.split(f'{v:.5f}', '.')
            final_str += '_' + k + split_str[0] + split_str[1]
        return final_str
