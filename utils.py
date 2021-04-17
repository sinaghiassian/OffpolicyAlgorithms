import numpy as np
import os


def get_save_value_function_steps(num_steps):
    return [int(num_steps * i) - 1 for i in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]]


def save_value_function(value_function, save_path, step, run):
    save_dir = os.path.join(save_path, 'Sample_value_function')
    res_path = os.path.join(save_dir, f"{step}_{run}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    np.save(res_path, value_function)


class Configuration(dict):
    def __str__(self):
        return f"{self.environment} {self.task} {self.algorithm}"

    def __getattr__(self, item):
        return self[item]


def find_all_experiment_configuration(experiments_path: str, ext='.json'):
    if experiments_path.endswith(ext):
        yield experiments_path
    for root, _, files in os.walk(experiments_path):
        for file in files:
            if file.endswith(ext):
                yield os.path.join(root, file)


class ImmutableDict(dict):
    def immutable(self):
        raise TypeError("%r objects are immutable" % self.__class__.__name__)

    def __setitem__(self, key, value):
        self.immutable()

    def __delitem__(self, key):
        self.immutable()

    def set_default(self, k, default):
        self.immutable()

    def update(self, __m, **kwargs):
        self.immutable()

    def clear(self) -> None:
        self.immutable()


def create_name_for_save_load(param_dict, excluded_params=None):
    if excluded_params is None:
        excluded_params = []
    final_str = ''
    for k, v in param_dict.items():
        if k in excluded_params:
            continue
        if k == 'alpha' or k == 'eta':
            split_str = str.split(f'{v:.10f}', '.')
        else:
            split_str = str.split(f'{v:.5f}', '.')
        final_str += '_' + k + split_str[0] + split_str[1]
    return final_str


def save_result(path, name, result_array, params, rerun):
    name_to_save = create_name_for_save_load(param_dict=params)
    path_and_name = os.path.join(path, name_to_save)
    final_name = f"{path_and_name}{name}"
    if rerun:
        final_name = f"{final_name}_rerun"
    np.save(final_name, result_array)


def generate_gif(frames, path, size=(180, 180, 3), duration=1 / 20):
    import imageio
    from skimage.transform import resize
    for idx, frame_idx in enumerate(frames):
        frames[idx] = resize(frame_idx, size, preserve_range=True, order=0).astype(np.uint8)
    imageio.mimsave(path, frames, duration=duration)
