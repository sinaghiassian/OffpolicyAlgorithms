import unittest
import numpy as np

from Algorithms.TD import TD
from Environments.Chain import Chain
from Tasks.EightStateCollision import EightStateCollision


class TestTD(unittest.TestCase):
    def setUp(self) -> None:
        params = {
            #'resource_root_path': '../../Resources',
            'alpha': 0.001953125,
            'lmbda': 0.9,
        }
        self.env = Chain()
        self.task = EightStateCollision(**params)
        self.task.reset()

        self.alg = TD(task=self.task, **params)

    def tearDown(self) -> None:
        ...

    def test_initial_w_is_zero(self):
        self.assertEqual(self.alg.w.sum(), 0)

    def test_initial_z_is_zero(self):
        self.assertEqual(self.alg.z.sum(), 0)

    def test_learn_single_policy_rmsve_after_num_steps(self):
        rmsve_of_run = np.zeros((self.task.num_policies, self.task.num_steps))
        np.random.seed(0)

        self.alg.state = self.env.reset()
        for step in range(self.task.num_steps):
            rmsve_of_run[:, step], error = self.alg.compute_rmsve()
            self.alg.action = self.alg.choose_behavior_action()
            self.alg.next_state, r, is_terminal, info = self.env.step(self.alg.action)
            self.alg.learn(self.alg.state, self.alg.next_state, r, is_terminal)
            if is_terminal:
                self.alg.state = self.env.reset()
                self.alg.reset()
                continue
            self.alg.state = self.alg.next_state
        self.assertTrue(abs(0.08319472840990755 - rmsve_of_run[0, -1]) <= 0.0000001)
