import unittest
from Tasks.EightStateCollision import EightStateCollision
from Environments.Chain import Chain


class TestEightStateCollision(unittest.TestCase):
    def setUp(self) -> None:
        params = {'resource_root_path': '../../Resources'}
        self.experiment = EightStateCollision(**params)
        self.experiment.reset()

    def tearDown(self) -> None:
        ...

    def test_load_feature_rep_evaluate_shape_is_(self):
        feature_rep_arr = self.experiment.load_feature_rep()
        self.assertEqual(feature_rep_arr.shape, (8, 6))

    def test_get_state_feature_rep_state_for_all_states(self):
        expected_states_feature_rep = [
            [0., 0., 1., 0., 1., 1.],
            [1., 1., 1., 0., 0., 0.],
            [0., 1., 1., 0., 0., 1.],
            [1., 0., 1., 1., 0., 0.],
            [1., 1., 0., 0., 1., 0.],
            [0., 1., 1., 1., 0., 0.],
            [1., 1., 0., 0., 0., 1.],
            [1., 0., 1., 0., 0., 1.]
        ]
        evaluated_states_feature_rep = []
        for state in range(self.experiment.N):
            evaluated_states_feature_rep.append(list(self.experiment.get_state_feature_rep(state)))
        self.assertListEqual(evaluated_states_feature_rep, expected_states_feature_rep)

    def test_load_behavior_dist_evaluate_shape_is_(self):
        behavior_dist = self.experiment.load_behavior_dist()
        self.assertEqual(behavior_dist.shape, (8,))

    def test_get_mu_for_right_action_in_initial_state_is_one(self):
        mu = self.experiment.get_mu(0, self.experiment.RIGHT_ACTION)
        self.assertEqual(mu, 1)

    def test_get_mu_for_retreat_action_in_initial_state_is_zero(self):
        mu = self.experiment.get_mu(0, self.experiment.RETREAT_ACTION)
        self.assertEqual(mu, 0)

    def test_get_mu_for_all_action_in_not_initial_state_is_one_half(self):
        mu = self.experiment.get_mu(5, self.experiment.RIGHT_ACTION)
        self.assertEqual(mu, 0.5)
        mu = self.experiment.get_mu(5, self.experiment.RETREAT_ACTION)
        self.assertEqual(mu, 0.5)

    def test_get_pi_for_right_action_is_one(self):
        pi = self.experiment.get_pi(0, self.experiment.RIGHT_ACTION)
        self.assertEqual(pi, 1)

    def test_get_pi_for_retreat_action_is_one(self):
        pi = self.experiment.get_pi(0, self.experiment.RETREAT_ACTION)
        self.assertEqual(pi, 0)
