import unittest

from Environments.Chain import Chain


class TestChain(unittest.TestCase):
    def setUp(self) -> None:
        self.env = Chain()
        self.env.reset()

    def tearDown(self) -> None:
        self.env.reset()

    def test_rest_initial_state_between_zero_three(self):
        self.env.reset()
        self.assertIn(self.env._state, [0, 1, 2, 3])

    def test_step_retreat_move_state_to_initial_state(self):
        self.env.reset()
        sp, r, is_done, _ = self.env.step(self.env.RETREAT_ACTION)
        self.assertEqual(is_done, True)

    def test_step_right_move_state_one_step_to_right(self):
        self.env.reset()
        s = self.env._state
        sp, r, is_done, _ = self.env.step(self.env.RIGHT_ACTION)
        self.assertEqual(sp - s, 1)
