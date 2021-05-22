import unittest
from Tasks.EightStateCollision import EightStateCollision


class TestEightStateCollision(unittest.TestCase):
    def setUp(self) -> None:
        params = {'resource_root_path': '../../Resources'}
        self.experiment = EightStateCollision(**params)
        self.experiment.reset()

    def tearDown(self) -> None:
        ...

    def test_load_feature_rep_evaluated_shape_is_(self):
        feature_rep_arr = self.experiment.load_feature_rep()
        self.assertEqual(feature_rep_arr.shape, (8, 6))
