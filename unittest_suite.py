import unittest
from Tests.Algorithms.TestTD import TestTD
from Tests.Environments.TestChain import TestChain
from Tests.Tasks.TestEightStateCollision import TestEightStateCollision

test_suite = unittest.TestSuite()
test_suite.addTest(unittest.makeSuite(TestChain))
test_suite.addTest(unittest.makeSuite(TestEightStateCollision))
test_suite.addTest(unittest.makeSuite(TestTD))
runner = unittest.TextTestRunner()
runner.run(test_suite)
