"""Regression checks for the experimental calibration solver scope."""
import unittest
from unittest.mock import patch
import osqp
from controllers.fixed_intercept import CalibrationMPC
from experiments.reuse_calibration import ReuseCalibrationMPC


class SolverPatchScopeTest(unittest.TestCase):
    def test_constructor_restored_when_calibration_raises(self):
        obj = ReuseCalibrationMPC.__new__(ReuseCalibrationMPC)
        original = osqp.OSQP
        with patch.object(CalibrationMPC, 'solve', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                obj.solve(None)
        self.assertIs(osqp.OSQP, original)


if __name__ == '__main__':
    unittest.main()
