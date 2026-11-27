"""Regression checks for the experimental calibration solver scope."""
import unittest
from unittest.mock import patch
import osqp
import numpy as np
from scipy import sparse
from controllers.fixed_intercept import CalibrationMPC
from experiments.reuse_calibration import ReuseCalibrationMPC, ResetRhoCalibrationMPC


class SolverPatchScopeTest(unittest.TestCase):
    def test_constructor_restored_when_calibration_raises(self):
        obj = ReuseCalibrationMPC.__new__(ReuseCalibrationMPC)
        original = osqp.OSQP
        with patch.object(CalibrationMPC, 'solve', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                obj.solve(None)
        self.assertIs(osqp.OSQP, original)

    def test_reuses_bounds_only_and_rebuilds_changed_matrix_or_settings(self):
        obj = ResetRhoCalibrationMPC.__new__(ResetRhoCalibrationMPC)
        obj.cache_hits = obj.cache_misses = 0
        def simulated_calibration(*args):
            P=sparse.csc_matrix([[2.]])
            A=sparse.csc_matrix([[1.]])
            for value, matrix, iterations in [(0.0,P,2000),(0.2,P,2000),
                                               (0.2,sparse.csc_matrix([[3.]]),2000),
                                               (0.2,P,1000)]:
                solver=osqp.OSQP()
                solver.setup(P=matrix,q=np.array([0.]),A=A,l=np.array([value]),
                             u=np.array([1.]),verbose=False,max_iter=iterations)
                result=solver.solve(raise_error=False)
                self.assertEqual(result.info.status_val,1)
                self.assertGreaterEqual(result.x[0],value-1e-3)
        with patch.object(CalibrationMPC,'solve',side_effect=simulated_calibration):
            obj.solve(None)
        self.assertEqual(obj.cache_hits,1)
        self.assertEqual(obj.cache_misses,3)


if __name__ == '__main__':
    unittest.main()
