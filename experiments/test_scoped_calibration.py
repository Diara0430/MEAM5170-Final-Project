"""Tests for isolated calibration solver workspaces."""
import concurrent.futures
import unittest
import numpy as np
import osqp
from scipy import sparse
from experiments.scoped_calibration import CalibrationWorkspace


class ScopedWorkspaceTest(unittest.TestCase):
    def test_rebuild_when_matrix_objective_or_settings_change(self):
        w=CalibrationWorkspace(initial_rho=.2)
        P=sparse.csc_matrix([[2.]])
        A=sparse.csc_matrix([[1.]])
        q=np.array([0.]); bounds=np.array([1.])
        for low in (0.,.2):
            result=w.setup(P,q,A,np.array([low]),bounds,verbose=False,max_iter=2000).solve(raise_error=False)
            self.assertEqual(result.info.status_val,1)
            self.assertGreaterEqual(result.x[0],low-1e-3)
        self.assertEqual((w.hits,w.misses),(1,1))
        w.setup(sparse.csc_matrix([[3.]]),q,A,np.array([0.]),bounds,verbose=False,max_iter=2000)
        w.setup(P,np.array([.3]),A,np.array([0.]),bounds,verbose=False,max_iter=2000)
        w.setup(P,np.array([.3]),A,np.array([0.]),bounds,verbose=False,max_iter=1000)
        self.assertEqual((w.hits,w.misses),(1,4))
        w.clear();self.assertIsNone(w.solver)

    def test_independent_workspaces_in_parallel(self):
        original=osqp.OSQP
        def job(value):
            w=CalibrationWorkspace();P=sparse.csc_matrix([[2.]])
            A=sparse.csc_matrix([[1.]]);q=np.array([0.]);upper=np.array([1.])
            for _ in range(4):
                result=w.setup(P,q,A,np.array([value]),upper,verbose=False).solve(raise_error=False)
                if result.info.status_val!=1: raise AssertionError(result.info.status)
            return result.x[0],w.hits,w.misses
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            result=list(pool.map(job,[.2,.7]))
        for (x,hits,misses),target in zip(result,[.2,.7]):
            self.assertAlmostEqual(x,target,delta=.002)
            self.assertEqual((hits,misses),(3,1))
        self.assertIs(osqp.OSQP,original)

    def test_rejects_invalid_rho(self):
        for value in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError): CalibrationWorkspace(value)

if __name__=='__main__': unittest.main()
