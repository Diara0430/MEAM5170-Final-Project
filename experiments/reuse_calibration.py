"""Experimental solver reuse inside a single full-grid calibration only.

This temporarily patches a module-global constructor; use only in a single-threaded
experiment. The normal controller is not changed.
"""
from unittest.mock import patch
import numpy as np
import osqp
from controllers.fixed_intercept import CalibrationMPC


class ReuseCalibrationMPC(CalibrationMPC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_hits = self.cache_misses = 0

    def solve(self, *args, **kwargs):
        original = osqp.OSQP
        owner = self
        cache = {}
        class ReusedSolver:
            def setup(self, *, P, q, A, l, u, **options):
                # 只有矩阵、线性目标和设置完全一致才允许更新上下界。
                key = (P.shape, P.indptr.tobytes(), P.indices.tobytes(), P.data.tobytes(),
                       q.tobytes(), A.shape, A.indptr.tobytes(), A.indices.tobytes(),
                       A.data.tobytes(), tuple(sorted(options.items())))
                if cache.get('key') == key:
                    self.solver = cache['solver']
                    self.solver.update(l=l, u=u)
                    # Match independent solver initialization; adaptive internal solver
                    # state can still differ and is checked experimentally.
                    self.solver.warm_start(x=np.zeros(P.shape[0]), y=np.zeros(A.shape[0]))
                    owner.cache_hits += 1
                else:
                    self.solver = original()
                    self.solver.setup(P=P, q=q, A=A, l=l, u=u, **options)
                    cache.update(key=key, solver=self.solver)
                    owner.cache_misses += 1
            def warm_start(self, *a, **k):
                return self.solver.warm_start(*a, **k)
            def solve(self, *a, **k):
                return self.solver.solve(*a, **k)
        with patch('controllers.catching_mpc.osqp.OSQP', ReusedSolver):
            return super().solve(*args, **kwargs)
