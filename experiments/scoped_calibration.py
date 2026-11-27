"""Planner-owned full-grid solver reuse for experimental calibration."""
import threading
import numpy as np
import osqp
from controllers.fixed_intercept import CalibrationMPC


class CalibrationWorkspace:
    """A single-owner OSQP workspace; callers synchronize concurrent access."""
    def __init__(self, initial_rho=0.1):
        if not np.isfinite(initial_rho) or initial_rho <= 0:
            raise ValueError('initial_rho must be finite and positive')
        self.initial_rho = float(initial_rho)
        self.key = self.solver = None
        self.hits = self.misses = 0

    def clear(self):
        self.key = self.solver = None

    def setup(self, P, q, A, l, u, **options):
        options = dict(options)
        options.setdefault('rho', self.initial_rho)
        rho = float(options['rho'])
        if not np.isfinite(rho) or rho <= 0:
            raise ValueError('rho must be finite and positive')
        key=(P.shape,P.indptr.tobytes(),P.indices.tobytes(),P.data.tobytes(),q.tobytes(),
             A.shape,A.indptr.tobytes(),A.indices.tobytes(),A.data.tobytes(),tuple(sorted(options.items())))
        if self.key == key:
            self.solver.update(l=l,u=u)
            self.solver.update_settings(rho=rho)
            self.solver.warm_start(x=np.zeros(P.shape[0]),y=np.zeros(A.shape[0]))
            self.hits += 1
        else:
            solver=osqp.OSQP()
            solver.setup(P=P,q=q,A=A,l=l,u=u,**options)
            self.key,self.solver=key,solver
            self.misses += 1
        return self.solver


class ScopedCalibrationMPC(CalibrationMPC):
    """Experimental: reuse only within one calibration call, with explicit rho."""
    def __init__(self,*args,initial_rho=0.1,**kwargs):
        super().__init__(*args,**kwargs)
        self.workspace=CalibrationWorkspace(initial_rho)
        self._calibration_lock=threading.Lock()

    @property
    def cache_hits(self): return self.workspace.hits

    @property
    def cache_misses(self): return self.workspace.misses

    def _setup_qp_solver(self,P,q,A,l,u,**options):
        return self.workspace.setup(P,q,A,l,u,**options)

    def solve(self,*args,**kwargs):
        # 同一规划器串行调用；不同规划器各自拥有独立求解器。
        with self._calibration_lock:
            self.workspace.clear()
            try:
                return super().solve(*args,**kwargs)
            finally:
                self.workspace.clear()
