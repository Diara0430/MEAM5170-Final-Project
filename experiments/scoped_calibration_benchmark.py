"""Compare planner-owned reuse with fresh-solver calibration."""
from unittest.mock import patch
from experiments.scoped_calibration import ScopedCalibrationMPC
from experiments import reuse_calibration_benchmark

if __name__ == '__main__':
    # The benchmark harness patches factories for instrumentation; the planner
    # itself uses no process-wide patch and is independently usable.
    with patch.object(reuse_calibration_benchmark,'ReuseCalibrationMPC',ScopedCalibrationMPC):
        reuse_calibration_benchmark.main()
