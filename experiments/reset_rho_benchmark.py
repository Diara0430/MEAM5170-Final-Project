"""Benchmark full-grid reuse with rho reset; single-threaded only."""
from unittest.mock import patch
from experiments.reuse_calibration import ResetRhoCalibrationMPC
from experiments import reuse_calibration_benchmark

if __name__ == "__main__":
    with patch.object(reuse_calibration_benchmark, "ReuseCalibrationMPC", ResetRhoCalibrationMPC):
        reuse_calibration_benchmark.main()
