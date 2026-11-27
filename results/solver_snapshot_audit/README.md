# Solver audit on identical planning states

## Controlled comparison

Six previously discrepant trajectory/observation cases are re-observed, and for each candidate speed the same `CatchingMPC.solve` input is evaluated sequentially with: (1) a fresh OSQP instance; (2) the earlier experimental reuse with zeroed primal and dual iterates; and (3) reuse with the same iterate reset plus `update_settings(rho=0.1)` before each reused solve. These are targeted diagnostic states, not independently sampled performance tests.

Each candidate uses an unchanged MuJoCo state (qpos, qvel, ctrl and qacc_warmstart checked by digest), identical arguments and identical sparse QP values, bounds and objective (digest checked). Full nonlinear position, collision, stopping and speed checks run unchanged. The process-wide OSQP constructor is patched temporarily and is **single-threaded only**. Diagnostic tracing and three solves make elapsed times and calibration deadlines invalid for performance evaluation; no catch rate or real-time claim is made here.

| Case | Candidates | Identical QP hashes | Fresh vs original reuse: different acceptance | Fresh vs reset-rho reuse: different acceptance |
|---|---:|---:|---:|---:|
| 1212, case 3, noisy | 16 | 16 | 2 | 0 |
| 1212, case 5, noisy | 16 | 16 | 1 | 0 |
| 1213, case 6, noisy | 16 | 16 | 1 | 0 |
| 1213, case 3, normal | 16 | 16 | 2 | 0 |
| 1213, case 6, normal | 16 | 16 | 2 | 0 |
| 1213, case 7, normal | 16 | 16 | 1 | 0 |
| **Total** | **96** | **96** | **9** | **0** |

On candidates for which fresh and reset-rho solvers both returned solved, the maximum absolute difference in the QP solution vector was 0.0 in this run. This is a diagnostic result for these states, not a proof of equivalence over all states, solver versions or execution conditions.

## Where the nine differences arose

Seven were rejected by one method at the nonlinear 12 mm terminal-position check; another failed the nonlinear terminal-speed tolerance; the last had different solver status. For example, seed 1212 case 3 noisy at 0.15 m/s yielded `solved` for both, but the reused solution had 20.246 mm nonlinear endpoint error and was rejected, while the fresh solution passed. At 0.45 m/s the reused endpoint error was 12.183 mm and was also rejected. Thus a `solved` QP status does not itself certify the robot's nonlinear endpoint.

The diagnostic supports retained/adaptive rho state as a plausible contributor, because explicitly resetting rho restored all recorded decisions and QP solutions in these six cases. It does not independently prove the complete internal causal mechanism. The tested reset value was 0.1, matching the OSQP default used for this run. No production control change is made.

## Data and reproduction

Each case directory contains `candidates.csv` with state and QP hashes, acceptance, rejection traces, status, residuals, iteration and rho update information. `qp_*.npz` files contain sparse QP inputs and the recorded MuJoCo state, flight estimate and reference values. All 96 archived QP hashes and fresh-solver statuses have been independently replay-checked. The solver settings are the same as in `controllers/catching_mpc.py`: max_iter=2000, eps_abs=eps_rel=1e-5, polishing disabled, remaining settings at installed OSQP defaults.

From the repository root:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
python -m experiments.solver_snapshot_audit --seed 1212 --case 3 --profile noisy --output-dir runs/snapshot_1212_3
python -m experiments.solver_snapshot_audit --seed 1213 --case 6 --profile normal --output-dir runs/snapshot_1213_6
```

Next verify reset-rho reuse on independent unseen states, including matrices and settings that change, and then benchmark it against the unmodified full-grid controller under identical 100 ms deadlines. Compare all candidate decisions, actual target speeds, timings and catch/constraint outcomes. A thread-safe implementation must replace this instrumentation before any default use. No machine learning is used.
