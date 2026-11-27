# Exploratory palm-normal rotation for rotating cuboids

This pilot rotates the desired palm frame about its local z axis by 0, -10, +10, -20, or +20 degrees. The ball-flight observations, launch seed, original hand closure, actuator limits, and controller are retained. Angles are applied in a rotated order for each launch. No machine learning is used.

## Screening results

Seed 1162, cases 0–7, normal observation (2 mm position noise and 20 ms latency):

| Angle (deg) | Applied plan | Held | Held and all constraints satisfied |
|---:|---:|---:|---:|
| 0 | 7/8 | 5/8 | 3/8 |
| -10 | 8/8 | 7/8 | 6/8 |
| +10 | 7/8 | 6/8 | 5/8 |
| -20 | 7/8 | 7/8 | 6/8 |
| +20 | 6/8 | 6/8 | 4/8 |

The data are exploratory only. Eight launches cannot establish superiority. Each angle independently runs the rolling planner, so selected intercepts, execution times and realized speeds can differ: these are not controlled fixed-point, fixed-time comparisons. Wall-clock command deadlines can also affect repeated outcomes. Contact quantities are absent if contact did not occur; avoid comparing medians over mismatched subsets. A larger development set, a genuinely fixed intercept and terminal-speed comparison, and independent confirmation are still needed. The pilot does not select a new default controller.

Full per-trial metrics, including contact-window peak acceleration, force and impulse, are in `screen/trials.csv`; the preliminary seed 1161 run is in `pilot/trials.csv`.

Run from the repository root (requires the scene dependencies):

```powershell
$env:OPENBLAS_NUM_THREADS='1'
python -m experiments.palm_angle_study --seed 1162 --cases 8 --profile normal --output-dir runs/palm_angle
```
