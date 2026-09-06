# TINTIN experiment harness

Reproducible baseline study for the TINTIN lunar-landing project. Regenerates
every figure used on <https://thejerrycheng.github.io/tintin.html>.

```bash
python scripts/tools/run_pid_baseline.py --trials 500      # ~4 min, NumPy only
```

- `scripts/tools/rocketgym_planar.py` — the planar (x, z, θ) slice of the 6-DoF
  RocketGym environment at the report's Table I parameters, plus the report's
  cascaded PD baseline. NumPy only; no MuJoCo needed.
- `scripts/tools/run_pid_baseline.py` — 500-descent Monte Carlo, the lateral
  gain grid, the deployment-envelope sweep, single-episode traces, and the
  propellant-fraction null result. Writes vector PDF + PNG into `docs/figures/`
  and a machine-readable `tintin_pid_results.json`.

Headline numbers (seed 0, 500 descents):

| metric | value |
| --- | --- |
| success rate | 70.2 % |
| median touchdown offset | 6.5 m |
| median touchdown tilt | 0.89° |
| median touchdown \|v_z\| | 1.00 m/s |
| median propellant used | 3.80 % of load |
| failures | 17.8 % drift-out, 10.8 % timeout, 1.2 % tip-over |

The browser sandbox on the website (`assets/js/tintin_lander.js` there) is a
line-by-line transliteration of `rocketgym_planar.py`; the two agree to within
Monte Carlo noise.
