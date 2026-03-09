# Decision Log: Hotspot Sensitivity and Lambda Validation

Date: 2026-02-23
Scope: N=4, alpha=0.8 workflow; tester/scanner/app plumbing; hotspot multiplier sensitivity.

## Mnemolog bootstrap used
Following `https://mnemolog.com/robots.txt`, the run used this discovery sequence before making code decisions:
1. `GET https://mnemolog.com/robots.txt`
2. `GET https://mnemolog.com/.well-known/agent.json`
3. `GET https://mnemolog.com/agents.txt`
4. `GET https://mnemolog.com/api/agents/capabilities`

This was used as a decision protocol: discover constraints first, then execute bounded changes with explicit artifacts.

## Decision framing
Observed high-priority repo risks:
- `tester.py` used a lambda conversion that algebraically reduces to identity:
  - `log2(1 + 2**depth - 1) == depth`
- Hotspot multiplier was hardcoded to `3.0` in both tester and scanner paths, preventing sensitivity checks.

Decision:
- Make hotspot multiplier configurable end-to-end.
- Remove the no-op lambda conversion in tester validation.
- Run an explicit sensitivity experiment for multipliers `{2.0, 3.0, 4.0}`.

## Code changes applied
- `tester.py`
  - Added `hotspot_multiplier` parameter to `QCATester`.
  - Switched hotspot prep to `lambda * hotspotMultiplier`.
  - Replaced no-op lambda conversion with direct use of measured depth.
- `scanners.py`
  - Added `hotspot_multiplier` to `scan_lambda_parallel`, `scan_2d_phase_space`, and `run_single_point`.
  - Removed hardcoded `3.0` in worker hotspot prep.
- `app.py`
  - Added CLI flag `--hotspot-multiplier` (default `3.0`).
  - Threaded multiplier through production run, scanner calls, tester calls, run tagging, and summary output.
- `README.md`
  - Documented `--hotspot-multiplier` and hotspot scaling semantics.
- `docs/results.md`
  - Added a hotspot sensitivity result block with run artifacts.

## Experiment execution
Command runtime:
- Interpreter: `.venv/bin/python`
- Scan: 3 x serial 40-point lambda sweeps, N=4, alpha=0.8, lambda in [0.1, 1.5]

Artifacts:
- `outputs/hotspot_sensitivity_20260222-192951_N4_alpha0.80_pts40/critical_points_hotspot_sensitivity.csv`
- `outputs/hotspot_sensitivity_20260222-192951_N4_alpha0.80_pts40/hotspot_multiplier_sensitivity.png`
- Per-multiplier scans in same directory:
  - `scan_hotspotx2.0.csv`
  - `scan_hotspotx3.0.csv`
  - `scan_hotspotx4.0.csv`

## Measured outcome

| hotspot_multiplier | lambda_c1 | lambda_revival | lambda_c2 |
|---:|---:|---:|---:|
| 2.0 | 0.357143 | 0.550000 | 0.633333 |
| 3.0 | 0.228571 | 0.357143 | 0.633333 |
| 4.0 | 0.164286 | 0.260714 | 0.633333 |

Interpretation:
- `lambda_revival` and `lambda_c1` shift strongly with multiplier.
- `lambda_c2` was stable in this 40-point sweep.
- The `3.0x` hotspot is therefore a tunable modeling choice, not a neutral constant.

## Next decisions queued
1. Repeat sensitivity at higher resolution (`points=100`) to reduce threshold quantization.
2. Run the same sweep across alpha in `{0.6, 1.0, 1.2}`.
3. Add multiplier and kappa sensitivity sections to `docs/status_202511.tex`.
