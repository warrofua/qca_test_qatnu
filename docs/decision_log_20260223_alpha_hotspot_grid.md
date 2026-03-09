# Decision Log: Alpha-Hotspot Grid Continuation

Date: 2026-02-23
Scope: Continue hotspot sensitivity work by adding alpha dependence and higher lambda resolution.

## Mnemolog decision protocol (applied)
Bootstrap reference from `https://mnemolog.com/robots.txt`:
1. `GET https://mnemolog.com/robots.txt`
2. `GET https://mnemolog.com/.well-known/agent.json`
3. `GET https://mnemolog.com/agents.txt`
4. `GET https://mnemolog.com/api/agents/capabilities`

Execution style: explicit objective, bounded experiment matrix, artifact-first outputs, post-run interpretation.

## Objective
Validate whether hotspot multiplier sensitivity survives when:
- Increasing scan density from 40 to 80 lambda points.
- Extending from a single alpha (`0.8`) to alpha grid `{0.6, 0.8, 1.0, 1.2}`.

## Experiment
- Engine: `ParameterScanner.scan_lambda_parallel` (existing production path).
- Matrix: 12 runs = 4 alpha values x 3 hotspot multipliers.
- Fixed parameters: `N=4`, `lambda_min=0.1`, `lambda_max=1.5`, `points=80`.
- Output root: `outputs/hotspot_alpha_multiplier_grid_20260222-193347_N4_pts80/`.

Generated artifacts:
- `critical_points_alpha_hotspot_grid.csv`
- `critical_point_spans_by_alpha.csv`
- `heatmap_lambda_c1.png`
- `heatmap_lambda_revival.png`
- `heatmap_lambda_c2.png`
- `lambda_revival_vs_hotspot_by_alpha.png`
- 12 per-run scan CSVs (`scan_a*_hotspotx*.csv`)

## Outcome summary

| alpha | span(lambda_c1) | span(lambda_revival_first) | span(lambda_revival_global) | span(lambda_c2) |
|---:|---:|---:|---:|---:|
| 0.6 | 0.225 | 0.799 | 0.911 | 0.633 |
| 0.8 | 0.193 | 0.305 | 0.305 | 0.000 |
| 1.0 | 0.161 | 1.008 | 0.257 | 0.000 |
| 1.2 | 0.161 | 0.280 | 0.225 | 0.260 |

Interpretation:
- Hotspot multiplier dependence is robust for `lambda_c1`.
- `lambda_revival_first` can jump between separated local minima.
- `lambda_revival_global` is often more stable but still multiplier-sensitive.
- `lambda_c2` stability is alpha-slice dependent, not universal.

## Decisions from this run
1. Keep `--hotspot-multiplier` exposed as a first-class experiment parameter.
2. Treat any single-value `3.0x` result as conditional unless accompanied by sensitivity brackets.
3. Flag revival-point extraction as method-sensitive in documentation and avoid over-claiming uniqueness.
4. Lock official reporting to `lambda_revival_global` (post-violation global minimum); retain first-local revival as diagnostic.

## Tooling hardening completed
- Added reusable runner: `scripts/hotspot_alpha_grid.py`.
- Fixed small-point scan edge case in `scanners.py`:
  - default-range densification now falls back to uniform spacing when `num_points < 10`.
- Added script robustness for missing critical points:
  - heatmap generation coerces non-numeric values to `NaN` before plotting.
- Added dual revival exports:
  - `lambda_revival_first`/`residual_min_first` and `lambda_revival_global`/`residual_min_global`.
  - New artifact: `heatmap_lambda_revival_global.png`.

## Recommended next run
1. Add a secondary revival detector (e.g., first post-violation minimum and global post-violation minimum side-by-side).
2. Repeat the same grid at `points=100` for tighter threshold placement.
3. Extend with `kappa` sensitivity to test coupling between hotspot tuning and numerical stabilizer terms.

## Follow-up executed (same date)
- Ran canonical high-resolution check:
  - `outputs/hotspot_alpha_multiplier_grid_20260222-200209_N4_pts100/`
- Observation:
  - For alpha `0.8`, multipliers `{2.0, 3.0, 4.0}`, `revival_gap=0` for all runs (`lambda_revival_first == lambda_revival_global`).
- Action:
  - Reporting rule lock to global minimum was applied without changing canonical alpha=0.8 conclusions.
