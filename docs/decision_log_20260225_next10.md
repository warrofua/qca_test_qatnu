# Decision Log: Next-10 Closure (2026-02-25)

## Scope
Close the ranked "next 10" items from:
- `outputs/critical_slowing_star_hotspot_sensitivity_N5_iter_20260224/next_10_steps_ranked.csv`

Main goals:
- finish star high-lambda extension checks,
- close `tester.py` lambda semantics,
- install backend parity gate,
- run kappa/deltaB sensitivity on refined star window,
- promote auto backend defaults for deep-time scans.

## Step 1 completion: star high-lambda extension

Runs:
- `outputs/critical_slowing_star_extend_hi_N5_hotspot3.0_chi4_iter_20260225/summary.csv`
- `outputs/critical_slowing_star_extend_hi_N5_hotspot3.0_chi5_iter_20260225/summary.csv`
- `outputs/critical_slowing_star_extend_hi_N5_hotspot4.0_chi4_iter_20260225/summary.csv`
- `outputs/critical_slowing_star_extend_hi_N5_hotspot4.0_chi5_iter_20260225/summary.csv`

Readout:
- `hotspot=3.0`: extension-window local max at `lambda=1.80` (`chi=4`) and `1.70` (`chi=5`).
- `hotspot=4.0`: extension-window local max at `lambda=1.75` (`chi=4`) and `1.80` (`chi=5`).

Decision:
- star peak is not reliably pinned within `[1.50,1.70]` under protocol/cutoff changes.

## Step 6 closure: tester lambda semantics

Code:
- `tester.py`
  - added explicit `_lambda_from_depth_metric` identity mapper.
  - validation now logs both `measuredDepth*` and `measuredLambda*`.

Canonical comparator artifact:
- `outputs/tester_lambda_semantics_20260225/summary.json`

Result:
- residual(current) = `0.193560504777191`
- residual(legacy expression) = `0.193560504777191`
- delta = `1.11e-16`

Decision:
- no physics change from legacy transform; issue is semantic clarity, not numeric bug.

## Step 7 closure: backend parity gate

Code:
- `scripts/backend_regression_check.py` (new)
  - runs dense and iterative scans on identical settings.
  - computes max absolute metric diff and fails if diff > threshold (default `1e-8`).

Anchor run:
- `outputs/backend_regression_check_N5_chi4_20260225/summary.json`
- `outputs/backend_regression_check_N5_chi4_20260225/report.md`

Result:
- PASS, `max_abs_diff = 1.1095e-10 < 1e-8`.

Decision:
- iterative backend accepted as production-safe for deep-time scan metrics at tested anchors.

## Steps 8-9 closure: kappa and deltaB sweeps

Kappa runs:
- `outputs/critical_slowing_star_kappa_N5_hotspot3_k0.05_chi4_iter_20260225/summary.csv`
- `outputs/critical_slowing_star_kappa_N5_hotspot3_k0.1_chi4_iter_20260225/summary.csv`
- `outputs/critical_slowing_star_kappa_N5_hotspot3_k0.2_chi4_iter_20260225/summary.csv`
- `outputs/critical_slowing_star_kappa_N5_hotspot3_k0.05_chi5_iter_20260225/summary.csv`
- `outputs/critical_slowing_star_kappa_N5_hotspot3_k0.1_chi5_iter_20260225/summary.csv`
- `outputs/critical_slowing_star_kappa_N5_hotspot3_k0.2_chi5_iter_20260225/summary.csv`

DeltaB runs:
- `outputs/critical_slowing_star_deltaB_N5_hotspot3_d4.0_chi4_iter_20260225/summary.csv`
- `outputs/critical_slowing_star_deltaB_N5_hotspot3_d5.0_chi4_iter_20260225/summary.csv`
- `outputs/critical_slowing_star_deltaB_N5_hotspot3_d6.0_chi4_iter_20260225/summary.csv`
- `outputs/critical_slowing_star_deltaB_N5_hotspot3_d4.0_chi5_iter_20260225/summary.csv`
- `outputs/critical_slowing_star_deltaB_N5_hotspot3_d5.0_chi5_iter_20260225/summary.csv`
- `outputs/critical_slowing_star_deltaB_N5_hotspot3_d6.0_chi5_iter_20260225/summary.csv`

Consolidation:
- `outputs/critical_slowing_next10_completion_20260225/star_peak_summary.csv`
- `outputs/critical_slowing_next10_completion_20260225/sensitivity_spans_by_chi.csv`
- `outputs/critical_slowing_next10_completion_20260225/report.md`

Peak-lambda spans (window-local):
- kappa: `0.10` (`chi=4`), `0.15` (`chi=5`)
- deltaB: `0.00` (`chi=4`), `0.05` (`chi=5`)
- hotspot extension (`3.0 -> 4.0`): `0.05` (`chi=4`), `0.10` (`chi=5`)

Decision:
- star peak placement is materially sensitive to phenomenological controls, especially `kappa` at `chi=5`.

## Step 10 closure: default backend promotion

Code:
- `scripts/critical_slowing_scan.py`
  - `--solver-backend` default: `auto` (previously `dense`)
  - `--auto-dense-threshold` default: `8000` (previously `12000`)

Docs:
- `README.md`: deep-time example updated to reflect auto-backend default and dense opt-in policy.
- `docs/results.md`: Feb 25 closure section added with artifacts and readout.

Decision:
- dense-only guidance for N5 deep-time scans is deprecated; iterative is now default-selected for N5 path/star chi4.
