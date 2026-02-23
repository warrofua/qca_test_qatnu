# Current Results Snapshot

## N = 4, α = 0.8, 100 λ points (run_20251116-154356_N4_alpha0.80)
- **Critical points**: λ_c1 ≈ 0.203, λ_rev ≈ 1.058 (residual ≈ 13.4%), λ_c2 ≈ 1.095.
- **SRQID checks**: v_LR ≈ 1.96, max no-signalling deviation ≈ 2.1×10⁻¹⁶, energy drift ≈ 1.4×10⁻¹⁴.
- **Mean-field overlay**: best-fit configuration at λ ≈ 1.058 shows exact and mean-field Ramsey traces re-locking during the revival.
- **Spin-2 PSD**: χ-informed PSD currently undershoots the expected 1/k² slope (measured power ≈ −0.024); further tuning of the χ→tier mapping is underway.

![N4 phase diagram](../figures/run_20251116-154356_N4_alpha0.80/phase_diagram_run_20251116-154356_N4_alpha0.80.png)

- Ramsey overlay: `figures/run_20251116-154356_N4_alpha0.80/ramsey_overlay_run_20251116-154356_N4_alpha0.80.png`
- Spin-2 PSD: `figures/run_20251116-154356_N4_alpha0.80/spin2_psd_run_20251116-154356_N4_alpha0.80.png`
- CSV data: `outputs/run_20251116-154356_N4_alpha0.80/scan_run_20251116-154356_N4_alpha0.80.csv`
- Summary log: `outputs/run_20251116-154356_N4_alpha0.80/summary_run_20251116-154356_N4_alpha0.80.txt`

## N = 5, α = 0.8, 50 λ points (run_20251116-160216_N5_alpha0.80)
- **Critical points**: λ_c1 ≈ 0.232, λ_rev ≈ 0.338 (residual ≈ 10.2%), λ_c2 ≈ 1.033.
- **SRQID checks**: v_LR ≈ 1.96, no-signalling ≈ 4.2×10⁻¹⁶, energy drift ≈ 5.0×10⁻¹⁴.

![N5 phase diagram](../figures/run_20251116-160216_N5_alpha0.80/phase_diagram_run_20251116-160216_N5_alpha0.80.png)

- Ramsey overlay: `figures/run_20251116-160216_N5_alpha0.80/ramsey_overlay_run_20251116-160216_N5_alpha0.80.png`
- Spin-2 PSD: `figures/run_20251116-160216_N5_alpha0.80/spin2_psd_run_20251116-160216_N5_alpha0.80.png`
- CSV data: `outputs/run_20251116-160216_N5_alpha0.80/scan_run_20251116-160216_N5_alpha0.80.csv`
- Summary log: `outputs/run_20251116-160216_N5_alpha0.80/summary_run_20251116-160216_N5_alpha0.80.txt`
- **2D phase space**: λ∈[0.1,1.2], α∈{0.2,…,1.2}; residual floor stays pinned at λ≈0.1 with magnitude {3.3, 6.6, 9.9, 13.1, 16.3, 19.5}×10⁻³ for α={0.2,…,1.2}. Heatmap: `figures/run_20251116-160216_N5_alpha0.80/phase_space_run_20251116-160216_N5_alpha0.80.png`; raw grid: `outputs/run_20251116-160216_N5_alpha0.80/phase_space_run_20251116-160216_N5_alpha0.80.csv`.

## N = 4 (cycle graph), α = 0.8, 40 λ points (run_20251117-170531_N4_cycle_alpha0.80)
- **Topology**: 4-site cycle (periodic boundary). Probes (0, 1) are equivalent by symmetry.
- **Critical points**: λ_c1 ≈ 0.261 (shifted right vs. open chain’s 0.203), λ_rev ≈ 0.486 with essentially zero residual (outer and inner clocks lock perfectly), λ_c2 ≈ 1.0 (slightly earlier catastrophic inversion).
- **SRQID checks**: v_LR ≈ 2.45 (higher due to degree=2 everywhere and the closed loop), no-signalling ≈ 1.1×10⁻¹⁵, energy drift ≈ 4.4×10⁻¹⁴.
- **Artifacts**: `figures/run_20251117-170531_N4_cycle_alpha0.80/phase_diagram_run_20251117-170531_N4_cycle_alpha0.80.png`, `outputs/run_20251117-170531_N4_cycle_alpha0.80/summary_run_20251117-170531_N4_cycle_alpha0.80.txt`.

## N = 4 (pyramid/star), α = 0.8, 40 λ points (run_20251117-182555_N4_pyramid_alpha0.80)
- **Topology**: central hub (degree 3) connected to 3 leaves; probes (0 hub, 1 leaf).
- **Critical points**: λ_c1 ≈ 0.164 (shifted left because the center accumulates Λ immediately), λ_rev ≈ 1.30 (residual ≈ 0.148), λ_c2 ≈ 1.40.
- **SRQID checks**: v_LR ≈ 1.05 (slower because only one promotion hub), no-signalling ≈ 4.2×10⁻¹⁶, energy drift ≈ 1.33×10⁻¹⁴.
- **Ω behavior**: inner (hub) clock runs much faster than leaf at revival, hence the large residual even at λ≈1.3.
- **Artifacts**: `figures/run_20251117-182555_N4_pyramid_alpha0.80/phase_diagram_run_20251117-182555_N4_pyramid_alpha0.80.png`, `outputs/run_20251117-182555_N4_pyramid_alpha0.80/summary_run_20251117-182555_N4_pyramid_alpha0.80.txt`.

## Hotspot multiplier sensitivity (N = 4, α = 0.8, 40 λ points each)
- **Experiment date**: February 23, 2026
- **Question**: does hardcoded hotspot scaling (`3.0×`) bias phase landmarks?
- **Runs**: hotspot multipliers `{2.0, 3.0, 4.0}` over λ∈[0.1, 1.5].
- **Critical-point shifts**:
  - `2.0×`: λ_c1 ≈ 0.357, λ_rev ≈ 0.550, λ_c2 ≈ 0.633
  - `3.0×`: λ_c1 ≈ 0.229, λ_rev ≈ 0.357, λ_c2 ≈ 0.633
  - `4.0×`: λ_c1 ≈ 0.164, λ_rev ≈ 0.261, λ_c2 ≈ 0.633
- **Conclusion**: λ_rev and λ_c1 shift materially with multiplier, so `3.0×` is a tuning choice rather than an invariant.
- **Artifacts**: `outputs/hotspot_sensitivity_20260222-192951_N4_alpha0.80_pts40/critical_points_hotspot_sensitivity.csv`, `outputs/hotspot_sensitivity_20260222-192951_N4_alpha0.80_pts40/hotspot_multiplier_sensitivity.png`.

## Hotspot multiplier × α grid (N = 4, 80 λ points each; run tag `hotspot_alpha_multiplier_grid_20260222-193347_N4_pts80`)
- **Experiment date**: February 23, 2026
- **Grid**: α∈{0.6, 0.8, 1.0, 1.2}, hotspot multiplier∈{2.0, 3.0, 4.0}, λ∈[0.1,1.5].
- **Primary observation**: phase landmarks remain hotspot-sensitive after increasing scan resolution.
- **Dual revival detector**:
  - `λ_rev_first`: first local minimum after residual crosses 10%; fallback to post-violation global minimum if no local minimum exists.
  - `λ_rev_global`: global minimum over the full post-violation region.
- **Reporting rule (locked)**: use `λ_rev_global` as the official `λ_rev` for summaries and figures; keep `λ_rev_first` as a diagnostic.
- **Drift spans across multipliers**:
  - α=0.6: span(λ_c1)=0.225, span(λ_rev_first)=0.799, span(λ_rev_global)=0.911, span(λ_c2)=0.633
  - α=0.8: span(λ_c1)=0.193, span(λ_rev_first)=0.305, span(λ_rev_global)=0.305, span(λ_c2)=0.000
  - α=1.0: span(λ_c1)=0.161, span(λ_rev_first)=1.008, span(λ_rev_global)=0.257, span(λ_c2)=0.000
  - α=1.2: span(λ_c1)=0.161, span(λ_rev_first)=0.280, span(λ_rev_global)=0.225, span(λ_c2)=0.260
- **Interpretation**:
  - λ_c1 shifts systematically with multiplier at every α.
  - `λ_rev_first` and `λ_rev_global` diverge in several α/multiplier slices, confirming detector-choice sensitivity.
  - λ_c2 is stable in some α slices (0.8, 1.0) but not universal.
- **Artifacts**:
  - Summary table: `outputs/hotspot_alpha_multiplier_grid_20260222-193347_N4_pts80/critical_points_alpha_hotspot_grid.csv`
  - Span metrics: `outputs/hotspot_alpha_multiplier_grid_20260222-193347_N4_pts80/critical_point_spans_by_alpha.csv`
  - Heatmaps: `outputs/hotspot_alpha_multiplier_grid_20260222-193347_N4_pts80/heatmap_lambda_c1.png`, `outputs/hotspot_alpha_multiplier_grid_20260222-193347_N4_pts80/heatmap_lambda_revival.png`, `outputs/hotspot_alpha_multiplier_grid_20260222-193347_N4_pts80/heatmap_lambda_revival_global.png`, `outputs/hotspot_alpha_multiplier_grid_20260222-193347_N4_pts80/heatmap_lambda_c2.png`
  - Revival trend plot: `outputs/hotspot_alpha_multiplier_grid_20260222-193347_N4_pts80/lambda_revival_vs_hotspot_by_alpha.png`

## Canonical high-resolution check (N = 4, α = 0.8, 100 λ points; run tag `hotspot_alpha_multiplier_grid_20260222-200209_N4_pts100`)
- **Experiment date**: February 23, 2026
- **Grid**: hotspot multipliers `{2.0, 3.0, 4.0}` with dual detector active.
- **Result**: `λ_rev_first == λ_rev_global` for all three multipliers (`revival_gap = 0`), so detector ambiguity is absent in the canonical setting.
- **Critical points (official λ_rev = global post-violation minimum)**:
  - `2.0×`: λ_c1 ≈ 0.344, λ_rev ≈ 0.550, λ_c2 ≈ 0.633
  - `3.0×`: λ_c1 ≈ 0.203, λ_rev ≈ 0.331, λ_c2 ≈ 0.633
  - `4.0×`: λ_c1 ≈ 0.151, λ_rev ≈ 0.241, λ_c2 ≈ 0.633
- **Interpretation**: hotspot sensitivity remains in λ_c1/λ_rev even after locking the revival rule.
- **Artifacts**:
  - Summary table: `outputs/hotspot_alpha_multiplier_grid_20260222-200209_N4_pts100/critical_points_alpha_hotspot_grid.csv`
  - Span metrics: `outputs/hotspot_alpha_multiplier_grid_20260222-200209_N4_pts100/critical_point_spans_by_alpha.csv`
  - Heatmaps: `outputs/hotspot_alpha_multiplier_grid_20260222-200209_N4_pts100/heatmap_lambda_revival.png`, `outputs/hotspot_alpha_multiplier_grid_20260222-200209_N4_pts100/heatmap_lambda_revival_global.png`

## Robustness sweep (δB, κ, χ) with locked global revival rule (`robustness_sweep_20260222-201426_N4_alpha0.80_pts24`)
- **Experiment date**: February 23, 2026
- **Grid**: `deltaB ∈ {3,5,7}`, `kappa ∈ {0.05,0.1,0.2}`, `bond_cutoff ∈ {3,4,5}` at `N=4`, `alpha=0.8`, `points=24`, `hotspot_multiplier=3.0`.
- **Global spans across all 27 combinations**:
  - span(λ_c1) = 0.625
  - span(λ_rev) = 1.400
  - span(λ_c2) = 1.344
  - span(mean residual) = 0.223
- **Parameter influence (mean-of-means span)**:
  - `deltaB`: λ_rev span ≈ 0.517, λ_c1 span ≈ 0.356, mean residual span ≈ 0.110
  - `kappa`: λ_rev span ≈ 0.251, λ_c1 span ≈ 0.299, mean residual span ≈ 0.123
  - `bond_cutoff` (3→5): λ_c1 span ≈ 0.0028, mean residual span ≈ 5.68e-4 (much smaller than `deltaB`/`kappa` effects in this sweep)
- **Interpretation**:
  - The phase landmarks are **not robust** against `deltaB`/`kappa` in this parameter window.
  - `bond_cutoff` contributes secondary shifts relative to `deltaB`/`kappa` here.
  - A subset of runs shows `revival_gap > 0`, reinforcing that first-local revival can diverge from global revival even though reporting is now locked to global.
- **Artifacts**:
  - Run report: `outputs/robustness_sweep_20260222-201426_N4_alpha0.80_pts24/robustness_report.md`
  - Summary table: `outputs/robustness_sweep_20260222-201426_N4_alpha0.80_pts24/robustness_summary.csv`
  - Baseline deltas: `outputs/robustness_sweep_20260222-201426_N4_alpha0.80_pts24/robustness_summary_with_deltas.csv`
  - Span table: `outputs/robustness_sweep_20260222-201426_N4_alpha0.80_pts24/robustness_spans.csv`

## No-retuning holdout protocol (N4 topology transfer check, February 23, 2026)
- **Question**: can one locked parameterization (no retuning) generalize across path/cycle/star holdouts?
- **Locked parameters**: `deltaB=6.5`, `kappa=0.2`, `hotspot_multiplier=1.5`, `k0=4`, `bond_cutoff=4`.
- **Scenarios**: `N4_cycle_alpha0.8`, `N4_star_alpha0.8`, `N4_path_alpha1.0`.
- **Protocol update**: adaptive refinement added to `scripts/no_retuning_holdout_test.py` with explicit metadata (`points_initial/final`, `refinement_steps`, `no_violation_detected`) to separate structural failures from undersampling.
- **Run command**: `QATNU_FORCE_MP=1 .venv/bin/python scripts/no_retuning_holdout_test.py --scenarios N4_cycle_alpha0.8,N4_star_alpha0.8,N4_path_alpha1.0 --max-points 6 --max-refinements 1 --refine-factor 2.0 --refine-max-points 12 --quiet-scanner`
- **Result**: `OVERALL=FAIL`.
- **Scenario outcomes**:
  - `N4_path_alpha1.0`: **PASS** (`lambda_c1=0.38`, `lambda_revival=0.94`, `lambda_c2=1.22`, `residual_min=0.112`).
  - `N4_star_alpha0.8`: **FAIL** (triplet found, but `residual_min=0.250 > 0.20` and target mismatch).
  - `N4_cycle_alpha0.8`: **FAIL** with `no_violation_detected=True` (residual stays at machine floor, so revival landmarks never activate under this lock).
- **Interpretation**: this lock does not transfer across topologies; at least one of the phenomenological knobs remains topology-dependent.
- **Artifacts**:
  - Summary CSV: `outputs/no_retuning_holdout_20260223-092900/holdout_summary.csv`
  - Protocol JSON: `outputs/no_retuning_holdout_20260223-092900/preregistered_protocol.json`
  - Report: `outputs/no_retuning_holdout_20260223-092900/holdout_report.md`

## Correlator spin-2 topology sweep (new proxy, February 23, 2026)
- **What changed**: spin-2 proxy now uses connected bond-bond correlators with coarse-graining (`geometry.py::spin2_from_bond_correlators`) instead of only χ-profile tiling.
- **N4 sweep A** (`hotspot_multiplier=1.5`, λ = `{0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.2,1.4}`):
  - `N4_star_alpha0.8`: best slope ≈ `-2.258` (best |slope+2| ≈ `0.258`)
  - `N4_path_alpha0.8`: best slope ≈ `-3.615` (best |slope+2| ≈ `1.615`)
  - `N4_cycle_alpha0.8`: best slope ≈ `-3.685` (best |slope+2| ≈ `1.685`)
- **N4 sweep B** (`hotspot_multiplier=3.0`, same λ grid):
  - `N4_star_alpha0.8`: best slope ≈ `-2.260` (best |slope+2| ≈ `0.260`)
  - `N4_cycle_alpha0.8`: best slope ≈ `-3.580`
  - `N4_path_alpha0.8`: best slope ≈ `-3.772`
- **N5 sparse anchor** (`N5_path_alpha0.8`, λ=`{0.2,0.6,1.0}`, hotspot `1.5`):
  - best slope ≈ `-3.118` (closer than N4 path, still far from `-2`)
- **N5 dense topology sweep** (`N5_path/cycle/star_alpha0.8`, λ=`{0.2,0.4,0.6,0.8,1.0,1.2}`, hotspot `1.5`, `bond_cutoff=3` for cycle tractability):
  - `N5_star_alpha0.8`: best slope ≈ `-2.407` (best |slope+2| ≈ `0.407`)
  - `N5_path_alpha0.8`: best slope ≈ `-3.118` (best |slope+2| ≈ `1.118`)
  - `N5_cycle_alpha0.8`: best slope ≈ `-3.755` (best |slope+2| ≈ `1.755`)
- **Interpretation**:
  - Star topology is consistently much closer to the spin-2 target slope than path/cycle at both N=4 and N=5.
  - This star advantage is robust across hotspot multipliers `1.5` and `3.0` (N4 checks).
  - Path-like/cycle-like topologies remain too steep under current coarse-graining.
- **Artifacts**:
  - Sweep A: `outputs/spin2_correlator_sweep_20260223-100252/spin2_points.csv`, `outputs/spin2_correlator_sweep_20260223-100252/spin2_summary.csv`, `outputs/spin2_correlator_sweep_20260223-100252/spin2_slope_vs_lambda.png`, `outputs/spin2_correlator_sweep_20260223-100252/spin2_report.md`
  - Sweep B: `outputs/spin2_correlator_sweep_20260223-101138/spin2_points.csv`, `outputs/spin2_correlator_sweep_20260223-101138/spin2_summary.csv`, `outputs/spin2_correlator_sweep_20260223-101138/spin2_slope_vs_lambda.png`, `outputs/spin2_correlator_sweep_20260223-101138/spin2_report.md`
  - N5 anchor: `outputs/spin2_correlator_sweep_20260223-100445/spin2_points.csv`, `outputs/spin2_correlator_sweep_20260223-100445/spin2_summary.csv`, `outputs/spin2_correlator_sweep_20260223-100445/spin2_report.md`
  - N5 dense topologies: `outputs/spin2_correlator_sweep_20260223-101802/spin2_points.csv`, `outputs/spin2_correlator_sweep_20260223-101802/spin2_summary.csv`, `outputs/spin2_correlator_sweep_20260223-101802/spin2_slope_vs_lambda.png`, `outputs/spin2_correlator_sweep_20260223-101802/spin2_report.md`
  - Consolidated comparison: `outputs/spin2_correlator_comparison_20260223.csv`

## Correlator spin-2 cutoff sensitivity (make-or-break check, February 23, 2026)
- **Question**: is star-near-`-2` behavior a bond-cutoff artifact, or topology-robust?
- **Code update**: `scripts/spin2_correlator_topology_sweep.py` now supports `--bond-cutoffs` for one-pass cutoff sweeps.
- **N4 sweep (`chi=3,4`)**: `outputs/spin2_correlator_sweep_20260223-robust_N4_chi34/`
  - `N4_star_alpha0.8`: best slope `-2.523` (`chi=3`) and `-2.326` (`chi=4`)
  - `N4_path_alpha0.8`: best slope `-3.415` (`chi=3`) and `-3.718` (`chi=4`)
  - `N4_cycle_alpha0.8`: best slope `-3.449` (`chi=3`) and `-3.685` (`chi=4`)
- **N4 extension (`chi=5`, path+star)**: `outputs/spin2_correlator_sweep_20260223-robust_N4_pathstar_chi5/`
  - `N4_star_alpha0.8`: best slope `-2.376` (`chi=5`)
  - `N4_path_alpha0.8`: best slope `-3.805` (`chi=5`)
- **N5 sweep (`chi=3,4`, path+star, λ=`{0.2,0.6,1.0}`)**: `outputs/spin2_correlator_sweep_20260223-robust_N5_pathstar_chi34/`
  - `N5_star_alpha0.8`: best slope `-2.536` (`chi=3`) and `-2.754` (`chi=4`)
  - `N5_path_alpha0.8`: best slope `-3.118` (`chi=3`) and `-3.118` (`chi=4`)
- **Interpretation**:
  - Topology separation is robust: star remains much closer to `-2` than path/cycle across tested cutoffs.
  - Path is effectively cutoff-insensitive at N5 in this λ set (`-3.118` at both `chi=3,4`).
  - Star degrades somewhat at higher cutoff in N5 (`-2.54 → -2.75`) but still stays far closer to target than path.
- **Artifacts**:
  - `outputs/spin2_correlator_sweep_20260223-robust_N4_chi34/spin2_summary.csv`
  - `outputs/spin2_correlator_sweep_20260223-robust_N4_pathstar_chi5/spin2_summary.csv`
  - `outputs/spin2_correlator_sweep_20260223-robust_N5_pathstar_chi34/spin2_summary.csv`
  - `outputs/spin2_correlator_cutoff_sensitivity_20260223.csv`
