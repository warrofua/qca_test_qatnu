# Decision Log: Topology-Conditioned Correlated Promotion (2026-03-08)

## Objective
Move the `gamma_corr` structural mechanism out of `v3over9000` and into the production research path, then test whether a star-only structural override can improve no-retuning transfer without disturbing the path scenario.

## Code changes
- `core_qca.py`
  - production Hamiltonian now supports:
    - `gamma_corr`
    - `gamma_corr_diag`
  - correlated pair terms are included for incident edge pairs in both dense and sparse builds.

- `scanners.py`
  - `scan_lambda_parallel` and `scan_2d_phase_space` now accept and forward:
    - `hotspot_time`
    - `gamma_corr`
    - `gamma_corr_diag`

- `tester.py`
  - validation path now carries:
    - `hotspot_time`
    - correlated-promotion parameters.

- `app.py`
  - new CLI flags:
    - `--hotspot-time`
    - `--gamma-corr`
    - `--gamma-corr-diag`

- `scripts/no_retuning_holdout_test.py`
  - new global flags:
    - `--hotspot-time`
    - `--gamma-corr`
    - `--gamma-corr-diag`
  - new graph-specific override flag:
    - `--graph-overrides-json`

## Verification
- Static compile check:
```bash
.venv/bin/python -m py_compile core_qca.py scanners.py tester.py app.py scripts/no_retuning_holdout_test.py
```

## First production-path test
Command:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/no_retuning_holdout_test.py \
  --scenarios N4_star_alpha0.8,N4_path_alpha1.0 \
  --deltaB 5.0 \
  --kappa 0.1 \
  --hotspot-multiplier 3.0 \
  --k0 4 \
  --bond-cutoff 4 \
  --graph-overrides-json '{"star":{"gamma_corr":-0.25}}' \
  --max-points 16 \
  --no-refine \
  --quiet-scanner \
  --output-dir outputs/no_retuning_holdout_20260308_star_gamma_pathstar16
```

Artifacts:
- `outputs/no_retuning_holdout_20260308_star_gamma_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_gamma_pathstar16/holdout_report.md`

## Result
- `N4_path_alpha1.0`: unchanged and still passes.
- `N4_star_alpha0.8`: residual floor improves:
  - from `0.159678...` in the prior 16-point locked run
  - to `0.102585...` with star-only `gamma_corr=-0.25`
- But star transfer still fails because:
  - `lambda_c1 = 0.19`
  - `lambda_revival = 0.19`
  - `lambda_c2 = 0.37`
  - ordering is no longer resolved on this grid.

## Decision
This is a useful negative result.

What improved:
- structural, topology-conditioned deformations are now first-class production features;
- star scalar residuals can be moved by a nontrivial structural channel without damaging the path control.

What did not improve:
- topology transfer is still not recovered;
- the current negative-`gamma_corr` channel appears to suppress scalar residuals more readily than it repairs revival placement.

Next implication:
- search for structural channels that disentangle residual suppression from landmark placement, rather than continuing scalar-only retuning.

## Follow-up 1: correlated diagonal scan
Command pattern:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/no_retuning_holdout_test.py \
  --scenarios N4_star_alpha0.8,N4_path_alpha1.0 \
  --deltaB 5.0 \
  --kappa 0.1 \
  --hotspot-multiplier 3.0 \
  --hotspot-time 1.0 \
  --k0 4 \
  --bond-cutoff 4 \
  --graph-overrides-json '{"star":{"gamma_corr":-0.25,"gamma_corr_diag":X}}' \
  --max-points 16 \
  --no-refine \
  --quiet-scanner
```

Artifacts:
- `outputs/no_retuning_holdout_20260308_star_gamma_diag_-0.20_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_gamma_diag_-0.10_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_gamma_diag_0.10_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_gamma_diag_0.20_pathstar16/holdout_summary.csv`

Result:
- positive `gamma_corr_diag` slightly lowers the star residual floor;
- no tested value separates `lambda_c1` from `lambda_revival`;
- this term is not the missing revival-placement mechanism.

## Follow-up 2: hotspot-time scan
Command pattern:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/no_retuning_holdout_test.py \
  --scenarios N4_star_alpha0.8,N4_path_alpha1.0 \
  --deltaB 5.0 \
  --kappa 0.1 \
  --hotspot-multiplier 3.0 \
  --hotspot-time 1.0 \
  --k0 4 \
  --bond-cutoff 4 \
  --graph-overrides-json '{"star":{"gamma_corr":-0.25,"hotspot_time":X}}' \
  --max-points 16 \
  --no-refine \
  --quiet-scanner
```

Quick-screen artifacts:
- `outputs/no_retuning_holdout_20260308_star_gamma_ht_0.50_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_gamma_ht_0.75_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_gamma_ht_1.25_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_gamma_ht_1.50_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_gamma_ht_2.00_pathstar16/holdout_summary.csv`

Refined artifact:
- `outputs/no_retuning_holdout_20260308_star_gamma_ht_0.50_pathstar48/holdout_summary.csv`

Result:
- `hotspot_time=0.50` reopens star phase ordering under the structural override;
- 48-point star readout:
  - `lambda_c1=0.184375`
  - `lambda_revival=0.26875`
  - `lambda_c2=0.4375`
  - `residual_min=0.104585`
- longer hotspot times do not help and can re-collapse the ordering.

Decision update:
- hotspot-time shaping is the first tested channel after `gamma_corr` that moves landmark geometry rather than only the residual floor;
- however, it moves the star revival in the wrong direction relative to the target window, so it still does not rescue no-retuning transfer.

## Follow-up 3: edge-local hotspot masks
Mechanism:
- `core_qca.py` now accepts `lambda_edge_weights`, allowing hotspot preparation to use per-edge promotion amplitudes while leaving the measured Hamiltonian unchanged.
- `scanners.py`, `tester.py`, `app.py`, and `scripts/no_retuning_holdout_test.py` now thread hotspot-only edge masks through the production path.

Quick-screen command pattern:
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python scripts/no_retuning_holdout_test.py \
  --scenarios N4_star_alpha0.8,N4_path_alpha1.0 \
  --deltaB 5.0 \
  --kappa 0.1 \
  --hotspot-multiplier 3.0 \
  --hotspot-time 1.0 \
  --k0 4 \
  --bond-cutoff 4 \
  --graph-overrides-json '{"star":{"gamma_corr":-0.25,"hotspot_time":0.5,"hotspot_edge_weights":[...]}}' \
  --max-points 16 \
  --no-refine \
  --quiet-scanner
```

Artifacts:
- `outputs/no_retuning_holdout_20260308_star_mask_anti_mild_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_mask_anti_strong_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_mask_probe_focus_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_mask_skew_pathstar16/holdout_summary.csv`

Result:
- anti-probe masks preserve relatively low residuals but leave the revival too early or collapse it again;
- a probe-focused mask `[1.5, 0.75, 0.75]` shifts the star landmarks later:
  - `lambda_c1=0.28`
  - `lambda_revival=0.46`
  - `lambda_c2=0.55`
- but that same move destroys residual quality:
  - `residual_min=0.361`

## Follow-up 4: probe-focused mask + positive gamma_corr_diag
Artifacts:
- `outputs/no_retuning_holdout_20260308_star_mask_probe_focus_gcd_0.10_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_mask_probe_focus_gcd_0.20_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_mask_probe_focus_gcd_0.40_pathstar16/holdout_summary.csv`

Result:
- increasing `gamma_corr_diag` from `0.10` to `0.40` does not materially repair the residual floor;
- the landmark geometry stays at the later `0.28, 0.46, 0.55` pattern;
- residual only improves from `0.3608` to `0.3589`, which is negligible for protocol purposes.

Decision update:
- edge-local hotspot shaping is the first mechanism that pushes the star landmarks later in the right qualitative direction;
- however, the current masks reveal a hard tradeoff between later revival placement and scalar residual quality;
- the next mechanism should target that tradeoff directly rather than continuing to sweep the current diagonal correlated term.

## Follow-up 5: two-stage hotspot preparation
Mechanism:
- hotspot preparation now supports sequential stages, each with its own multiplier, duration, and optional edge mask.
- this was added to:
  - `scanners.py`
  - `tester.py`
  - `app.py`
  - `scripts/no_retuning_holdout_test.py`

16-point stage-screen artifacts:
- `outputs/no_retuning_holdout_20260308_star_2stage_probe_then_anti_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_2stage_probe_then_neutral_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_2stage_anti_then_probe_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_2stage_probe_short_anti_long_pathstar16/holdout_summary.csv`

Refined artifacts:
- `outputs/no_retuning_holdout_20260308_star_2stage_probe_then_neutral_pathstar48/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_2stage_probe_then_anti_pathstar48/holdout_summary.csv`

Refined results:
- `probe_then_neutral`:
  - `lambda_c1=0.2125`
  - `lambda_revival=0.353125`
  - `lambda_c2=0.465625`
  - `residual_min=0.111793`
- `probe_then_anti`:
  - `lambda_c1=0.184375`
  - `lambda_revival=0.26875`
  - `lambda_c2=0.465625`
  - `residual_min=0.100128`

Interpretation:
- these two-stage protocols are the first clear improvement over the one-stage edge-mask scans;
- `probe_then_neutral` is the best “later landmark” candidate that still preserves an acceptable residual floor;
- `probe_then_anti` gives the best residual floor while preserving the later `lambda_c2` gained from the probe-focused branch;
- ordering of stages matters: `probe_then_neutral` outperforms `anti_then_probe`, so the protocol is genuinely dynamical rather than just an average over masks.

Decision update:
- multi-stage source preparation is the first tested mechanism that improves the geometry/residual tradeoff in a durable way;
- it still falls far short of the target star revival window, so the missing mechanism is not merely better source shaping;
- the next mechanism should modify the star dynamics after source preparation, not only the source protocol itself.

## Follow-up 6: readout-only correlated dynamics
Mechanism:
- the readout Hamiltonian now supports correlated terms separate from the source-preparation Hamiltonian.
- this was added to:
  - `scanners.py`
  - `tester.py`
  - `app.py`
  - `scripts/no_retuning_holdout_test.py`

16-point artifacts:
- `outputs/no_retuning_holdout_20260308_star_readout_rgc_-0.50_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_readout_rgc_-0.25_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_readout_rgc_0.25_pathstar16/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_readout_rgc_0.50_pathstar16/holdout_summary.csv`

Refined artifacts:
- `outputs/no_retuning_holdout_20260308_star_readout_rgc_0.25_pathstar48/holdout_summary.csv`
- `outputs/no_retuning_holdout_20260308_star_readout_rgc_0.25_rgd_0.20_pathstar48/holdout_summary.csv`

Refined result:
- baseline two-stage source:
  - `lambda_c1=0.2125`
  - `lambda_revival=0.353125`
  - `lambda_c2=0.465625`
  - `residual_min=0.111793`
- with `readout_gamma_corr=+0.25`:
  - `lambda_c1=0.2125`
  - `lambda_revival=0.353125`
  - `lambda_c2=0.55`
  - `residual_min=0.111793`
- adding `readout_gamma_corr_diag=+0.20` produces no measurable change in this window.

Interpretation:
- this is the first clear indication that different mechanisms control different landmarks:
  - multi-stage source shaping moves the lower landmarks and the overall residual tradeoff,
  - readout-only correlated dynamics can move `lambda_c2` independently.
- negative readout correlation is actively harmful (`residual_min≈0.369` at `readout_gamma_corr=-0.50`).

Decision update:
- the star failure is not one monolithic mismatch; the lower and upper landmarks can be moved by different structural channels;
- the next mechanism should target `lambda_revival` specifically, because:
  - source shaping already improves `lambda_c1`,
  - readout-only correlation already improves `lambda_c2`,
  - `lambda_revival` remains the dominant unresolved gap.
