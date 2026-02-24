# Decision Log: v3over9000 (TT Spin-2 + Self-Energy Alpha + Correlated Promotions)

Date: 2026-02-23  
Scope: Create a new forward branch for "full physical story" experiments, not just parameter retuning.

## Why this branch exists
Recent findings showed:
- spin-2 remains unresolved in path/cycle topologies,
- topology transfer without retuning fails,
- phase landmarks are sensitive to phenomenological knobs.

This branch targets structural math changes:
1. TT-projected tensor observable (not scalar proxy only),
2. alpha extraction from self-energy response slope,
3. explicit correlated promotion channel in the Hamiltonian.

## New code workspace
- `v3over9000/__init__.py`
- `v3over9000/embeddings.py`
- `v3over9000/tensor_spin2.py`
- `v3over9000/alpha_self_energy.py`
- `v3over9000/correlated_qca.py`
- `v3over9000/run_reality_experiment.py`
- `v3over9000/README.md`

## Implementation details
- `tensor_spin2.py`:
  - Builds connected bond-bond covariance `C_ef`.
  - Lifts to rank-4 tensor correlator via edge direction tensors.
  - Applies 3D TT projector and fits `S_TT(k) ~ 1/k^p`.
- `alpha_self_energy.py`:
  - Estimates `sigma(0) = omega - omega_eff`.
  - Computes `alpha_self_energy ~ (1/omega) * d sigma / d Lambda`.
  - Uses FFT peak interpolation for sub-bin frequency resolution.
- `correlated_qca.py`:
  - Adds correlated double-promotion/demotion terms between incident edge pairs.
  - Controlled by `gamma_corr` (plus optional diagonal `gamma_corr_diag`).
- `run_reality_experiment.py`:
  - Runs baseline vs correlated model side-by-side.
  - Emits summary CSV, TT spectra CSV, alpha JSON, plot, and markdown report.

## Smoke runs completed
1. `outputs/v3over9000_smoke_20260223/`
   - Minimal N3 path sanity run.
2. `outputs/v3over9000_smoke_20260223_N4/`
   - N4 path sanity run.
3. `outputs/v3over9000_smoke_20260223_N4_refined/`
   - N4 path with refined Ramsey resolution.
   - Produced non-trivial self-energy alpha estimate:
     - baseline: `alpha_self_energy ≈ 0.0347` (`R^2 ≈ 0.998`)
     - correlated: `alpha_self_energy ≈ 0.0350` (`R^2 ≈ 0.998`)

## Initial interpretation
- The branch is operational and numerically stable for small-N smoke runs.
- TT spin-2 power in these quick path smokes is still far from target `p=2`.
- Correlated-promotion term is now available for systematic scans, which was missing before.

## Sweep H: topology x gamma_corr (positive-only first pass)
Command:
` .venv/bin/python v3over9000/sweep_tt_gamma.py --N 4 --alpha 0.8 --topologies path,cycle,star --gammas 0.0,0.02,0.05,0.1 --lambdas 0.2,0.4,0.6,0.8,1.0 --bond-cutoff 4 --hotspot-multiplier 1.5 --frustration-time 1.0 --k-modes 14 --k-angles 20 --ramsey-tmax 12 --ramsey-points 72 --workers 4 --output-dir outputs/v3over9000_gamma_sweep_20260223_N4`

Outcome:
- Positive `gamma_corr` did **not** improve TT spin-2 residual.
- Best rows remained at `gamma_corr=0.0` for all topologies in this pass.

Artifacts:
- `outputs/v3over9000_gamma_sweep_20260223_N4/points.csv`
- `outputs/v3over9000_gamma_sweep_20260223_N4/summary.csv`
- `outputs/v3over9000_gamma_sweep_20260223_N4/spin2_residual_vs_gamma.png`
- `outputs/v3over9000_gamma_sweep_20260223_N4/report.md`

## Sweep I: gamma sign scan (negative to positive)
Command:
` .venv/bin/python v3over9000/sweep_tt_gamma.py --N 4 --alpha 0.8 --topologies path,cycle,star --gammas -0.1,-0.05,-0.02,0.0,0.02,0.05,0.1 --lambdas 0.2,0.6,1.0 --bond-cutoff 4 --hotspot-multiplier 1.5 --frustration-time 1.0 --k-modes 14 --k-angles 20 --ramsey-tmax 12 --ramsey-points 72 --workers 4 --output-dir outputs/v3over9000_gamma_signscan_20260223_N4`

Outcome:
- Negative `gamma_corr` improved TT residual consistently across tested topologies.
- Best-by-topology:
  - `path`: `gamma_corr=-0.1`, best residual `1.9878` (vs `2.0006` at `gamma=0`)
  - `cycle`: `gamma_corr=-0.1`, best residual `1.9773` (vs `1.9896`)
  - `star`: `gamma_corr=-0.1`, best residual `1.9920` (vs `2.0008`)
- Postulate residual behavior stayed qualitatively unchanged in this scan:
  - cycle near machine precision,
  - path moderate,
  - star large at high `lambda` as before.

Artifacts:
- `outputs/v3over9000_gamma_signscan_20260223_N4/points.csv`
- `outputs/v3over9000_gamma_signscan_20260223_N4/summary.csv`
- `outputs/v3over9000_gamma_signscan_20260223_N4/spin2_residual_vs_gamma.png`
- `outputs/v3over9000_gamma_signscan_20260223_N4/report.md`

## Sweep J: deeper negative gamma scan
Command:
` .venv/bin/python v3over9000/sweep_tt_gamma.py --N 4 --alpha 0.8 --topologies path,cycle,star --gammas -0.2,-0.15,-0.1,-0.05,0.0 --lambdas 0.2,0.6,1.0 --bond-cutoff 4 --hotspot-multiplier 1.5 --frustration-time 1.0 --k-modes 14 --k-angles 20 --ramsey-tmax 12 --ramsey-points 72 --workers 4 --output-dir outputs/v3over9000_gamma_negdeep_20260223_N4`

Outcome:
- Improvement was monotonic toward more negative coupling in this window.
- Best-by-topology at `gamma_corr=-0.2`:
  - `cycle`: residual `1.9643` (gain `~0.0254` vs gamma `0`)
  - `path`: residual `1.9682` (gain `~0.0323`)
  - `star`: residual `1.9791` (gain `~0.0217`)
- This is still far from target `|power-2| -> 0`, but it is the first branch-internal directional gain from a structural Hamiltonian change.

Artifacts:
- `outputs/v3over9000_gamma_negdeep_20260223_N4/points.csv`
- `outputs/v3over9000_gamma_negdeep_20260223_N4/summary.csv`
- `outputs/v3over9000_gamma_negdeep_20260223_N4/spin2_residual_vs_gamma.png`
- `outputs/v3over9000_gamma_negdeep_20260223_N4/report.md`

## Sweep K: finite-size carryover test (N=5, chi=3)
Command:
` .venv/bin/python v3over9000/sweep_tt_gamma.py --N 5 --alpha 0.8 --topologies path,cycle,star --gammas -0.2,-0.1,0.0 --lambdas 0.2,0.6,1.0 --bond-cutoff 3 --hotspot-multiplier 1.5 --frustration-time 1.0 --k-modes 14 --k-angles 20 --ramsey-tmax 12 --ramsey-points 72 --workers 3 --output-dir outputs/v3over9000_gamma_N5_chi3_20260223`

Outcome:
- The negative-coupling trend survives finite-size growth (at tested `N=5`, `chi=3`):
  - `path`: best residual improves from `2.0006` (`gamma=0`) to `1.9687` (`gamma=-0.2`)
  - `cycle`: `2.0004` -> `1.9797`
  - `star`: `2.0038` -> `1.9538`
- Best `lambda` in this sparse anchor set is `0.2` for all three topologies.
- Postulate residual behavior remains topology-dependent:
  - cycle near numerical floor,
  - path moderate,
  - star high at larger `lambda`, but low near `lambda=0.2`.

Artifacts:
- `outputs/v3over9000_gamma_N5_chi3_20260223/points.csv`
- `outputs/v3over9000_gamma_N5_chi3_20260223/summary.csv`
- `outputs/v3over9000_gamma_N5_chi3_20260223/spin2_residual_vs_gamma.png`
- `outputs/v3over9000_gamma_N5_chi3_20260223/report.md`

## Sweep L: dense lambda check (N=5 path/star, chi=3)
Command:
` .venv/bin/python v3over9000/sweep_tt_gamma.py --N 5 --alpha 0.8 --topologies path,star --gammas -0.25,-0.2,0.0 --lambdas 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 --bond-cutoff 3 --hotspot-multiplier 1.5 --frustration-time 1.0 --k-modes 16 --k-angles 24 --ramsey-tmax 12 --ramsey-points 72 --workers 3 --output-dir outputs/v3over9000_gamma_dense_N5_pathstar_chi3_20260223`

Outcome:
- Trend remains robust under denser lambda sampling.
- Best residuals (both at `lambda=0.2`):
  - `path`: `gamma=-0.25` -> `1.9565` (vs `2.0006` at `gamma=0`)
  - `star`: `gamma=-0.25` -> `1.9382` (vs `2.0039`)
- Postulate residual profile remains similar:
  - path mean residual around `~0.098`,
  - star remains high at larger lambdas (`max ~0.75`).

Artifacts:
- `outputs/v3over9000_gamma_dense_N5_pathstar_chi3_20260223/points.csv`
- `outputs/v3over9000_gamma_dense_N5_pathstar_chi3_20260223/summary.csv`
- `outputs/v3over9000_gamma_dense_N5_pathstar_chi3_20260223/spin2_residual_vs_gamma.png`
- `outputs/v3over9000_gamma_dense_N5_pathstar_chi3_20260223/report.md`

## Sweep M: cutoff carryover (N=5 path/star, chi=4)
Command:
` .venv/bin/python v3over9000/sweep_tt_gamma.py --N 5 --alpha 0.8 --topologies path,star --gammas -0.25,-0.2,0.0 --lambdas 0.2,0.6,1.0 --bond-cutoff 4 --hotspot-multiplier 1.5 --frustration-time 1.0 --k-modes 14 --k-angles 20 --ramsey-tmax 12 --ramsey-points 72 --workers 2 --output-dir outputs/v3over9000_gamma_N5_pathstar_chi4_20260223`

Outcome:
- Negative-coupling trend survives cutoff increase (`chi=3 -> 4`).
- Best-by-topology shifts to `gamma=-0.25`:
  - `path`: best residual `1.9580` at `chi=4` vs `1.9687` (`chi=3`, sparse set)
  - `star`: `1.9391` at `chi=4` vs `1.9538` (`chi=3`, sparse set)
- Best lambda remained `0.2` across tested settings.
- Postulate-residual means for best rows remained close for path; star remained high in high-lambda region.

Artifacts:
- `outputs/v3over9000_gamma_N5_pathstar_chi4_20260223/points.csv`
- `outputs/v3over9000_gamma_N5_pathstar_chi4_20260223/summary.csv`
- `outputs/v3over9000_gamma_N5_pathstar_chi4_20260223/spin2_residual_vs_gamma.png`
- `outputs/v3over9000_gamma_N5_pathstar_chi4_20260223/report.md`

## Sweep N: lambda-dependent gamma schedule (taper) to protect Postulate residual
Motivation:
Negative constant `gamma_corr` improves TT residual but can increase star mean postulate residual.

Code update:
- `v3over9000/sweep_tt_gamma.py` now supports:
  - `--gamma-mode {constant,taper_high}`
  - `--gamma-lambda-low`, `--gamma-lambda-high`, `--gamma-taper-power`
- Added summary guardrails:
  - `p90_postulate_residual`
  - `frac_postulate_gt_0_2`
  - effective gamma stats (`mean/min/max_gamma_corr_effective`)

### N1: taper schedule A (`low=0.35`, `high=0.70`)
Command:
` .venv/bin/python v3over9000/sweep_tt_gamma.py --N 5 --alpha 0.8 --topologies path,star --gammas -0.25,-0.2,0.0 --lambdas 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 --gamma-mode taper_high --gamma-lambda-low 0.35 --gamma-lambda-high 0.70 --gamma-taper-power 1.0 --bond-cutoff 3 --hotspot-multiplier 1.5 --frustration-time 1.0 --k-modes 16 --k-angles 24 --ramsey-tmax 12 --ramsey-points 72 --workers 3 --output-dir outputs/v3over9000_gamma_taper_N5_pathstar_chi3_20260223`

Result:
- TT best residuals unchanged (best still at `lambda=0.2`).
- Star postulate guardrail improved for `gamma=-0.2` relative to constant:
  - mean postulate `~0.476 -> ~0.398`
  - max postulate `~0.734 -> ~0.706`
- For `gamma=-0.25`, guardrail did not improve enough.

Artifacts:
- `outputs/v3over9000_gamma_taper_N5_pathstar_chi3_20260223/summary.csv`
- `outputs/v3over9000_gamma_taper_N5_pathstar_chi3_20260223/report.md`

### N2: taper schedule B (more aggressive: `low=0.30`, `high=0.55`)
Command:
` .venv/bin/python v3over9000/sweep_tt_gamma.py --N 5 --alpha 0.8 --topologies path,star --gammas -0.25,-0.2,0.0 --lambdas 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 --gamma-mode taper_high --gamma-lambda-low 0.30 --gamma-lambda-high 0.55 --gamma-taper-power 1.0 --bond-cutoff 3 --hotspot-multiplier 1.5 --frustration-time 1.0 --k-modes 16 --k-angles 24 --ramsey-tmax 12 --ramsey-points 72 --workers 3 --output-dir outputs/v3over9000_gamma_taper2_N5_pathstar_chi3_20260223`

Result:
- Preserved best TT residual for `gamma=-0.25` (`1.9382`) at `lambda=0.2`.
- Brought star postulate mean essentially back to baseline while keeping TT gain:
  - star `gamma=-0.25`: mean postulate `~0.470` (constant) -> `~0.406` (taper B), with same best TT residual.
- Path metrics remained nearly unchanged from constant schedules.

Artifacts:
- `outputs/v3over9000_gamma_taper2_N5_pathstar_chi3_20260223/summary.csv`
- `outputs/v3over9000_gamma_taper2_N5_pathstar_chi3_20260223/report.md`

## Sweep O: long-run reliability + throughput updates (2026-02-24)
Code updates in `v3over9000/sweep_tt_gamma.py`:
- Added `--resume` and `--checkpoint-every` for interruption-safe partial progress.
- Added effective-gamma deduplication: logical points with identical `(topology, lambda, gamma_eff, physics config)` are computed once, then expanded.
- Optimized probe frequency extraction by reusing pulse/eigenbasis work across times.

Validation:
- Resume smoke: `outputs/v3over9000_resume_smoke_20260224/`
- Dedupe smoke (3->1 unique at high lambda): `outputs/v3over9000_dedupe_smoke_20260224/`

Runtime profile (single N5/star/chi4 point):
- `outputs/v3over9000_perf_probe_star_N5_chi4_20260224/`
- Dominant cost is dense diagonalization/evolution, not tensor or Ramsey loops:
  - ground-state diagonalization: `~75.1 s`
  - hotspot evolution (second diagonalization path): `~75.9 s`
  - tensor spectrum + postulate bookkeeping: `<< 1 s`
  - total: `~152.4 s` per unique point at this setting.

Implication:
- Current dense solver makes full N5/chi4+ sweeps expensive on present hardware.
- Deduplication and resume/checkpoint are now mandatory for practical long runs.

## Sweep P: N5 star chi4 micro-grid with taper schedule B (`low=0.30`, `high=0.55`)
Command:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python v3over9000/sweep_tt_gamma.py --N 5 --topologies star --gammas -0.25,-0.2,0.0 --lambdas 0.2,0.8 --gamma-mode taper_high --gamma-lambda-low 0.30 --gamma-lambda-high 0.55 --gamma-taper-power 1.0 --bond-cutoff 4 --k-modes 8 --k-angles 12 --ramsey-tmax 10 --ramsey-points 48 --workers 2 --checkpoint-every 1 --resume --output-dir outputs/v3over9000_gamma_taper2_N5_star_chi4_micro_20260224`

Outcome:
- Low lambda retains the strong negative-gamma TT improvement:
  - `lambda=0.2`: `gamma=-0.25` -> `spin2_residual=1.9398`, `postulate_residual=0.0088`
  - `lambda=0.2`: `gamma=0.0` -> `spin2_residual=2.0038`, `postulate_residual=0.0042`
- At `lambda=0.8`, taper-B forces `gamma_eff=0` for all gammas, so all collapse to baseline:
  - `spin2_residual=2.0426`, `postulate_residual=0.4094`.

Artifacts:
- `outputs/v3over9000_gamma_taper2_N5_star_chi4_micro_20260224/points.csv`
- `outputs/v3over9000_gamma_taper2_N5_star_chi4_micro_20260224/summary.csv`
- `outputs/v3over9000_gamma_taper2_N5_star_chi4_micro_20260224/report.md`

## Sweep Q: high-lambda constant controls (N5 star chi4)
Goal:
Test whether taper-to-zero is too aggressive in high lambda.

Runs:
- `outputs/v3over9000_gamma_constant_N5_star_chi4_lambda08_20260224/`
- `outputs/v3over9000_gamma_constant_N5_star_chi4_lambda09_20260224/`
- `outputs/v3over9000_gamma_constant_N5_star_chi4_lambda10_20260224/`

Key results:
- `lambda=0.8`: negative gamma improves both TT and postulate vs `gamma=0`
  - `gamma=-0.25`: `spin2_residual=2.0073`, `postulate_residual=0.2848`
  - `gamma=0.0`: `spin2_residual=2.0426`, `postulate_residual=0.4094`
- `lambda=0.9`: same directional improvement for negative gamma
  - `gamma=-0.2`: `spin2_residual=2.0184`, `postulate_residual=0.2916`
  - `gamma=0.0`: `spin2_residual=2.0474`, `postulate_residual=0.3530`
- `lambda=1.0`: mixed/crossover behavior
  - `gamma=-0.25`: `postulate_residual=0.3682` (worse than `gamma=0`)
  - `gamma=0.0`: `postulate_residual=0.2989` (best postulate at this lambda in this slice)

Interpretation:
- High-lambda behavior is not monotonic: negative gamma helps at `0.8-0.9` but can hurt by `1.0`.
- This falsifies the simplistic "always taper to zero by 0.55" rule at chi=4.

## Sweep R: delayed taper candidate C (`low=0.30`, `high=1.00`, `gamma=-0.25`)
Command:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python v3over9000/sweep_tt_gamma.py --N 5 --topologies star --gammas -0.25 --lambdas 0.8,0.9,1.0 --gamma-mode taper_high --gamma-lambda-low 0.30 --gamma-lambda-high 1.00 --gamma-taper-power 1.0 --bond-cutoff 4 --k-modes 8 --k-angles 12 --ramsey-tmax 10 --ramsey-points 48 --workers 2 --checkpoint-every 1 --output-dir outputs/v3over9000_gamma_taperC_N5_star_chi4_hilambda_20260224`

Outcome:
- `lambda=0.8` (`gamma_eff=-0.071`): `spin2_residual=2.0331`, `postulate_residual=0.1998` (best postulate seen at this lambda across tested schedules).
- `lambda=0.9` (`gamma_eff=-0.036`): `spin2_residual=2.0425`, `postulate_residual=0.3497` (near baseline).
- `lambda=1.0` (`gamma_eff=0.0`): baseline recovery (`spin2_residual=2.0500`, `postulate_residual=0.2989`).

Interpretation:
- Delaying taper appears superior to early-zero taper for high-lambda star behavior in this chi=4 slice.
- This supports an adaptive schedule family (nonzero until near `lambda~1.0`) instead of hard early shutoff.

Artifacts:
- `outputs/v3over9000_gamma_taperC_N5_star_chi4_hilambda_20260224/points.csv`
- `outputs/v3over9000_gamma_taperC_N5_star_chi4_hilambda_20260224/summary.csv`
- `outputs/v3over9000_gamma_taperC_N5_star_chi4_hilambda_20260224/report.md`

## Sweep S: frozen-matter screening extraction (Poisson vs Yukawa, 2026-02-24)
Motivation:
The docs call for a scalar-sector make/break test: whether static Lambda response is unscreened (`Delta Lambda ~ -rho`) or screened (`(Delta - mu^2)Lambda ~ -rho`).

New script:
- `scripts/frozen_screening_scan.py`
  - freezes matter via static `z_i in {+1,-1}` assignments,
  - derives edge frustration `F_ij = (1 - z_i z_j)/2`,
  - solves bond-only ground states,
  - measures `Lambda_i` (log or linear proxy),
  - fits massless vs screened operators per topology and lambda.

Command:
` .venv/bin/python scripts/frozen_screening_scan.py --N 4 --topologies path,cycle,star --lambdas 0.1,0.2,0.35,0.5,0.7,1.0,1.2,1.4 --bond-cutoff 4 --output-dir outputs/frozen_screening_N4_path_cycle_star_20260224`

Outcome summary:
- `cycle`: massless fit is exact to machine precision across lambdas; `mu^2 ~ 0`.
- `star`: screened term does not improve fit under nonnegative `mu^2` constraint; `mu^2=0` selected across tested lambdas.
- `path`: unconstrained screened fit yields small positive `mu^2` but only marginal RSS gains (below selection threshold), so massless remains preferred in this first pass.

Artifacts:
- `outputs/frozen_screening_N4_path_cycle_star_20260224/frozen_cases.csv`
- `outputs/frozen_screening_N4_path_cycle_star_20260224/frozen_fits.csv`
- `outputs/frozen_screening_N4_path_cycle_star_20260224/mu2_vs_lambda.png`
- `outputs/frozen_screening_N4_path_cycle_star_20260224/fit_residuals_vs_lambda.png`
- `outputs/frozen_screening_N4_path_cycle_star_20260224/report.md`

Interpretation:
- This first frozen-matter scan provides initial evidence against a strong screened (`mu^2>0`) scalar response in these N4 settings.
- The decisive follow-up is dynamic: test deep-time relaxation/critical slowing near `lambda_rev` to see whether screening behavior changes at/near revival.

## Sweep T: deep-time critical-slowing scan (2026-02-24)
Motivation:
Follow up the frozen-matter static check with the dynamic test suggested in Appendix-D:
measure whether bond-sector relaxation/dephasing time peaks near revival windows.

New script:
- `scripts/critical_slowing_scan.py`
  - protocol: ground state of nominal `H(lambda)` -> quench to hotspot `H(lambda * hotspot_multiplier)` -> deep-time evolution
  - observables:
    - global `Lambda(t)` mean
    - probe-gap `Lambda_in(t) - Lambda_out(t)`
  - metrics:
    - strict equilibration time (`tau_eq`) via sustained tail-band criterion
    - dephasing time (`tau_dephase`) via running-average convergence (robust for closed finite systems)

### T1: hotspot multiplier 1.5 (dense near revival bands)
Command:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/critical_slowing_scan.py --N 4 --topologies path,cycle,star --lambdas 'path:0.8,0.9,1.0,1.1,1.2,1.3;cycle:0.35,0.45,0.5,0.55,0.65;star:1.1,1.2,1.3,1.4,1.5' --bond-cutoff 4 --hotspot-multiplier 1.5 --t-max 60 --n-times 140 --output-dir outputs/critical_slowing_N4_dense_revival_20260224`

Readout:
- strict `tau_eq` is mostly undefined (persistent oscillations), consistent with closed-system dynamics.
- dephasing times (`tau_dephase_probe`) are finite and topology-dependent:
  - `path`: max `~10.79` at `lambda=0.9`
  - `star`: max `~11.65` at `lambda=1.1`
  - `cycle`: probe-gap is nearly symmetry-suppressed; global metric is more informative.

Artifacts:
- `outputs/critical_slowing_N4_dense_revival_20260224/summary.csv`
- `outputs/critical_slowing_N4_dense_revival_20260224/tau_eq_probe_vs_lambda.png`
- `outputs/critical_slowing_N4_dense_revival_20260224/report.md`

### T2: hotspot multiplier 3.0 (historical-control alignment)
Command:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/critical_slowing_scan.py --N 4 --topologies path,cycle,star --lambdas 'path:0.8,1.03,1.2;cycle:0.35,0.49,0.65;star:1.1,1.3,1.5' --bond-cutoff 4 --hotspot-multiplier 3.0 --t-max 60 --n-times 140 --output-dir outputs/critical_slowing_N4_revival_hotspot3_20260224`

Readout:
- clear cycle signal near revival:
  - `cycle`: `tau_dephase_probe ~31.08` at `lambda=0.49` (revival-adjacent), then drops strongly by `lambda=0.65`.
- `path` and `star` do not show a clean single revival-centered peak in this coarse slice:
  - `path`: max at `lambda=1.2`
  - `star`: increases toward `lambda=1.5` in tested points.

Interpretation:
- The critical-slowing picture is partially supported: strong in `cycle` under hotspot-3 control, ambiguous in `path/star` with current observable/protocol.
- This suggests topology-sensitive slowing behavior, not a universal one-curve phenomenon yet.

Artifacts:
- `outputs/critical_slowing_N4_revival_hotspot3_20260224/summary.csv`
- `outputs/critical_slowing_N4_revival_hotspot3_20260224/tau_eq_probe_vs_lambda.png`
- `outputs/critical_slowing_N4_revival_hotspot3_20260224/report.md`

## Sweep U: observable refinement + noise-floor guard (2026-02-24)
Goal:
Resolve the path/star ambiguity from Sweep T by adding topology-aware observables and removing numerical-noise false positives.

Code updates:
- `scripts/critical_slowing_scan.py`
  - added spatial observables:
    - `site_var(t) = Var_i[Lambda_i(t)]`
    - graph-Laplacian mode amplitudes from centered `Lambda_i(t)`
      - `mode1_amp(t)` (first nontrivial mode)
      - `mode_dom_amp(t)` (nontrivial mode with largest temporal fluctuation)
  - added robust dephasing guard: `--min-scale` (default `1e-10`) so symmetry-suppressed/noise-floor channels return `NaN` instead of fake long taus.
  - added `tau_dephase_vs_lambda.png` artifact.

### U1: path/star dense hotspot-3 refinement
Command:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/critical_slowing_scan.py --N 4 --topologies path,star --lambdas 'path:0.85,0.9,0.95,1.0,1.03,1.08,1.12,1.2;star:1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5' --bond-cutoff 4 --hotspot-multiplier 3.0 --t-max 70 --n-times 180 --output-dir outputs/critical_slowing_obsref_N4_pathstar_hotspot3_minscale_20260224`

Readout:
- `mode1` channel is symmetry-suppressed (amplitude ~`1e-15`) and correctly rejected by `--min-scale`.
- consistent finite-signal peaks:
  - `path`: `tau_dephase_probe` and `tau_dephase_mode_dom` both peak at `lambda=1.12` (`~9.39`)
  - `star`: both peak at `lambda=1.30` (`~7.04`)
  - `site_var` confirms same peak locations with smaller magnitudes.

Artifacts:
- `outputs/critical_slowing_obsref_N4_pathstar_hotspot3_minscale_20260224/summary.csv`
- `outputs/critical_slowing_obsref_N4_pathstar_hotspot3_minscale_20260224/tau_dephase_vs_lambda.png`
- `outputs/critical_slowing_obsref_N4_pathstar_hotspot3_minscale_20260224/report.md`

### U2: cycle control under same estimator
Command:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/critical_slowing_scan.py --N 4 --topologies cycle --lambdas 'cycle:0.35,0.45,0.49,0.55,0.65' --bond-cutoff 4 --hotspot-multiplier 3.0 --t-max 70 --n-times 180 --output-dir outputs/critical_slowing_obsref_N4_cycle_hotspot3_minscale_20260224`

Readout:
- probe/mode channels are symmetry-suppressed for cycle (all `NaN` after noise-floor guard).
- global channel still captures the known revival-adjacent slowdown:
  - `tau_dephase_global` peaks at `lambda=0.49` (`~7.82`) in this scan.

Interpretation:
- Observable choice is topology-dependent:
  - cycle: use global channel;
  - path/star: probe or dominant Laplacian mode are informative and consistent.
- Ambiguity is reduced: path/star now show internal-consistent peaks, but these peaks are not aligned to a single universal revival lambda across topologies.

## Sweep V: N5 finite-size carryover (2026-02-24)
Goal:
Test whether the N4 topology-specific dephasing peaks persist at larger system size under the same hotspot protocol and estimator.

### V1: N5 path/star carryover (`chi=3`, hotspot=3.0)
Command:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/critical_slowing_scan.py --N 5 --topologies path,star --lambdas 'path:0.9,1.0,1.1,1.2,1.3;star:1.1,1.2,1.3,1.4,1.5' --bond-cutoff 3 --hotspot-multiplier 3.0 --t-max 70 --n-times 180 --output-dir outputs/critical_slowing_obsref_N5_pathstar_hotspot3_chi3_20260224`

Readout:
- `path`:
  - `tau_dephase_probe` peak at `lambda=1.10` (`~15.25`)
  - `tau_dephase_mode_dom` near-coincident peak around `lambda=1.00-1.10` (`~17.21` at `1.00`).
- `star`:
  - `tau_dephase_probe` peak at `lambda=1.50` (`~8.21`)
  - `tau_dephase_mode_dom` also peaks at `lambda=1.50` (`~8.21`).

Interpretation:
- `path` peak location is stable from N4 to N5 (small shift).
- `star` peak moves upward in this slice (from `~1.30` at N4 to `~1.50` at N5), indicating stronger finite-size dependence.

Artifacts:
- `outputs/critical_slowing_obsref_N5_pathstar_hotspot3_chi3_20260224/summary.csv`
- `outputs/critical_slowing_obsref_N5_pathstar_hotspot3_chi3_20260224/tau_dephase_vs_lambda.png`
- `outputs/critical_slowing_obsref_N5_pathstar_hotspot3_chi3_20260224/report.md`

### V2: N5 cycle control + refinement (`chi=3`, hotspot=3.0)
Commands:
- coarse:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/critical_slowing_scan.py --N 5 --topologies cycle --lambdas 'cycle:0.35,0.45,0.49,0.55,0.65' --bond-cutoff 3 --hotspot-multiplier 3.0 --t-max 70 --n-times 180 --output-dir outputs/critical_slowing_obsref_N5_cycle_hotspot3_chi3_20260224`
- refined:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/critical_slowing_scan.py --N 5 --topologies cycle --lambdas 'cycle:0.49,0.53,0.57,0.61,0.65,0.70' --bond-cutoff 3 --hotspot-multiplier 3.0 --t-max 70 --n-times 180 --output-dir outputs/critical_slowing_obsref_N5_cycle_hotspot3_chi3_refine_20260224`

Readout:
- probe/mode channels remain symmetry-suppressed for cycle (all `NaN` after `--min-scale`).
- global channel carries cycle slowing signal:
  - refined peak `tau_dephase_global ~6.26` at `lambda=0.61`.

Interpretation:
- cycle also shifts upward relative to N4 (`0.49 -> 0.61`) in this tested setting, with smaller peak magnitude.
- no evidence for a universal topology-independent peak lambda at N5.

Artifacts:
- `outputs/critical_slowing_obsref_N5_cycle_hotspot3_chi3_20260224/summary.csv`
- `outputs/critical_slowing_obsref_N5_cycle_hotspot3_chi3_refine_20260224/summary.csv`

N4 vs N5 peak snapshot (same estimator family):
- `path` (`tau_dephase_probe`): `1.12 -> 1.10`
- `star` (`tau_dephase_probe`): `1.30 -> 1.50`
- `cycle` (`tau_dephase_global`): `0.49 -> 0.61`

## Immediate next experiment
Run a cutoff-upgrade check (`N5`, `chi=4`) on a sparse lambda anchor set around these N5 peaks to determine whether observed peak shifts are true finite-size physics or `chi=3` truncation artifacts.
