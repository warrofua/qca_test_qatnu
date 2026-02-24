# v3over9000

Purpose: push beyond parameter retuning toward a falsifiable "full story" testbed.

This folder introduces three concrete upgrades:

1. `tensor_spin2.py`
- Computes a TT-projected tensor spectrum from connected bond-bond covariance.
- Uses direction-aware edge tensors and a transverse-traceless projector.
- Fits `S_TT(k) ~ 1/k^p` with target `p=2`.

2. `alpha_self_energy.py`
- Estimates susceptibility from static self-energy response:
  - `sigma(0) = omega - omega_eff`
  - `alpha ~ (1/omega) * d sigma / d Lambda` near dilute regime.
- Avoids directly fitting Postulate-1 as the only extraction path.

3. `correlated_qca.py`
- Extends `ExactQCA` with correlated edge-pair promotion/demotion channels.
- Adds structural bond-bond coupling controlled by `gamma_corr`.

## Quick start

```bash
.venv/bin/python v3over9000/run_reality_experiment.py \
  --N 4 \
  --topology path \
  --alpha 0.8 \
  --lambda-value 0.6 \
  --bond-cutoff 4 \
  --hotspot-multiplier 1.5 \
  --gamma-corr 0.05
```

Outputs:
- `v3over9000_summary.csv` (baseline vs correlated comparison)
- `tt_spin2_spectra.csv`
- `alpha_self_energy.json`
- `tt_spin2_spectrum.png`
- `report.md`

## Topology x gamma sweep

```bash
.venv/bin/python v3over9000/sweep_tt_gamma.py \
  --N 4 \
  --topologies path,cycle,star \
  --gammas 0.0,0.02,0.05,0.1 \
  --lambdas 0.2,0.4,0.6,0.8,1.0 \
  --bond-cutoff 4 \
  --workers 4
```

Sweep outputs:
- `points.csv`
- `summary.csv`
- `spin2_residual_vs_gamma.png`
- `report.md`

Lambda-dependent gamma schedule example:

```bash
.venv/bin/python v3over9000/sweep_tt_gamma.py \
  --N 5 \
  --topologies path,star \
  --gammas -0.25,-0.2,0.0 \
  --lambdas 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --gamma-mode taper_high \
  --gamma-lambda-low 0.35 \
  --gamma-lambda-high 0.70 \
  --gamma-taper-power 1.0
```

Long-run reliability/performance flags:
- `--resume`: skip already completed `(topology, gamma_corr, lambda)` rows in `points.csv`.
- `--checkpoint-every N`: write `points.csv`/`errors.csv` every `N` completed computations.
- Effective-gamma dedupe is automatic: if multiple logical rows map to the same `gamma_eff` (common in taper schedules), the solver computes once and expands rows.

For heavy N5/chi4+ runs, pin BLAS threads to avoid oversubscription:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python v3over9000/sweep_tt_gamma.py \
  --N 5 \
  --topologies star \
  --gammas -0.25,-0.2,0.0 \
  --lambdas 0.2,0.8 \
  --gamma-mode taper_high \
  --gamma-lambda-low 0.30 \
  --gamma-lambda-high 0.55 \
  --workers 2 \
  --resume \
  --checkpoint-every 1 \
  --output-dir outputs/v3over9000_example_resume
```

## Interpretation guardrails

- This is an experimental branch-in-place, not yet a replacement for the canonical pipeline.
- Results should be compared against existing run tags in `docs/results.md`.
- A meaningful success criterion is not "better fit at one point", but stable TT power-law behavior across topology/cutoff/size.
