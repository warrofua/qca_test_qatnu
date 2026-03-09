# Decision Log: March 8 Redteam of Tensor-Sector Conclusions

Date: 2026-03-08

## Goal

Recheck the strongest recent tensor-side conclusions one assumption at a time:

1. claim linkage to the older star correlator hint
2. source-choice dependence
3. permutation / relabeling dependence
4. harmonic-rank dependence
5. fit-resolution fragility

## 1. Claim Linkage

Result:

- the older positive-looking star signal came from the earlier bond-correlator proxy
- the newer TT pipeline is a different observable
- in the newer TT pipeline, the raw signal is already near zero at the anchored star points

Consequence:

- it is too strong to say "background subtraction killed the old star signal"
- the defensible statement is:
  - the newer TT observable does not reproduce the old star hint
  - shell-level subtraction within that newer TT observable does not recover it

## 2. Source-Choice Dependence

Artifacts:

- `outputs/source_mode_dependence_20260308/source_mode_points.csv`
- `outputs/source_mode_dependence_20260308/source_mode_summary.csv`

Result:

- covariance-background power is very stable to source choice across the anchored star/cycle cases
- harmonic-background power is:
  - stable for cycle at the anchored points
  - modestly source-sensitive for star because `center` changes the shell count
- shell edge-field subtraction is the least stable:
  - `N5 cycle` at one anchor shows a meaningful source-choice shift

Consequence:

- the shell edge-field observable should be treated as source-gauge-sensitive
- covariance-background is the most source-robust of the tested subtraction schemes
- harmonic-background is not source-free on star because its rank is tied to shell count

## 3. Permutation / Relabeling Dependence

Artifacts:

- `outputs/permutation_dependence_20260308/points.csv`
- `outputs/permutation_dependence_20260308/summary.csv`

Result:

- covariance-background and harmonic-background observables are effectively invariant under:
  - cycle automorphisms
  - star leaf permutations
- shell edge-field subtraction is not fully invariant:
  - `N5 cycle` shows a nontrivial span under relabeling

Consequence:

- the current tensor-side conclusions are not primarily artifacts of graph labeling
- the main arbitrariness is source choice and projector choice, not encoding of the graph itself

## 4. Harmonic-Rank Dependence

Artifacts:

- `outputs/harmonic_rank_dependence_20260308/points.csv`
- `outputs/harmonic_rank_dependence_20260308/summary.csv`

Result:

- `N4 cycle` near-target behavior appears only at exactly the current `shell_count` projector:
  - keep `2`: power about `-0.351`
  - keep `3`: power about `2.537`
  - keep `4`: power about `0`
- `N5 cycle` remains non-target even when the rank is varied:
  - keep `2`: about `-0.29`
  - keep `3` or `4`: about `0.338`
- star does not become near-target under nearby rank changes

Consequence:

- the `N4 cycle` harmonic result is highly projector-specific
- this strongly supports the symmetry-artifact reading
- the harmonic observable should not be treated as a stable tensor diagnostic yet

## 5. Fit-Resolution Fragility

Artifacts:

- `outputs/fit_fragility_harmonic_20260308/points.csv`
- `outputs/fit_fragility_harmonic_20260308/summary.csv`

Result:

- `N4 cycle` harmonic fitted power moves materially with `n_modes` / `n_angles`:
  - span about `1.07`
  - range about `1.70` to `2.77`
- `N5 cycle` and `N5 star` are more stable:
  - `N5 cycle` span about `0.066`
  - `N5 star` span about `0.015`

Consequence:

- the fitted "near-target" power for `N4 cycle` is not numerically rigid
- this makes it even less appropriate to treat that point as serious evidence for a tensor sector

## Net Verdict

After the redteam checks:

- shell edge-field subtraction remains a useful hard negative control, but it is source-sensitive and should not be over-interpreted
- covariance-background subtraction remains the most robust tested background definition so far
- harmonic-background subtraction found a real small-`N` cycle structure, but:
  - it is projector-specific
  - it is fit-sensitive at `N4`
  - it fails `N4 -> N5` carryover

So the redteam verdict is:

- no tested tensor observable yet supports a topology-stable, finite-size-stable TT sector
- the scalar / topology-dependent story remains the strongest supported physics in the repo

## Action

The decision logs should now use the following hierarchy:

1. shell TT:
   - new TT observable fails to reproduce the old star hint
2. covariance TT:
   - most robust tested subtraction; still no convincing tensor sector
3. harmonic TT:
   - interesting `N4 cycle` artifact, not a scalable positive result
