# Decision Log: Explaining the `N4 Cycle` Harmonic Signal

Date: 2026-03-08

## Question

Was the near-target `N4 cycle` result under harmonic-background subtraction a real tensor signal, or just a symmetry artifact of the line-graph mode structure?

## Method

I added a dedicated mode-decomposition script:

- `scripts/harmonic_mode_decomposition.py`

and ran:

- `outputs/harmonic_mode_decomposition_cycle_anchor_20260308/summary.csv`
- `outputs/harmonic_mode_decomposition_cycle_anchor_20260308/modes.csv`
- `outputs/harmonic_mode_decomposition_cycle_anchor_20260308/report.md`

The script:

1. builds the bond covariance `C`
2. diagonalizes the line-graph Laplacian
3. splits the mode basis into:
   - low modes removed by the harmonic background projector
   - high modes left in the residual
4. measures:
   - residual subspace dimension
   - residual covariance rank
   - mode weights inside the residual block
   - resulting harmonic-background TT power

## Main Result

### `N4 cycle`

For `N4 cycle`:

- edge count = `4`
- shell count = `3`
- harmonic subtraction removes the lowest `3` line-graph modes
- therefore only **one** high mode remains

Numerically:

- `high_mode_dim = 1`
- `harmonic_residual_rank_numeric = 1`
- `top_high_mode_weight_fraction = 1.0`
- `harmbg_power = 2.536688` at every tested `lambda`

Laplacian eigenvalues:

- `[0, 2, 2, 4]` up to floating-point noise

So the residual is literally the single highest-frequency alternating mode of the `C4` line graph.

### `N5 cycle`

For `N5 cycle`:

- edge count = `5`
- shell count = `3`
- harmonic subtraction removes the lowest `3` modes
- therefore **two** high modes remain

Numerically:

- `high_mode_dim = 2`
- `harmonic_residual_rank_numeric = 2`
- the two high modes split the residual weight almost exactly `50/50`
- `harmbg_power = 0.338374`

Laplacian eigenvalues:

- `[0, 1.381966, 1.381966, 3.618034, 3.618034]`

So the special `N4` single-mode leftover is gone immediately at `N5`.

## Interpretation

This explains the `N4 cycle` near-target harmonic signal.

It is not behaving like a tensor sector that responds meaningfully to `lambda` or scales upward in system size. It is behaving like a projector artifact on a very small symmetric graph:

- `N4 cycle` leaves one alternating line-graph mode
- that leftover mode always carries the entire residual
- its TT power therefore stays fixed

That is exactly what a symmetry artifact should look like.

## Conclusion

For the current harmonic projector choice (`keep = shell_count`), the `N4 cycle` harmonic-background signal is best interpreted as a small-`N` line-graph symmetry artifact, not as evidence for a robust tensor sector.

This substantially narrows the most interesting loophole opened by the harmonic-background follow-up:

- the observable was not completely trivial
- but its strongest positive-looking result is now explicitly explained by mode counting and symmetry

So the tensor verdict remains negative:

- no tested observable has yet produced a topology-stable, finite-size-stable TT sector

while the scalar/topology story remains the strongest supported physics in the repo.
