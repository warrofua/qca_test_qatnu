# Decision Log: Harmonic-Background TT Follow-up

Date: 2026-03-08

## Why This Follow-up Was Needed

The covariance-background follow-up improved on shell edge-field subtraction, but it still imposed a radial shell ansatz directly in covariance space. That left one more reasonable objection open:

> perhaps the right scalar background is not shell-averaged at all, but the low-frequency topology-native sector of the edge dynamics.

So the next step was to define the background using the line-graph Laplacian rather than shell averages.

## New Observable

For each graph:

1. construct the line graph on edges
2. build its Laplacian
3. take the lowest `m` eigenmodes as the background subspace, with `m = shell_count`
4. form the residual covariance

`C_res = (I - P_low) C (I - P_low)`

5. TT-project only `C_res`

This is more topology-native than shell averaging because it removes the smooth low-frequency edge background without assuming the residual should be radial.

## Runs

Smoke:

- `outputs/tt_background_harmonic_smoke_20260308/`

Anchored `N4` matrix:

- `outputs/tt_background_harmonic_N4_anchor_auto_20260308/summary.csv`

Anchored `N5` matrix:

- `outputs/tt_background_harmonic_N5_anchor_auto_20260308/summary.csv`

Combined comparison:

- `outputs/tt_background_harmonic_anchor_comparison_20260308.csv`

Anchors matched the same earlier favorable windows used in the previous follow-ups:

- `N4`: `lambda=0.4, 1.0`, `chi=4,5`
- `N5`: `lambda=0.6, 0.8`, `chi=4`

## Main Results

### N4

- `cycle`
  - harmonic-background power is near target and cutoff-stable:
    - `2.537` at `lambda=0.4`
    - `2.537` at `lambda=1.0`
- `star`
  - remains clearly non-target:
    - `-0.352` at `lambda=0.4`
    - `-0.352` at `lambda=1.0`
- `path`
  - collapses essentially to zero

### N5

- `cycle`
  - does **not** keep the `N4` near-target result:
    - `0.338` at `lambda=0.6`
    - `0.338` at `lambda=0.8`
- `star`
  - remains non-target:
    - `-0.549` at `lambda=0.6`
    - `-0.549` at `lambda=0.8`
- `path`
  - becomes strongly non-target:
    - `-5.815` at both anchors

## Interpretation

This is the first tensor follow-up in this cycle that materially changes the topology story.

Under harmonic-background subtraction:

- the strongest `N4` signal is on `cycle`, not `star`
- the older star-first narrative is therefore not robust even qualitatively across observables

But the decisive point is the finite-size carryover:

- the `N4 cycle` near-target harmonic signal fails at `N5`

So the best current reading is:

- the harmonic-background observable found a genuine small-`N` structure
- but that structure does not yet scale
- the most likely explanation is a finite-size / symmetry artifact rather than a stable tensor sector

Important caveat:

- this interpretation depends on the current harmonic projector choice (`keep = shell_count`)
- a later redteam check should verify rank sensitivity before treating the artifact diagnosis as closed

## Scientific Consequence

This is still a negative result for the gravity claim, but it is a useful one:

- the tensor problem is not merely "star failed"
- the observable itself can move the apparent best topology
- any serious emergent spin-2 claim must now survive:
  - background subtraction
  - topology controls
  - `N4 -> N5` carryover

At present, none of the tested observables meet that standard.

## What This Suggests Next

The next tensor-facing experiment should not be another broad sweep. It should be one of:

1. explain the `N4 cycle` harmonic signal as a symmetry artifact by decomposing it more explicitly into line-graph modes
2. test whether a slightly larger cycle-like case or alternative cutoff reproduces the effect
3. if not, stop treating the tensor sector as primary and formalize the scalar/topology story as the core result

Until then, the honest status remains:

- scalar sector: real and structured
- tensor sector: unresolved and currently unsupported as a robust claim
