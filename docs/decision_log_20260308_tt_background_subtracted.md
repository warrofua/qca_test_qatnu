# Decision Log: Background-Subtracted TT Anchor Test

Date: 2026-03-08

## Question

Does a newer TT-style tensor observable reproduce the earlier star-topology near-`1/k^2` correlator hint once the scalar background is measured first, shell-averaged, subtracted, and only the remaining fluctuation field is analyzed?

## Why This Test

The February 23 correlator sweeps improved the spin-2 story from "flat PSD everywhere" to "star graphs are materially closer to the target than path/cycle." But that result was based on the older bond-correlator proxy rather than the newer TT observable explicitly defined about a measured scalar background.

Given the newer robustness failures:

- hotspot preparation moves phase landmarks materially
- no-retuning topology transfer still fails
- star high-`lambda` behavior is sensitive to cutoff and phenomenological knobs

the right next question was no longer whether a pretty slope can be found, but whether the star signal survives a hard falsification test after scalar-background subtraction.

## What Was Implemented

Three code changes were added to support this test:

- `geometry.py`
  - `site_lambda_profile(...)`
  - `vertex_graph_distances(...)`
  - `shell_average(...)`
- `v3over9000/tensor_spin2.py`
  - `compute_spectrum_from_covariance(...)`
  - `compute_spectrum_from_edge_field(...)`
- `scripts/tt_background_subtracted_matrix.py`
  - preregistered matrix runner over:
    - topology / `N`
    - `lambda`
    - `bond_cutoff`
    - hotspot multiplier
    - `deltaB`
    - `kappa`
    - backend

Protocol per point:

1. build the promoted background with the same hotspot protocol used in the earlier correlator sweep
2. measure sitewise operational `Lambda_i`
3. choose a source site from the measured profile (`lambda_max`)
4. shell-average `Lambda_i` into `Lambda_bg(r)`
5. define `delta Lambda_i = Lambda_i - Lambda_bg(r_i)`
6. map this to an edge fluctuation field
7. compute TT power only from that background-subtracted field

## Anchored Runs

### Smoke

- `outputs/tt_background_subtracted_smoke_20260308/`

This confirmed the pipeline was numerically well-defined and that symmetric cases can collapse to near-zero after shell subtraction.

### N4 anchor matrix

- `outputs/tt_background_subtracted_N4_anchor_matrix_20260308/summary.csv`

Anchors were chosen from the earlier star-favorable windows:

- `chi=4`, `lambda=0.4`
- `chi=5`, `lambda=1.0`

Controls:

- path
- cycle

Backends:

- `auto`
- `dense`

### N5 anchor matrix

- `outputs/tt_background_subtracted_N5_anchor_auto_20260308/summary.csv`

Anchors:

- `chi=4`, `lambda=0.6`
- `chi=4`, `lambda=0.8`

Controls:

- path
- cycle

Backend:

- `auto` (`sparse` / iterative path)

Note: an attempted dense-parity extension for the `N5 cycle` control was abandoned as pure backend-cost overhead once it became clear that it was not needed for the first physics readout.

## Main Results

### N4

- `star`
  - raw TT power is already near zero:
    - `-0.015` at `lambda=0.4`
    - `-0.038` at `lambda=1.0`
  - background-subtracted TT power collapses to numerical zero at both anchors
- `path`
  - raw TT power is near zero
  - background-subtracted TT power becomes nonzero but non-tensor-like:
    - about `-0.660`
- `cycle`
  - background-subtracted TT power collapses to numerical zero

### N5

- `star`
  - raw TT power remains small:
    - `-0.144` at `lambda=0.6`
    - `-0.132` at `lambda=0.8`
  - background-subtracted TT power again collapses to numerical zero
- `path`
  - background-subtracted TT power collapses to numerical zero at both anchors
- `cycle`
  - `lambda=0.8` collapses to numerical zero
  - `lambda=0.6` leaves a small remainder:
    - about `-0.334`
  - still far from the tensor target

## Interpretation

This is the strongest negative update so far for the tensor sector from the new TT pipeline.

The newer TT observable does **not** reproduce the earlier raw star correlator hint at the anchored points that previously looked most favorable, and shell-level background subtraction does not recover it.

The cleanest current reading is:

- the older raw star correlator signal does not automatically carry over to this TT observable
- within this TT observable, once a measured scalar background is removed, the star advantage disappears
- what remains is either numerical zero or a small non-target anisotropic remainder

That does **not** prove that every possible emergent tensor observable is dead. It does mean the burden of proof has moved sharply:

- the project can no longer treat the raw star correlator slope as headline tensor evidence
- any future spin-2 claim must survive explicit background subtraction and negative controls

## Consequence For Project Priorities

The repo center of gravity moves further toward:

- scalar clock renormalization
- topology-dependent critical dynamics
- topology-conditioned structural mechanisms

and away from any claim that a universal tensor sector is already in hand.

The next serious tensor step is therefore not another attractive raw correlator sweep. It is one of:

1. a better background definition than simple shell averaging, if there is a principled reason for it
2. a true fluctuation covariance observable about that background, not just edge-field subtraction
3. a TT observable tied to a more defensible coarse-grained geometry variable

Until one of those works under the same negative controls, the honest statement is that the scalar sector is carrying the project and the tensor sector remains open.
