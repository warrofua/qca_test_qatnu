# Decision Log: Covariance-Background TT Follow-up

Date: 2026-03-08

## Why This Follow-up Was Needed

The first background-subtracted TT test used shell-averaged subtraction at the edge-field level. That was a legitimate falsification pass, but it was also maximally aggressive:

- measure sitewise scalar `Lambda`
- shell-average it around the source
- map `delta Lambda` onto edges
- TT-project the resulting edge field

At the anchored `N4` and `N5` star points, that procedure collapsed the earlier raw star hint to numerical zero. The result was important, but it left one reasonable objection open:

> perhaps the shell subtraction was too crude because it removed the signal before covariance structure had a chance to organize.

So the next test was to move the subtraction into covariance space rather than field space.

## New Observable

The follow-up observable keeps the same measured source site and radial shell construction, but changes what is subtracted.

Per point:

1. measure connected bond covariance `C_ef`
2. assign each edge a radial shell from the measured source site
3. build an isotropic background covariance `C_bg(r_e, r_f)` by shell-pair averaging
4. define `delta C = C - C_bg`
5. TT-project only `delta C`

This is more defensible than simple edge-field subtraction because it preserves covariance structure while removing only the isotropic radial component.

## Runs

Smoke:

- `outputs/tt_background_covariance_smoke2_20260308/`

Anchored `N4` matrix:

- `outputs/tt_background_covariance_N4_anchor_auto_20260308/summary.csv`

Anchored `N5` matrix:

- `outputs/tt_background_covariance_N5_anchor_auto_20260308/summary.csv`

Combined comparison:

- `outputs/tt_background_covariance_anchor_comparison_20260308.csv`

The anchors match the earlier star-favorable windows:

- `N4 star/path/cycle`, `lambda=0.4, 1.0`, `chi=4,5`
- `N5 star/path/cycle`, `lambda=0.6, 0.8`, `chi=4`

All runs used the same hotspot / `deltaB` / `kappa` settings as the earlier correlator sweep:

- hotspot multiplier `1.5`
- `deltaB=6.5`
- `kappa=0.2`

## Main Results

### Smoke

The smoke run already showed that the covariance residual behaves differently from the shell edge-field residual:

- it does **not** collapse trivially to zero on star
- it leaves a real anisotropic covariance remainder

So this was a real observable change, not a no-op.

### N4 Anchors

- `star`
  - covariance-subtracted power:
    - `-0.352` at `lambda=0.4`
    - `-0.352` at `lambda=1.0`
- `cycle`
  - covariance-subtracted power:
    - `-0.351` at `lambda=0.4`
    - `-0.333` at `lambda=1.0`
- `path`
  - covariance-subtracted power:
    - `-1.799` at `lambda=0.4`
    - `-1.869` at `lambda=1.0`

### N5 Anchors

- `star`
  - covariance-subtracted power:
    - `-0.529` at `lambda=0.6`
    - `-0.529` at `lambda=0.8`
- `cycle`
  - covariance-subtracted power:
    - `-0.390` at `lambda=0.6`
    - `-0.387` at `lambda=0.8`
- `path`
  - covariance-subtracted power:
    - `-1.012` at `lambda=0.6`
    - `-0.977` at `lambda=0.8`

## Interpretation

This follow-up changes the technical reading, but not the overall conclusion.

What changed:

- the shell edge-field subtraction was not the only possible background definition
- the covariance-based subtraction leaves a genuine residual covariance signal instead of forcing trivial zero

What did **not** change:

- the star residual is still nowhere near the target power `2`
- star does not cleanly separate from cycle
- path remains strongly non-target

So the tensor story is still not rescued.

The right updated statement is:

- the first shell-subtracted falsification pass may have been too severe
- but the stronger covariance-background follow-up still does not reveal a convincing TT sector at the anchored star points

## Consequence

This narrows the next move further.

The project should not return to raw star correlator slopes as primary evidence.

The next serious tensor experiment should now be one of:

1. a still better coarse-grained background model with a principled justification
2. a fluctuation covariance built from a more defensible geometry variable than the present shell-defined scalar proxy
3. a benchmarked tensor observable that can show star separation from cycle under fixed negative controls

Until that happens, the tensor claim remains open and weak, while the scalar / causal / topology-dependent story remains the strongest part of the repo.
