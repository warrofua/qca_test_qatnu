# No-Retuning Holdout Report

Generated: 2026-02-23T09:22:37

## Locked Parameterization

- deltaB: 6.5
- kappa: 0.2
- hotspot_multiplier: 1.5
- k0: 4
- bond_cutoff: 4

## Pass Criteria

- Complete critical triplet required (`lambda_c1`, `lambda_revival`, `lambda_c2` non-null).
- Phase ordering required: `lambda_c1 < lambda_revival < lambda_c2`.
- Residual gate: `residual_min <= 0.2`.
- Targeted scenarios additionally require |Δc1|<= 0.15, |Δrev|<= 0.3, |Δc2|<= 0.3.
- Adaptive refinement: rerun failed structural detections up to 1 additional attempts (factor=2.0, max_points=12).

## Preregistered Holdouts

- `N4_cycle_alpha0.8`: N=4, graph=cycle, alpha=0.8, lambda_range=[0.1, 1.5], points=6
  targets: c1=0.261, revival=0.486, c2=1.0

## Results

```text
      scenario_id  scenario_pass lambda_c1 lambda_revival lambda_c2 residual_min  phase_order_ok  residual_ok  target_match_ok  points_initial  points_final  refinement_steps  resolution_ok  insufficient_resolution err_lambda_c1 err_lambda_revival err_lambda_c2
N4_cycle_alpha0.8          False        NA             NA        NA           NA           False        False            False               6            12                 1          False                     True            NA                 NA            NA
```

## Verdict

- OVERALL: FAIL

