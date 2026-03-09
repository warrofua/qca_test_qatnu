# No-Retuning Holdout Report

Generated: 2026-02-23T09:34:11

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
- `no_violation_detected=True` means residual never crossed 10%, so revival detection cannot activate.

## Preregistered Holdouts

- `N4_cycle_alpha0.8`: N=4, graph=cycle, alpha=0.8, lambda_range=[0.1, 1.5], points=6
  targets: c1=0.261, revival=0.486, c2=1.0
- `N4_star_alpha0.8`: N=4, graph=star, alpha=0.8, lambda_range=[0.1, 1.5], points=6
  targets: c1=0.164, revival=1.3, c2=1.4
- `N4_path_alpha1.0`: N=4, graph=path, alpha=1.0, lambda_range=[0.1, 1.5], points=6

## Results

```text
      scenario_id  scenario_pass lambda_c1 lambda_revival lambda_c2 residual_min  phase_order_ok  residual_ok target_match_ok  points_initial  points_final  refinement_steps  resolution_ok  insufficient_resolution  no_violation_detected err_lambda_c1 err_lambda_revival err_lambda_c2
N4_cycle_alpha0.8          False        NA             NA        NA           NA           False        False           False               6            12                 1          False                    False                   True            NA                 NA            NA
 N4_path_alpha1.0           True      0.38           0.94      1.22     0.112106            True         True              NA               6             6                 0           True                    False                  False            NA                 NA            NA
 N4_star_alpha0.8          False      0.38           0.66      0.94     0.250054            True        False           False               6             6                 0           True                    False                  False         0.216               0.64          0.46
```

## Verdict

- OVERALL: FAIL

