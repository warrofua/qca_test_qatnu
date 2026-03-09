# No-Retuning Holdout Report

Generated: 2026-02-25T09:03:04

## Locked Parameterization

- deltaB: 5.0
- kappa: 0.1
- hotspot_multiplier: 3.0
- k0: 4
- bond_cutoff: 4

## Pass Criteria

- Complete critical triplet required (`lambda_c1`, `lambda_revival`, `lambda_c2` non-null).
- Phase ordering required: `lambda_c1 < lambda_revival < lambda_c2`.
- Residual gate: `residual_min <= 0.2`.
- Targeted scenarios additionally require |Δc1|<= 0.15, |Δrev|<= 0.3, |Δc2|<= 0.3.
- Adaptive refinement disabled.
- `no_violation_detected=True` means residual never crossed 10%, so revival detection cannot activate.

## Preregistered Holdouts

- `N4_cycle_alpha0.8`: N=4, graph=cycle, alpha=0.8, lambda_range=[0.1, 1.5], points=8
  targets: c1=0.261, revival=0.486, c2=1.0
- `N4_star_alpha0.8`: N=4, graph=star, alpha=0.8, lambda_range=[0.1, 1.5], points=8
  targets: c1=0.164, revival=1.3, c2=1.4
- `N4_path_alpha1.0`: N=4, graph=path, alpha=1.0, lambda_range=[0.1, 1.5], points=8

## Results

```text
      scenario_id  scenario_pass lambda_c1 lambda_revival lambda_c2 residual_min  phase_order_ok  residual_ok target_match_ok  points_initial  points_final  refinement_steps  resolution_ok  insufficient_resolution  no_violation_detected err_lambda_c1 err_lambda_revival err_lambda_c2
N4_cycle_alpha0.8          False        NA             NA        NA           NA           False        False           False               8             8                 0          False                    False                   True            NA                 NA            NA
 N4_path_alpha1.0          False       0.3            0.3       0.7     0.107576           False         True              NA               8             8                 0          False                    False                  False            NA                 NA            NA
 N4_star_alpha0.8          False       0.3            0.3       0.5     0.174303           False         True           False               8             8                 0          False                    False                  False         0.136                1.0           0.9
```

## Verdict

- OVERALL: FAIL

