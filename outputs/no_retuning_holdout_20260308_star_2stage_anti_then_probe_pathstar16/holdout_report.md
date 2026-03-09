# No-Retuning Holdout Report

Generated: 2026-03-08T18:03:17

## Locked Parameterization

- deltaB: 5.0
- kappa: 0.1
- hotspot_multiplier: 3.0
- hotspot_time: 1.0
- hotspot_edge_weights: None
- hotspot_stages: None
- k0: 4
- bond_cutoff: 4
- gamma_corr: 0.0
- gamma_corr_diag: 0.0
- graph_overrides: {"star": {"gamma_corr": -0.25, "hotspot_stages": [{"edge_weights": [0.5, 1.25, 1.25], "multiplier": 3.0, "time": 0.25}, {"edge_weights": [1.5, 0.75, 0.75], "multiplier": 3.0, "time": 0.25}]}}

## Pass Criteria

- Complete critical triplet required (`lambda_c1`, `lambda_revival`, `lambda_c2` non-null).
- Phase ordering required: `lambda_c1 < lambda_revival < lambda_c2`.
- Residual gate: `residual_min <= 0.2`.
- Targeted scenarios additionally require |Δc1|<= 0.15, |Δrev|<= 0.3, |Δc2|<= 0.3.
- Adaptive refinement disabled.
- `no_violation_detected=True` means residual never crossed 10%, so revival detection cannot activate.

## Preregistered Holdouts

- `N4_star_alpha0.8`: N=4, graph=star, alpha=0.8, lambda_range=[0.1, 1.5], points=16
  targets: c1=0.164, revival=1.3, c2=1.4
- `N4_path_alpha1.0`: N=4, graph=path, alpha=1.0, lambda_range=[0.1, 1.5], points=16

## Results

```text
     scenario_id  scenario_pass  lambda_c1  lambda_revival  lambda_c2  residual_min  phase_order_ok  residual_ok target_match_ok  points_initial  points_final  refinement_steps  resolution_ok  insufficient_resolution  no_violation_detected err_lambda_c1 err_lambda_revival err_lambda_c2
N4_path_alpha1.0           True       0.19            0.37   0.657143      0.135876            True         True              NA              16            16                 0           True                    False                  False            NA                 NA            NA
N4_star_alpha0.8          False       0.19            0.28   0.460000      0.117992            True         True           False              16            16                 0           True                    False                  False         0.026               1.02          0.94
```

## Verdict

- OVERALL: FAIL

