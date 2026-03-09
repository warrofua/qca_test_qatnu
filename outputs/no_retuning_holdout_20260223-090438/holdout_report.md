# No-Retuning Holdout Report

Generated: 2026-02-23T09:04:40

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

## Preregistered Holdouts

- `N4_path_alpha1.0`: N=4, graph=path, alpha=1.0, lambda_range=[0.1, 1.5], points=3

## Results

```text
     scenario_id  scenario_pass  lambda_c1  lambda_revival  lambda_c2  residual_min  phase_order_ok  residual_ok target_match_ok err_lambda_c1 err_lambda_revival err_lambda_c2
N4_path_alpha1.0          False        0.8             0.8        1.5      0.129581           False         True              NA            NA                 NA            NA
```

## Verdict

- OVERALL: FAIL

