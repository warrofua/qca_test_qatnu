# v3over9000 gamma sweep

## Configuration
- N: 5
- alpha: 0.8
- topologies: star
- gammas: -0.25,-0.2,0.0
- lambdas: 0.2,0.8
- gamma_mode: taper_high
- gamma_lambda_low: 0.3
- gamma_lambda_high: 0.55
- gamma_taper_power: 1.0
- bond_cutoff: 4
- hotspot_multiplier: 1.5
- frustration_time: 1.0
- workers: 2
- resume: True
- checkpoint_every: 1
- pending_logical_points: 6
- pending_unique_computations: 4
- dedupe_saved: 2
- interrupted: False

## Summary

```csv
topology,gamma_corr,n_points,best_lambda,best_spin2_power,best_spin2_slope,best_spin2_residual,best_spin2_r2,best_postulate_residual,mean_postulate_residual,max_postulate_residual,p90_postulate_residual,frac_postulate_gt_0_2,mean_gamma_corr_effective,min_gamma_corr_effective,max_gamma_corr_effective
star,-0.25,2,0.2,0.060167363997975475,-0.060167363997975475,1.9398326360020246,0.9795546027112905,0.008822478380173582,0.20912589164697082,0.40942930491376806,0.3693686222604086,0.5,-0.125,-0.25,0.0
star,-0.2,2,0.2,0.0457279197574206,-0.0457279197574206,1.9542720802425795,0.979960415463341,0.0064706970012455844,0.20795000095750682,0.40942930491376806,0.3691334441225158,0.5,-0.1,-0.2,0.0
star,0.0,2,0.2,-0.0037665635352809052,0.0037665635352809052,2.003766563535281,0.9813205872108406,0.0041863261050989475,0.2068078155094335,0.40942930491376806,0.36890500703290113,0.5,0.0,0.0,0.0
```

## Artifacts
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/v3over9000_gamma_taper2_N5_star_chi4_micro_20260224/points.csv`
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/v3over9000_gamma_taper2_N5_star_chi4_micro_20260224/summary.csv`
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/v3over9000_gamma_taper2_N5_star_chi4_micro_20260224/spin2_residual_vs_gamma.png`
