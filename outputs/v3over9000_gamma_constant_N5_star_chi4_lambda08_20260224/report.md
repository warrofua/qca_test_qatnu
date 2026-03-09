# v3over9000 gamma sweep

## Configuration
- N: 5
- alpha: 0.8
- topologies: star
- gammas: -0.25,-0.2,0.0
- lambdas: 0.8
- gamma_mode: constant
- gamma_lambda_low: 0.4
- gamma_lambda_high: 0.8
- gamma_taper_power: 1.0
- bond_cutoff: 4
- hotspot_multiplier: 1.5
- frustration_time: 1.0
- workers: 2
- resume: False
- checkpoint_every: 1
- pending_logical_points: 3
- pending_unique_computations: 3
- dedupe_saved: 0
- interrupted: False

## Summary

```csv
topology,gamma_corr,n_points,best_lambda,best_spin2_power,best_spin2_slope,best_spin2_residual,best_spin2_r2,best_postulate_residual,mean_postulate_residual,max_postulate_residual,p90_postulate_residual,frac_postulate_gt_0_2,mean_gamma_corr_effective,min_gamma_corr_effective,max_gamma_corr_effective
star,-0.25,1,0.8,-0.007333338557873244,0.007333338557873244,2.0073333385578733,0.9814167627762787,0.2847928711688825,0.2847928711688825,0.2847928711688825,0.2847928711688825,1.0,-0.25,-0.25,-0.25
star,-0.2,1,0.8,-0.01476581427113057,0.01476581427113057,2.0147658142711307,0.9816163779713378,0.270187560556169,0.270187560556169,0.270187560556169,0.270187560556169,1.0,-0.2,-0.2,-0.2
star,0.0,1,0.8,-0.04259346539017206,0.04259346539017206,2.042593465390172,0.9823541946484661,0.40942930491376806,0.40942930491376806,0.40942930491376806,0.40942930491376806,1.0,0.0,0.0,0.0
```

## Artifacts
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/v3over9000_gamma_constant_N5_star_chi4_lambda08_20260224/points.csv`
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/v3over9000_gamma_constant_N5_star_chi4_lambda08_20260224/summary.csv`
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/v3over9000_gamma_constant_N5_star_chi4_lambda08_20260224/spin2_residual_vs_gamma.png`
