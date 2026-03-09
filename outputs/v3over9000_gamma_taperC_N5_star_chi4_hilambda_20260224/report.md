# v3over9000 gamma sweep

## Configuration
- N: 5
- alpha: 0.8
- topologies: star
- gammas: -0.25
- lambdas: 0.8,0.9,1.0
- gamma_mode: taper_high
- gamma_lambda_low: 0.3
- gamma_lambda_high: 1.0
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
star,-0.25,3,0.8,-0.03305525376547639,0.03305525376547639,2.0330552537654762,0.9821029990234267,0.1997544360070691,0.28279977627034175,0.34973349000301934,0.3395690725626028,0.6666666666666666,-0.035714285714285705,-0.07142857142857141,0.0
```

## Artifacts
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/v3over9000_gamma_taperC_N5_star_chi4_hilambda_20260224/points.csv`
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/v3over9000_gamma_taperC_N5_star_chi4_hilambda_20260224/summary.csv`
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/v3over9000_gamma_taperC_N5_star_chi4_hilambda_20260224/spin2_residual_vs_gamma.png`
