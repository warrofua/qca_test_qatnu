# v3over9000 gamma sweep

## Configuration
- N: 4
- alpha: 0.8
- topologies: path
- gammas: 0.0,-0.2
- lambdas: 0.2
- gamma_mode: constant
- gamma_lambda_low: 0.4
- gamma_lambda_high: 0.8
- gamma_taper_power: 1.0
- bond_cutoff: 4
- hotspot_multiplier: 1.5
- frustration_time: 1.0
- workers: 1
- resume: True
- checkpoint_every: 1
- interrupted: False

## Summary

```csv
topology,gamma_corr,n_points,best_lambda,best_spin2_power,best_spin2_slope,best_spin2_residual,best_spin2_r2,best_postulate_residual,mean_postulate_residual,max_postulate_residual,p90_postulate_residual,frac_postulate_gt_0_2,mean_gamma_corr_effective,min_gamma_corr_effective,max_gamma_corr_effective
path,-0.2,1,0.2,0.0325557458913642,-0.0325557458913642,1.9674442541086357,0.974885971512378,0.0103965840393555,0.0103965840393555,0.0103965840393555,0.0103965840393555,0.0,-0.2,-0.2,-0.2
path,0.0,1,0.2,-0.0005875499436571,0.0005875499436571,2.000587549943657,0.9755887212062484,0.0090524337298889,0.0090524337298889,0.0090524337298889,0.0090524337298889,0.0,0.0,0.0,0.0
```

## Artifacts
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/v3over9000_resume_smoke_20260224/points.csv`
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/v3over9000_resume_smoke_20260224/summary.csv`
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/v3over9000_resume_smoke_20260224/spin2_residual_vs_gamma.png`
