# v3over9000 gamma sweep

## Configuration
- N: 5
- alpha: 0.8
- topologies: star
- gammas: -0.25,-0.2,0.0
- lambdas: 0.9
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
star,-0.25,1,0.9,-0.010549675434807476,0.010549675434807476,2.0105496754348073,0.9815032765266191,0.31397710603891626,0.31397710603891626,0.31397710603891626,0.31397710603891626,1.0,-0.25,-0.25,-0.25
star,-0.2,1,0.9,-0.01838485247581649,0.01838485247581649,2.0183848524758163,0.9817131855350795,0.2916406801279985,0.2916406801279985,0.2916406801279985,0.2916406801279985,1.0,-0.2,-0.2,-0.2
star,0.0,1,0.9,-0.04741069530451013,0.04741069530451013,2.04741069530451,0.9824803866073202,0.3530468250855916,0.3530468250855916,0.3530468250855916,0.3530468250855916,1.0,0.0,0.0,0.0
```

## Artifacts
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/v3over9000_gamma_constant_N5_star_chi4_lambda09_20260224/points.csv`
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/v3over9000_gamma_constant_N5_star_chi4_lambda09_20260224/summary.csv`
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/v3over9000_gamma_constant_N5_star_chi4_lambda09_20260224/spin2_residual_vs_gamma.png`
