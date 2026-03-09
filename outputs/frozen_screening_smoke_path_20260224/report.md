# Frozen-Matter Screening Scan

Generated: 2026-02-24T09:16:02

## Configuration
- N: 4
- topologies: path
- lambdas: 0.2,0.6,1.0
- bond_cutoff: 4
- deltaB: 5.0
- kappa: 0.1
- k0: 4.0
- lambda_proxy: log
- center_source: True
- constrain_mu_nonnegative: True
- min_improvement: 0.05

## Aggregate Fits

```csv
topology,lambda,n_patterns,dim_bond,kappa_massless,resid_massless,kappa_screened,mu2_screened,resid_screened,rss_massless,rss_screened,improvement_frac,best_model
path,0.2,7,64,-0.010420108814331377,0.5361052186829485,-0.009839987769672853,0.09087091686924978,0.5307029260761282,0.0003503435397785624,0.00034331834337944403,0.02005230752523284,massless
path,0.6,7,64,-0.08077871722338123,0.5286544432568142,-0.07706662280040041,0.07423552407767646,0.5248510687814592,0.020247850450789306,0.019957554451375197,0.014337126803640214,massless
path,1.0,7,64,-0.17864489578409595,0.5196160550582134,-0.17256932270850464,0.05417220674416124,0.5174373789990472,0.09443086685556694,0.09364065662386697,0.008368134890772523,massless
```

## Case Count

- total cases: 21

## Artifacts
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/frozen_screening_smoke_path_20260224/frozen_cases.csv`
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/frozen_screening_smoke_path_20260224/frozen_fits.csv`
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/frozen_screening_smoke_path_20260224/mu2_vs_lambda.png`
- `/Users/joshuafarrow/Projects/qca_test_qatnu/outputs/frozen_screening_smoke_path_20260224/fit_residuals_vs_lambda.png`
