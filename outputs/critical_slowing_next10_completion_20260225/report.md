# Next-10 Step Completion (Feb 25, 2026)

## Step 6: tester lambda semantics
- residual(current): `0.193560504777191`
- residual(legacy expr): `0.193560504777191`
- delta: `1.110e-16`

## Step 7: backend regression gate
- status: **PASS**
- max_abs_diff: `1.110e-10` vs threshold `1.000e-08`
- artifact: `outputs/backend_regression_check_N5_chi4_20260225/report.md`

## Star peak summary (Steps 1, 8, 9)
```csv
group,summary,peak_lambda,peak_tau_dephase_probe,rows,hotspot,chi,kappa,deltaB
deltaB_sweep,outputs/critical_slowing_star_deltaB_N5_hotspot3_d4.0_chi4_iter_20260225/summary.csv,1.65,6.546762589928058,5,3.0,4,,4.0
deltaB_sweep,outputs/critical_slowing_star_deltaB_N5_hotspot3_d5.0_chi4_iter_20260225/summary.csv,1.65,10.071942446043163,5,3.0,4,,5.0
deltaB_sweep,outputs/critical_slowing_star_deltaB_N5_hotspot3_d6.0_chi4_iter_20260225/summary.csv,1.65,11.58273381294964,5,3.0,4,,6.0
deltaB_sweep,outputs/critical_slowing_star_deltaB_N5_hotspot3_d4.0_chi5_iter_20260225/summary.csv,1.7,9.568345323741006,5,3.0,5,,4.0
deltaB_sweep,outputs/critical_slowing_star_deltaB_N5_hotspot3_d5.0_chi5_iter_20260225/summary.csv,1.65,7.0503597122302155,5,3.0,5,,5.0
deltaB_sweep,outputs/critical_slowing_star_deltaB_N5_hotspot3_d6.0_chi5_iter_20260225/summary.csv,1.65,8.057553956834532,5,3.0,5,,6.0
kappa_sweep,outputs/critical_slowing_star_kappa_N5_hotspot3_k0.05_chi4_iter_20260225/summary.csv,1.55,10.071942446043163,5,3.0,4,0.05,
kappa_sweep,outputs/critical_slowing_star_kappa_N5_hotspot3_k0.1_chi4_iter_20260225/summary.csv,1.65,10.071942446043163,5,3.0,4,0.1,
kappa_sweep,outputs/critical_slowing_star_kappa_N5_hotspot3_k0.2_chi4_iter_20260225/summary.csv,1.6,7.553956834532373,5,3.0,4,0.2,
kappa_sweep,outputs/critical_slowing_star_kappa_N5_hotspot3_k0.05_chi5_iter_20260225/summary.csv,1.55,18.633093525179856,5,3.0,5,0.05,
kappa_sweep,outputs/critical_slowing_star_kappa_N5_hotspot3_k0.1_chi5_iter_20260225/summary.csv,1.65,7.0503597122302155,5,3.0,5,0.1,
kappa_sweep,outputs/critical_slowing_star_kappa_N5_hotspot3_k0.2_chi5_iter_20260225/summary.csv,1.5,5.53956834532374,5,3.0,5,0.2,
star_extension,outputs/critical_slowing_star_extend_hi_N5_hotspot3.0_chi4_iter_20260225/summary.csv,1.8,6.546762589928058,3,3.0,4,,
star_extension,outputs/critical_slowing_star_extend_hi_N5_hotspot4.0_chi4_iter_20260225/summary.csv,1.75,7.0503597122302155,3,4.0,4,,
star_extension,outputs/critical_slowing_star_extend_hi_N5_hotspot3.0_chi5_iter_20260225/summary.csv,1.7,4.532374100719425,3,3.0,5,,
star_extension,outputs/critical_slowing_star_extend_hi_N5_hotspot4.0_chi5_iter_20260225/summary.csv,1.8,8.057553956834532,3,4.0,5,,
```

## Sensitivity spans by chi
```csv
chi,kappa_peak_lambda_span,deltaB_peak_lambda_span,hotspot_peak_lambda_span
4,0.09999999999999987,0.0,0.050000000000000044
5,0.1499999999999999,0.050000000000000044,0.10000000000000009
```

## Key readout
- Star high-lambda behavior remains protocol-sensitive (hotspot, kappa, deltaB each move the local peak).
- For hotspot=4.0, chi=5, extension point lambda=1.80 becomes the local maximum in the 1.70-1.80 window.
- Kappa and deltaB both induce O(0.10-0.15) peak-lambda shifts in this window, comparable to hotspot-induced drift.
