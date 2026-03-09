# Critical Slowing Scan

Generated: 2026-02-24T11:28:40

## Configuration
- N: 3
- topologies: path
- lambdas: 1.0
- bond_cutoff: 3
- hotspot_multiplier: 1.5
- t_max: 6.0
- n_times: 30
- lambda_proxy: log
- tail_frac: 0.2
- rel_tol: 0.08
- sustain_points: 8
- min_scale: 1e-10
- solver_backend: auto
- auto_dense_threshold: 50
- iterative_tol: 1e-09
- iterative_maxiter: 0

## Summary

```csv
topology,backend,hamiltonian_mode,lambda,tau_eq_global,tau_eq_probe,tau_dephase_global,tau_dephase_probe,tau_eq_site_var,tau_dephase_site_var,tau_eq_mode1,tau_dephase_mode1,tau_eq_mode_dom,tau_dephase_mode_dom,tail_mean_global,tail_std_global,threshold_global,tail_mean_probe,tail_std_probe,threshold_probe,tail_mean_site_var,tail_std_site_var,threshold_site_var,tail_mean_mode1,tail_std_mode1,threshold_mode1,tail_mean_mode_dom,tail_std_mode_dom,threshold_mode_dom,final_global_lambda,final_probe_gap,max_abs_probe_gap,final_site_var,final_mode1_amp,final_mode_dom_amp,max_abs_mode1_amp,max_abs_mode_dom_amp,mode1_index,mode_dom_index,mode1_eigval,mode_dom_eigval,dim_nominal,dim_hotspot
path,iterative,sparse,1.0,,,2.0689655172413794,2.0689655172413794,,,,,,2.0689655172413794,0.14265441788733305,0.038095993469217604,0.0068299160180169466,0.10699081341550327,0.02857199510192718,0.005122437013509353,0.002725198457647565,0.00133467023712176,0.00018520535685589209,4.9237767725916056e-15,7.749884205507059e-14,,-0.08735763334443186,0.02332893631099586,0.004182452307550761,0.18661144537022714,0.13995858402760078,0.13995858402760078,0.004352978942895658,-9.838159403170626e-14,-0.11427570533007866,1.554690049233572e-13,0.11427570533007866,1,2,0.9999999999999997,3.0,72,72
```

## Artifacts
- `outputs/critical_slowing_auto_backend_smoke_iter_20260224/summary.csv`
- `outputs/critical_slowing_auto_backend_smoke_iter_20260224/tau_eq_probe_vs_lambda.png`
- `outputs/critical_slowing_auto_backend_smoke_iter_20260224/tau_dephase_vs_lambda.png`
- `outputs/critical_slowing_auto_backend_smoke_iter_20260224/timeseries_samples.json`
