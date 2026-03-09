# Critical Slowing Scan

Generated: 2026-02-24T11:26:34

## Configuration
- N: 5
- topologies: path,star
- lambdas: path:1.0;star:1.6
- bond_cutoff: 4
- hotspot_multiplier: 3.0
- t_max: 40.0
- n_times: 80
- lambda_proxy: log
- tail_frac: 0.2
- rel_tol: 0.08
- sustain_points: 8
- min_scale: 1e-08
- solver_backend: dense
- auto_dense_threshold: 12000
- iterative_tol: 1e-09
- iterative_maxiter: 0

## Summary

```csv
topology,backend,hamiltonian_mode,lambda,tau_eq_global,tau_eq_probe,tau_dephase_global,tau_dephase_probe,tau_eq_site_var,tau_dephase_site_var,tau_eq_mode1,tau_dephase_mode1,tau_eq_mode_dom,tau_dephase_mode_dom,tail_mean_global,tail_std_global,threshold_global,tail_mean_probe,tail_std_probe,threshold_probe,tail_mean_site_var,tail_std_site_var,threshold_site_var,tail_mean_mode1,tail_std_mode1,threshold_mode1,tail_mean_mode_dom,tail_std_mode_dom,threshold_mode_dom,final_global_lambda,final_probe_gap,max_abs_probe_gap,final_site_var,final_mode1_amp,final_mode_dom_amp,max_abs_mode1_amp,max_abs_mode_dom_amp,mode1_index,mode_dom_index,mode1_eigval,mode_dom_eigval,dim_nominal,dim_hotspot
path,dense,dense,1.0,,,7.59493670886076,11.645569620253166,,23.797468354430382,,,,17.21518987341772,0.43962452579672406,0.05944406872554703,0.028126670453159735,0.2924451823929284,0.041495341009699795,0.018028557918537476,0.023237344196241808,0.0069505885539987555,0.0017444050264955728,1.7280589833864837e-15,2.8751065155821353e-15,,-0.32163270095061547,0.06043958937716723,0.019017683127397537,0.4475560990389404,0.29977057756186276,0.4175782745266171,0.023747344152628486,6.794618810198662e-15,-0.3321245450782905,6.794618810198662e-15,0.45324595636872284,1,2,0.381966011250105,1.3819660112501055,8192,8192
star,dense,dense,1.6,,,4.556962025316456,4.556962025316456,,2.0253164556962027,,,,4.556962025316456,0.7213770427904409,0.18894281521756534,0.031644452432407634,-1.3525819552320764,0.3542677785329352,0.059333348310764276,0.3127973767241742,0.15467065057288978,0.02496915451436273,3.3306690738754696e-16,6.360547609623277e-16,,-1.2097860788154002,0.3168667340149934,0.05306936006221652,0.6102171355123456,-1.1441571290856487,1.9762840891331077,0.20945528576600198,-6.938893903907228e-16,-1.0233652470306043,1.887379141862766e-15,1.7676422264611529,1,4,1.0,5.0,8192,8192
```

## Artifacts
- `outputs/critical_slowing_backendcheck_N5_chi4_dense_20260224/summary.csv`
- `outputs/critical_slowing_backendcheck_N5_chi4_dense_20260224/tau_eq_probe_vs_lambda.png`
- `outputs/critical_slowing_backendcheck_N5_chi4_dense_20260224/tau_dephase_vs_lambda.png`
- `outputs/critical_slowing_backendcheck_N5_chi4_dense_20260224/timeseries_samples.json`
