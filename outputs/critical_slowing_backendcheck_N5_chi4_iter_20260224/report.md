# Critical Slowing Scan

Generated: 2026-02-24T11:21:30

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
- solver_backend: iterative
- auto_dense_threshold: 12000
- iterative_tol: 1e-10
- iterative_maxiter: 0

## Summary

```csv
topology,backend,hamiltonian_mode,lambda,tau_eq_global,tau_eq_probe,tau_dephase_global,tau_dephase_probe,tau_eq_site_var,tau_dephase_site_var,tau_eq_mode1,tau_dephase_mode1,tau_eq_mode_dom,tau_dephase_mode_dom,tail_mean_global,tail_std_global,threshold_global,tail_mean_probe,tail_std_probe,threshold_probe,tail_mean_site_var,tail_std_site_var,threshold_site_var,tail_mean_mode1,tail_std_mode1,threshold_mode1,tail_mean_mode_dom,tail_std_mode_dom,threshold_mode_dom,final_global_lambda,final_probe_gap,max_abs_probe_gap,final_site_var,final_mode1_amp,final_mode_dom_amp,max_abs_mode1_amp,max_abs_mode_dom_amp,mode1_index,mode_dom_index,mode1_eigval,mode_dom_eigval,dim_nominal,dim_hotspot
path,iterative,sparse,1.0,,,7.59493670886076,11.645569620253166,,23.797468354430382,,,,17.21518987341772,0.43962452578736533,0.05944406872232574,0.028126670453399703,0.292445182387404,0.04149534101099986,0.018028557920215304,0.023237344195513786,0.006950588553598382,0.0017444050264660918,1.6429839039472812e-11,1.73914791178947e-11,,-0.3216327009464735,0.06043958937461876,0.01901768312790163,0.4475560990309441,0.2997705775540728,0.417578274508321,0.023747344151895514,4.4232532593508086e-11,-0.33212454507332945,5.902071208156505e-11,0.45324595636056986,1,2,0.381966011250105,1.3819660112501055,8192,8192
star,iterative,sparse,1.6,,,4.556962025316456,4.556962025316456,,2.0253164556962027,,,,4.556962025316456,0.7213770427907099,0.18894281521765188,0.03164445243250947,-1.3525819552326335,0.3542677785330146,0.05933334831099664,0.31279737672441077,0.1546706505730185,0.024969154514350438,-6.049848122469115e-14,3.7343761036702094e-13,,-1.2097860788158514,0.31686673401513854,0.0530693600623873,0.610217135512351,-1.1441571290856964,1.9762840891330373,0.20945528576600578,-4.3853809472693683e-14,-1.0233652470306136,8.159584119482588e-13,1.76764222646127,1,4,1.0,5.0,8192,8192
```

## Artifacts
- `outputs/critical_slowing_backendcheck_N5_chi4_iter_20260224/summary.csv`
- `outputs/critical_slowing_backendcheck_N5_chi4_iter_20260224/tau_eq_probe_vs_lambda.png`
- `outputs/critical_slowing_backendcheck_N5_chi4_iter_20260224/tau_dephase_vs_lambda.png`
- `outputs/critical_slowing_backendcheck_N5_chi4_iter_20260224/timeseries_samples.json`
