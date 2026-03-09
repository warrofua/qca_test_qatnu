# Critical Slowing Scan

Generated: 2026-02-24T11:27:01

## Configuration
- N: 5
- topologies: cycle
- lambdas: cycle:0.55,0.61,0.65
- bond_cutoff: 3
- hotspot_multiplier: 3.0
- t_max: 60.0
- n_times: 100
- lambda_proxy: log
- tail_frac: 0.2
- rel_tol: 0.08
- sustain_points: 8
- min_scale: 1e-08
- solver_backend: iterative
- auto_dense_threshold: 12000
- iterative_tol: 1e-08
- iterative_maxiter: 0

## Summary

```csv
topology,backend,hamiltonian_mode,lambda,tau_eq_global,tau_eq_probe,tau_dephase_global,tau_dephase_probe,tau_eq_site_var,tau_dephase_site_var,tau_eq_mode1,tau_dephase_mode1,tau_eq_mode_dom,tau_dephase_mode_dom,tail_mean_global,tail_std_global,threshold_global,tail_mean_probe,tail_std_probe,threshold_probe,tail_mean_site_var,tail_std_site_var,threshold_site_var,tail_mean_mode1,tail_std_mode1,threshold_mode1,tail_mean_mode_dom,tail_std_mode_dom,threshold_mode_dom,final_global_lambda,final_probe_gap,max_abs_probe_gap,final_site_var,final_mode1_amp,final_mode_dom_amp,max_abs_mode1_amp,max_abs_mode_dom_amp,mode1_index,mode_dom_index,mode1_eigval,mode_dom_eigval,dim_nominal,dim_hotspot
cycle,iterative,sparse,0.55,,,4.848484848484849,,,,,,,,0.3132592985184269,0.08156570384840536,0.02181863159580195,-1.083180933836303e-10,2.7253092138925765e-10,,9.495464660466382e-20,9.342535216893193e-20,,-2.1610219505969634e-11,5.44394659659038e-10,,-2.1610219505969634e-11,5.44394659659038e-10,,0.39172284441209604,1.3676981769350505e-10,6.244850192516083e-10,4.199022042104852e-20,-7.640417105736039e-11,-7.640417105736039e-11,1.2471169470690962e-09,1.2471169470690962e-09,1,1,1.3819660112501047,1.3819660112501047,7776,7776
cycle,iterative,sparse,0.61,,,9.090909090909092,,,,,,,,0.3584475292184708,0.10750530213418047,0.024687991171818496,2.962794815042358e-10,1.1990774343810642e-09,,7.54570717024308e-19,6.418732172177253e-19,,1.5178816358165604e-10,1.4981233543444562e-09,,1.5178816358165604e-10,1.4981233543444562e-09,,0.510056141479338,2.2375195030122086e-09,3.1293309055158147e-09,1.993674995084036e-18,1.7121592369940905e-09,1.7121592369940905e-09,3.123446520715603e-09,3.123446520715603e-09,1,1,1.3819660112501047,1.3819660112501047,7776,7776
cycle,iterative,sparse,0.65,,,6.0606060606060606,,,,,,,,0.3616811285834829,0.08181109977755098,0.02440732391616067,-3.0467738054706926e-11,1.6946436750080803e-09,,2.1760899386189007e-18,1.991681807328952e-18,,-3.4873509569397944e-11,3.1680590994966693e-09,,-3.4873509569397944e-11,3.1680590994966693e-09,,0.2821414982142843,1.5900508687494153e-10,3.5926430719257496e-09,1.5949996611799314e-19,6.479719202260425e-10,6.479719202260425e-10,6.495452042223197e-09,6.495452042223197e-09,1,1,1.3819660112501047,1.3819660112501047,7776,7776
```

## Artifacts
- `outputs/critical_slowing_convmat_N5_cycle_chi3_iter_triplet_20260224/summary.csv`
- `outputs/critical_slowing_convmat_N5_cycle_chi3_iter_triplet_20260224/tau_eq_probe_vs_lambda.png`
- `outputs/critical_slowing_convmat_N5_cycle_chi3_iter_triplet_20260224/tau_dephase_vs_lambda.png`
- `outputs/critical_slowing_convmat_N5_cycle_chi3_iter_triplet_20260224/timeseries_samples.json`
