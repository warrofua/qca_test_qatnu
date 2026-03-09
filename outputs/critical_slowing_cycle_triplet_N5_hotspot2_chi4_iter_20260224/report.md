# Critical Slowing Scan

Generated: 2026-02-24T21:25:39

## Configuration
- N: 5
- topologies: cycle
- lambdas: cycle:0.55,0.61,0.65
- bond_cutoff: 4
- hotspot_multiplier: 2.0
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
cycle,iterative,sparse,0.55,,,4.848484848484849,,,,,,,,0.16895709457492486,0.04775217748872933,0.010274006782334295,-9.002639814181813e-11,3.0106488735995496e-10,,1.343385720221192e-19,1.4504414868128097e-19,,-2.0437754243718223e-10,7.150839597028583e-10,,-2.0437754243718223e-10,7.150839597028583e-10,,0.10272644554916033,-2.0605811501539506e-10,6.052260637101625e-10,1.167715738093769e-19,-7.321735023421669e-10,-7.321735023421669e-10,1.928096110722301e-09,1.928096110722301e-09,1,1,1.3819660112501047,1.3819660112501047,32768,32768
cycle,iterative,sparse,0.61,,,4.848484848484849,,,,,,,,0.2033225236524739,0.05518686478275239,0.012277166478332576,-5.53399132074972e-12,7.148126854095039e-11,,4.552164000753215e-21,3.3545146312942843e-21,,-2.883462658448541e-11,5.909270666908524e-11,,1.1618971288162743e-11,1.2707353140384602e-10,,0.1943693291193735,1.101893021271394e-10,1.6772894184668985e-10,1.3003086022360702e-20,-1.0925658177983453e-10,2.2284888450590704e-10,1.6435487335419095e-10,2.65162860701592e-10,1,2,1.3819660112501047,1.3819660112501055,32768,32768
cycle,iterative,sparse,0.65,,,12.727272727272728,,,,,,,,0.23475766106497659,0.06235367175558354,0.014252251056870745,-7.287240255671179e-13,7.914142049275464e-10,,5.78170779612801e-19,5.452148118682597e-19,,-1.1832948806951435e-10,1.539343957386438e-09,,-1.1832948806951435e-10,1.539343957386438e-09,,0.29531876621829617,1.3171899126973585e-09,1.8788388067569883e-09,1.3129172455990754e-18,2.5323369389297627e-09,2.5323369389297627e-09,3.3347969074993286e-09,3.3347969074993286e-09,1,1,1.3819660112501047,1.3819660112501047,32768,32768
```

## Artifacts
- `outputs/critical_slowing_cycle_triplet_N5_hotspot2_chi4_iter_20260224/summary.csv`
- `outputs/critical_slowing_cycle_triplet_N5_hotspot2_chi4_iter_20260224/tau_eq_probe_vs_lambda.png`
- `outputs/critical_slowing_cycle_triplet_N5_hotspot2_chi4_iter_20260224/tau_dephase_vs_lambda.png`
- `outputs/critical_slowing_cycle_triplet_N5_hotspot2_chi4_iter_20260224/timeseries_samples.json`
