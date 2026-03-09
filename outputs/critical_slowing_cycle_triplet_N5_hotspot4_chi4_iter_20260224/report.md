# Critical Slowing Scan

Generated: 2026-02-24T21:26:55

## Configuration
- N: 5
- topologies: cycle
- lambdas: cycle:0.55,0.61,0.65
- bond_cutoff: 4
- hotspot_multiplier: 4.0
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
cycle,iterative,sparse,0.55,,,9.696969696969697,,,,,,,,0.466080430890451,0.09022188984746696,0.0340438736328399,9.86482590059623e-11,4.680453039925133e-10,,1.6543797162223972e-19,1.5057269551399965e-19,,-6.423128402082284e-11,3.447442391236919e-10,,1.5014266826079144e-10,8.005152764628257e-10,,0.6308244133572469,7.058647000235396e-10,1.8311596683417974e-09,3.831566832620636e-19,-1.323744364768801e-10,1.3707779400238205e-09,7.150564516162517e-10,2.12590235979965e-09,1,2,1.3819660112501047,1.3819660112501055,32768,32768
cycle,iterative,sparse,0.61,,,36.96969696969697,,,,,,,,0.5051984956835242,0.11177475268420478,0.036427244188008696,5.737120223336944e-11,3.0588053598967754e-10,,1.4406020077887092e-19,1.581467090473283e-19,,-1.021840580158721e-10,4.512741940290467e-10,,1.5484371753564295e-10,6.65686617244335e-10,,0.5745370999961066,-3.9519520989017565e-10,1.0660516358562688e-09,3.229085187701709e-19,7.718592484971681e-10,-1.0072119724801581e-09,1.5924402257584254e-09,2.2843145261234434e-09,1,2,1.3819660112501047,1.3819660112501055,32768,32768
cycle,iterative,sparse,0.65,,,21.81818181818182,,,,,,,,0.4946631960953619,0.09214744454197216,0.03504469387563168,-1.775934954650893e-12,1.0058815275977053e-10,,4.0307520584932314e-21,2.625803540853069e-21,,3.742406544456677e-12,7.96264986956786e-11,,-2.9092277417260234e-12,1.1373982354796349e-10,,0.5031593566512681,4.6984194312926775e-11,1.9732038225583892e-10,6.514200464535155e-22,4.203248174713803e-11,3.7117438528519515e-11,2.0116985956532327e-10,2.5481157015747165e-10,1,2,1.3819660112501047,1.3819660112501055,32768,32768
```

## Artifacts
- `outputs/critical_slowing_cycle_triplet_N5_hotspot4_chi4_iter_20260224/summary.csv`
- `outputs/critical_slowing_cycle_triplet_N5_hotspot4_chi4_iter_20260224/tau_eq_probe_vs_lambda.png`
- `outputs/critical_slowing_cycle_triplet_N5_hotspot4_chi4_iter_20260224/tau_dephase_vs_lambda.png`
- `outputs/critical_slowing_cycle_triplet_N5_hotspot4_chi4_iter_20260224/timeseries_samples.json`
