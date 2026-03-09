# Critical Slowing Scan

Generated: 2026-02-24T21:26:26

## Configuration
- N: 5
- topologies: cycle
- lambdas: cycle:0.55,0.61,0.65
- bond_cutoff: 5
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
- iterative_tol: 5e-08
- iterative_maxiter: 0

## Summary

```csv
topology,backend,hamiltonian_mode,lambda,tau_eq_global,tau_eq_probe,tau_dephase_global,tau_dephase_probe,tau_eq_site_var,tau_dephase_site_var,tau_eq_mode1,tau_dephase_mode1,tau_eq_mode_dom,tau_dephase_mode_dom,tail_mean_global,tail_std_global,threshold_global,tail_mean_probe,tail_std_probe,threshold_probe,tail_mean_site_var,tail_std_site_var,threshold_site_var,tail_mean_mode1,tail_std_mode1,threshold_mode1,tail_mean_mode_dom,tail_std_mode_dom,threshold_mode_dom,final_global_lambda,final_probe_gap,max_abs_probe_gap,final_site_var,final_mode1_amp,final_mode_dom_amp,max_abs_mode1_amp,max_abs_mode_dom_amp,mode1_index,mode_dom_index,mode1_eigval,mode_dom_eigval,dim_nominal,dim_hotspot
cycle,iterative,sparse,0.55,,,4.848484848484849,,,,,,,,0.16895736799009028,0.04775562545138575,0.010274027943095518,-1.9724299485379682e-10,3.485261459668684e-09,,5.652641678521317e-18,4.502815315426512e-18,,-2.0016607900937305e-10,4.98001259425605e-10,,-2.435666203130808e-10,5.247865806164456e-09,,0.10267118409918394,1.788265591162741e-09,6.450569051130728e-09,1.936115476817104e-18,-3.3187578868886187e-10,3.0932914780289917e-09,1.2427637729416398e-09,9.056135651265394e-09,1,2,1.3819660112501047,1.3819660112501055,100000,100000
cycle,iterative,sparse,0.61,,,4.848484848484849,,,,,,,,0.20332314521209205,0.055193599076801304,0.012277214511086005,7.688002318095854e-12,4.913247291077758e-10,,9.429382926919491e-20,7.26131681651994e-20,,3.117147978730761e-11,3.3381892707894906e-10,,-1.0720431002356904e-11,5.53520395348453e-10,,0.19427194925758356,-1.0680962503339941e-09,1.0680962503339941e-09,2.0172030882480856e-19,-3.3128050414699346e-10,-8.835329283891492e-10,9.095563966782627e-10,1.06125264260411e-09,1,2,1.3819660112501047,1.3819660112501055,100000,100000
cycle,iterative,sparse,0.65,,,12.727272727272728,,,,,,,,0.2347694230851614,0.06235032305178409,0.01425318919964301,6.2071195405799525e-12,2.8130249041942315e-10,,5.041072524082771e-20,3.5279789203609535e-20,,-2.2095632793620398e-11,1.6621513225437363e-10,,2.2105522264806837e-11,3.435804586851298e-10,,0.29549683783944386,9.690748203894373e-11,8.956867469933627e-10,1.0647677224839914e-19,-2.41620805246861e-10,6.33334224203534e-10,5.197265506776102e-10,8.362480678085904e-10,1,2,1.3819660112501047,1.3819660112501055,100000,100000
```

## Artifacts
- `outputs/critical_slowing_cycle_triplet_N5_hotspot2_chi5_iter_20260224/summary.csv`
- `outputs/critical_slowing_cycle_triplet_N5_hotspot2_chi5_iter_20260224/tau_eq_probe_vs_lambda.png`
- `outputs/critical_slowing_cycle_triplet_N5_hotspot2_chi5_iter_20260224/tau_dephase_vs_lambda.png`
- `outputs/critical_slowing_cycle_triplet_N5_hotspot2_chi5_iter_20260224/timeseries_samples.json`
