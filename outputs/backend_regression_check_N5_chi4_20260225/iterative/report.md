# Critical Slowing Scan

Generated: 2026-02-24T21:38:13

## Configuration
- N: 5
- topologies: path,star
- lambdas: path:1.0;star:1.6
- bond_cutoff: 4
- hotspot_multiplier: 3.0
- t_max: 70.0
- n_times: 140
- lambda_proxy: log
- tail_frac: 0.2
- rel_tol: 0.08
- sustain_points: 8
- min_scale: 1e-08
- solver_backend: iterative
- auto_dense_threshold: 12000
- iterative_tol: 1e-09
- iterative_maxiter: 0

## Summary

```csv
topology,backend,hamiltonian_mode,lambda,tau_eq_global,tau_eq_probe,tau_dephase_global,tau_dephase_probe,tau_eq_site_var,tau_dephase_site_var,tau_eq_mode1,tau_dephase_mode1,tau_eq_mode_dom,tau_dephase_mode_dom,tail_mean_global,tail_std_global,threshold_global,tail_mean_probe,tail_std_probe,threshold_probe,tail_mean_site_var,tail_std_site_var,threshold_site_var,tail_mean_mode1,tail_std_mode1,threshold_mode1,tail_mean_mode_dom,tail_std_mode_dom,threshold_mode_dom,final_global_lambda,final_probe_gap,max_abs_probe_gap,final_site_var,final_mode1_amp,final_mode_dom_amp,max_abs_mode1_amp,max_abs_mode_dom_amp,mode1_index,mode_dom_index,mode1_eigval,mode_dom_eigval,dim_nominal,dim_hotspot
path,iterative,sparse,1.0,,48.84892086330935,8.057553956834532,14.60431654676259,,26.690647482014388,,,,27.697841726618705,0.44197234787352235,0.0550391397397545,0.02831449622251078,0.29540182191527997,0.030930588923995934,0.018265089080951504,0.02354288403523394,0.005716524077947741,0.0017688482135706628,-1.069225109343618e-11,3.150503861013726e-11,,-0.3265421020417848,0.04482012995625646,0.01941043521308698,0.3903849923583006,0.2597327369737678,0.41649208666223286,0.01765766160341403,1.3640700881186559e-11,-0.28570586033945367,9.759259442330506e-11,0.4574085663933617,1,2,0.381966011250105,1.3819660112501055,8192,8192
star,iterative,sparse,1.6,,,7.0503597122302155,7.0503597122302155,,2.014388489208633,,,,7.0503597122302155,0.7338781932232904,0.11880343919092207,0.03264454446709636,-1.3760216122937783,0.22275644848297438,0.061208520876025484,0.3108889460544031,0.09943327781637382,0.02453762335477039,-1.2591316878030057e-13,1.3062383899287925e-12,,-1.230751145439002,0.1992394244937483,0.05474656539220658,0.6426754349380037,-1.2050164405082653,1.9647029641730338,0.2323303395034233,5.682676551543864e-13,-1.0777994699929652,3.0471181133862046e-12,1.757283753394757,1,4,1.0,5.0,8192,8192
```

## Artifacts
- `outputs/backend_regression_check_N5_chi4_20260225/iterative/summary.csv`
- `outputs/backend_regression_check_N5_chi4_20260225/iterative/tau_eq_probe_vs_lambda.png`
- `outputs/backend_regression_check_N5_chi4_20260225/iterative/tau_dephase_vs_lambda.png`
- `outputs/backend_regression_check_N5_chi4_20260225/iterative/timeseries_samples.json`
