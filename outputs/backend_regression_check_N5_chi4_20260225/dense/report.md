# Critical Slowing Scan

Generated: 2026-02-24T21:38:09

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
- solver_backend: dense
- auto_dense_threshold: 12000
- iterative_tol: 1e-09
- iterative_maxiter: 0

## Summary

```csv
topology,backend,hamiltonian_mode,lambda,tau_eq_global,tau_eq_probe,tau_dephase_global,tau_dephase_probe,tau_eq_site_var,tau_dephase_site_var,tau_eq_mode1,tau_dephase_mode1,tau_eq_mode_dom,tau_dephase_mode_dom,tail_mean_global,tail_std_global,threshold_global,tail_mean_probe,tail_std_probe,threshold_probe,tail_mean_site_var,tail_std_site_var,threshold_site_var,tail_mean_mode1,tail_std_mode1,threshold_mode1,tail_mean_mode_dom,tail_std_mode_dom,threshold_mode_dom,final_global_lambda,final_probe_gap,max_abs_probe_gap,final_site_var,final_mode1_amp,final_mode_dom_amp,max_abs_mode1_amp,max_abs_mode_dom_amp,mode1_index,mode_dom_index,mode1_eigval,mode_dom_eigval,dim_nominal,dim_hotspot
path,dense,dense,1.0,,48.84892086330935,8.057553956834532,14.60431654676259,,26.690647482014388,,,,27.697841726618705,0.4419723479575803,0.0550391397570509,0.02831449622602823,0.29540182196025444,0.030930588934204875,0.01826508908392356,0.023542884040481216,0.005716524079966652,0.0017688482140347254,-2.776529420589604e-16,2.7937873401945904e-15,,-0.3265421020723255,0.04482012996530051,0.01941043521713434,0.3903849924115523,0.25973273693816523,0.41649208669679183,0.017657661601272846,3.3691146776988888e-15,-0.2857058603100807,6.532853270192844e-15,0.4574085665043162,1,2,0.381966011250105,1.3819660112501055,8192,8192
star,dense,dense,1.6,,,7.0503597122302155,7.0503597122302155,,2.014388489208633,,,,7.0503597122302155,0.7338781932256678,0.11880343919179993,0.03264454446722579,-1.376021612298127,0.22275644848462453,0.061208520876048333,0.3108889460564833,0.09943327781744808,0.024537623354728748,2.978768025891603e-16,1.1859187231480997e-15,,-1.230751145442989,0.19923942449522056,0.05474656539242362,0.6426754349387681,-1.2050164405101922,1.9647029641758076,0.23233033950397583,-2.248201624865942e-15,-1.0777994699942468,2.581268532253489e-15,1.7572837533969758,1,4,1.0,5.0,8192,8192
```

## Artifacts
- `outputs/backend_regression_check_N5_chi4_20260225/dense/summary.csv`
- `outputs/backend_regression_check_N5_chi4_20260225/dense/tau_eq_probe_vs_lambda.png`
- `outputs/backend_regression_check_N5_chi4_20260225/dense/tau_dephase_vs_lambda.png`
- `outputs/backend_regression_check_N5_chi4_20260225/dense/timeseries_samples.json`
