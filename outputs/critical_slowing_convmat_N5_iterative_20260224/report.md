# N5 Iterative Convergence Matrix

## Inputs
- path/star anchors: lambda={1.0,1.1} for path, {1.5,1.6} for star
- cycle anchors: lambda={0.55,0.61,0.65}
- hotspot_multiplier=3.0, t_max=60, n_times=100 (chi3/4/5 matrix)

## Peak Summary

```csv
chi,topology,metric,peak_lambda,peak_tau
3,cycle,tau_dephase_global,0.61,9.090909090909092
4,cycle,tau_dephase_global,0.61,9.090909090909092
5,cycle,tau_dephase_global,0.61,9.090909090909092
3,path,tau_dephase_probe,1.0,15.126050420168069
4,path,tau_dephase_probe,1.0,15.630252100840336
5,path,tau_dephase_probe,1.0,18.181818181818183
3,star,tau_dephase_probe,1.5,9.07563025210084
4,star,tau_dephase_probe,1.6,7.0588235294117645
5,star,tau_dephase_probe,1.6,7.272727272727273
```

## Backend Parity (N5 chi4 anchors)

```csv
metric,max_abs_diff
max_abs_mode1_amp,5.901391746275485e-11
final_mode1_amp,4.422573797469789e-11
max_abs_probe_gap,1.829608686776396e-11
tail_std_mode1,1.7388604011379116e-11
tail_mean_mode1,1.6428110980489427e-11
tail_mean_global,9.35873600838022e-12
max_abs_mode_dom_amp,8.152978292486068e-12
final_global_lambda,7.996270312560227e-12
final_probe_gap,7.789879852282411e-12
tail_mean_probe,5.5243587482323164e-12
```

## Artifacts
- `outputs/critical_slowing_convmat_N5_iterative_20260224/pathstar_anchor_by_chi.csv`
- `outputs/critical_slowing_convmat_N5_iterative_20260224/cycle_triplet_by_chi.csv`
- `outputs/critical_slowing_convmat_N5_iterative_20260224/peaks_by_chi.csv`
- `outputs/critical_slowing_convmat_N5_iterative_20260224/backend_parity_dense_vs_iterative_chi4.csv`
