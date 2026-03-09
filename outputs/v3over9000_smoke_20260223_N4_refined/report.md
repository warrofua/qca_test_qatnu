# v3over9000 Reality Check

## Configuration
- N: 4
- topology: path
- alpha(input): 0.8
- lambda_value: 0.6
- bond_cutoff: 3
- hotspot_multiplier: 1.5
- frustration_time: 1.0
- gamma_corr: 0.05
- gamma_corr_diag: 0.0

## Summary (baseline vs correlated)

```csv
model,N,topology,bond_cutoff,lambda_value,hotspot_multiplier,gamma_corr,gamma_corr_diag,spin2_measured_power,spin2_measured_slope,spin2_residual_power,spin2_r2,spin2_fit_quality,alpha_self_energy,alpha_fit_r2,alpha_fit_points
baseline,4,path,3,0.6,1.5,0.0,0.0,-0.004935320754556447,0.004935320754556447,2.0049353207545564,0.9707849430763551,True,0.03471043825253391,0.9980024704917887,3
correlated,4,path,3,0.6,1.5,0.05,0.0,-0.009004500809706864,0.009004500809706864,2.009004500809707,0.9709693823039992,True,0.03501788987686343,0.9984499918736883,3
```

## Artifacts
- `outputs/v3over9000_smoke_20260223_N4_refined/tt_spin2_spectra.csv`
- `outputs/v3over9000_smoke_20260223_N4_refined/v3over9000_summary.csv`
- `outputs/v3over9000_smoke_20260223_N4_refined/alpha_self_energy.json`
- `outputs/v3over9000_smoke_20260223_N4_refined/tt_spin2_spectrum.png`
