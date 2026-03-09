# v3over9000 Reality Check

## Configuration
- N: 3
- topology: path
- alpha(input): 0.8
- lambda_value: 0.4
- bond_cutoff: 3
- hotspot_multiplier: 1.2
- frustration_time: 1.0
- gamma_corr: 0.03
- gamma_corr_diag: 0.0

## Summary (baseline vs correlated)

```csv
model,N,topology,bond_cutoff,lambda_value,hotspot_multiplier,gamma_corr,gamma_corr_diag,spin2_measured_power,spin2_measured_slope,spin2_residual_power,spin2_r2,spin2_fit_quality,alpha_self_energy,alpha_fit_r2,alpha_fit_points
baseline,3,path,3,0.4,1.2,0.0,0.0,-0.0008569781030111291,0.0008569781030111291,2.000856978103011,0.9953197251077484,True,0.0,0.0,3
correlated,3,path,3,0.4,1.2,0.03,0.0,-0.002960369521499575,0.002960369521499575,2.0029603695214995,0.9953352347503769,True,5.317867321027988e-14,0.0,3
```

## Artifacts
- `outputs/v3over9000_smoke_20260223/tt_spin2_spectra.csv`
- `outputs/v3over9000_smoke_20260223/v3over9000_summary.csv`
- `outputs/v3over9000_smoke_20260223/alpha_self_energy.json`
- `outputs/v3over9000_smoke_20260223/tt_spin2_spectrum.png`
