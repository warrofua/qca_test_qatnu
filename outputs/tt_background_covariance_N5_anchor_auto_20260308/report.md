# TT Background-Subtracted Matrix

Generated: 2026-03-08T20:02:52

## Configuration

- scenarios: ['N5_star_alpha0.8', 'N5_path_alpha0.8', 'N5_cycle_alpha0.8']
- lambdas: [0.6, 0.8]
- bond_cutoffs: [4]
- hotspot_multipliers: [1.5]
- deltaBs: [6.5]
- kappas: [0.2]
- backends: ['auto']
- frustration_time: 1.0
- source_mode: lambda_max
- n_modes: 16
- n_angles: 24
- auto_dense_threshold: 8000

## Best Background-Subtracted Rows

- `N5_cycle_alpha0.8` chi=4 hotspot=1.50 dB=6.50 kappa=0.200 backend=auto: best_bg_lambda=0.600, best_bg_power=0.000, best_bg_residual=2.000; best_covbg_lambda=0.800, best_covbg_power=-0.387, best_covbg_residual=2.387; raw_residual=2.016
- `N5_path_alpha0.8` chi=4 hotspot=1.50 dB=6.50 kappa=0.200 backend=auto: best_bg_lambda=0.800, best_bg_power=-0.000, best_bg_residual=2.000; best_covbg_lambda=0.800, best_covbg_power=-0.977, best_covbg_residual=2.977; raw_residual=2.023
- `N5_star_alpha0.8` chi=4 hotspot=1.50 dB=6.50 kappa=0.200 backend=auto: best_bg_lambda=0.600, best_bg_power=-0.000, best_bg_residual=2.000; best_covbg_lambda=0.600, best_covbg_power=-0.529, best_covbg_residual=2.529; raw_residual=2.132
