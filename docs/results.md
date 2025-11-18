# Current Results Snapshot

## N = 4, α = 0.8, 100 λ points (run_20251116-154356_N4_alpha0.80)
- **Critical points**: λ_c1 ≈ 0.203, λ_rev ≈ 1.058 (residual ≈ 13.4%), λ_c2 ≈ 1.095.
- **SRQID checks**: v_LR ≈ 1.96, max no-signalling deviation ≈ 2.1×10⁻¹⁶, energy drift ≈ 1.4×10⁻¹⁴.
- **Mean-field overlay**: best-fit configuration at λ ≈ 1.058 shows exact and mean-field Ramsey traces re-locking during the revival.
- **Spin-2 PSD**: χ-informed PSD currently undershoots the expected 1/k² slope (measured power ≈ −0.024); further tuning of the χ→tier mapping is underway.

![N4 phase diagram](../figures/run_20251116-154356_N4_alpha0.80/phase_diagram_run_20251116-154356_N4_alpha0.80.png)

- Ramsey overlay: `figures/run_20251116-154356_N4_alpha0.80/ramsey_overlay_run_20251116-154356_N4_alpha0.80.png`
- Spin-2 PSD: `figures/run_20251116-154356_N4_alpha0.80/spin2_psd_run_20251116-154356_N4_alpha0.80.png`
- CSV data: `outputs/run_20251116-154356_N4_alpha0.80/scan_run_20251116-154356_N4_alpha0.80.csv`
- Summary log: `outputs/run_20251116-154356_N4_alpha0.80/summary_run_20251116-154356_N4_alpha0.80.txt`

## N = 5, α = 0.8, 50 λ points (run_20251116-160216_N5_alpha0.80)
- **Critical points**: λ_c1 ≈ 0.232, λ_rev ≈ 0.338 (residual ≈ 10.2%), λ_c2 ≈ 1.033.
- **SRQID checks**: v_LR ≈ 1.96, no-signalling ≈ 4.2×10⁻¹⁶, energy drift ≈ 5.0×10⁻¹⁴.

![N5 phase diagram](../figures/run_20251116-160216_N5_alpha0.80/phase_diagram_run_20251116-160216_N5_alpha0.80.png)

- Ramsey overlay: `figures/run_20251116-160216_N5_alpha0.80/ramsey_overlay_run_20251116-160216_N5_alpha0.80.png`
- Spin-2 PSD: `figures/run_20251116-160216_N5_alpha0.80/spin2_psd_run_20251116-160216_N5_alpha0.80.png`
- CSV data: `outputs/run_20251116-160216_N5_alpha0.80/scan_run_20251116-160216_N5_alpha0.80.csv`
- Summary log: `outputs/run_20251116-160216_N5_alpha0.80/summary_run_20251116-160216_N5_alpha0.80.txt`
- **2D phase space**: λ∈[0.1,1.2], α∈{0.2,…,1.2}; residual floor stays pinned at λ≈0.1 with magnitude {3.3, 6.6, 9.9, 13.1, 16.3, 19.5}×10⁻³ for α={0.2,…,1.2}. Heatmap: `figures/run_20251116-160216_N5_alpha0.80/phase_space_run_20251116-160216_N5_alpha0.80.png`; raw grid: `outputs/run_20251116-160216_N5_alpha0.80/phase_space_run_20251116-160216_N5_alpha0.80.csv`.

## N = 4 (cycle graph), α = 0.8, 40 λ points (run_20251117-170531_N4_cycle_alpha0.80)
- **Topology**: 4-site cycle (periodic boundary). Probes (0, 1) are equivalent by symmetry.
- **Critical points**: λ_c1 ≈ 0.261 (shifted right vs. open chain’s 0.203), λ_rev ≈ 0.486 with essentially zero residual (outer and inner clocks lock perfectly), λ_c2 ≈ 1.0 (slightly earlier catastrophic inversion).
- **SRQID checks**: v_LR ≈ 2.45 (higher due to degree=2 everywhere and the closed loop), no-signalling ≈ 1.1×10⁻¹⁵, energy drift ≈ 4.4×10⁻¹⁴.
- **Artifacts**: `figures/run_20251117-170531_N4_cycle_alpha0.80/phase_diagram_run_20251117-170531_N4_cycle_alpha0.80.png`, `outputs/run_20251117-170531_N4_cycle_alpha0.80/summary_run_20251117-170531_N4_cycle_alpha0.80.txt`.

## N = 4 (pyramid/star), α = 0.8, 40 λ points (run_20251117-182555_N4_pyramid_alpha0.80)
- **Topology**: central hub (degree 3) connected to 3 leaves; probes (0 hub, 1 leaf).
- **Critical points**: λ_c1 ≈ 0.164 (shifted left because the center accumulates Λ immediately), λ_rev ≈ 1.30 (residual ≈ 0.148), λ_c2 ≈ 1.40.
- **SRQID checks**: v_LR ≈ 1.05 (slower because only one promotion hub), no-signalling ≈ 4.2×10⁻¹⁶, energy drift ≈ 1.33×10⁻¹⁴.
- **Ω behavior**: inner (hub) clock runs much faster than leaf at revival, hence the large residual even at λ≈1.3.
- **Artifacts**: `figures/run_20251117-182555_N4_pyramid_alpha0.80/phase_diagram_run_20251117-182555_N4_pyramid_alpha0.80.png`, `outputs/run_20251117-182555_N4_pyramid_alpha0.80/summary_run_20251117-182555_N4_pyramid_alpha0.80.txt`.
