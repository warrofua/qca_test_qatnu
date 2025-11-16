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

## N = 5, α = 0.8, 100 λ points (legacy run)
- **Critical points** (from `legacy outputs-figures/summary_N5.txt`): λ_c1 ≈ 0.219, λ_rev ≈ 0.338 (residual ≈ 10.2%), λ_c2 ≈ 0.984.
- **SRQID checks**: v_LR ≈ 1.96, no-signalling ≈ 4.2×10⁻¹⁶, energy drift ≈ 5.0×10⁻¹⁴.

![N5 phase diagram](../legacy%20outputs-figures/phase_diagram_N5_alpha0.8.png)

- Scan CSV: `legacy outputs-figures/scan_N5_alpha0.8.csv`
- Summary: `legacy outputs-figures/summary_N5.txt`

The fully timestamped N=5 scan (run_20251116-160216_N5_alpha0.80) is currently running; results will be appended once the job completes.

## How to extend
1. Run `python app.py --N 4 --alpha 0.8 --points 100 --phase-space` for the canonical dataset.
2. For N=5, reduce `--points` (e.g., 40–50) and focus density around λ∈[0.5,0.9] to keep runtimes manageable.
3. After each run, copy the summary lines into this document and link the new figure/CSV paths so collaborators can trace the provenance.
