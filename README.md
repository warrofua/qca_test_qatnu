# QATNU / SRQID Unified Simulation Framework

This repository houses the `app.py` workflow that unifies the exact QCA diagonalization, λ scans, SRQID structural tests, mean-field overlays, and spin-2 diagnostics used in the recent QATNU/SRQID studies.

## Key capabilities
- **Exact QCA engine** (`core_qca.py`) takes the many-body Hamiltonian, constructs the full matter+bond Hilbert space, and exposes time-evolution and observable evaluation.
- **Parallel λ scans** (`scanners.py`) sweep promotion strength to generate phase diagrams with critical points, ω scaling, and phase classification (figures saved under `figures/run_*`).
- **SRQID validators** (`srqid.py`) run Lieb–Robinson, no-signalling, and energy-drift checks in the same Hamiltonian.
- **Mean-field comparator** (`mean_field.py`, `tester.py`) overlays exact Ramsey traces with the Λ-informed semiclassical chain.
- **Spin-2 PSD** (`geometry.py`) derives a PSD directly from the measured χ profile to cross-check emergent-geometry expectations.

## Environment
```bash
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

The workflow assumes an Apple Silicon BLAS stack (Accelerate) but will fall back to OpenBLAS on other systems. macOS requires the `forkserver` start method, which `app.py` sets automatically.

## Running the production workflow
```bash
python app.py --N 4 --alpha 0.8 --points 100 --phase-space
```

Command-line options:
- `--N` : number of matter sites (tested up to N=5)
- `--alpha` : Postulate‑1 coupling in `ω/(1+αΛ)`
- `--points` : number of λ samples in the 1-D scan
- `--phase-space` : additionally generate the 2-D (α, λ) heatmap

Each run creates timestamped directories:
```
outputs/run_YYYYMMDD-HHMMSS_N{N}_alpha{α}
figures/run_YYYYMMDD-HHMMSS_N{N}_alpha{α}
```
containing CSVs, summaries, phase diagrams, mean-field overlays, and spin-2 PSD plots. See `docs/results.md` for the latest run highlights.

## Project layout
```
app.py                # orchestration CLI
core_qca.py           # exact diagonalization engine
mean_field.py         # semiclassical comparator
scanners.py           # parallel λ scans
phase_analysis.py     # plotting + critical point logic
srqid.py              # structural validators
geometry.py           # spin-2 PSD utilities
tester.py             # single-point experiments & overlays
dataverse_files/      # ancillary SRQID/QATNU scripts from Harvard Dataverse
figures/              # generated plots (timestamped subfolders)
outputs/              # CSVs & summary logs (timestamped subfolders)
```

To regenerate a figure mentioned in the docs, rerun `app.py` with the appropriate flags; the CLI prints the run tag plus the saved artifact paths.

## Ancillary Dataverse scripts
The `dataverse_files/` folder contains the original toy-model numerics (Ising SRQID checks, Hadamard light-cone, stochastic spin-2 PSD, etc.). They are preserved for reference but are not part of the main workflow. You can run them via the included `Makefile` inside that folder if you need the legacy comparison plots.
