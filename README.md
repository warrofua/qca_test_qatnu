# QATNU / SRQID Unified Simulation Framework

This repository houses the `app.py` workflow that unifies the exact QCA diagonalization, λ scans, SRQID structural tests, mean-field overlays, and spin-2 diagnostics used in the recent QATNU/SRQID studies.

## Physics overview (why this exists)
The simulator implements the August 2025 QATNU/SRQID construction in which a measurement-free promotion ladder renormalizes local clock rates according to Postulate 1,
\[
\omega_{\mathrm{eff}}^{(i)} = \frac{\omega}{1+\alpha \Lambda_i},
\]
with $\Lambda_i$ extracted from the logarithmic bond-depth profile of the SRQID Hamiltonian.  Each production run follows the “GR-style” workflow documented in `docs/status_202511.tex`:

1. **Geometry from source:** build the exact many-body Hamiltonian on the requested graph, evolve its ground state under a hotspot (3λ) promotion, and measure $\Lambda_{\text{out/in}}$ from that frustrated background.
2. **Test clocks vs. embedded clocks:** by default (Protocol A) the Ramsey probes are clean π/2 excitations on the ground state, mirroring GR’s “test clock” assumption.  The optional `--embedded-clocks` flag switches to Protocol B, exciting the probes directly on the promoted background to explore the breakdown of Postulate 1 in the fully many-body regime.
3. **Postulate 1 validation:** for each λ sample we compare the measured frequency ratio (FFT of the Ramsey traces) to $(1+\alpha\Lambda_{\text{out}})/(1+\alpha\Lambda_{\text{in}})$, producing the residual/phase diagrams cited in the status paper.
4. **SRQID + spin‑2 checks:** Lieb–Robinson velocity, no-signalling, energy drift, mean-field overlays, and χ→spin-2 PSD diagnostics run within the same Hamiltonian so the entire QATNU/SRQID story stays in one script.

The goal is to give the “unified app” promised in the preprints: phase diagrams, critical points, Ramsey overlays, SRQID metrics, and spin‑2 figures all emerge from a single exact-diagonalization workflow while remaining faithful to the theory narrative.

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
- `--graph` : topology of the matter-bond network (`path`, `cycle`, `diamond`, `bowtie`, `pyramid/star`, etc.)
- `--probes` : override the outer/inner probe vertices (defaults are topology-dependent)
- `--embedded-clocks` : excite the Ramsey probes on the frustrated background instead of the ground state (Protocol B); default runs use clean test clocks (Protocol A).

Each run creates timestamped directories:
```
outputs/run_YYYYMMDD-HHMMSS_N{N}_alpha{α}
figures/run_YYYYMMDD-HHMMSS_N{N}_alpha{α}
```
containing CSVs, summaries, phase diagrams, mean-field overlays, and spin-2 PSD plots. See `docs/results.md` for the latest run highlights.

Optional extras:
- `--graph ... --probes ...` : explore alternate geometries (cycle, diamond, bowtie, pyramid/star) and choose which vertices serve as “outer”/“inner” probes.
- `--embedded-clocks` : toggle the “embedded clock” experiment discussed in the papers to stress-test Postulate 1 beyond the test-particle limit.
- `--legacy-viz` : after the main workflow, run `dataverse_files/qatnu_poc.py` to regenerate the Dataverse light-cone/dispersion/MERA figures alongside the new outputs.

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

To regenerate a figure mentioned in the docs, rerun `app.py` with the appropriate flags; the CLI prints the run tag plus the saved artifact paths. For snapshots of recent runs (including embedded figures), read [docs/results.md](docs/results.md).

## Ancillary Dataverse scripts
The `dataverse_files/` folder contains the original toy-model numerics (Ising SRQID checks, Hadamard light-cone, stochastic spin-2 PSD, etc.). They are preserved for reference but are not part of the main workflow. You can run them via the included `Makefile` inside that folder if you need the legacy comparison plots.
