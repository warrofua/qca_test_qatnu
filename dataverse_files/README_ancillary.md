# SRQID & QATNU Reproducibility (Ancillary Files)

This archive reproduces numerics cited in the SRQID and QATNU papers.

## What’s included
- **SRQID** (`src/srqid_numerics.py`): 
  - LR velocity via commutator-growth thresholds
  - No-signalling local quench
  - Energy-drift bound
- **QATNU** (`src/qatnu_poc.py`):
  - Local unitarity check on an 8-qubit brickwork layer (‖U†U−I‖₂)
  - 1-D Hadamard QCA: causal light-cone + dispersion
  - Spin-2 tail: PSD ∝ 1/k² (1-D, 2-D, 3-D; toy sizes)
  - MERA coarse-graining: Hausdorff dimension flow
  - Back-reaction: δχ vs. T correlation

## Layout
- `Makefile` – automation wrapper
- `requirements_qatn.txt` – pinned Python deps
- `src/srqid_numerics.py` – SRQID numerics
- `src/qatnu_poc.py` – QATNU proof-of-concept tests
- `notebooks/` – optional; you may add a notebook to be executed by `make run_nb`
- `outputs/` – created on first run (CSVs + summaries)
- `figures/` – created on first run (PDF plots)

## Quick start
```bash
make run_all      # run SRQID + QATNU
# or individually:
make run_srqid
make run_qatnu
```

### Environment variables (optional)
You can tune sizes/time by exporting, e.g.:
- `QATNU_SIZE_2D=96` (default 128)
- `QATNU_SIZE_3D=32` (default 32)
- `QATNU_STEPS=2000` (default 2000–4000 depending on task)

## Outputs (typical files)
- `outputs/lr_fit.csv`, `outputs/quench.csv`, `outputs/energy.csv`, `outputs/summary.txt` (SRQID)
- `outputs/qatnu_summary.txt` with key scalars
- `figures/`:
  - `qca_lightcone.pdf`, `dispersion.pdf`
  - `graviton_psd_1d.pdf`, `graviton_psd_2d.pdf`, `graviton_psd_3d.pdf`
  - `mera_dim_flow_1d.pdf`, `mera_dim_flow_3d.pdf`
  - `dchi_vs_T_scatter.pdf`
