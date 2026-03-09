# Research Console

Read-only visualizer for the current notebook state, centered on the scalar/topology result and tensor falsification panels.

## Purpose

This console is intentionally truth-first:

- scalar/topology results are the headline view
- tensor is presented as a falsification track
- redteam sensitivity checks are visible alongside claims

## Launch

From the repo root:

```bash
python3 Simulator/run_research_console.py
```

If `textual` is missing in the active environment, install the simulator requirements first:

```bash
python3 -m pip install -r Simulator/requirements.txt
```

## Views

1. `Scalar / Transfer`
   - locked no-retuning holdout summary
   - scalar/topology verdict

2. `Tensor Falsification`
   - covariance-background TT comparison across path/cycle/star
   - current tensor verdict

3. `Critical Slowing`
   - star sensitivity matrix from the February 25 N5 scans

4. `Redteam Checks`
   - harmonic-rank sensitivity table
   - reminder that the tensor verdict weakened under stress tests

## Interaction

- Use `1` / `2` / `3` / `4` to switch views quickly.
- The left table is the active dataset for that view.
- The right-hand panels show:
  - an ASCII topology sketch inferred from the selected row
  - a view-specific visual summary:
    - scalar landmark timeline
    - TT comparison bars
    - critical-slowing peak bars
    - redteam rank-span bars
  - row-specific detail for the currently highlighted result
- Use arrow keys inside the table to move between rows.

## Data Sources

- `docs/theory_status_20260308.md`
- `docs/results.md`
- `outputs/no_retuning_holdout_20260225_locked_d5_k01_h3_pathstar16/holdout_summary.csv`
- `outputs/tt_background_covariance_anchor_comparison_20260308.csv`
- `outputs/tt_background_harmonic_anchor_comparison_20260308.csv`
- `outputs/critical_slowing_next10_completion_20260225/star_peak_summary.csv`
- `outputs/harmonic_rank_dependence_20260308/summary.csv`

## Current Design Constraint

The existing simulator entrypoints in `Simulator/backend/main.py`, `Simulator/backend/models.py`, and `Simulator/frontend/app.py` already have local edits. The research console is therefore implemented as a separate read-only frontend so it can evolve without colliding with those files.
