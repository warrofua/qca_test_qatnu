# Research Visualizer

Standalone browser visualizer for the current QATNU/SRQID notebook state.

## Purpose

This is not the simulator control app. It is a graph-first explainer for the current scientific state:

- scalar/topology result as the headline
- tensor as a falsification/comparison track
- critical slowing and redteam stress tests shown in the same visual language

## Files

- `index.html` – UI shell
- `styles.css` – visual system and layout
- `app.js` – rendering and interactions
- `data/current_state.json` – built snapshot from the repo docs/outputs

## Build the data snapshot

From the repo root:

```bash
python3 /Users/joshuafarrow/Projects/qca_test_qatnu/scripts/build_research_visualizer_data.py
```

## Run locally

```bash
python3 -m http.server 8765 -d /Users/joshuafarrow/Projects/qca_test_qatnu/research_visualizer
```

Then open:

- `http://127.0.0.1:8765`

## Notes

- The app is intentionally standalone so it does not collide with the dirty simulator backend/frontend entrypoints.
- The current data payload is a distilled snapshot of canonical docs and outputs, not a live database-backed feed.
