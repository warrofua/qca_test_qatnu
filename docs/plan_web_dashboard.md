# QATNU Dashboard Implementation Plan

## Overview

A single-window, real-time web dashboard for QATNU/SRQID quantum simulations using **FastAPI + WebSocket + D3.js SVG** with **SQLite** as the single database.

**Key Principles:**
- Reuse all existing physics code (`core_qca.py`, `scanners.py`, etc.) without rewriting
- Expose all hardcoded parameters with current defaults
- Real-time visualization via WebSocket streaming
- Dark sci-fi aesthetic
- Agent-accessible via structured API

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Browser (Single Window)                                    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  D3.js SVG Charts (Real-time)                         │  │
│  │  ┌─────────┬─────────┬─────────┐                     │  │
│  │  │ Phase   │ Freq    │ Λ       │                     │  │
│  │  │ Diagram │ Scaling │ Proxy   │                     │  │
│  │  ├─────────┼─────────┼─────────┤                     │  │
│  │  │ Ratio   │ Phase   │ Ramsey  │                     │  │
│  │  │ Dev     │ Class   │ Overlay │                     │  │
│  │  └─────────┴─────────┴─────────┘                     │  │
│  │  ┌─────────┬─────────┬─────────┐                     │  │
│  │  │ Spin-2  │ SRQID   │ Progress│                     │  │
│  │  │ PSD     │ Metrics │ Bar     │                     │  │
│  │  └─────────┴─────────┴─────────┘                     │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Control Panel [N] [α] [χ_max] [Start] [Pause] [Stop] │  │
│  │  [Advanced ▼] {hotspot, deltaB, kappa, k0, J0...}    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ WebSocket (real-time)
┌─────────────────────────────────────────────────────────────┐
│  FastAPI Backend                                            │
│  - /api/runs (CRUD)                                         │
│  - /ws (WebSocket streaming)                                │
│  - SQLite integration                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  SQLite Database (qca.db)                                   │
│  - runs                                                     │
│  - lambda_points                                            │
│  - validations                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Database Schema

### Tables

```sql
-- Runs table (one per simulation)
CREATE TABLE runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    status TEXT CHECK(status IN ('queued', 'running', 'paused', 'completed', 'failed', 'cancelled')),
    
    -- Core parameters
    N INTEGER NOT NULL,
    alpha REAL NOT NULL,
    bond_cutoff INTEGER NOT NULL DEFAULT 4,
    topology TEXT NOT NULL DEFAULT 'path',
    
    -- Lambda scan range
    lambda_min REAL DEFAULT 0.1,
    lambda_max REAL DEFAULT 1.5,
    num_points INTEGER DEFAULT 100,
    
    -- Advanced parameters (exposed with defaults)
    hotspot_multiplier REAL DEFAULT 3.0,
    omega REAL DEFAULT 1.0,
    deltaB REAL DEFAULT 5.0,
    kappa REAL DEFAULT 0.1,
    k0 INTEGER DEFAULT 4,
    J0 REAL DEFAULT 0.01,
    gamma REAL DEFAULT 0.0,
    tMax REAL DEFAULT 20.0,
    
    -- Probe positions (computed from topology)
    probe_out INTEGER,
    probe_in INTEGER,
    
    -- Critical points (extracted after scan)
    lambda_c1 REAL,
    lambda_revival REAL,
    lambda_c2 REAL,
    residual_min REAL,
    
    -- Full config backup
    config_json TEXT
);

-- Lambda points (one per scan point)
CREATE TABLE lambda_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    point_index INTEGER NOT NULL,
    
    -- Scan value
    lambda_val REAL NOT NULL,
    
    -- Postulate 1 metrics
    residual REAL,
    omega_out REAL,
    omega_in REAL,
    predicted_omega_out REAL,
    predicted_omega_in REAL,
    
    -- Entanglement proxies
    lambda_out REAL,  -- Λ_out
    lambda_in REAL,   -- Λ_in
    
    -- Energy spectrum
    E0 REAL, E1 REAL, E2 REAL, E3 REAL, E4 REAL, E5 REAL,
    gap01 REAL, gap12 REAL, gap23 REAL, gap34 REAL, gap45 REAL,
    min_gap REAL,
    
    -- Phase classification
    status TEXT CHECK(status IN ('✓', '~', '✗')),
    
    UNIQUE(run_id, point_index)
);

-- Validations (one per run)
CREATE TABLE validations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL UNIQUE REFERENCES runs(id) ON DELETE CASCADE,
    
    -- SRQID metrics
    v_lr REAL,
    no_signalling_max REAL,  -- max|Δ⟨Z_r⟩|
    energy_drift REAL,
    
    -- Mean-field comparison
    lambda_focus REAL,
    residual_at_focus REAL,
    measured_freq_out REAL,
    measured_freq_in REAL,
    predicted_freq_out REAL,
    predicted_freq_in REAL,
    
    -- Spin-2 PSD
    spin2_measured_power REAL,
    spin2_expected_power REAL DEFAULT 2.0,
    spin2_residual REAL
);

-- Chart cache (pre-computed data for fast loading)
CREATE TABLE chart_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    chart_type TEXT NOT NULL,  -- 'phase_diagram', 'ramsey', 'spin2_psd'
    data_json TEXT NOT NULL,   -- Structured data for D3
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(run_id, chart_type)
);

-- Indices for performance
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_timestamp ON runs(timestamp DESC);
CREATE INDEX idx_lambda_points_run_id ON lambda_points(run_id);
CREATE INDEX idx_lambda_points_run_index ON lambda_points(run_id, point_index);
```

---

## API Specification

### HTTP Endpoints

```yaml
# Runs
POST   /api/runs              # Create and queue new run
GET    /api/runs              # List runs (query: status, N, alpha, limit)
GET    /api/runs/{id}         # Get run details with current status
DELETE /api/runs/{id}         # Delete run and all associated data

# Run Control
POST   /api/runs/{id}/start   # Start queued run
POST   /api/runs/{id}/pause   # Pause running run
POST   /api/runs/{id}/resume  # Resume paused run
POST   /api/runs/{id}/cancel  # Cancel running/paused run

# Data
GET    /api/runs/{id}/lambda_points           # All λ points (array)
GET    /api/runs/{id}/lambda_points/{index}   # Single point
GET    /api/runs/{id}/progress                # {completed, total, current_lambda, eta}
GET    /api/runs/{id}/critical_points         # {lambda_c1, lambda_revival, lambda_c2, residual_min}
GET    /api/runs/{id}/validations             # SRQID + mean-field + spin2 results

# Charts (structured data for D3)
GET    /api/runs/{id}/charts/phase_diagram    # 6-panel data structure
GET    /api/runs/{id}/charts/ramsey           # Ramsey overlay data
GET    /api/runs/{id}/charts/spin2_psd        # PSD data

# Comparison
POST   /api/compare                           # Compare multiple runs
GET    /api/parameters/defaults               # Get all default parameter values

# Export (for agent accessibility)
GET    /api/runs/{id}/export?format=json      # Full structured data
GET    /api/runs/{id}/export?format=markdown  # Human-readable summary
```

### WebSocket Protocol

**Connection:** `ws://localhost:8000/ws`

**Client → Server:**
```json
{"type": "subscribe_run", "run_id": 2847}
{"type": "unsubscribe_run", "run_id": 2847}
{"type": "control", "run_id": 2847, "action": "pause"}
{"type": "control", "run_id": 2847, "action": "resume"}
{"type": "control", "run_id": 2847, "action": "cancel"}
```

**Server → Client:**
```json
// Point completed
{
  "type": "lambda_point_complete",
  "run_id": 2847,
  "point_index": 45,
  "total": 100,
  "lambda": 0.634,
  "residual": 0.134,
  "omega_out": 0.826,
  "omega_in": 0.734,
  "lambda_out": 0.263,
  "lambda_in": 0.470,
  "status": "✓"
}

// Critical points detected
{
  "type": "critical_points_found",
  "run_id": 2847,
  "critical_points": {
    "lambda_c1": 0.203,
    "lambda_revival": 1.058,
    "lambda_c2": 1.095,
    "residual_min": 0.134
  }
}

// Validation complete
{
  "type": "validation_complete",
  "run_id": 2847,
  "validations": {
    "v_lr": 1.96,
    "no_signalling_max": 5.1e-16,
    "energy_drift": 9.77e-15,
    "lambda_focus": 1.058,
    "residual_at_focus": 0.134
  }
}

// Chart data ready
{
  "type": "chart_ready",
  "run_id": 2847,
  "chart_type": "ramsey",
  "data": {...}
}

// Status change
{
  "type": "status_change",
  "run_id": 2847,
  "status": "completed",
  "timestamp": "2026-02-02T22:15:32Z"
}
```

---

## Parameters to Expose (with Current Defaults)

All previously hardcoded values are now exposed in UI with these defaults:

| Parameter | Current Default | UI Location | Physics Meaning |
|-----------|----------------|-------------|-----------------|
| **N** | 4 | Main panel | Number of matter sites |
| **α (alpha)** | 0.8 | Main panel | Postulate 1 susceptibility |
| **χ_max (bond_cutoff)** | 4 | Main panel | Max bond dimension |
| **Topology** | 'path' | Main panel | Graph structure |
| **λ_min** | 0.1 | Main panel | Scan start |
| **λ_max** | 1.5 | Main panel | Scan end |
| **Points** | 100 | Main panel | Number of λ samples |
| **Hotspot Multiplier** | 3.0 | Advanced ▼ | Frustration protocol strength |
| **ω (omega)** | 1.0 | Advanced ▼ | Bare clock frequency |
| **δB (deltaB)** | 5.0 | Advanced ▼ | Bond energy spacing |
| **κ (kappa)** | 0.1 | Advanced ▼ | Degree penalty strength |
| **k0** | 4 | Advanced ▼ | Target coordination number |
| **J0** | 0.01 | Advanced ▼ | Matter-matter coupling |
| **γ (gamma)** | 0.0 | Advanced ▼ | Decoherence rate |
| **tMax** | 20.0 | Advanced ▼ | Time evolution window |

### Advanced Panel Layout

```
[Advanced Parameters ▼]
┌─────────────────────────────────────┐
│ Frustration Protocol                │
│  Hotspot multiplier: [3.0    ]      │
│                                     │
│ Hamiltonian Parameters              │
│  ω (bare freq):     [1.0     ]      │
│  δB (bond spacing): [5.0     ]      │
│  κ (penalty):       [0.1     ]      │
│  k0 (target deg):   [4       ]      │
│  J0 (coupling):     [0.01    ]      │
│  γ (decoherence):   [0.0     ]      │
│                                     │
│ Scan Parameters                     │
│  tMax (evolution):  [20.0    ]      │
└─────────────────────────────────────┘
```

---

## Frontend Components

### Chart Panels (D3.js SVG)

1. **Phase Diagram (6 Sub-panels)**
   - Residual vs λ (main, with thresholds)
   - Frequency scaling (measured vs predicted)
   - Frequency inversion (ω_out vs ω_in)
   - Entanglement proxy Λ (Λ_out vs Λ_in)
   - Frequency ratio (measured/predicted)
   - Phase classification timeline (✓/~/-✗ markers)

2. **λ Scan Progress (Large)**
   - Real-time residual curve drawing
   - Progress bar with %
   - Current λ value (large display)
   - ETA calculation
   - Points colored by phase status

3. **Ramsey Overlay**
   - Exact vs mean-field traces
   - Time on x-axis, ⟨Z⟩ on y-axis
   - Interactive legend

4. **Spin-2 PSD**
   - Log-log plot
   - Measured PSD line
   - 1/k² reference line (dashed)
   - Slope indicator

5. **SRQID Metrics (Gauges)**
   - v_LR: Gauge 0-4, target zone 1.9-2.1 (green)
   - No-signalling: Log gauge, target <1e-15
   - Energy drift: Linear gauge, target ~1e-14

### Control Components

- **Start/Stop/Pause/Resume buttons** (state-aware)
- **Parameter inputs** (sliders + number fields)
- **Topology selector** (with visual diagram)
- **Run history table** (click to load old runs)
- **Comparison mode** (select multiple runs)

---

## Backend Runner Architecture

### Pause/Resume Implementation

```python
class PausableRunner:
    def __init__(self, run_id, db, ws_manager):
        self.run_id = run_id
        self.db = db
        self.ws = ws_manager
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Initially not paused
        self._cancelled = False
    
    async def run(self, config):
        run = self.db.query(Run).get(self.run_id)
        run.status = 'running'
        self.db.commit()
        
        lambda_vals = self._build_grid(config)
        
        for idx, lam in enumerate(lambda_vals):
            # Check for cancellation
            if self._cancelled:
                run.status = 'cancelled'
                self.db.commit()
                return
            
            # Wait if paused
            await self._pause_event.wait()
            
            # Run physics (existing code)
            result = await self._run_single_point(lam, config)
            
            # Save to DB
            self._save_point(idx, lam, result)
            
            # Stream to frontend
            await self.ws.broadcast(self.run_id, {
                'type': 'lambda_point_complete',
                ...
            })
        
        run.status = 'completed'
        self.db.commit()
    
    def pause(self):
        self._pause_event.clear()
        run = self.db.query(Run).get(self.run_id)
        run.status = 'paused'
        self.db.commit()
    
    def resume(self):
        self._pause_event.set()
        run = self.db.query(Run).get(self.run_id)
        run.status = 'running'
        self.db.commit()
    
    def cancel(self):
        self._cancelled = True
        self._pause_event.set()  # Unblock if paused
```

---

## File Structure

```
qca_dashboard/
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, routes
│   ├── database.py             # SQLAlchemy setup
│   ├── models.py               # Table definitions
│   ├── schemas.py              # Pydantic request/response models
│   ├── websocket_manager.py    # WebSocket connection handling
│   ├── runner.py               # PausableRunner class
│   └── api/
│       ├── __init__.py
│       ├── runs.py             # Run CRUD endpoints
│       ├── charts.py           # Chart data endpoints
│       └── export.py           # JSON/Markdown export
│
├── frontend/
│   ├── index.html              # Single-page dashboard
│   ├── css/
│   │   └── theme.css           # Dark sci-fi styles
│   └── js/
│       ├── app.js              # Main application logic
│       ├── api.js              # HTTP API client
│       ├── websocket.js        # WebSocket client
│       ├── charts.js           # D3.js chart components
│       └── controls.js         # UI controls & state
│
├── qca_core/                   # EXISTING CODE (symlink or copy)
│   ├── core_qca.py
│   ├── scanners.py
│   ├── phase_analysis.py
│   ├── srqid.py
│   ├── tester.py
│   ├── geometry.py
│   └── topologies.py
│
├── docs/
│   └── plan.md                 # This file
│
├── qca.db                      # SQLite database (created on first run)
├── run.py                      # Entry point: python run.py
└── requirements.txt            # FastAPI, SQLAlchemy, D3 (CDN), etc.
```

---

## Implementation Phases

### Phase 1: Foundation (Days 1-3)
- [ ] SQLite schema + SQLAlchemy models
- [ ] FastAPI skeleton with basic routes
- [ ] WebSocket connection manager
- [ ] Test: Can create run via API?

### Phase 2: Runner Integration (Days 4-6)
- [ ] PausableRunner wrapper around existing scanners.py
- [ ] Pause/resume/cancel logic
- [ ] Database writes per λ point
- [ ] WebSocket streaming
- [ ] Test: Can run N=4, 10 points, see real-time updates?

### Phase 3: Frontend Core (Days 7-10)
- [ ] HTML structure + CSS theme
- [ ] D3.js real-time progress chart
- [ ] WebSocket client integration
- [ ] Control panel (start/pause/resume/stop)
- [ ] Test: Can see chart drawing live in browser?

### Phase 4: Full Visualization (Days 11-14)
- [ ] 6-panel phase diagram
- [ ] Ramsey overlay panel
- [ ] Spin-2 PSD panel
- [ ] SRQID metrics gauges
- [ ] Run history table
- [ ] Test: All physics outputs visible?

### Phase 5: Polish (Days 15-17)
- [ ] Advanced parameters panel (collapsible)
- [ ] Comparison mode
- [ ] Export functionality (JSON/Markdown)
- [ ] Error handling & recovery
- [ ] Test: Can compare bond_cutoff=3 vs 4?

### Phase 6: Documentation (Days 18-20)
- [ ] API documentation
- [ ] Agent guide (how to query without UI)
- [ ] Physics parameter guide
- [ ] Migration guide from old CLI

---

## Testing Checklist

### Physics Correctness
- [ ] λ_c1, λ_revival, λ_c2 match old CLI outputs
- [ ] Residuals identical for same parameters
- [ ] SRQID validators pass (v_LR ~ 2, no-sig < 1e-15)
- [ ] Ramsey overlay matches old plots
- [ ] Spin-2 PSD slope same as before

### Control Flow
- [ ] Can pause mid-scan, resume continues from same point
- [ ] Can cancel, status updates correctly
- [ ] Can delete completed run, removes all data
- [ ] Multiple runs can be queued
- [ ] Browser refresh reconnects WebSocket, shows current state

### Agent Accessibility
- [ ] Can GET /api/runs/{id}/export?format=json and parse
- [ ] Can query SQLite directly for analysis
- [ ] WebSocket messages are structured and complete
- [ ] No screenshots needed to understand state

---

## Migration from CLI

Old workflow:
```bash
python app.py --N 4 --alpha 0.8 --points 100 --bond-cutoff 4
# Check outputs/run_*/ folder
# Open PNG files
```

New workflow:
```bash
python run.py
# Opens browser to localhost:8000
# Set parameters in UI
# Click Start
# Watch real-time charts
# Click comparison mode to overlay old runs
```

Backwards compatibility:
- CLI still works via `python -m qca_core.app`
- Old CSV outputs can be imported to dashboard database
- Dashboard can generate old-style PNGs if needed

---

## Success Metrics

1. **Single window**: All visualizations in one browser tab
2. **Single database**: SQLite file is only persistence
3. **Real-time**: Charts update within 100ms of point completion
4. **No hardcoded params**: All values exposed with documented defaults
5. **Pause/resume**: Works correctly, no data loss
6. **Agent accessible**: Can query /api/ or SQLite without UI
7. **Physics preserved**: Identical results to old CLI
8. **Reuses code**: <500 lines of new physics code (mostly wrappers)

---

## Notes

- Keep existing `scanners.py` logic, wrap it in async runner
- Use D3.js v7 from CDN (no build step)
- SQLite WAL mode for concurrent reads during writes
- WebSocket auto-reconnect on disconnect
- All times in UTC, displayed in local timezone
- Colorblind-friendly palette (test with simulator)
