"""FastAPI backend for QATNU First Principles Simulator."""

import json
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import get_db, db
from .models import Run, LambdaPoint, Validation, DerivationStep

# Import derivation and agreement modules
from .derivation.coordinator import DerivationCoordinator, DerivedParameters
from .derivation.anchoring import compute_all_scales
from .agreement import AgreementCalculator, PhysicsPriority, GradeScale

app = FastAPI(
    title="QATNU First Principles Simulator",
    description="Validate QATNU/SRQID theory through first-principles derivation",
    version="0.1.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== WebSocket Manager ==============

class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        # run_id -> list of websockets
        self.active_connections: Dict[int, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, run_id: int):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, run_id: int):
        if run_id in self.active_connections:
            self.active_connections[run_id].remove(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]
    
    async def broadcast(self, run_id: int, message: dict):
        """Send message to all clients watching a run."""
        if run_id not in self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections[run_id]:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn, run_id)


manager = ConnectionManager()


# ============== API Endpoints ==============

@app.get("/")
async def root():
    return {
        "message": "QATNU First Principles Simulator API",
        "docs": "/docs",
        "websocket": "ws://localhost:8000/ws/runs/{run_id}"
    }


@app.get("/api/runs")
async def list_runs(
    status: Optional[str] = None,
    limit: int = 50,
    db_session: Session = Depends(get_db)
):
    """List recent runs with optional status filter."""
    query = db_session.query(Run)
    
    if status:
        query = query.filter(Run.status == status)
    
    runs = query.order_by(Run.timestamp.desc()).limit(limit).all()
    
    return {
        "runs": [
            {
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "status": r.status,
                "N": r.N,
                "topology": r.topology,
                "chi_max": r.chi_max,
                "agreement_score": r.agreement_score,
                "lambda_c1": r.lambda_c1,
                "lambda_revival": r.lambda_revival,
                "lambda_c2": r.lambda_c2,
            }
            for r in runs
        ]
    }


@app.post("/api/runs")
async def create_run(
    config: dict,
    db_session: Session = Depends(get_db)
):
    """Create a new run with theory inputs."""
    run = Run(
        status="queued",
        N=config.get("N", 4),
        topology=config.get("topology", "path"),
        chi_max=config.get("chi_max", 4),
        G_eff=config.get("G_eff", 6.674e-11),
        c=config.get("c", 1.0),
        omega=config.get("omega", 1.0),
        config_json=config,
    )
    
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)
    
    return {
        "id": run.id,
        "status": run.status,
        "message": "Run created. Connect to WebSocket to start."
    }


@app.get("/api/runs/{run_id}")
async def get_run(run_id: int, db_session: Session = Depends(get_db)):
    """Get full run details."""
    run = db_session.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return {
        "id": run.id,
        "timestamp": run.timestamp.isoformat(),
        "status": run.status,
        "inputs": {
            "N": run.N,
            "topology": run.topology,
            "chi_max": run.chi_max,
            "G_eff": run.G_eff,
            "c": run.c,
            "omega": run.omega,
        },
        "scales": {
            "lattice_spacing": run.lattice_spacing,
            "time_step": run.time_step,
        },
        "critical_points": {
            "lambda_c1": run.lambda_c1,
            "lambda_revival": run.lambda_revival,
            "lambda_c2": run.lambda_c2,
            "residual_min": run.residual_min,
        },
        "agreement_score": run.agreement_score,
        "point_count": len(run.lambda_points),
    }


@app.get("/api/runs/{run_id}/lambda_points")
async def get_lambda_points(
    run_id: int,
    db_session: Session = Depends(get_db)
):
    """Get all lambda points for a run (for plotting)."""
    points = db_session.query(LambdaPoint).filter(
        LambdaPoint.run_id == run_id
    ).order_by(LambdaPoint.point_index).all()
    
    return {
        "points": [
            {
                "index": p.point_index,
                "lambda": p.lambda_val,
                "theory": {
                    "alpha": p.theory_alpha,
                    "omega_out": p.theory_omega_out,
                    "omega_in": p.theory_omega_in,
                },
                "measured": {
                    "omega_out": p.measured_omega_out,
                    "omega_in": p.measured_omega_in,
                    "lambda_out": p.measured_lambda_out,
                    "lambda_in": p.measured_lambda_in,
                },
                "residual": p.residual,
                "alpha_error": p.alpha_error,
                "phase_status": p.phase_status,
            }
            for p in points
        ]
    }


@app.get("/api/runs/{run_id}/derivation")
async def get_derivation_steps(
    run_id: int,
    db_session: Session = Depends(get_db)
):
    """Get step-by-step derivation log."""
    steps = db_session.query(DerivationStep).filter(
        DerivationStep.run_id == run_id
    ).order_by(DerivationStep.step_number).all()
    
    return {
        "steps": [
            {
                "number": s.step_number,
                "parameter": s.parameter_name,
                "formula": s.formula_used,
                "inputs": s.inputs,
                "output": s.output_value,
                "details": s.intermediate_steps,
            }
            for s in steps
        ]
    }


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: int, db_session: Session = Depends(get_db)):
    """Delete a run and all associated data."""
    run = db_session.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    db_session.delete(run)
    db_session.commit()
    
    return {"message": f"Run {run_id} deleted"}


# ============== Derivation Endpoints ==============

@app.post("/api/derive")
async def derive_parameters(config: dict):
    """
    Derive all parameters from first principles.
    
    Returns theory-predicted values without running simulation.
    """
    chi_max = config.get("chi_max", 4)
    G_eff = config.get("G_eff", 6.674e-11)
    topology = config.get("topology", "path")
    N = config.get("N", 4)
    lambda_val = config.get("lambda", 0.5)
    
    # Compute scale anchoring
    scales = compute_all_scales(chi_max=chi_max)
    
    # Create coordinator for parameter derivation
    from .derivation.perturbative import compute_perturbative_parameters
    from .derivation.frustration import compute_hotspot_simple, estimate_frustration_timescale
    from .derivation.newtonian import compute_kappa_from_Geff, compute_target_degree
    
    # Derive perturbative parameters
    pert = compute_perturbative_parameters(
        lambda_val=lambda_val,
        omega=1.0,
        lambda_max=1.5,
        avg_frustration=0.5
    )
    
    # Derive hotspot and Newtonian
    hotspot = compute_hotspot_simple(lambda_val, chi_max)
    t_scale = estimate_frustration_timescale(lambda_val)
    kappa = compute_kappa_from_Geff(
        G_eff=G_eff,
        alpha=pert["alpha_pert"],
        c=1.0,
        a=scales["lattice_spacing"]
    )
    k0 = compute_target_degree(topology, N)
    
    return {
        "scales": scales,
        "perturbative": {
            "alpha": pert["alpha_pert"],
            "deltaB": pert["deltaB"],
            "validity_ratio": pert["validity_ratio"],
            "is_valid": pert["is_valid"]
        },
        "frustration": {
            "hotspot_multiplier": hotspot,
            "timescale": t_scale
        },
        "newtonian": {
            "kappa": kappa,
            "k0": k0
        },
        "summary": {
            "alpha_derived": pert["alpha_pert"],
            "lattice_spacing": scales["lattice_spacing"],
            "time_step": scales["time_step"]
        }
    }


@app.post("/api/agreement")
async def compute_agreement(data: dict):
    """
    Compute agreement between theory and measurement.
    
    Input: {"theory": [...], "measured": [...], "priority": "balanced"}
    Output: Agreement grade and metrics
    """
    import numpy as np
    
    theory = np.array(data.get("theory", []))
    measured = np.array(data.get("measured", []))
    priority_str = data.get("priority", "balanced")
    
    # Parse priority
    try:
        priority = PhysicsPriority(priority_str)
    except ValueError:
        priority = PhysicsPriority.BALANCED
    
    # Compute agreement
    calculator = AgreementCalculator(priority)
    agreement = calculator.compute_quantity_agreement("quantity", theory, measured)
    
    return {
        "grade": agreement.grade,
        "score": agreement.score,
        "is_acceptable": agreement.is_acceptable,
        "relative_difference": agreement.relative_difference,
        "color": GradeScale.get_color(agreement.grade),
        "metrics": {k: {
            "value": v.value,
            "normalized": v.normalized,
            "acceptable": v.is_acceptable
        } for k, v in agreement.metrics.items()}
    }


@app.get("/api/runs/{run_id}/agreement")
async def get_run_agreement(run_id: int, db_session: Session = Depends(get_db)):
    """Get agreement report for a completed run."""
    run = db_session.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    if run.agreement_score is None:
        return {"status": "pending", "message": "Run not yet completed"}
    
    return {
        "status": "completed",
        "overall_score": run.agreement_score,
        "grade": GradeScale.get_grade(run.agreement_score or 0),
        "color": GradeScale.get_color(GradeScale.get_grade(run.agreement_score or 0)),
        "critical_points": {
            "lambda_c1": run.lambda_c1,
            "lambda_revival": run.lambda_revival,
            "lambda_c2": run.lambda_c2
        }
    }


# ============== WebSocket Endpoint ==============

@app.websocket("/ws/runs/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: int):
    """WebSocket for real-time run updates."""
    await manager.connect(websocket, run_id)
    
    try:
        while True:
            # Receive commands from client
            data = await websocket.receive_json()
            
            if data.get("type") == "start_validation":
                # Start the validation run
                asyncio.create_task(
                    run_validation_task(run_id, data.get("config", {}))
                )
            
            elif data.get("type") == "pause":
                await set_run_status(run_id, "paused")
            
            elif data.get("type") == "resume":
                await set_run_status(run_id, "running")
            
            elif data.get("type") == "cancel":
                await set_run_status(run_id, "cancelled")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, run_id)


# ============== Background Tasks ==============

async def set_run_status(run_id: int, status: str):
    """Update run status and notify clients."""
    with db.session() as session:
        run = session.query(Run).filter(Run.id == run_id).first()
        if run:
            run.status = status
            session.commit()
    
    await manager.broadcast(run_id, {
        "type": "status_change",
        "run_id": run_id,
        "status": status,
        "timestamp": datetime.utcnow().isoformat()
    })


async def run_validation_task(run_id: int, config: dict):
    """Background task to run the validation with real physics."""
    import numpy as np
    
    await set_run_status(run_id, "deriving")
    
    # Get run configuration
    with db.session() as session:
        run = session.query(Run).filter(Run.id == run_id).first()
        N = run.N
        chi_max = run.chi_max
        G_eff = run.G_eff
        topology = run.topology
    
    # Step 1: Derive parameters from first principles
    scales = compute_all_scales(chi_max=chi_max)
    
    from .derivation.perturbative import compute_perturbative_parameters
    from .derivation.frustration import compute_hotspot_simple
    from .derivation.newtonian import compute_kappa_from_Geff
    
    # Store derived scales in run
    with db.session() as session:
        run = session.query(Run).filter(Run.id == run_id).first()
        run.lattice_spacing = scales["lattice_spacing"]
        run.time_step = scales["time_step"]
        session.commit()
    
    await manager.broadcast(run_id, {
        "type": "derivation_complete",
        "scales": scales
    })
    
    # Step 2: Lambda scan
    await set_run_status(run_id, "running")
    
    lambda_min = 0.1
    lambda_max = 1.5
    num_points = config.get("num_points", 30)
    lambda_vals = np.linspace(lambda_min, lambda_max, num_points)
    
    # Import physics engine
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from core_qca import ExactQCA
    
    comparisons = []
    
    for i, lambda_val in enumerate(lambda_vals):
        # Check control signals
        with db.session() as session:
            run = session.query(Run).filter(Run.id == run_id).first()
            if run.status == "cancelled":
                return
            if run.status == "paused":
                while run.status == "paused":
                    await asyncio.sleep(0.5)
                    session.refresh(run)
        
        # Derive parameters at this lambda
        pert = compute_perturbative_parameters(lambda_val, omega=1.0, lambda_max=lambda_max)
        alpha = pert["alpha_pert"]
        
        # Theory prediction for Lambda
        hotspot = compute_hotspot_simple(lambda_val, chi_max)
        chi_eff = min(chi_max, 1.0 + (hotspot * lambda_val) ** 2)
        Lambda_theory = np.full(N, 2 * np.log2(chi_eff))  # Simplified
        
        # Run simulation (simplified for now)
        try:
            qca = ExactQCA(
                N=N,
                config={"lambda": lambda_val, "alpha": alpha},
                bond_cutoff=chi_max
            )
            
            # Get ground state
            state = qca.get_ground_state()
            
            # Measure Lambda from state
            from .extraction.bubble_density import extract_lambda_measured
            lambda_result = extract_lambda_measured(state, qca)
            Lambda_measured = lambda_result["Lambda"]
            
            # Compute discrepancy
            abs_error = float(np.mean(np.abs(Lambda_measured - Lambda_theory)))
            rel_error = abs_error / (np.mean(np.abs(Lambda_theory)) + 1e-10)
            
            # Store lambda point
            with db.session() as session:
                point = LambdaPoint(
                    run_id=run_id,
                    point_index=i,
                    lambda_val=float(lambda_val),
                    theory_alpha=float(alpha),
                    measured_lambda_out=float(np.mean(Lambda_measured))
                )
                session.add(point)
                session.commit()
            
            comparisons.append({
                "lambda": float(lambda_val),
                "theory": float(np.mean(Lambda_theory)),
                "measured": float(np.mean(Lambda_measured)),
                "rel_error": float(rel_error)
            })
            
        except Exception as e:
            # Simulation failed at this point
            await manager.broadcast(run_id, {
                "type": "error",
                "message": f"Simulation failed at λ={lambda_val:.3f}: {e}"
            })
            continue
        
        # Progress update
        await manager.broadcast(run_id, {
            "type": "lambda_point_complete",
            "run_id": run_id,
            "point_index": i,
            "total": num_points,
            "lambda": float(lambda_val),
            "alpha": float(alpha),
            "residual": float(rel_error),
            "theory_mean": float(np.mean(Lambda_theory)),
            "measured_mean": float(np.mean(Lambda_measured)) if 'Lambda_measured' in locals() else 0.0
        })
    
    # Step 3: Compute agreement
    await set_run_status(run_id, "grading")
    
    if comparisons:
        theory_vals = np.array([c["theory"] for c in comparisons])
        measured_vals = np.array([c["measured"] for c in comparisons])
        
        calculator = AgreementCalculator(PhysicsPriority.BALANCED)
        agreement = calculator.compute_quantity_agreement("Lambda", theory_vals, measured_vals)
        
        # Update run with final score
        with db.session() as session:
            run = session.query(Run).filter(Run.id == run_id).first()
            run.agreement_score = agreement.score
            session.commit()
        
        await manager.broadcast(run_id, {
            "type": "agreement_complete",
            "grade": agreement.grade,
            "score": agreement.score,
            "color": GradeScale.get_color(agreement.grade)
        })
    
    await set_run_status(run_id, "completed")


# ============== Startup ==============

@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    db.connect()
    print(f"Database connected: {db.db_path}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
