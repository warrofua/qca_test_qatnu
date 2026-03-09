#!/usr/bin/env python3
"""
No-retuning holdout protocol.

Locks one parameterization and evaluates out-of-sample phase predictions
across pre-registered holdout scenarios with fixed pass/fail criteria.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# Allow running directly from scripts/ while importing repo modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase_analysis import PhaseAnalyzer
from scanners import ParameterScanner
from topologies import get_topology


DEFAULT_HOLDOUTS: List[Dict[str, Any]] = [
    {
        "id": "N5_path_alpha0.8",
        "N": 5,
        "graph": "path",
        "alpha": 0.8,
        "lambda_min": 0.1,
        "lambda_max": 1.5,
        "points": 40,
        "targets": {
            "lambda_c1": 0.232,
            "lambda_revival": 0.338,
            "lambda_c2": 1.033,
        },
    },
    {
        "id": "N4_cycle_alpha0.8",
        "N": 4,
        "graph": "cycle",
        "alpha": 0.8,
        "lambda_min": 0.1,
        "lambda_max": 1.5,
        "points": 80,
        "targets": {
            "lambda_c1": 0.261,
            "lambda_revival": 0.486,
            "lambda_c2": 1.000,
        },
    },
    {
        "id": "N4_star_alpha0.8",
        "N": 4,
        "graph": "star",
        "alpha": 0.8,
        "lambda_min": 0.1,
        "lambda_max": 1.5,
        "points": 80,
        "targets": {
            "lambda_c1": 0.164,
            "lambda_revival": 1.300,
            "lambda_c2": 1.400,
        },
    },
    {
        "id": "N4_path_alpha1.0",
        "N": 4,
        "graph": "path",
        "alpha": 1.0,
        "lambda_min": 0.1,
        "lambda_max": 1.5,
        "points": 80,
        "targets": None,
    },
]

ALLOWED_LOCK_KEYS = {
    "deltaB",
    "kappa",
    "hotspot_multiplier",
    "hotspot_time",
    "hotspot_edge_weights",
    "hotspot_stages",
    "k0",
    "bond_cutoff",
    "gamma_corr",
    "gamma_corr_diag",
    "readout_gamma_corr",
    "readout_gamma_corr_diag",
}


@dataclass
class HoldoutTolerances:
    c1: float
    revival: float
    c2: float
    residual: float


@dataclass
class RefinementConfig:
    enabled: bool
    max_refinements: int
    factor: float
    max_points: int


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _phase_order_ok(c1: Optional[float], revival: Optional[float], c2: Optional[float]) -> bool:
    if c1 is None or revival is None or c2 is None:
        return False
    return c1 < revival < c2


def _target_errors(
    crit: Dict[str, Any],
    targets: Dict[str, float],
) -> Dict[str, Optional[float]]:
    errors: Dict[str, Optional[float]] = {}
    for key in ("lambda_c1", "lambda_revival", "lambda_c2"):
        observed = _safe_float(crit.get(key))
        target = _safe_float(targets.get(key))
        if observed is None or target is None:
            errors[f"err_{key}"] = None
        else:
            errors[f"err_{key}"] = abs(observed - target)
    return errors


def _load_holdouts(
    holdout_file: Optional[Path],
    scenario_ids: Optional[Iterable[str]],
    max_points: Optional[int],
) -> List[Dict[str, Any]]:
    if holdout_file:
        payload = json.loads(holdout_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Holdout file must contain a JSON list of scenario objects.")
        holdouts = payload
    else:
        holdouts = copy.deepcopy(DEFAULT_HOLDOUTS)

    if scenario_ids:
        wanted = {s.strip() for s in scenario_ids if s.strip()}
        holdouts = [row for row in holdouts if row.get("id") in wanted]

    if max_points is not None:
        for row in holdouts:
            row["points"] = min(int(row.get("points", max_points)), max_points)

    if not holdouts:
        raise ValueError("No holdout scenarios selected.")
    return holdouts


def _parse_graph_overrides(raw: str) -> Dict[str, Dict[str, Any]]:
    if not raw.strip():
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("graph-overrides-json must decode to an object keyed by graph.")
    out: Dict[str, Dict[str, Any]] = {}
    for graph_name, overrides in payload.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"Overrides for graph '{graph_name}' must be an object.")
        clean: Dict[str, Any] = {}
        for key, value in overrides.items():
            if key not in ALLOWED_LOCK_KEYS:
                raise ValueError(f"Unsupported graph override key '{key}'.")
            if key == "hotspot_edge_weights" and value is not None and not isinstance(value, list):
                raise ValueError("hotspot_edge_weights overrides must be JSON lists.")
            if key == "hotspot_stages" and value is not None and not isinstance(value, list):
                raise ValueError("hotspot_stages overrides must be JSON lists.")
            clean[key] = value
        out[str(graph_name).lower()] = clean
    return out


def _parse_optional_json_list(raw: str, flag_name: str) -> Optional[List[float]]:
    if not raw.strip():
        return None
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"{flag_name} must decode to a JSON list.")
    return [float(item) for item in value]


def _parse_optional_hotspot_stages(raw: str, flag_name: str) -> Optional[List[Dict[str, Any]]]:
    if not raw.strip():
        return None
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"{flag_name} must decode to a JSON list of stage objects.")
    stages: List[Dict[str, Any]] = []
    for idx, stage in enumerate(value):
        if not isinstance(stage, dict):
            raise ValueError(f"{flag_name} stage {idx} must be an object.")
        clean = dict(stage)
        if "multiplier" in clean:
            clean["multiplier"] = float(clean["multiplier"])
        if "time" in clean:
            clean["time"] = float(clean["time"])
        if "edge_weights" in clean and clean["edge_weights"] is not None:
            if not isinstance(clean["edge_weights"], list):
                raise ValueError(f"{flag_name} stage {idx} edge_weights must be a list.")
            clean["edge_weights"] = [float(item) for item in clean["edge_weights"]]
        stages.append(clean)
    return stages


def _effective_locked_for_scenario(scenario: Dict[str, Any], locked: Dict[str, Any]) -> Dict[str, Any]:
    effective = dict(locked)
    graph_overrides = effective.pop("graph_overrides", {})
    if isinstance(graph_overrides, dict):
        effective.update(graph_overrides.get(str(scenario["graph"]).lower(), {}))
    return effective


def _run_scan(
    scanner: ParameterScanner,
    *,
    scenario: Dict[str, Any],
    locked: Dict[str, Any],
    output_dir: Path,
    quiet_scanner: bool,
    attempt_idx: int = 0,
) -> pd.DataFrame:
    topology = get_topology(str(scenario["graph"]), int(scenario["N"]))
    run_tag = (
        f"holdout_{scenario['id']}_p{int(scenario['points'])}_a{attempt_idx}_"
        f"dB{locked['deltaB']:.2f}_k{locked['kappa']:.3f}_hs{locked['hotspot_multiplier']:.2f}"
    )
    if abs(float(locked.get("hotspot_time", 1.0) - 1.0)) > 1e-12:
        run_tag += f"_ht{float(locked['hotspot_time']):.2f}"
    if locked.get("hotspot_edge_weights") is not None:
        run_tag += "_hew"
    if locked.get("hotspot_stages") is not None:
        run_tag += "_hstg"
    if abs(float(locked.get("gamma_corr", 0.0))) > 1e-12:
        run_tag += f"_gc{float(locked['gamma_corr']):+.2f}"
    if abs(float(locked.get("gamma_corr_diag", 0.0))) > 1e-12:
        run_tag += f"_gcd{float(locked['gamma_corr_diag']):+.2f}"
    if abs(float(locked.get("readout_gamma_corr", locked.get("gamma_corr", 0.0)))) > 1e-12:
        run_tag += f"_rgc{float(locked['readout_gamma_corr']):+.2f}"
    if abs(float(locked.get("readout_gamma_corr_diag", locked.get("gamma_corr_diag", 0.0)))) > 1e-12:
        run_tag += f"_rgd{float(locked['readout_gamma_corr_diag']):+.2f}"
    kwargs = dict(
        N=int(scenario["N"]),
        alpha=float(scenario["alpha"]),
        lambda_min=float(scenario["lambda_min"]),
        lambda_max=float(scenario["lambda_max"]),
        num_points=int(scenario["points"]),
        bond_cutoff=int(locked["bond_cutoff"]),
        output_dir=str(output_dir),
        run_tag=run_tag,
        edges=topology.edges,
        probes=topology.probes,
        hotspot_multiplier=float(locked["hotspot_multiplier"]),
        hotspot_time=float(locked.get("hotspot_time", 1.0)),
        hotspot_edge_weights=locked.get("hotspot_edge_weights"),
        hotspot_stages=locked.get("hotspot_stages"),
        deltaB=float(locked["deltaB"]),
        kappa=float(locked["kappa"]),
        k0=int(locked["k0"]),
        gamma_corr=float(locked.get("gamma_corr", 0.0)),
        gamma_corr_diag=float(locked.get("gamma_corr_diag", 0.0)),
        readout_gamma_corr=float(locked.get("readout_gamma_corr", locked.get("gamma_corr", 0.0))),
        readout_gamma_corr_diag=float(locked.get("readout_gamma_corr_diag", locked.get("gamma_corr_diag", 0.0))),
    )
    if quiet_scanner:
        with open("/dev/null", "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return scanner.scan_lambda_parallel(**kwargs)
    return scanner.scan_lambda_parallel(**kwargs)


def _evaluate_one(
    *,
    scanner: ParameterScanner,
    scenario: Dict[str, Any],
    locked: Dict[str, Any],
    tolerances: HoldoutTolerances,
    refinement: RefinementConfig,
    output_dir: Path,
    quiet_scanner: bool,
) -> Dict[str, Any]:
    effective_locked = _effective_locked_for_scenario(scenario, locked)
    initial_points = int(scenario["points"])
    points = initial_points
    attempts: List[Dict[str, Any]] = []

    for attempt_idx in range(max(1, refinement.max_refinements + 1)):
        scenario_attempt = dict(scenario)
        scenario_attempt["points"] = points

        df = _run_scan(
            scanner,
            scenario=scenario_attempt,
            locked=effective_locked,
            output_dir=output_dir,
            quiet_scanner=quiet_scanner,
            attempt_idx=attempt_idx,
        )
        crit = PhaseAnalyzer.analyze_critical_points(df)

        c1 = _safe_float(crit.get("lambda_c1"))
        revival = _safe_float(crit.get("lambda_revival"))
        c2 = _safe_float(crit.get("lambda_c2"))
        residual_min = _safe_float(crit.get("residual_min"))
        has_complete_triplet = c1 is not None and revival is not None and c2 is not None
        phase_order_ok = _phase_order_ok(c1, revival, c2)
        residual_ok = residual_min is not None and residual_min <= tolerances.residual

        attempts.append(
            {
                "attempt_idx": attempt_idx,
                "points": points,
                "lambda_c1": c1,
                "lambda_revival": revival,
                "lambda_c2": c2,
                "residual_min": residual_min,
                "complete_triplet": has_complete_triplet,
                "phase_order_ok": phase_order_ok,
                "residual_ok": residual_ok,
                "revival_method": crit.get("revival_method"),
                "revival_reporting_rule": crit.get("revival_reporting_rule"),
                "violation_detected": bool((df["residual"] > 0.10).any()),
                "mean_residual": float(df["residual"].mean()),
                "max_residual": float(df["residual"].max()),
                "min_residual": float(df["residual"].min()),
            }
        )

        needs_structural_refine = (
            (not has_complete_triplet) or (not phase_order_ok) or (residual_min is None)
        )
        can_refine = (
            refinement.enabled
            and attempt_idx < refinement.max_refinements
            and points < refinement.max_points
        )
        if not (needs_structural_refine and can_refine):
            break

        next_points = int(np.ceil(points * refinement.factor))
        if next_points <= points:
            next_points = points + 1
        points = min(next_points, refinement.max_points)

    final = attempts[-1]
    c1 = _safe_float(final["lambda_c1"])
    revival = _safe_float(final["lambda_revival"])
    c2 = _safe_float(final["lambda_c2"])
    residual_min = _safe_float(final["residual_min"])
    has_complete_triplet = bool(final["complete_triplet"])
    phase_order_ok = bool(final["phase_order_ok"])
    residual_ok = bool(final["residual_ok"])
    violation_detected = bool(final["violation_detected"])
    mean_residual = float(final["mean_residual"])
    max_residual = float(final["max_residual"])
    min_residual = float(final["min_residual"])

    target_errors = {
        "err_lambda_c1": None,
        "err_lambda_revival": None,
        "err_lambda_c2": None,
    }
    target_match_ok: Optional[bool] = None
    targets = scenario.get("targets")
    if isinstance(targets, dict):
        target_errors = _target_errors(crit, targets)
        target_match_ok = (
            target_errors["err_lambda_c1"] is not None
            and target_errors["err_lambda_revival"] is not None
            and target_errors["err_lambda_c2"] is not None
            and target_errors["err_lambda_c1"] <= tolerances.c1
            and target_errors["err_lambda_revival"] <= tolerances.revival
            and target_errors["err_lambda_c2"] <= tolerances.c2
        )

    scenario_pass = has_complete_triplet and phase_order_ok and residual_ok
    if target_match_ok is not None:
        scenario_pass = scenario_pass and target_match_ok

    resolution_ok = has_complete_triplet and phase_order_ok
    insufficient_resolution = (
        not resolution_ok
        and points >= refinement.max_points
        and refinement.enabled
        and violation_detected
    )
    no_violation_detected = not violation_detected
    refinement_steps = max(len(attempts) - 1, 0)

    return {
        "scenario_id": scenario["id"],
        "graph": scenario["graph"],
        "N": int(scenario["N"]),
        "alpha": float(scenario["alpha"]),
        "deltaB": float(effective_locked["deltaB"]),
        "kappa": float(effective_locked["kappa"]),
        "hotspot_multiplier": float(effective_locked["hotspot_multiplier"]),
        "hotspot_time": float(effective_locked.get("hotspot_time", 1.0)),
        "hotspot_edge_weights": json.dumps(effective_locked["hotspot_edge_weights"]) if effective_locked.get("hotspot_edge_weights") is not None else "",
        "hotspot_stages": json.dumps(effective_locked["hotspot_stages"]) if effective_locked.get("hotspot_stages") is not None else "",
        "k0": int(effective_locked["k0"]),
        "bond_cutoff": int(effective_locked["bond_cutoff"]),
        "gamma_corr": float(effective_locked.get("gamma_corr", 0.0)),
        "gamma_corr_diag": float(effective_locked.get("gamma_corr_diag", 0.0)),
        "readout_gamma_corr": float(effective_locked.get("readout_gamma_corr", effective_locked.get("gamma_corr", 0.0))),
        "readout_gamma_corr_diag": float(effective_locked.get("readout_gamma_corr_diag", effective_locked.get("gamma_corr_diag", 0.0))),
        "lambda_min": float(scenario["lambda_min"]),
        "lambda_max": float(scenario["lambda_max"]),
        "points": int(points),
        "points_initial": int(initial_points),
        "points_final": int(points),
        "refinement_steps": int(refinement_steps),
        "refinement_enabled": bool(refinement.enabled),
        "refinement_trace": "; ".join(f"{a['points']}pts" for a in attempts),
        "resolution_ok": bool(resolution_ok),
        "insufficient_resolution": bool(insufficient_resolution),
        "no_violation_detected": bool(no_violation_detected),
        "lambda_c1": c1,
        "lambda_revival": revival,
        "lambda_c2": c2,
        "residual_min": residual_min,
        "phase_order_ok": phase_order_ok,
        "complete_triplet": has_complete_triplet,
        "residual_ok": residual_ok,
        "target_match_ok": target_match_ok,
        "revival_method": final["revival_method"],
        "revival_reporting_rule": final["revival_reporting_rule"],
        "mean_residual": mean_residual,
        "max_residual": max_residual,
        "min_residual": min_residual,
        "scenario_pass": bool(scenario_pass),
        **target_errors,
    }


def _write_report(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    locked: Dict[str, Any],
    holdouts: List[Dict[str, Any]],
    tolerances: HoldoutTolerances,
    refinement: RefinementConfig,
) -> None:
    verdict = bool(summary["scenario_pass"].all())
    lines: List[str] = []
    lines.append("# No-Retuning Holdout Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Locked Parameterization")
    lines.append("")
    lines.append(f"- deltaB: {locked['deltaB']}")
    lines.append(f"- kappa: {locked['kappa']}")
    lines.append(f"- hotspot_multiplier: {locked['hotspot_multiplier']}")
    lines.append(f"- hotspot_time: {locked.get('hotspot_time', 1.0)}")
    lines.append(f"- hotspot_edge_weights: {locked.get('hotspot_edge_weights')}")
    lines.append(f"- hotspot_stages: {locked.get('hotspot_stages')}")
    lines.append(f"- k0: {locked['k0']}")
    lines.append(f"- bond_cutoff: {locked['bond_cutoff']}")
    lines.append(f"- gamma_corr: {locked.get('gamma_corr', 0.0)}")
    lines.append(f"- gamma_corr_diag: {locked.get('gamma_corr_diag', 0.0)}")
    lines.append(f"- readout_gamma_corr: {locked.get('readout_gamma_corr', locked.get('gamma_corr', 0.0))}")
    lines.append(f"- readout_gamma_corr_diag: {locked.get('readout_gamma_corr_diag', locked.get('gamma_corr_diag', 0.0))}")
    if locked.get("graph_overrides"):
        lines.append(f"- graph_overrides: {json.dumps(locked['graph_overrides'], sort_keys=True)}")
    lines.append("")
    lines.append("## Pass Criteria")
    lines.append("")
    lines.append("- Complete critical triplet required (`lambda_c1`, `lambda_revival`, `lambda_c2` non-null).")
    lines.append("- Phase ordering required: `lambda_c1 < lambda_revival < lambda_c2`.")
    lines.append(f"- Residual gate: `residual_min <= {tolerances.residual}`.")
    lines.append(
        f"- Targeted scenarios additionally require |Δc1|<= {tolerances.c1}, "
        f"|Δrev|<= {tolerances.revival}, |Δc2|<= {tolerances.c2}."
    )
    lines.append(
        "- Adaptive refinement: rerun failed structural detections "
        f"up to {refinement.max_refinements} additional attempts (factor={refinement.factor}, max_points={refinement.max_points})."
        if refinement.enabled
        else "- Adaptive refinement disabled."
    )
    lines.append("- `no_violation_detected=True` means residual never crossed 10%, so revival detection cannot activate.")
    lines.append("")
    lines.append("## Preregistered Holdouts")
    lines.append("")
    for row in holdouts:
        lines.append(
            f"- `{row['id']}`: N={row['N']}, graph={row['graph']}, alpha={row['alpha']}, "
            f"lambda_range=[{row['lambda_min']}, {row['lambda_max']}], points={row['points']}"
        )
        if isinstance(row.get("targets"), dict):
            tgt = row["targets"]
            lines.append(
                f"  targets: c1={tgt.get('lambda_c1')}, revival={tgt.get('lambda_revival')}, c2={tgt.get('lambda_c2')}"
            )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    display_cols = [
        "scenario_id",
        "scenario_pass",
        "lambda_c1",
        "lambda_revival",
        "lambda_c2",
        "residual_min",
        "phase_order_ok",
        "residual_ok",
        "target_match_ok",
        "points_initial",
        "points_final",
        "refinement_steps",
        "resolution_ok",
        "insufficient_resolution",
        "no_violation_detected",
        "err_lambda_c1",
        "err_lambda_revival",
        "err_lambda_c2",
    ]
    display_df = summary[display_cols].copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].map(
            lambda x: "NA" if x is None or (isinstance(x, float) and not np.isfinite(x)) else x
        )
    lines.append("```text")
    lines.append(display_df.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- OVERALL: {'PASS' if verdict else 'FAIL'}")
    lines.append("")
    (output_dir / "holdout_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_protocol(
    *,
    holdouts: List[Dict[str, Any]],
    locked: Dict[str, Any],
    tolerances: HoldoutTolerances,
    refinement: RefinementConfig,
    output_dir: Path,
    quiet_scanner: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scans_dir = output_dir / "scans"
    scans_dir.mkdir(parents=True, exist_ok=True)

    scanner = ParameterScanner()
    rows: List[Dict[str, Any]] = []

    for idx, scenario in enumerate(holdouts, start=1):
        print(
            f"[{idx}/{len(holdouts)}] {scenario['id']} "
            f"(N={scenario['N']}, graph={scenario['graph']}, alpha={scenario['alpha']}, points={scenario['points']})",
            flush=True,
        )
        row = _evaluate_one(
            scanner=scanner,
            scenario=scenario,
            locked=locked,
            tolerances=tolerances,
            refinement=refinement,
            output_dir=scans_dir,
            quiet_scanner=quiet_scanner,
        )
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("scenario_id")
    summary_path = output_dir / "holdout_summary.csv"
    summary.to_csv(summary_path, index=False)

    prereg_payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "locked_parameterization": locked,
        "tolerances": {
            "lambda_c1": tolerances.c1,
            "lambda_revival": tolerances.revival,
            "lambda_c2": tolerances.c2,
            "residual_min": tolerances.residual,
        },
        "refinement": {
            "enabled": refinement.enabled,
            "max_refinements": refinement.max_refinements,
            "factor": refinement.factor,
            "max_points": refinement.max_points,
        },
        "graph_overrides": locked.get("graph_overrides", {}),
        "holdouts": holdouts,
    }
    prereg_path = output_dir / "preregistered_protocol.json"
    prereg_path.write_text(json.dumps(prereg_payload, indent=2), encoding="utf-8")

    _write_report(
        output_dir=output_dir,
        summary=summary,
        locked=locked,
        holdouts=holdouts,
        tolerances=tolerances,
        refinement=refinement,
    )

    overall_pass = bool(summary["scenario_pass"].all())
    print("\nProtocol completed.")
    print(summary.to_string(index=False))
    print(f"\nOVERALL={'PASS' if overall_pass else 'FAIL'}")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"SUMMARY={summary_path}")
    print(f"PREREG={prereg_path}")
    print(f"REPORT={output_dir / 'holdout_report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-retuning holdout protocol for QATNU/SRQID")
    parser.add_argument("--holdout-file", type=str, default=None, help="Optional JSON file with scenario list")
    parser.add_argument(
        "--scenarios",
        type=str,
        default="",
        help="Comma-separated scenario ids to run (default: all preregistered)",
    )
    parser.add_argument("--max-points", type=int, default=None, help="Optional cap on points per scenario")
    parser.add_argument("--deltaB", type=float, default=6.5, help="Locked deltaB")
    parser.add_argument("--kappa", type=float, default=0.2, help="Locked kappa")
    parser.add_argument("--hotspot-multiplier", type=float, default=1.5, help="Locked hotspot multiplier")
    parser.add_argument("--hotspot-time", type=float, default=1.0, help="Locked hotspot preparation time")
    parser.add_argument(
        "--hotspot-edge-weights-json",
        type=str,
        default="",
        help="Optional global JSON list of hotspot edge weights.",
    )
    parser.add_argument(
        "--hotspot-stages-json",
        type=str,
        default="",
        help="Optional global JSON list of hotspot stage objects.",
    )
    parser.add_argument("--k0", type=int, default=4, help="Locked k0")
    parser.add_argument("--bond-cutoff", type=int, default=4, help="Locked chi cutoff")
    parser.add_argument("--gamma-corr", type=float, default=0.0, help="Global correlated-promotion amplitude")
    parser.add_argument("--gamma-corr-diag", type=float, default=0.0, help="Global correlated diagonal amplitude")
    parser.add_argument("--readout-gamma-corr", type=float, default=None, help="Optional readout-only correlated amplitude")
    parser.add_argument("--readout-gamma-corr-diag", type=float, default=None, help="Optional readout-only correlated diagonal amplitude")
    parser.add_argument(
        "--graph-overrides-json",
        type=str,
        default="",
        help="Optional JSON object keyed by graph name with per-graph locked-parameter overrides.",
    )
    parser.add_argument("--c1-tol", type=float, default=0.15, help="Tolerance for lambda_c1 target error")
    parser.add_argument("--rev-tol", type=float, default=0.30, help="Tolerance for lambda_revival target error")
    parser.add_argument("--c2-tol", type=float, default=0.30, help="Tolerance for lambda_c2 target error")
    parser.add_argument("--residual-threshold", type=float, default=0.20, help="Maximum allowed residual_min")
    parser.add_argument(
        "--max-refinements",
        type=int,
        default=2,
        help="Max additional refinement attempts per scenario",
    )
    parser.add_argument(
        "--refine-factor",
        type=float,
        default=2.0,
        help="Point multiplier between refinement attempts",
    )
    parser.add_argument(
        "--refine-max-points",
        type=int,
        default=120,
        help="Hard point cap during adaptive refinement",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable adaptive refinement and evaluate each scenario once.",
    )
    parser.add_argument("--output-dir", type=str, default="", help="Optional explicit output directory")
    parser.add_argument(
        "--quiet-scanner",
        action="store_true",
        help="Suppress scanner worker logs for cleaner output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario_ids = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    holdout_file = Path(args.holdout_file) if args.holdout_file else None
    holdouts = _load_holdouts(holdout_file, scenario_ids, args.max_points)

    locked = {
        "deltaB": float(args.deltaB),
        "kappa": float(args.kappa),
        "hotspot_multiplier": float(args.hotspot_multiplier),
        "hotspot_time": float(args.hotspot_time),
        "hotspot_edge_weights": _parse_optional_json_list(args.hotspot_edge_weights_json, "--hotspot-edge-weights-json"),
        "hotspot_stages": _parse_optional_hotspot_stages(args.hotspot_stages_json, "--hotspot-stages-json"),
        "k0": int(args.k0),
        "bond_cutoff": int(args.bond_cutoff),
        "gamma_corr": float(args.gamma_corr),
        "gamma_corr_diag": float(args.gamma_corr_diag),
        "readout_gamma_corr": float(args.readout_gamma_corr) if args.readout_gamma_corr is not None else float(args.gamma_corr),
        "readout_gamma_corr_diag": float(args.readout_gamma_corr_diag) if args.readout_gamma_corr_diag is not None else float(args.gamma_corr_diag),
        "graph_overrides": _parse_graph_overrides(args.graph_overrides_json),
    }
    tolerances = HoldoutTolerances(
        c1=float(args.c1_tol),
        revival=float(args.rev_tol),
        c2=float(args.c2_tol),
        residual=float(args.residual_threshold),
    )
    refinement = RefinementConfig(
        enabled=not bool(args.no_refine),
        max_refinements=max(0, int(args.max_refinements)),
        factor=max(1.01, float(args.refine_factor)),
        max_points=max(2, int(args.refine_max_points)),
    )

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path("outputs") / f"no_retuning_holdout_{ts}"

    run_protocol(
        holdouts=holdouts,
        locked=locked,
        tolerances=tolerances,
        refinement=refinement,
        output_dir=out_dir,
        quiet_scanner=args.quiet_scanner,
    )


if __name__ == "__main__":
    main()
