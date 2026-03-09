"""Data loading helpers for the read-only research console."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
OUTPUTS_DIR = REPO_ROOT / "outputs"


@dataclass
class TheoryStatus:
    one_line: str
    established: List[str]
    indicated: List[str]
    not_supported: List[str]
    thesis: str


@dataclass
class TableDataset:
    name: str
    source: str
    columns: List[str]
    rows: List[List[str]]
    raw_rows: List[Dict[str, str]]


@dataclass
class ResearchSnapshot:
    theory: TheoryStatus
    headline_metrics: Dict[str, str]
    tables: Dict[str, TableDataset]
    file_index: Dict[str, str]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _section_lines(text: str, heading: str) -> List[str]:
    lines = text.splitlines()
    capture = False
    depth = None
    out: List[str] = []
    target = heading.strip()
    for line in lines:
        if line.startswith("#"):
            stripped = line.lstrip("#").strip()
            if stripped == target:
                capture = True
                depth = len(line) - len(line.lstrip("#"))
                continue
            if capture:
                current_depth = len(line) - len(line.lstrip("#"))
                if current_depth <= (depth or 0):
                    break
        if capture:
            out.append(line.rstrip())
    return out


def _bullet_items(lines: Iterable[str]) -> List[str]:
    items: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items


def _paragraph_after_heading(text: str, heading: str) -> str:
    lines = _section_lines(text, heading)
    parts: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if parts:
                break
            continue
        if stripped.startswith("#"):
            break
        parts.append(stripped)
    return " ".join(parts)


def load_theory_status() -> TheoryStatus:
    path = DOCS_DIR / "theory_status_20260308.md"
    text = _read_text(path)
    one_line = _paragraph_after_heading(text, "One-Line Status")
    thesis = _paragraph_after_heading(text, "Most Honest Current Thesis")
    return TheoryStatus(
        one_line=one_line,
        established=_bullet_items(_section_lines(text, "Established")),
        indicated=_bullet_items(_section_lines(text, "Indicated")),
        not_supported=_bullet_items(_section_lines(text, "Not Supported")),
        thesis=thesis,
    )


def _format_float(value: str, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if abs(number) >= 1000 or (0 < abs(number) < 1e-3):
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _table_from_rows(
    name: str,
    source: Path,
    rows: List[Dict[str, str]],
    columns: List[str],
    formatters: Optional[Dict[str, int]] = None,
) -> TableDataset:
    formatters = formatters or {}
    table_rows: List[List[str]] = []
    for row in rows:
        rendered: List[str] = []
        for col in columns:
            value = row.get(col, "")
            if col in formatters:
                value = _format_float(value, formatters[col])
            rendered.append(value)
        table_rows.append(rendered)
    return TableDataset(
        name=name,
        source=str(source),
        columns=columns,
        rows=table_rows,
        raw_rows=rows,
    )


def load_tables() -> Dict[str, TableDataset]:
    tables: Dict[str, TableDataset] = {}

    tt_cov_path = OUTPUTS_DIR / "tt_background_covariance_anchor_comparison_20260308.csv"
    tt_cov_rows = _read_csv_rows(tt_cov_path)
    tables["tensor_covariance"] = _table_from_rows(
        "Tensor Falsification: Covariance Background",
        tt_cov_path,
        tt_cov_rows,
        [
            "scenario_id",
            "topology",
            "lambda",
            "bond_cutoff",
            "effective_backend",
            "raw_power",
            "bg_power",
            "covbg_power",
        ],
        {
            "lambda": 3,
            "raw_power": 3,
            "bg_power": 3,
            "covbg_power": 3,
        },
    )

    tt_harm_path = OUTPUTS_DIR / "tt_background_harmonic_anchor_comparison_20260308.csv"
    tt_harm_rows = _read_csv_rows(tt_harm_path)
    tables["tensor_harmonic"] = _table_from_rows(
        "Tensor Falsification: Harmonic Background",
        tt_harm_path,
        tt_harm_rows,
        [
            "scenario_id",
            "topology",
            "lambda",
            "bond_cutoff",
            "effective_backend",
            "harmbg_power",
            "harmbg_residual",
        ],
        {
            "lambda": 3,
            "harmbg_power": 3,
            "harmbg_residual": 3,
        },
    )

    holdout_path = OUTPUTS_DIR / "no_retuning_holdout_20260225_locked_d5_k01_h3_pathstar16" / "holdout_summary.csv"
    holdout_rows = _read_csv_rows(holdout_path)
    tables["holdout"] = _table_from_rows(
        "No-Retuning Holdout",
        holdout_path,
        holdout_rows,
        [
            "scenario_id",
            "graph",
            "lambda_c1",
            "lambda_revival",
            "lambda_c2",
            "residual_min",
            "scenario_pass",
        ],
        {
            "lambda_c1": 3,
            "lambda_revival": 3,
            "lambda_c2": 3,
            "residual_min": 3,
        },
    )

    critical_path = OUTPUTS_DIR / "critical_slowing_next10_completion_20260225" / "star_peak_summary.csv"
    critical_rows = _read_csv_rows(critical_path)
    tables["critical_slowing"] = _table_from_rows(
        "Critical Slowing: Star Sensitivity",
        critical_path,
        critical_rows,
        [
            "group",
            "hotspot",
            "chi",
            "kappa",
            "deltaB",
            "peak_lambda",
            "peak_tau_dephase_probe",
        ],
        {
            "hotspot": 2,
            "kappa": 2,
            "deltaB": 2,
            "peak_lambda": 3,
            "peak_tau_dephase_probe": 3,
        },
    )

    redteam_path = OUTPUTS_DIR / "harmonic_rank_dependence_20260308" / "summary.csv"
    redteam_rows = _read_csv_rows(redteam_path)
    tables["harmonic_rank"] = _table_from_rows(
        "Redteam: Harmonic Rank Sensitivity",
        redteam_path,
        redteam_rows,
        [
            "scenario_id",
            "keep_modes",
            "power_min",
            "power_max",
            "power_span",
        ],
        {
            "power_min": 3,
            "power_max": 3,
            "power_span": 3,
        },
    )

    return tables


def _count_rows(table: TableDataset) -> str:
    return str(len(table.rows))


def load_research_snapshot() -> ResearchSnapshot:
    theory = load_theory_status()
    tables = load_tables()
    file_index = {
        "Theory status": str(DOCS_DIR / "theory_status_20260308.md"),
        "Results snapshot": str(DOCS_DIR / "results.md"),
        "Consolidated paper": str(DOCS_DIR / "qatnu_consolidated_2026_rewrite.tex"),
        "Redteam log": str(DOCS_DIR / "decision_log_20260308_redteam.md"),
        "TT covariance comparison": str(OUTPUTS_DIR / "tt_background_covariance_anchor_comparison_20260308.csv"),
    }
    headline_metrics = {
        "Scalar core": "Supported",
        "Tensor headline": "Not supported",
        "Topology transfer": "Fails",
        "Tensor controls": _count_rows(tables["tensor_covariance"]),
        "Critical-slowing scans": _count_rows(tables["critical_slowing"]),
    }
    return ResearchSnapshot(
        theory=theory,
        headline_metrics=headline_metrics,
        tables=tables,
        file_index=file_index,
    )
