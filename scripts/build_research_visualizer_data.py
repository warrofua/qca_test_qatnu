from __future__ import annotations

import csv
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / 'docs'
OUTPUTS_DIR = REPO_ROOT / 'outputs'
TARGET = REPO_ROOT / 'research_visualizer' / 'data' / 'current_state.json'


def read_csv(path: Path):
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def section_lines(text: str, heading: str) -> list[str]:
    lines = text.splitlines()
    capture = False
    depth = None
    out: list[str] = []
    for line in lines:
        if line.startswith('#'):
            stripped = line.lstrip('#').strip()
            if stripped == heading:
                capture = True
                depth = len(line) - len(line.lstrip('#'))
                continue
            if capture:
                current_depth = len(line) - len(line.lstrip('#'))
                if current_depth <= (depth or 0):
                    break
        if capture:
            out.append(line.rstrip())
    return out


def paragraph_after_heading(text: str, heading: str) -> str:
    parts: list[str] = []
    for line in section_lines(text, heading):
        stripped = line.strip()
        if not stripped:
            if parts:
                break
            continue
        parts.append(stripped)
    return ' '.join(parts)


def bullet_items(lines: Iterable[str]) -> list[str]:
    return [line.strip()[2:].strip() for line in lines if line.strip().startswith('- ')]


def load_theory() -> dict:
    text = (DOCS_DIR / 'theory_status_20260308.md').read_text(encoding='utf-8')
    return {
        'one_line': paragraph_after_heading(text, 'One-Line Status'),
        'thesis': paragraph_after_heading(text, 'Most Honest Current Thesis'),
        'established': bullet_items(section_lines(text, 'Established')),
        'indicated': bullet_items(section_lines(text, 'Indicated')),
        'not_supported': bullet_items(section_lines(text, 'Not Supported')),
    }


def main() -> None:
    payload = {
        'generated_at': datetime.now(UTC).isoformat(),
        'theory': load_theory(),
        'scalar': {
            'holdout': read_csv(OUTPUTS_DIR / 'no_retuning_holdout_20260225_locked_d5_k01_h3_pathstar16' / 'holdout_summary.csv'),
        },
        'tensor': {
            'covariance': read_csv(OUTPUTS_DIR / 'tt_background_covariance_anchor_comparison_20260308.csv'),
            'harmonic': read_csv(OUTPUTS_DIR / 'tt_background_harmonic_anchor_comparison_20260308.csv'),
        },
        'critical': {
            'star_sensitivity': read_csv(OUTPUTS_DIR / 'critical_slowing_next10_completion_20260225' / 'star_peak_summary.csv'),
        },
        'redteam': {
            'harmonic_rank': read_csv(OUTPUTS_DIR / 'harmonic_rank_dependence_20260308' / 'summary.csv'),
        },
        'sources': {
            'theory_status': str(DOCS_DIR / 'theory_status_20260308.md'),
            'results': str(DOCS_DIR / 'results.md'),
        },
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(TARGET)


if __name__ == '__main__':
    main()
