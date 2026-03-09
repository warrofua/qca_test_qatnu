"""Read-only research console for the notebook results."""

from __future__ import annotations

from typing import Dict, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Label, Select, Static

from topologies import get_topology

from .research_data import ResearchSnapshot, TableDataset, load_research_snapshot


class ResearchConsole(App):
    """Truth-first visualizer for scalar/topology results and tensor falsification."""

    CSS = """
    Screen {
        background: #06131a;
        color: #e8f1ef;
    }

    #root {
        layout: vertical;
    }

    #hero {
        height: auto;
        padding: 1 2;
        border-bottom: heavy #8fb996;
        background: #10242d;
    }

    .title {
        text-style: bold;
        color: #f4d35e;
    }

    .subtitle {
        color: #b8d8d8;
        margin-top: 1;
    }

    #body {
        height: 1fr;
    }

    #sidebar {
        width: 34;
        min-width: 34;
        border-right: heavy #8fb996;
        background: #0b1b22;
        padding: 1;
    }

    .metric {
        border: round #3fa7d6;
        margin-bottom: 1;
        padding: 0 1;
        height: auto;
    }

    .metric-key {
        color: #9db4c0;
    }

    .metric-value {
        color: #f4f1de;
        text-style: bold;
    }

    #main {
        padding: 1;
    }

    #view-select {
        width: 40;
        margin-bottom: 1;
    }

    .panel {
        border: round #8fb996;
        padding: 1;
        margin-bottom: 1;
        background: #10242d;
    }

    .panel-title {
        color: #f4d35e;
        text-style: bold;
        margin-bottom: 1;
    }

    DataTable {
        height: 1fr;
    }

    #status-pane {
        height: 11;
    }

    #content-row {
        height: 1fr;
    }

    #table-pane {
        width: 1.35fr;
    }

    #detail-column {
        width: 42;
        min-width: 42;
        margin-left: 1;
    }

    #graph-pane {
        height: 14;
    }

    #visual-pane {
        height: 14;
    }

    #detail-pane {
        height: 1fr;
    }

    #files-pane {
        height: 8;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("1", "show_scalar", "Scalar"),
        Binding("2", "show_tensor", "Tensor"),
        Binding("3", "show_critical", "Critical"),
        Binding("4", "show_redteam", "Redteam"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.snapshot: ResearchSnapshot = load_research_snapshot()
        self.current_view: str = "scalar"
        self.current_dataset: Optional[TableDataset] = None
        self.current_dataset_key: str = "holdout"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="root"):
            with Container(id="hero"):
                yield Static("QATNU Research Console", classes="title")
                yield Static(
                    self.snapshot.theory.one_line,
                    classes="subtitle",
                    id="hero-subtitle",
                )
            with Horizontal(id="body"):
                with Vertical(id="sidebar"):
                    yield Label("Truth Status", classes="panel-title")
                    for key, value in self.snapshot.headline_metrics.items():
                        yield Static(
                            f"[{key}] {value}",
                            classes="metric",
                        )
                    yield Label("Theory Verdict", classes="panel-title")
                    yield Static(self.snapshot.theory.thesis, classes="panel", id="thesis")
                with Vertical(id="main"):
                    yield Select(
                        [
                            ("Scalar / Transfer", "scalar"),
                            ("Tensor Falsification", "tensor"),
                            ("Critical Slowing", "critical"),
                            ("Redteam Checks", "redteam"),
                        ],
                        value="scalar",
                        id="view-select",
                    )
                    yield Static("", id="status-pane", classes="panel")
                    with Horizontal(id="content-row"):
                        yield DataTable(id="table-pane")
                        with Vertical(id="detail-column"):
                            yield Static("", id="graph-pane", classes="panel")
                            yield Static("", id="visual-pane", classes="panel")
                            yield Static("", id="detail-pane", classes="panel")
                    yield Static("", id="files-pane", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        self._render_view("scalar")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "view-select" and event.value is not None:
            self._render_view(str(event.value))

    def action_show_scalar(self) -> None:
        self.query_one("#view-select", Select).value = "scalar"
        self._render_view("scalar")

    def action_show_tensor(self) -> None:
        self.query_one("#view-select", Select).value = "tensor"
        self._render_view("tensor")

    def action_show_critical(self) -> None:
        self.query_one("#view-select", Select).value = "critical"
        self._render_view("critical")

    def action_show_redteam(self) -> None:
        self.query_one("#view-select", Select).value = "redteam"
        self._render_view("redteam")

    def _set_table(self, dataset: TableDataset) -> None:
        table = self.query_one("#table-pane", DataTable)
        table.clear(columns=True)
        table.add_columns(*dataset.columns)
        for row in dataset.rows:
            table.add_row(*row)
        if dataset.rows:
            table.move_cursor(row=0, column=0)

    def _render_view(self, view: str) -> None:
        status = self.query_one("#status-pane", Static)
        files = self.query_one("#files-pane", Static)

        if view == "scalar":
            dataset_key = "holdout"
            dataset = self.snapshot.tables[dataset_key]
            self._set_table(dataset)
            status.update(
                "[b]Scalar / topology core[/b]\n"
                "Established:\n- local unitary backbone\n- scalar clock renormalization\n- topology dependence\n\n"
                "Current read:\n- locked no-retuning transfer still fails\n- path can pass while star misses badly\n- topology is physics, not nuisance"
            )
        elif view == "tensor":
            dataset_key = "tensor_covariance"
            dataset = self.snapshot.tables[dataset_key]
            self._set_table(dataset)
            status.update(
                "[b]Tensor falsification track[/b]\n"
                "Best current read:\n"
                "- shell subtraction is a hard negative control but brittle\n"
                "- covariance subtraction is the most robust tested TT observable\n"
                "- no tested TT observable is background-robust, topology-robust, and finite-size-stable"
            )
        elif view == "critical":
            dataset_key = "critical_slowing"
            dataset = self.snapshot.tables[dataset_key]
            self._set_table(dataset)
            status.update(
                "[b]Critical-slowing diagnostics[/b]\n"
                "What matters here:\n"
                "- star peak remains protocol-sensitive\n"
                "- kappa, deltaB, and hotspot move the star window\n"
                "- backend parity is now good enough for these comparisons"
            )
        else:
            dataset_key = "harmonic_rank"
            dataset = self.snapshot.tables[dataset_key]
            self._set_table(dataset)
            status.update(
                "[b]Redteam checks[/b]\n"
                "Stress-tested:\n"
                "- source choice\n"
                "- permutation dependence\n"
                "- harmonic projector rank\n"
                "- fit fragility\n\n"
                "Outcome:\n- tensor verdict got weaker, scalar/topology verdict got stronger"
            )

        self.current_view = view
        self.current_dataset = dataset
        self.current_dataset_key = dataset_key
        files.update(
            "[b]Current dataset[/b]\n"
            f"{dataset.name}\n"
            f"Source: {dataset.source}\n\n"
            "[b]Canonical docs[/b]\n"
            + "\n".join(f"- {name}: {path}" for name, path in self.snapshot.file_index.items())
        )
        self._update_detail(0)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "table-pane":
            self._update_detail(event.cursor_row)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "table-pane":
            self._update_detail(event.cursor_row)

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        if event.data_table.id == "table-pane":
            self._update_detail(event.coordinate.row)

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        if event.data_table.id == "table-pane":
            self._update_detail(event.coordinate.row)

    def _update_detail(self, row_index: int) -> None:
        if self.current_dataset is None or not self.current_dataset.raw_rows:
            return
        row_index = max(0, min(row_index, len(self.current_dataset.raw_rows) - 1))
        raw_row = self.current_dataset.raw_rows[row_index]
        graph = self.query_one("#graph-pane", Static)
        visual = self.query_one("#visual-pane", Static)
        detail = self.query_one("#detail-pane", Static)
        graph.update(self._graph_panel_text(raw_row))
        visual.update(self._visual_panel_text(raw_row))
        detail.update(self._detail_panel_text(raw_row))

    def _graph_panel_text(self, raw_row: Dict[str, str]) -> str:
        topology, n_sites = self._resolve_topology(raw_row)
        title = f"[b]Topology[/b]\n{topology}, N={n_sites}\n\n"
        return title + self._ascii_topology(topology, n_sites)

    def _detail_panel_text(self, raw_row: Dict[str, str]) -> str:
        if self.current_dataset_key == "holdout":
            return (
                "[b]Selected holdout point[/b]\n"
                f"scenario: {raw_row.get('scenario_id', '-')}\n"
                f"lambda_c1: {raw_row.get('lambda_c1', '-')}\n"
                f"lambda_revival: {raw_row.get('lambda_revival', '-')}\n"
                f"lambda_c2: {raw_row.get('lambda_c2', '-')}\n"
                f"residual_min: {raw_row.get('residual_min', '-')}\n"
                f"scenario_pass: {raw_row.get('scenario_pass', '-')}"
            )
        if self.current_dataset_key == "tensor_covariance":
            return (
                "[b]Selected tensor point[/b]\n"
                f"scenario: {raw_row.get('scenario_id', '-')}\n"
                f"lambda: {raw_row.get('lambda', '-')}\n"
                f"raw_power: {raw_row.get('raw_power', '-')}\n"
                f"bg_power: {raw_row.get('bg_power', '-')}\n"
                f"covbg_power: {raw_row.get('covbg_power', '-')}\n"
                f"effective_backend: {raw_row.get('effective_backend', '-')}"
            )
        if self.current_dataset_key == "critical_slowing":
            return (
                "[b]Selected sensitivity slice[/b]\n"
                f"group: {raw_row.get('group', '-')}\n"
                f"hotspot: {raw_row.get('hotspot', '-')}\n"
                f"chi: {raw_row.get('chi', '-')}\n"
                f"kappa: {raw_row.get('kappa', '-')}\n"
                f"deltaB: {raw_row.get('deltaB', '-')}\n"
                f"peak_lambda: {raw_row.get('peak_lambda', '-')}\n"
                f"peak_tau: {raw_row.get('peak_tau_dephase_probe', '-')}"
            )
        return (
            "[b]Selected redteam slice[/b]\n"
            f"scenario: {raw_row.get('scenario_id', '-')}\n"
            f"keep_modes: {raw_row.get('keep_modes', '-')}\n"
            f"power_min: {raw_row.get('power_min', '-')}\n"
            f"power_max: {raw_row.get('power_max', '-')}\n"
            f"power_span: {raw_row.get('power_span', '-')}"
        )

    def _visual_panel_text(self, raw_row: Dict[str, str]) -> str:
        if self.current_dataset_key == "holdout":
            return self._holdout_visual(raw_row)
        if self.current_dataset_key == "tensor_covariance":
            return self._tensor_visual(raw_row)
        if self.current_dataset_key == "critical_slowing":
            return self._critical_visual(raw_row)
        return self._redteam_visual(raw_row)

    def _holdout_visual(self, raw_row: Dict[str, str]) -> str:
        c1 = self._safe_float(raw_row.get("lambda_c1"))
        revival = self._safe_float(raw_row.get("lambda_revival"))
        c2 = self._safe_float(raw_row.get("lambda_c2"))
        residual = self._safe_float(raw_row.get("residual_min"))
        scenario = raw_row.get("scenario_id", "-")
        graph = raw_row.get("graph", "-")
        pass_flag = raw_row.get("scenario_pass", "-")
        lines = [
            "[b]Phase landmarks[/b]",
            self._timeline(
                [("c1", c1), ("rev", revival), ("c2", c2)],
                limit=max(1.5, c2 * 1.1 if c2 is not None else 1.5),
            ),
            "",
            f"residual {self._bar(residual, max_value=0.40)} {self._fmt_num(residual)}",
            "",
            "[b]Transfer snapshot[/b]",
        ]
        if self.current_dataset:
            for row in self.current_dataset.raw_rows:
                lines.append(
                    f"{row.get('graph','-'):>5} "
                    f"{self._pass_chip(row.get('scenario_pass', '-'))} "
                    f"rev {self._fmt_num(self._safe_float(row.get('lambda_revival')))}"
                )
        lines.extend(["", f"selected: {scenario} ({graph}) {self._pass_chip(pass_flag)}"])
        return "\n".join(lines)

    def _tensor_visual(self, raw_row: Dict[str, str]) -> str:
        raw_power = self._safe_float(raw_row.get("raw_power"))
        bg_power = self._safe_float(raw_row.get("bg_power"))
        covbg_power = self._safe_float(raw_row.get("covbg_power"))
        topology = raw_row.get("topology", "-")
        lam = raw_row.get("lambda", "-")
        bond_cutoff = raw_row.get("bond_cutoff", "-")
        scenario = raw_row.get("scenario_id", "-")
        lines = [
            "[b]TT observable bars[/b]",
            f" raw  {self._signed_bar(raw_power, scale=2.5)} {self._fmt_num(raw_power)}",
            f" shell{self._signed_bar(bg_power, scale=2.5)} {self._fmt_num(bg_power)}",
            f" cov  {self._signed_bar(covbg_power, scale=2.5)} {self._fmt_num(covbg_power)}",
            "",
            "[b]Same-slice topology compare[/b]",
        ]
        if self.current_dataset:
            n_label = "N5" if "N5" in scenario else "N4"
            peers = [
                row for row in self.current_dataset.raw_rows
                if (("N5" in row.get("scenario_id", "")) == ("N5" in scenario))
                and row.get("lambda") == lam
                and row.get("bond_cutoff") == bond_cutoff
            ]
            for row in peers:
                topo = row.get("topology", "-")
                value = self._safe_float(row.get("covbg_power"))
                marker = ">" if topo == topology else " "
                lines.append(f"{marker}{topo:<5} {self._signed_bar(value, scale=2.5)} {self._fmt_num(value)}")
            lines.extend(["", f"slice: {n_label}, lambda={lam}, chi={bond_cutoff}"])
        return "\n".join(lines)

    def _critical_visual(self, raw_row: Dict[str, str]) -> str:
        peak_lambda = self._safe_float(raw_row.get("peak_lambda"))
        peak_tau = self._safe_float(raw_row.get("peak_tau_dephase_probe"))
        group = raw_row.get("group", "-")
        lines = [
            "[b]Peak location[/b]",
            self._timeline([("peak", peak_lambda)], limit=1.8),
            "",
            f"tau   {self._bar(peak_tau, max_value=8.0)} {self._fmt_num(peak_tau)}",
            "",
            "[b]Group compare[/b]",
        ]
        if self.current_dataset:
            peers = [row for row in self.current_dataset.raw_rows if row.get("group") == group]
            for row in peers[:6]:
                label = row.get("deltaB") or row.get("kappa") or row.get("hotspot") or row.get("chi") or "slice"
                value = self._safe_float(row.get("peak_lambda"))
                lines.append(f"{label:>5} {self._bar(value, max_value=1.8)} {self._fmt_num(value)}")
        return "\n".join(lines)

    def _redteam_visual(self, raw_row: Dict[str, str]) -> str:
        power_min = self._safe_float(raw_row.get("power_min"))
        power_max = self._safe_float(raw_row.get("power_max"))
        power_span = self._safe_float(raw_row.get("power_span"))
        scenario = raw_row.get("scenario_id", "-")
        lines = [
            "[b]Rank sensitivity[/b]",
            f"min  {self._signed_bar(power_min, scale=3.0)} {self._fmt_num(power_min)}",
            f"max  {self._signed_bar(power_max, scale=3.0)} {self._fmt_num(power_max)}",
            f"span {self._bar(power_span, max_value=3.0)} {self._fmt_num(power_span)}",
            "",
            "[b]Scenario compare[/b]",
        ]
        if self.current_dataset:
            peers = [row for row in self.current_dataset.raw_rows if row.get("scenario_id") == scenario]
            for row in peers[:6]:
                keep = row.get("keep_modes", "-")
                value = self._safe_float(row.get("power_span"))
                lines.append(f"m={keep:<3} {self._bar(value, max_value=3.0)} {self._fmt_num(value)}")
        return "\n".join(lines)

    def _resolve_topology(self, raw_row: Dict[str, str]) -> tuple[str, int]:
        topology = raw_row.get("topology") or raw_row.get("graph")
        scenario_id = raw_row.get("scenario_id", "")
        if not topology:
            if "_cycle_" in scenario_id:
                topology = "cycle"
            elif "_star_" in scenario_id or "_pyramid_" in scenario_id:
                topology = "star"
            elif "_diamond_" in scenario_id:
                topology = "diamond"
            else:
                topology = "path"
        n_raw = raw_row.get("N")
        n_sites = int(float(n_raw)) if n_raw else (5 if "N5" in scenario_id else 4)
        if self.current_dataset_key == "critical_slowing":
            topology = "star"
            n_sites = 5
        return topology, n_sites

    def _ascii_topology(self, topology: str, n_sites: int) -> str:
        try:
            topo = get_topology(topology, n_sites)
        except Exception:
            return f"edges unavailable\n{topology}, N={n_sites}"

        if topo.name == "path":
            return "  " + "──".join(str(i) for i in range(n_sites))
        if topo.name == "cycle":
            if n_sites == 4:
                return "\n".join(
                    [
                        "  0──1",
                        "  │  │",
                        "  3──2",
                    ]
                )
            return "cycle\n" + "edges: " + ", ".join(f"{a}-{b}" for a, b in topo.edges)
        if topo.name in {"star", "pyramid"}:
            lines = [f"    {leaf}" for leaf in range(1, n_sites)]
            if n_sites >= 4:
                return "\n".join(
                    [
                        f"    {1}",
                        "    │",
                        f"{2}──0──{3}" if n_sites >= 4 else "1──0",
                    ]
                    + ([f"    │\n    {4}"] if n_sites >= 5 else [])
                )
            return "star\n" + "edges: " + ", ".join(f"{a}-{b}" for a, b in topo.edges)
        if topo.name == "diamond":
            return "\n".join(
                [
                    "   0",
                    "  ╱ ╲",
                    " 3   1",
                    "  ╲ ╱",
                    "   2",
                ]
            )
        return "edges:\n" + "\n".join(f"{a}──{b}" for a, b in topo.edges)

    def _safe_float(self, value: Optional[str]) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _fmt_num(self, value: Optional[float]) -> str:
        if value is None:
            return "-"
        if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
            return f"{value:.2e}"
        return f"{value:.3f}"

    def _bar(self, value: Optional[float], *, max_value: float, width: int = 18) -> str:
        if value is None:
            return " " * width
        clamped = max(0.0, min(value, max_value))
        filled = int(round((clamped / max_value) * width))
        return "█" * filled + "·" * (width - filled)

    def _signed_bar(self, value: Optional[float], *, scale: float, width: int = 18) -> str:
        if value is None:
            return " " * width
        half = width // 2
        magnitude = min(abs(value) / scale, 1.0)
        filled = int(round(magnitude * half))
        if value >= 0:
            return "·" * half + "│" + "█" * filled + "·" * (half - filled)
        return "·" * (half - filled) + "█" * filled + "│" + "·" * half

    def _timeline(self, markers: list[tuple[str, Optional[float]]], *, limit: float, width: int = 26) -> str:
        axis = ["─"] * width
        labels = [" "] * width
        for name, value in markers:
            if value is None:
                continue
            pos = int(round(max(0.0, min(value / limit, 1.0)) * (width - 1)))
            axis[pos] = "●"
            for offset, char in enumerate(name[:3]):
                idx = min(width - 1, pos + offset)
                labels[idx] = char
        return f"0 {' '.join([])}{''.join(axis)} {limit:.1f}\n  {''.join(labels)}"

    def _pass_chip(self, value: str) -> str:
        lowered = str(value).lower()
        if lowered == "true":
            return "PASS"
        if lowered == "false":
            return "FAIL"
        return str(value)


def main() -> None:
    ResearchConsole().run()


if __name__ == "__main__":
    main()
