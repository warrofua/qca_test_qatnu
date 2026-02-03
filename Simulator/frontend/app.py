"""Textual frontend for QATNU First Principles Simulator."""

import asyncio
import json
from typing import Optional

import websockets
try:
    import aiohttp
except ImportError:
    aiohttp = None
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Header, Footer, Static, Button, Input, Select, 
    DataTable, ProgressBar, Label, Collapsible, RichLog, ContentSwitcher
)
from textual.binding import Binding


class QATNUSimulator(App):
    """Terminal-based first principles physics validator."""
    
    CSS = """
    Screen { align: center middle; }
    
    .title { 
        text-align: center; 
        text-style: bold underline; 
        color: cyan;
        margin: 1 0;
    }
    
    .subtitle {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }
    
    .panel {
        border: solid cyan;
        padding: 1;
    }
    
    .input-group {
        margin: 1 0;
    }
    
    .agreement { 
        text-align: center; 
        text-style: bold;
        padding: 1;
        border: solid green;
    }
    
    .agreement.good { border: solid green; color: green; }
    .agreement.warning { border: solid yellow; color: yellow; }
    .agreement.bad { border: solid red; color: red; }
    
    .status-bar {
        dock: bottom;
        height: 1;
        background: $surface-darken-1;
        color: $text;
        content-align: center middle;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh_runs", "Refresh"),
    ]
    
    # Reactive state
    current_run_id: reactive[Optional[int]] = reactive(None)
    connection_status: reactive[str] = reactive("disconnected")
    scan_progress: reactive[float] = reactive(0.0)
    current_lambda: reactive[float] = reactive(0.0)
    
    def __init__(self, api_url: str = "ws://localhost:8000"):
        super().__init__()
        self.api_url = api_url
        self.ws = None
        self.ws_task = None
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container():
            # Title
            yield Static("QATNU FIRST PRINCIPLES SIMULATOR", classes="title")
            yield Static("Validate theory through first-principles derivation", classes="subtitle")
            
            # Main layout: sidebar + content
            with Horizontal():
                # Left panel: Inputs
                with Vertical(classes="panel", id="input-panel"):
                    yield Label("Ground Truth Inputs", classes="title")
                    
                    with Container(classes="input-group"):
                        yield Label("Topology:")
                        yield Select(
                            [(t, t) for t in ["path", "cycle", "star", "diamond"]],
                            id="topology",
                            value="path"
                        )
                    
                    with Container(classes="input-group"):
                        yield Label("N (sites):")
                        yield Input(value="4", id="n-input")
                    
                    with Container(classes="input-group"):
                        yield Label("χ_max (bond cutoff):")
                        yield Input(value="4", id="chi-input")
                    
                    with Container(classes="input-group"):
                        yield Label("G_eff:")
                        yield Input(value="6.674e-11", id="g-input")
                    
                    yield Button("Validate Theory", id="validate-btn", variant="primary")
                    yield Button("Cancel", id="cancel-btn", variant="error", disabled=True)
                
                # Right panel: Results
                with Vertical(classes="panel", id="results-panel"):
                    yield Label("Results", classes="title")
                    
                    # Agreement display (hidden initially)
                    yield Static(id="agreement-display", classes="agreement")
                    
                    # Progress
                    with Container():
                        yield Label("Scan Progress:")
                        yield ProgressBar(id="scan-progress", total=100)
                        yield Static(id="progress-status", content="λ = 0.000")
                    
                    # Comparison table
                    yield DataTable(id="comparison-table")
                    
                    # Derivation log
                    with Collapsible(title="Derivation Steps", collapsed=True):
                        yield RichLog(id="derivation-log", markup=True)
            
            # Recent runs list
            with Container(id="runs-panel"):
                yield Label("Recent Runs", classes="title")
                yield DataTable(id="runs-table")
        
        # Status bar
        yield Static(id="status-bar", content="Disconnected", classes="status-bar")
        yield Footer()
    
    async def on_mount(self) -> None:
        """Initialize on mount."""
        # Setup comparison table
        table = self.query_one("#comparison-table", DataTable)
        table.add_columns("lambda", "alpha_theory", "alpha_meas", "Error", "Status")
        
        # Setup runs table
        runs_table = self.query_one("#runs-table", DataTable)
        runs_table.add_columns("ID", "Time", "N", "Topology", "χ_max", "Status", "Agreement")
        
        # Connect to backend
        asyncio.create_task(self.connect_websocket())
        
        # Load recent runs
        await self.load_recent_runs()
    
    async def connect_websocket(self):
        """Connect to backend WebSocket."""
        try:
            self.ws = await websockets.connect(f"{self.api_url}/ws/runs/0")
            self.ws_task = asyncio.create_task(self.listen_websocket())
            self.connection_status = "connected"
            self.update_status("Connected to backend")

            # Listen for messages
            # Listen handled by separate task

        except Exception as e:
            self.connection_status = "error"
            self.update_status(f"Connection failed: {e}")
    
    async def handle_ws_message(self, data: dict):
        """Handle incoming WebSocket messages."""
        msg_type = data.get("type")
        
        if msg_type == "derivation_complete":
            # Show derived parameters
            scales = data.get("scales", {})
            self.update_status(f"Derived: a={scales.get('lattice_spacing', 0):.3f}, "
                             f"τ={scales.get('time_step', 0):.3f}")
            
            # Log derivation
            log = self.query_one("#derivation-log", RichLog)
            log.write("[bold cyan]Parameter Derivation Complete[/]")
            log.write(f"  Lattice spacing: {scales.get('lattice_spacing', 0):.4f} ℓ_P")
            log.write(f"  Time step: {scales.get('time_step', 0):.4f} t_P")
            log.write(f"  UV energy: {scales.get('UV_energy', 0):.4f} E_P")
        
        elif msg_type == "lambda_point_complete":
            self.scan_progress = (data["point_index"] / data["total"]) * 100
            self.current_lambda = data["lambda"]
            
            # Update UI
            progress = self.query_one("#scan-progress", ProgressBar)
            progress.advance(1)
            
            status = self.query_one("#progress-status", Static)
            status.update(f"λ = {data['lambda']:.3f} | α = {data.get('alpha', 0):.4f} | "
                         f"Residual = {data.get('residual', 0):.4f}")
            
            # Add to table
            table = self.query_one("#comparison-table", DataTable)
            table.add_row(
                f"{data['lambda']:.3f}",
                f"{data.get('alpha', 0):.4f}",
                f"{data.get('measured_mean', 0):.4f}",
                f"{data.get('residual', 0):.4f}",
                "✓" if data.get('residual', 1.0) < 0.2 else "⚠"
            )
        
        elif msg_type == "agreement_complete":
            # Show final agreement grade
            grade = data.get("grade", "F")
            score = data.get("score", 0.0)
            color = data.get("color", "white")
            
            agreement_display = self.query_one("#agreement-display", Static)

            # Style based on grade
            agreement_display.remove_class("good")
            agreement_display.remove_class("warning")
            agreement_display.remove_class("bad")
            if score >= 0.9:
                agreement_display.add_class("good")
            elif score >= 0.7:
                agreement_display.add_class("warning")
            else:
                agreement_display.add_class("bad")

            agreement_display.update(f"GRADE {grade} | Score: {score:.1%}")
            self.update_status(f"Validation complete: Grade {grade}")
            
        elif msg_type == "status_change":
            status = data["status"]
            self.update_status(f"Run {data['run_id']}: {status}")
            
            if status == "completed":
                await self.load_recent_runs()
            elif status == "deriving":
                self.update_status("Deriving parameters from first principles...")
            elif status == "grading":
                self.update_status("Computing agreement scores...")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "validate-btn":
            await self.start_validation()
        elif event.button.id == "cancel-btn":
            await self.cancel_validation()
    
    async def start_validation(self):
        """Start a new validation run."""
        # Validate connection
        if self.connection_status != "connected":
            self.update_status("Error: Not connected to backend. Start the backend first.")
            return
        
        # Get inputs
        topology_select = self.query_one("#topology", Select)
        topology = topology_select.value
        if isinstance(topology, tuple):
            topology = topology[1]  # Get value from (label, value) tuple
        if topology is None:
            topology = "path"  # Default
            
        try:
            N = int(self.query_one("#n-input", Input).value)
            chi_max = int(self.query_one("#chi-input", Input).value)
            G_eff = float(self.query_one("#g-input", Input).value)
        except ValueError as e:
            self.update_status(f"Invalid input: {e}")
            return
        
        self.update_status("Creating run...")
        
        if aiohttp is None:
            self.update_status("Error: aiohttp not installed (pip install aiohttp)")
            return
        
        # Create run via HTTP API
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8000/api/runs",
                    json={
                        "topology": topology,
                        "N": N,
                        "chi_max": chi_max,
                        "G_eff": G_eff,
                    }
                ) as resp:
                    result = await resp.json()
                    self.current_run_id = result["id"]
                    
                    self.update_status(f"Run {self.current_run_id} created, starting validation...")
                    
                    # Connect to run-specific WebSocket
                    if self.ws:
                        await self.ws.close()
                        if self.ws_task:
                            self.ws_task.cancel()
                    self.ws = await websockets.connect(
                        f"{self.api_url}/ws/runs/{self.current_run_id}"
                    )
                    self.ws_task = asyncio.create_task(self.listen_websocket())
                    
                    # Send start command
                    await self.ws.send(json.dumps({
                        "type": "start_validation",
                        "config": {"num_points": 30}
                    }))
        
        except Exception as e:
            self.update_status(f"Error creating run: {e}")
            return
        
        # Update UI state
        self.query_one("#validate-btn", Button).disabled = True
        self.query_one("#cancel-btn", Button).disabled = False
        
        # Reset progress
        progress = self.query_one("#scan-progress", ProgressBar)
        progress.update(total=30, progress=0)
        
        # Clear table (keep columns)
        table = self.query_one("#comparison-table", DataTable)
        table.clear()
        
        # Clear agreement display
        agreement = self.query_one("#agreement-display", Static)
        agreement.update("")
        agreement.remove_class("good", "warning", "bad")
    
    async def listen_websocket(self):
        """Listen for WebSocket messages."""
        try:
            async for message in self.ws:
                try:
                    await self.handle_ws_message(json.loads(message))
                except json.JSONDecodeError:
                    pass  # Ignore invalid messages
        except websockets.exceptions.ConnectionClosed:
            self.connection_status = "disconnected"
            self.update_status("Connection closed")
        except asyncio.CancelledError:
            pass  # Task was cancelled
    
    async def cancel_validation(self):
        """Cancel current run."""
        if self.current_run_id and self.ws:
            try:
                await self.ws.send(json.dumps({
                    "type": "cancel",
                    "run_id": self.current_run_id
                }))
            except Exception as e:
                self.update_status(f"Cancel failed: {e}")
        
        self.query_one("#validate-btn", Button).disabled = False
        self.query_one("#cancel-btn", Button).disabled = True
        self.current_run_id = None
        self.update_status("Cancelled")
    
    async def load_recent_runs(self):
        """Load and display recent runs from API."""
        table = self.query_one("#runs-table", DataTable)
        table.clear()
        
        if aiohttp is None:
            table.add_row("-", "Install aiohttp", "-", "-", "-", "-", "-")
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8000/api/runs?limit=10") as resp:
                    data = await resp.json()
                    runs = data.get("runs", [])
                    
                    for run in runs:
                        agreement = f"{run.get('agreement_score', 0)*100:.1f}%" if run.get('agreement_score') else "-"
                        table.add_row(
                            str(run["id"]),
                            run["timestamp"][:16] if run.get("timestamp") else "-",
                            str(run["N"]),
                            run["topology"],
                            str(run["chi_max"]),
                            run["status"],
                            agreement
                        )
        except Exception as e:
            self.update_status(f"Failed to load runs: {e}")
    
    def update_status(self, message: str):
        """Update status bar."""
        status_bar = self.query_one("#status-bar", Static)
        status_bar.update(f"[{self.connection_status}] {message}")
    
    def action_refresh_runs(self):
        """Refresh runs list (keybinding: r)."""
        asyncio.create_task(self.load_recent_runs())
    
    async def on_unmount(self) -> None:
        """Cleanup on exit."""
        if self.ws_task:
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                pass
        if self.ws:
            await self.ws.close()


def main():
    """Entry point."""
    app = QATNUSimulator()
    app.run()


if __name__ == "__main__":
    main()
