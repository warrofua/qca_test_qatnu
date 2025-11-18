"""
Parallel parameter scanning utilities for QATNU/SRQID.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tqdm

from core_qca import ExactQCA


class ParameterScanner:
    """Parallel parameter scanning with adaptive sampling."""

    @staticmethod
    def run_single_point(
        args: Tuple[
            int,
            float,
            float,
            float,
            int,
            Tuple[Tuple[int, int], ...],
            Tuple[int, int],
            bool,
        ]
    ) -> Dict:
        import sys
        from copy import deepcopy

        N, alpha, lambda_param, tMax, bond_cutoff, edges, probes, embedded_clocks = args

        print(
            f"🔄 Worker {os.getpid()} starting: N={N}, λ={lambda_param:.3f}",
            file=sys.stderr,
            flush=True,
        )

        config = {
            "omega": 1.0,
            "deltaB": 5.0,
            "lambda": lambda_param,
            "kappa": 0.1,
            "k0": 4,
            "bondCutoff": bond_cutoff,
            "J0": 0.01,
            "gamma": 0.0,
            "probeOut": probes[0],
            "probeIn": probes[1],
            "edges": list(edges),
        }

        edge_list = list(edges)

        qca = ExactQCA(N, config, bond_cutoff=bond_cutoff, edges=edge_list)
        hotspot_config = deepcopy(config)
        hotspot_config["lambda"] = lambda_param * 3.0
        qca_hotspot = ExactQCA(N, hotspot_config, bond_cutoff=bond_cutoff, edges=edge_list)
        ground_state = qca.get_ground_state()
        frustrated_state = qca_hotspot.evolve_state(ground_state, 1.0)

        t_grid = np.linspace(0, tMax, 80)

        probe_sites = list(probes)
        source_state = frustrated_state if embedded_clocks else ground_state
        measured_freqs: List[float] = []

        for site in probe_sites:
            psi = qca.apply_pi2_pulse(source_state.copy(), site)
            Z_signal = []

            for t in t_grid:
                psi_t = qca.evolve_state(psi, t)
                Z_t = qca.measure_Z(psi_t, site)
                Z_signal.append(Z_t)

            Z_signal = np.array(Z_signal)
            Z_detrended = Z_signal - np.mean(Z_signal)

            fft = np.fft.fft(Z_detrended)
            freqs = np.fft.fftfreq(len(t_grid), d=t_grid[1] - t_grid[0])
            pos_mask = freqs > 0
            peak_idx = np.argmax(np.abs(fft[pos_mask]))
            freq = 2 * np.pi * freqs[pos_mask][peak_idx]

            measured_freqs.append(np.clip(freq, 0.1, 5.0))

        lambda_out = qca.count_circuit_depth(probes[0], frustrated_state)
        lambda_in = qca.count_circuit_depth(probes[1], frustrated_state)

        predicted_out = 1.0 / (1.0 + alpha * lambda_out)
        predicted_in = 1.0 / (1.0 + alpha * lambda_in)

        residual = abs(
            (measured_freqs[1] / measured_freqs[0]) - (predicted_in / predicted_out)
        )

        return {
            "lambda": lambda_param,
            "residual": residual,
            "omega_out": measured_freqs[0],
            "omega_in": measured_freqs[1],
            "predicted_omega_out": predicted_out,
            "predicted_omega_in": predicted_in,
            "lambda_out": lambda_out,
            "lambda_in": lambda_in,
            "status": "✓" if residual < 0.05 else "~" if residual < 0.10 else "✗",
        }

    def scan_lambda_parallel(
        self,
        N: int = 4,
        alpha: float = 0.8,
        lambda_min: float = 0.1,
        lambda_max: float = 1.5,
        num_points: int = 100,
        bond_cutoff: int = 4,
        output_dir: Optional[str] = None,
        run_tag: Optional[str] = None,
        edges: Optional[List[Tuple[int, int]]] = None,
        probes: Optional[Tuple[int, int]] = None,
        embedded_clocks: bool = False,
    ) -> pd.DataFrame:
        print(f"\n🔬 Parallel λ scan: N={N}, α={alpha}, points={num_points}")
        print(f"Using {os.cpu_count()} cores")
        print(f"Environment: VECLIB_MAXIMUM_THREADS={os.environ.get('VECLIB_MAXIMUM_THREADS')}")

        if edges is None:
            edges = [(i, i + 1) for i in range(max(N - 1, 0))]
        if probes is None:
            probes = (0, 1 if N > 1 else 0)
        edges_tuple = tuple(tuple(edge) for edge in edges)

        lambda_vals = np.unique(
            np.concatenate(
                [
                    np.linspace(lambda_min, 0.55, int(num_points * 0.35), endpoint=False),
                    np.linspace(0.55, 0.8, int(num_points * 0.45), endpoint=False),
                    np.linspace(0.8, lambda_max, int(num_points * 0.20)),
                ]
            )
        )

        args_list = [
            (N, alpha, l, 20.0, bond_cutoff, edges_tuple, probes, embedded_clocks)
            for l in lambda_vals
        ]

        results = []
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(self.run_single_point, args): args[2] for args in args_list}

            with tqdm.tqdm(total=len(lambda_vals), desc="λ scan") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"  ⚠️ Failed at λ={futures[future]:.3f}: {e}")
                    pbar.update(1)

        df = pd.DataFrame(results).sort_values("lambda")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            tag = run_tag or f"N{N}_alpha{alpha:.1f}"
            df.to_csv(os.path.join(output_dir, f"scan_{tag}.csv"), index=False)

        return df

    def scan_2d_phase_space(
        self,
        N: int = 4,
        lambda_vals: Optional[np.ndarray] = None,
        alpha_vals: Optional[np.ndarray] = None,
        bond_cutoff: int = 4,
        output_dir: Optional[str] = None,
        run_tag: Optional[str] = None,
        edges: Optional[List[Tuple[int, int]]] = None,
        probes: Optional[Tuple[int, int]] = None,
        embedded_clocks: bool = False,
    ) -> pd.DataFrame:
        if lambda_vals is None:
            lambda_vals = np.linspace(0.1, 1.2, 12)
        if alpha_vals is None:
            alpha_vals = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        if edges is None:
            edges = [(i, i + 1) for i in range(max(N - 1, 0))]
        if probes is None:
            probes = (0, 1 if N > 1 else 0)
        edges_tuple = tuple(tuple(edge) for edge in edges)

        print(f"\n🗺️ 2D phase space: {len(lambda_vals)}×{len(alpha_vals)}={len(lambda_vals)*len(alpha_vals)} points")

        args_list = [
            (N, a, l, 15.0, bond_cutoff, edges_tuple, probes, embedded_clocks)
            for a in alpha_vals
            for l in lambda_vals
        ]

        results = []
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(self.run_single_point, args): (args[1], args[2]) for args in args_list
            }

            with tqdm.tqdm(total=len(args_list), desc="2D scan") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        alpha, lam = futures[future]
                        result["alpha"] = alpha
                        results.append(result)
                    except Exception as e:
                        alpha, lam = futures[future]
                        print(f"  ⚠️ Failed at α={alpha}, λ={lam:.3f}: {e}")
                    pbar.update(1)

        df = pd.DataFrame(results)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            tag = run_tag or f"N{N}"
            df.to_csv(os.path.join(output_dir, f"phase_space_{tag}.csv"), index=False)
        return df
