"""
Parallel parameter scanning utilities for QATNU/SRQID.
"""
from __future__ import annotations

import os
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tqdm

from core_qca import ExactQCA


def _apply_hotspot_protocol(
    *,
    N: int,
    base_config: Dict,
    bond_cutoff: int,
    edges: List[Tuple[int, int]],
    ground_state: np.ndarray,
    hotspot_multiplier: float,
    hotspot_time: float,
    hotspot_edge_weights: Optional[Sequence[float]],
    hotspot_stages: Optional[Sequence[Dict]],
) -> np.ndarray:
    from copy import deepcopy

    if hotspot_stages:
        state = ground_state
        for stage in hotspot_stages:
            stage_config = deepcopy(base_config)
            stage_config["lambda"] = float(base_config["lambda"]) * float(stage.get("multiplier", hotspot_multiplier))
            stage_weights = stage.get("edge_weights")
            if stage_weights is not None:
                stage_config["lambda_edge_weights"] = list(stage_weights)
            qca_hotspot = ExactQCA(N, stage_config, bond_cutoff=bond_cutoff, edges=edges)
            state = qca_hotspot.evolve_state(state, float(stage.get("time", hotspot_time)))
        return state

    hotspot_config = deepcopy(base_config)
    hotspot_config["lambda"] = float(base_config["lambda"]) * hotspot_multiplier
    if hotspot_edge_weights is not None:
        hotspot_config["lambda_edge_weights"] = list(hotspot_edge_weights)
    qca_hotspot = ExactQCA(N, hotspot_config, bond_cutoff=bond_cutoff, edges=edges)
    return qca_hotspot.evolve_state(ground_state, hotspot_time)


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
            float,
            float,
            float,
            int,
            float,
            float,
            float,
            Optional[Tuple[float, ...]],
            Optional[Tuple[Tuple[Tuple[str, object], ...], ...]],
            float,
            float,
        ]
    ) -> Dict:
        import sys

        (
            N,
            alpha,
            lambda_param,
            tMax,
            bond_cutoff,
            edges,
            probes,
            embedded_clocks,
            hotspot_multiplier,
            deltaB,
            kappa,
            k0,
            gamma_corr,
            gamma_corr_diag,
            hotspot_time,
            hotspot_edge_weights,
            hotspot_stages,
            readout_gamma_corr,
            readout_gamma_corr_diag,
        ) = args

        print(
            f"🔄 Worker {os.getpid()} starting: N={N}, λ={lambda_param:.3f}",
            file=sys.stderr,
            flush=True,
        )

        config = {
            "omega": 1.0,
            "deltaB": deltaB,
            "lambda": lambda_param,
            "kappa": kappa,
            "k0": k0,
            "gamma_corr": gamma_corr,
            "gamma_corr_diag": gamma_corr_diag,
            "bondCutoff": bond_cutoff,
            "J0": 0.01,
            "gamma": 0.0,
            "probeOut": probes[0],
            "probeIn": probes[1],
            "edges": list(edges),
        }

        edge_list = list(edges)

        qca = ExactQCA(N, config, bond_cutoff=bond_cutoff, edges=edge_list)
        readout_config = dict(config)
        readout_config["gamma_corr"] = readout_gamma_corr
        readout_config["gamma_corr_diag"] = readout_gamma_corr_diag
        qca_readout = ExactQCA(N, readout_config, bond_cutoff=bond_cutoff, edges=edge_list)
        ground_state = qca.get_ground_state()
        stage_payload = None
        if hotspot_stages is not None:
            stage_payload = [dict(stage) for stage in hotspot_stages]
        frustrated_state = _apply_hotspot_protocol(
            N=N,
            base_config=config,
            bond_cutoff=bond_cutoff,
            edges=edge_list,
            ground_state=ground_state,
            hotspot_multiplier=hotspot_multiplier,
            hotspot_time=hotspot_time,
            hotspot_edge_weights=hotspot_edge_weights,
            hotspot_stages=stage_payload,
        )

        t_grid = np.linspace(0, tMax, 80)

        probe_sites = list(probes)
        source_state = frustrated_state if embedded_clocks else ground_state
        measured_freqs: List[float] = []

        for site in probe_sites:
            psi = qca_readout.apply_pi2_pulse(source_state.copy(), site)
            Z_signal = []

            for t in t_grid:
                psi_t = qca_readout.evolve_state(psi, t)
                Z_t = qca_readout.measure_Z(psi_t, site)
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
        
        # --- Low-lying spectrum diagnostics (first few levels) ---
        eig = qca_readout.diagonalize()
        eigs = eig["eigenvalues"]

        # Convert to NumPy array and take up to 6 lowest levels (E0..E5)
        E = np.asarray(eigs, dtype=float)
        max_levels = min(6, E.shape[0])
        E = E[:max_levels]

        # Individual energies (some may not exist for very tiny Hilbert spaces)
        E0 = float(E[0]) if max_levels > 0 else np.nan
        E1 = float(E[1]) if max_levels > 1 else np.nan
        E2 = float(E[2]) if max_levels > 2 else np.nan
        E3 = float(E[3]) if max_levels > 3 else np.nan
        E4 = float(E[4]) if max_levels > 4 else np.nan
        E5 = float(E[5]) if max_levels > 5 else np.nan

        # Nearest-neighbor gaps: [E1-E0, E2-E1, ..., E5-E4]
        gaps = np.diff(E) if max_levels > 1 else np.array([], dtype=float)

        gap01 = float(gaps[0]) if gaps.size > 0 else np.nan
        gap12 = float(gaps[1]) if gaps.size > 1 else np.nan
        gap23 = float(gaps[2]) if gaps.size > 2 else np.nan
        gap34 = float(gaps[3]) if gaps.size > 3 else np.nan
        gap45 = float(gaps[4]) if gaps.size > 4 else np.nan

        min_gap = float(gaps.min()) if gaps.size > 0 else np.nan
        
        return {
            "lambda": lambda_param,
            "residual": residual,
            "omega_out": measured_freqs[0],
            "omega_in": measured_freqs[1],
            "predicted_omega_out": predicted_out,
            "predicted_omega_in": predicted_in,
            "lambda_out": lambda_out,
            "lambda_in": lambda_in,
            "E0": E0,
            "E1": E1,
            "E2": E2,
            "E3": E3,
            "E4": E4,
            "E5": E5,
            "gap01": gap01,
            "gap12": gap12,
            "gap23": gap23,
            "gap34": gap34,
            "gap45": gap45,
            "min_gap": min_gap,
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
        hotspot_multiplier: float = 3.0,
        deltaB: float = 5.0,
        kappa: float = 0.1,
        k0: int = 4,
        gamma_corr: float = 0.0,
        gamma_corr_diag: float = 0.0,
        hotspot_time: float = 1.0,
        hotspot_edge_weights: Optional[Sequence[float]] = None,
        hotspot_stages: Optional[Sequence[Dict]] = None,
        readout_gamma_corr: float = 0.0,
        readout_gamma_corr_diag: float = 0.0,
    ) -> pd.DataFrame:
        print(f"\n🔬 Parallel λ scan: N={N}, α={alpha}, points={num_points}")
        print(f"Using {os.cpu_count()} cores")
        print(f"Hotspot multiplier: {hotspot_multiplier:.2f}×")
        print(f"Hotspot time: {hotspot_time:.2f}")
        if hotspot_edge_weights is not None:
            print(f"Hotspot edge weights: {list(hotspot_edge_weights)}")
        if hotspot_stages is not None:
            print(f"Hotspot stages: {list(hotspot_stages)}")
        print(f"Hamiltonian params: deltaB={deltaB}, kappa={kappa}, k0={k0}")
        if gamma_corr != 0.0 or gamma_corr_diag != 0.0:
            print(f"Correlated params: gamma_corr={gamma_corr}, gamma_corr_diag={gamma_corr_diag}")
        if readout_gamma_corr != gamma_corr or readout_gamma_corr_diag != gamma_corr_diag:
            print(
                "Readout correlated params: "
                f"gamma_corr={readout_gamma_corr}, gamma_corr_diag={readout_gamma_corr_diag}"
            )
        print(f"Environment: VECLIB_MAXIMUM_THREADS={os.environ.get('VECLIB_MAXIMUM_THREADS')}")

        if edges is None:
            edges = [(i, i + 1) for i in range(max(N - 1, 0))]
        if probes is None:
            probes = (0, 1 if N > 1 else 0)
        edges_tuple = tuple(tuple(edge) for edge in edges)
        
        # If we are in the default full-range scan, use the 3-part densified grid.
        # For very small num_points, fall back to uniform spacing to avoid empty segments.
        if (
            abs(lambda_min - 0.1) < 1e-9
            and abs(lambda_max - 1.5) < 1e-9
            and num_points >= 10
        ):
            lambda_vals = np.unique(
                np.concatenate(
                    [
                        np.linspace(lambda_min, 0.55, int(num_points * 0.35), endpoint=False),
                        np.linspace(0.55, 0.8, int(num_points * 0.45), endpoint=False),
                        np.linspace(0.8, lambda_max, int(num_points * 0.20)),
                    ]
                )
            )
        else:
            # Custom zoom window: respect the user’s range exactly
            lambda_vals = np.linspace(lambda_min, lambda_max, num_points)
        args_list = [
            (
                N,
                alpha,
                l,
                20.0,
                bond_cutoff,
                edges_tuple,
                probes,
                embedded_clocks,
                hotspot_multiplier,
                deltaB,
                kappa,
                k0,
                gamma_corr,
                gamma_corr_diag,
                hotspot_time,
                tuple(float(x) for x in hotspot_edge_weights) if hotspot_edge_weights is not None else None,
                tuple(tuple(stage.items()) for stage in hotspot_stages) if hotspot_stages is not None else None,
                readout_gamma_corr,
                readout_gamma_corr_diag,
            )
            for l in lambda_vals
        ]

        results: List[Dict] = []

        # Decide whether to use multiprocessing
        use_mp = True
        if platform.system() == "Darwin" and os.environ.get("QATNU_FORCE_MP") != "1":
            print(
                "⚠️ macOS detected: running λ scan serially to avoid NumPy longdouble "
                "issues with multiprocessing."
            )
            use_mp = False

        if use_mp:
            with ProcessPoolExecutor(max_workers=6) as executor:
                futures = {
                    executor.submit(ParameterScanner.run_single_point, args):args[2]
                    for args in args_list
                }

                with tqdm.tqdm(total=len(lambda_vals), desc="λ scan") as pbar:
                    for future in as_completed(futures):
                        lam_val = futures[future]
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as e:
                            print(f"  ⚠️ Failed at λ={lam_val:.3f}: {e}")
                        pbar.update(1)
        else:
            # Pure serial fallback – same work, no subprocesses
            with tqdm.tqdm(total=len(lambda_vals), desc="λ scan (serial)") as pbar:
                for args in args_list:
                    lam_val = args[2]
                    try:
                        result = ParameterScanner.run_single_point(args)
                        results.append(result)
                    except Exception as e:
                        print(f"  ⚠️ Failed at λ={lam_val:.3f}: {e}")
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
        hotspot_multiplier: float = 3.0,
        deltaB: float = 5.0,
        kappa: float = 0.1,
        k0: int = 4,
        gamma_corr: float = 0.0,
        gamma_corr_diag: float = 0.0,
        hotspot_time: float = 1.0,
        hotspot_edge_weights: Optional[Sequence[float]] = None,
        hotspot_stages: Optional[Sequence[Dict]] = None,
        readout_gamma_corr: float = 0.0,
        readout_gamma_corr_diag: float = 0.0,
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
            (
                N,
                a,
                l,
                15.0,
                bond_cutoff,
                edges_tuple,
                probes,
                embedded_clocks,
                hotspot_multiplier,
                deltaB,
                kappa,
                k0,
                gamma_corr,
                gamma_corr_diag,
                hotspot_time,
                tuple(float(x) for x in hotspot_edge_weights) if hotspot_edge_weights is not None else None,
                tuple(tuple(stage.items()) for stage in hotspot_stages) if hotspot_stages is not None else None,
                readout_gamma_corr,
                readout_gamma_corr_diag,
            )
            for a in alpha_vals
            for l in lambda_vals
        ]

        results = []

        # NEW: macOS serial fallback, mirroring scan_lambda_parallel
        use_mp = True
        if platform.system() == "Darwin" and os.environ.get("QATNU_FORCE_MP") != "1":
            print(
                "⚠️ macOS detected: running 2D scan serially to avoid multiprocessing issues."
            )
            use_mp = False

        if use_mp:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import tqdm

            with ProcessPoolExecutor(max_workers=6) as executor:
                futures = {
                    executor.submit(ParameterScanner.run_single_point, args): (args[1], args[2])
                    for args in args_list
                }

                with tqdm.tqdm(total=len(args_list), desc="2D scan") as pbar:
                    for future in as_completed(futures):
                        alpha, lam = futures[future]
                        try:
                            result = future.result()
                            result["alpha"] = alpha
                            results.append(result)
                        except Exception as e:
                            print(f"  ⚠️ Failed at α={alpha}, λ={lam:.3f}: {e}")
                        pbar.update(1)
        else:
            import tqdm

            with tqdm.tqdm(total=len(args_list), desc="2D scan (serial)") as pbar:
                for args in args_list:
                    alpha, lam = args[1], args[2]
                    try:
                        result = ParameterScanner.run_single_point(args)
                        result["alpha"] = alpha
                        results.append(result)
                    except Exception as e:
                        print(f"  ⚠️ Failed at α={alpha}, λ={lam:.3f}: {e}")
                    pbar.update(1)

        df = pd.DataFrame(results)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            tag = run_tag or f"N{N}"
            df.to_csv(os.path.join(output_dir, f"phase_space_{tag}.csv"), index=False)

        return df
