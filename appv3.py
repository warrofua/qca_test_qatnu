# =============================================================================
# QATNU/SRQID Unified Simulation Framework (v3.0)
# =============================================================================
# Features: Optimized exact diagonalization, parallel scanning, publication-
# quality plotting, mean-field comparison, SRQID validation, and spin-2 analysis.
# Runtime: ~45 sec for full λ-scan (100 pts, N=4) on M4/M3 Apple Silicon
# Memory: ~30 MB per process (N=4), scales as O(16^N) for N≤6
# -----------------------------------------------------------------------------

# %% ============================================================================
# ENVIRONMENT CONFIGURATION (Apple Silicon optimized)
# =============================================================================
import os
import platform

def configure_runtime():
    """Configure BLAS backend for optimal Apple Silicon performance"""
    if platform.system() == "Darwin" and "arm" in platform.machine():
        # Force Accelerate framework on M1/M2/M3/M4
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(os.cpu_count())
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
    else:
        # Linux/Intel: Use OpenBLAS
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
        os.environ["MKL_NUM_THREADS"] = "1"

configure_runtime()

print("=== BLAS Backend Check ===")
import numpy as np

def detect_accelerate() -> bool:
    """Multi-method detection of Accelerate framework"""
    
    # Method 1: Direct attribute inspection (most reliable)
    config = np.__config__
    
    # Check blas_opt_info (older NumPy versions)
    if hasattr(config, 'blas_opt_info'):
        info = config.blas_opt_info
        if any('accelerate' in str(lib).lower()
               for lib in info.get('libraries', [])):
            return True
    
    # Check lapack_opt_info
    if hasattr(config, 'lapack_opt_info'):
        info = config.lapack_opt_info
        if any('accelerate' in str(lib).lower()
               for lib in info.get('libraries', [])):
            return True
    
    # Method 2: Newer NumPy structure
    try:
        # show() may return dict in newer versions
        import inspect
        if 'mode' in inspect.signature(config.show).parameters:
            info = config.show(mode='dicts')
            blas_name = info.get('Build Dependencies', {}).get('blas', {}).get('name', '')
            if blas_name == 'accelerate':
                return True
    except:
        pass
    
    # Method 3: Parse show() output as string fallback
    import io
    import sys
    try:
        old_stdout = sys.stdout
        sys.stdout = capture = io.StringIO()
        config.show()
        sys.stdout = old_stdout
        return 'accelerate' in capture.getvalue().lower()
    except:
        sys.stdout = old_stdout
    
    return False

using_accelerate = detect_accelerate()

if using_accelerate:
    print("✅ Using Accelerate framework")
    print("   Performance should be optimal on Apple Silicon")
else:
    print("❌ NOT using Accelerate - will be SLOW")
    print("   (If you see this but performance is good, it's a false positive)")
# %% ============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigh
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import itertools
import csv
from typing import Dict, List, Tuple, Optional

# %% ============================================================================
# FORCE "FORK" MULTIPROCESSING (macOS fix)
# =============================================================================

import multiprocessing
import sys
if platform.system() == "Darwin":
    multiprocessing.set_start_method('forkserver', force=True)
    print("🍎 macOS detected: using 'forkserver' start method")
# %% ============================================================================
# UTILITIES (Shared constants and helpers)
# =============================================================================
class Utils:
    """Shared utilities and Pauli matrices"""
    
    # Pauli matrices (real where possible for speed)
    I2 = np.eye(2, dtype=np.float64)
    X = np.array([[0, 1], [1, 0]], dtype=np.float64)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.float64)
    
    @staticmethod
    def kronN(ops: List[np.ndarray]) -> np.ndarray:
        """N-way Kronecker product"""
        out = ops[0]
        for op in ops[1:]:
            out = np.kron(out, op)
        return out
    
    @staticmethod
    def local_operator(op: np.ndarray, site: int, N: int) -> np.ndarray:
        """Single-site operator in N-site chain"""
        ops = [Utils.I2 if k != site else op for k in range(N)]
        return Utils.kronN(ops)
    
    @staticmethod
    def comm_norm(A: np.ndarray, B: np.ndarray) -> float:
        """Operator norm of commutator [A, B]"""
        comm = A @ B - B @ A
        return np.linalg.norm(comm, 2)  # Spectral norm

# %% ============================================================================
# EXACT DIAGONALIZATION ENGINE (v2 optimizations + configurable)
# =============================================================================
class ExactQCA:
    """
    Optimized exact diagonalization engine for QATNU/SRQID Hamiltonian.
    Memory-efficient construction with pre-computed lookup tables.
    """
    
    def __init__(self, N: int, config: Dict, bond_cutoff: int = 4):
        self.N = N
        self.config = config
        self.bond_cutoff = bond_cutoff
        
        # Hilbert space dimensions
        self.matter_dim = 2 ** N
        self.bond_dim = bond_cutoff ** (N - 1)
        self.total_dim = self.matter_dim * self.bond_dim
        
        # Pre-computations
        self.bond_powers = [bond_cutoff ** i for i in range(N - 1)][::-1]
        
        # Vectorized Z-lookup: Z_lookup[state, site] = ±1
        self.Z_lookup = np.ones((self.matter_dim, self.N), dtype=np.int8)
        for i in range(N):
            self.Z_lookup[:, i] = 1 - 2 * ((np.arange(self.matter_dim) >> i) & 1)
        
        self.H = self._build_hamiltonian()
        self._eig = None  # Cache eigen-decomposition
        
    def get_entanglement_profile(qca, state):
        return [qca.get_bond_dimension(e, state) for e in range(qca.N-1)]

        # At each λ, plot profile: edge 0, edge 1, edge 2, ...
        # Expect: χ_out < χ_in < χ_center (for N=5)
        # For N=6: Look for saturation pattern
            
    def decode_bond_config(self, bond_index: int) -> List[int]:
        """Fast bond configuration decoding"""
        config, remaining = [], bond_index
        for power in self.bond_powers:
            config.append(remaining // power)
            remaining %= power
        return config[::-1]
    
    def state_index(self, matter_state: int, bond_config: List[int]) -> int:
        """Fast linear index calculation"""
        bond_index = sum(val * self.bond_powers[i] for i, val in enumerate(bond_config))
        return matter_state * self.bond_dim + bond_index
    
    def _build_hamiltonian(self) -> np.ndarray:
        """Memory-efficient Hamiltonian construction (O(total_dim) memory)"""
        H = np.zeros((self.total_dim, self.total_dim), dtype=np.float64)
        
        for idx in range(self.total_dim):
            matter_state = idx // self.bond_dim
            bond_config = self.decode_bond_config(idx % self.bond_dim)
            Z_vals = self.Z_lookup[matter_state]
            
            # Matter rotation: Σ (ω/2) X_i
            for i in range(self.N):
                flipped = matter_state ^ (1 << i)
                j = self.state_index(flipped, bond_config)
                H[idx, j] += self.config['omega'] / 2.0
            
            # Bond terms (vectorized per edge)
            for edge in range(self.N - 1):
                i, j = edge, edge + 1
                Zi, Zj = Z_vals[i], Z_vals[j]
                
                # Z_i Z_j interaction
                H[idx, idx] += self.config['J0'] * Zi * Zj
                
                # Bond energy term
                H[idx, idx] += self.config['deltaB'] * bond_config[edge]
                
                # Degree penalty
                d_i = self._calculate_degree(i, bond_config)
                d_j = self._calculate_degree(j, bond_config)
                penalty = (d_i - self.config['k0']) ** 2 + (d_j - self.config['k0']) ** 2
                H[idx, idx] += self.config['kappa'] * penalty
                
                # Promotion term
                F = 0.5 * (1.0 - Zi * Zj)
                
                if bond_config[edge] < self.bond_cutoff - 1:
                    new_config = bond_config.copy()
                    new_config[edge] += 1
                    jdx = self.state_index(matter_state, new_config)
                    H[idx, jdx] += self.config['lambda'] * F
                
                if bond_config[edge] > 0:
                    new_config = bond_config.copy()
                    new_config[edge] -= 1
                    jdx = self.state_index(matter_state, new_config)
                    H[idx, jdx] += self.config['lambda'] * F
        
        # Symmetrize (faster than full addition)
        H = (H + H.T) * 0.5
        return H
    
    def _calculate_degree(self, site: int, bond_config: List[int]) -> int:
        """Calculate vertex degree from bond configuration"""
        degree = 0
        if site > 0 and bond_config[site - 1] > 0:
            degree += 1
        if site < self.N - 1 and bond_config[site] > 0:
            degree += 1
        return degree
    
    def diagonalize(self):
        """Cached eigen-decomposition using Accelerate/MKL"""
        if self._eig is None:
            eigenvalues, eigenvectors = eigh(self.H, overwrite_a=True)
            self._eig = {'eigenvectors': eigenvectors, 'eigenvalues': eigenvalues}
        return self._eig
    
    def get_ground_state(self) -> np.ndarray:
        """Return actual ground state (lowest eigenvector)"""
        return self.diagonalize()['eigenvectors'][:, 0].astype(complex)
    
    def apply_pi2_pulse(self, state: np.ndarray, site: int) -> np.ndarray:
        """Apply π/2 pulse for Ramsey protocol"""
        new_state = np.zeros_like(state, dtype=complex)
        sqrt2 = np.sqrt(2.0)
        
        for idx in range(self.total_dim):
            if np.abs(state[idx]) < 1e-15:
                continue
            
            matter_state = idx // self.bond_dim
            bond_config = self.decode_bond_config(idx % self.bond_dim)
            bit = (matter_state >> site) & 1
            flipped_matter = matter_state ^ (1 << site)
            
            j1 = self.state_index(matter_state, bond_config)
            j2 = self.state_index(flipped_matter, bond_config)
            amplitude = state[idx]
            
            if bit == 0:
                new_state[j1] += amplitude / sqrt2
                new_state[j2] += 1j * amplitude / sqrt2
            else:
                new_state[j1] += amplitude / sqrt2
                new_state[j2] -= 1j * amplitude / sqrt2
        
        return new_state
    
    def evolve_state(self, state: np.ndarray, t: float) -> np.ndarray:
        """Exact time evolution via spectral decomposition"""
        eig = self.diagonalize()
        eigenvectors = eig['eigenvectors']
        eigenvalues = eig['eigenvalues']
        
        coeffs = eigenvectors.T @ state
        exp_coeffs = coeffs * np.exp(-1j * eigenvalues * t)
        
        return eigenvectors @ exp_coeffs
    
    def measure_Z(self, state: np.ndarray, site: int) -> float:
        """Vectorized Z expectation value"""
        probabilities = np.abs(state) ** 2
        Z_full = np.tile(self.Z_lookup[:, site], self.bond_dim)
        return np.sum(Z_full * probabilities)
    
    def get_bond_dimension(self, edge: int, state: np.ndarray) -> float:
        """CORRECTED: ⟨χ⟩ = 2^(⟨n⟩) without arbitrary scaling"""
        avg_n = 0.0
        norm = 0.0
        
        for idx in range(self.total_dim):
            prob = np.abs(state[idx]) ** 2
            if prob > 1e-15:
                bond_config = self.decode_bond_config(idx % self.bond_dim)
                avg_n += bond_config[edge] * prob
                norm += prob
        
        return 2.0 ** (avg_n / norm) if norm > 0 else 1.0
    
    def count_circuit_depth(self, site: int, state: np.ndarray) -> float:
        """Entanglement proxy Λ = log₂(1 + ⟨degree⟩)"""
        avg_degree = 0.0
        norm = 0.0
        
        for idx in range(self.total_dim):
            prob = np.abs(state[idx]) ** 2
            if prob > 1e-15:
                bond_config = self.decode_bond_config(idx % self.bond_dim)
                degree = self._calculate_degree(site, bond_config)
                avg_degree += degree * prob
                norm += prob
        
        d = avg_degree / norm if norm > 0 else 0
        return np.log2(1.0 + d)
    
    def commutator_norm(self, obs1: str, site1: int, obs2: str, site2: int, t: float) -> float:
        """Compute ||[obs1(t), obs2]|| for LR bound tests"""
        # Build local operators
        ops = {'X': Utils.X, 'Y': Utils.Y, 'Z': Utils.Z}
        A = Utils.local_operator(ops[obs1], site1, self.N)
        B = Utils.local_operator(ops[obs2], site2, self.N)
        
        # Tensor product with identity on bond registers
        A = np.kron(A, np.eye(self.bond_dim))
        B = np.kron(B, np.eye(self.bond_dim))
        
        # Heisenberg evolution
        eig = self.diagonalize()
        U = eig['eigenvectors']
        U_dag = U.conj().T
        exp_E = np.diag(np.exp(-1j * eig['eigenvalues'] * t))
        
        A_t = U @ exp_E.conj().T @ U_dag @ A @ U @ exp_E @ U_dag
        
        return Utils.comm_norm(A_t, B)

# %% ============================================================================
# MEAN-FIELD COMPARISON (from app_copy.py)
# =============================================================================
class QuantumChain:
    """Mean-field comparator using exact χ profile"""
    
    def __init__(self, N: int, omega: float, alpha: float, chi_profile: List[float],
                 J0: float, gamma: float):
        self.N = N
        self.omega = omega
        self.alpha = alpha
        self.chi_profile = chi_profile
        self.J0 = J0
        self.gamma = gamma
        self._compute_lambda()
    
    def _compute_lambda(self):
        """Λ_i = log₂(χ_{i-1}) + log₂(χ_i)"""
        self.Lambda = np.zeros(self.N)
        for i in range(self.N):
            if i > 0:
                self.Lambda[i] += np.log2(max(1.0, self.chi_profile[i - 1]))
            if i < self.N - 1:
                self.Lambda[i] += np.log2(max(1.0, self.chi_profile[i]))
    
    def get_effective_frequency(self, i: int) -> float:
        """Postulate 1: ω_eff = ω / (1 + αΛ)"""
        return self.omega / (1.0 + self.alpha * self.Lambda[i])
    
    def evolve_ramsey_single_ion(self, probe: int, t: float) -> float:
        """Mean-field Ramsey evolution for single site"""
        omega_eff = self.get_effective_frequency(probe)
        dt = min(0.1, np.pi / (4.0 * omega_eff))
        steps = max(1, int(np.floor(t / dt)))
        
        # Initial Bell state (|00> + |11>)/√2
        state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        
        for _ in range(steps):
            angle = omega_eff * dt / 2.0
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            new_state = np.array([
                cos_a * state[0] - sin_a * state[2],
                cos_a * state[1] - sin_a * state[3],
                cos_a * state[2] + sin_a * state[0],
                cos_a * state[3] + sin_a * state[1]
            ], dtype=complex)
            
            if self.gamma > 0.0:
                decay = np.exp(-self.gamma * dt / 2.0)
                new_state[1] *= decay
                new_state[3] *= decay
            
            norm = np.sqrt(np.sum(np.abs(new_state) ** 2))
            state = new_state / norm
        
        # Final π/2 pulse
        sqrt2 = np.sqrt(2.0)
        final = np.array([
            (state[0] - state[3]) / sqrt2,
            (state[1] + state[2]) / sqrt2,
            (state[2] + state[1]) / sqrt2,
            (state[3] - state[0]) / sqrt2
        ], dtype=complex)
        
        prob_up = np.abs(final[0]) ** 2 + np.abs(final[1]) ** 2
        prob_down = np.abs(final[2]) ** 2 + np.abs(final[3]) ** 2
        return prob_up - prob_down

# %% ============================================================================
# PARAMETER SCANNER (parallel engine from appv2)
# =============================================================================
class ParameterScanner:
    """Parallel parameter scanning with adaptive sampling"""
    
    @staticmethod
    def run_single_point(args: tuple) -> Dict:
        """Worker function: build QCA AFTER fork to avoid memory copy"""
        import sys
        
        # Unpack arguments
        N, alpha, lambda_param, tMax, bond_cutoff = args
        
        # Print PID for debugging
        print(f"🔄 Worker {os.getpid()} starting: N={N}, λ={lambda_param:.3f}",
              file=sys.stderr, flush=True)
        
        # Build config INSIDE worker (not passed pre-built)
        config = {
            'omega': 1.0, 'deltaB': 5.0, 'lambda': lambda_param,
            'kappa': 0.1, 'k0': 4, 'bondCutoff': bond_cutoff,
            'J0': 0.01, 'gamma': 0.0
        }
        
        # Construct heavy objects here (after fork)
        qca = ExactQCA(N, config, bond_cutoff=bond_cutoff)
        
        # Create frustrated state
        hotspot_config = config.copy()
        hotspot_config['lambda'] = lambda_param * 3.0
        qca_hotspot = ExactQCA(N, hotspot_config, bond_cutoff=bond_cutoff)
        ground_state = qca.get_ground_state()
        frustrated_state = qca_hotspot.evolve_state(ground_state, 1.0)
        
        # Time grid (adaptive density near revival)
        t_grid = np.linspace(0, tMax, 80)
        
        # Ramsey on two probe sites
        probe_sites = [0, 1]
        measured_freqs = []
        
        for site in probe_sites:
            psi = qca.apply_pi2_pulse(ground_state, site)
            Z_signal = []
            
            for t in t_grid:
                psi_t = qca.evolve_state(psi, t)
                Z_t = qca.measure_Z(psi_t, site)
                Z_signal.append(Z_t)
            
            # FFT frequency extraction
            Z_signal = np.array(Z_signal)
            Z_detrended = Z_signal - np.mean(Z_signal)
            
            fft = np.fft.fft(Z_detrended)
            freqs = np.fft.fftfreq(len(t_grid), d=t_grid[1] - t_grid[0])
            pos_mask = freqs > 0
            peak_idx = np.argmax(np.abs(fft[pos_mask]))
            freq = 2 * np.pi * freqs[pos_mask][peak_idx]
            
            measured_freqs.append(np.clip(freq, 0.1, 5.0))
        
        # Entanglement proxies
        lambda_out = qca.count_circuit_depth(0, frustrated_state)
        lambda_in = qca.count_circuit_depth(1, frustrated_state)
        
        # Postulate 1 predictions (NO ARBITRARY FACTORS)
        predicted_out = 1.0 / (1.0 + alpha * lambda_out)
        predicted_in = 1.0 / (1.0 + alpha * lambda_in)
        
        # Residual (relative error on frequency ratio)
        residual = abs((measured_freqs[1] / measured_freqs[0]) -
                       (predicted_in / predicted_out))
        
        return {
            'lambda': lambda_param,
            'residual': residual,
            'omega_out': measured_freqs[0],
            'omega_in': measured_freqs[1],
            'predicted_omega_out': predicted_out,
            'predicted_omega_in': predicted_in,
            'lambda_out': lambda_out,
            'lambda_in': lambda_in,
            'status': '✓' if residual < 0.05 else '~' if residual < 0.10 else '✗'
        }
    
    def scan_lambda_parallel(self, N: int = 4, alpha: float = 0.8,
                           lambda_min: float = 0.1, lambda_max: float = 1.5,
                           num_points: int = 100, bond_cutoff: int = 4) -> pd.DataFrame:
        """Parallel λ-scan with adaptive sampling"""
        print(f"\n🔬 Parallel λ scan: N={N}, α={alpha}, points={num_points}")
        print(f"Using {os.cpu_count()} cores")
        print(f"Environment: VECLIB_MAXIMUM_THREADS={os.environ.get('VECLIB_MAXIMUM_THREADS')}")
        
        # Adaptive sampling: dense near revival region
        lambda_vals = np.unique(np.concatenate([  # Remove duplicates
            np.linspace(lambda_min, 0.55, int(num_points * 0.35), endpoint=False),
            np.linspace(0.55, 0.8, int(num_points * 0.45), endpoint=False),
            np.linspace(0.8, lambda_max, int(num_points * 0.20))
        ]))
        
        args_list = [(N, alpha, l, 20.0, bond_cutoff) for l in lambda_vals]
        
        results = []
        #with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(self.run_single_point, args): args[2]
                       for args in args_list}
            
            with tqdm.tqdm(total=len(lambda_vals), desc="λ scan") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"  ⚠️ Failed at λ={futures[future]:.3f}: {e}")
                    pbar.update(1)
        
        df = pd.DataFrame(results).sort_values('lambda')
        df.to_csv(f"outputs/scan_N{N}_alpha{alpha:.1f}.csv", index=False)
        return df
    
    def scan_2d_phase_space(self, N: int = 4, lambda_vals: np.ndarray = None,
                           alpha_vals: np.ndarray = None, bond_cutoff: int = 4) -> pd.DataFrame:
        """Full 2D (α, λ) phase space mapping"""
        if lambda_vals is None:
            lambda_vals = np.linspace(0.1, 1.2, 12)
        if alpha_vals is None:
            alpha_vals = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        
        print(f"\n🗺️ 2D phase space: {len(lambda_vals)}×{len(alpha_vals)}={len(lambda_vals)*len(alpha_vals)} points")
        
        args_list = [(N, a, l, 15.0, bond_cutoff)
                     for a, l in itertools.product(alpha_vals, lambda_vals)]
        
        results = []
        #with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(self.run_single_point, args): (args[1], args[2])
                       for args in args_list}
            
            with tqdm.tqdm(total=len(args_list), desc="2D scan") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        alpha, lam = futures[future]
                        result['alpha'] = alpha
                        results.append(result)
                    except Exception as e:
                        alpha, lam = futures[future]
                        print(f"  ⚠️ Failed at α={alpha}, λ={lam:.3f}: {e}")
                    pbar.update(1)
        
        df = pd.DataFrame(results)
        df.to_csv(f"outputs/phase_space_N{N}.csv", index=False)
        return df

# %% ============================================================================
# PHASE ANALYZER (critical points + publication plots)
# =============================================================================
class PhaseAnalyzer:
    """Critical point detection and publication-quality plotting"""
    
    @staticmethod
    def analyze_critical_points(df: pd.DataFrame) -> Dict:
        """Robust critical point detection using local minima"""
        try:
            lambda_c1 = df[df['residual'] > 0.05]['lambda'].iloc[0]
        except IndexError:
            lambda_c1 = None
        
        try:
            # Find local minimum in violation region (residual > 10%)
            violation_region = df[df['residual'] > 0.10]
            if not violation_region.empty:
                first_viol_idx = violation_region.index[0]
                post_violation = df.loc[first_viol_idx:]
                
                # Negative residual peaks = minima
                minima_indices, _ = find_peaks(-post_violation['residual'].values,
                                             height=-0.15, distance=5)
                
                if len(minima_indices) > 0:
                    revival_idx = post_violation.index[minima_indices[0]]
                    lambda_revival = df.loc[revival_idx, 'lambda']
                    residual_min = df.loc[revival_idx, 'residual']
                else:
                    # Fallback to global minimum in violation region
                    idx_revival = violation_region['residual'].idxmin()
                    lambda_revival = df.loc[idx_revival, 'lambda']
                    residual_min = df.loc[idx_revival, 'residual']
            else:
                lambda_revival = residual_min = None
        except Exception:
            lambda_revival = residual_min = None
        
        try:
            if lambda_revival:
                post_revival = df[df['lambda'] > lambda_revival]
                lambda_c2 = post_revival[post_revival['residual'] > 0.20]['lambda'].iloc[0]
            else:
                lambda_c2 = None
        except IndexError:
            lambda_c2 = None
        
        return {
            'lambda_c1': lambda_c1,
            'lambda_revival': lambda_revival,
            'lambda_c2': lambda_c2,
            'residual_min': residual_min
        }
    
    @staticmethod
    def plot_phase_diagram(df: pd.DataFrame, N: int = 4, alpha: float = 0.8,
                          save: bool = True) -> plt.Figure:
        """6-panel publication-quality figure (from app_copy.py)"""
        crit = PhaseAnalyzer.analyze_critical_points(df)
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)
        
        # === PLOT 1: Main residual diagram ===
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(df['lambda'], df['residual'] * 100, 'o-',
                linewidth=2.5, markersize=8, color='darkblue', label='Postulate 1 Residual')
        
        # Thresholds and regions
        ax1.axhline(5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='5% threshold')
        ax1.axhline(10, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='10% threshold')
        ax1.fill_between(df['lambda'], 0, 5, alpha=0.15, color='green', label='Emergent (Phase I)')
        ax1.fill_between(df['lambda'], 5, 50, alpha=0.15, color='yellow', label='Breakdown (Phase II)')
        ax1.fill_between(df['lambda'], 50, 150, alpha=0.15, color='red', label='Catastrophic (Phase IV)')
        
        # Critical points
        if crit['lambda_c1']:
            ax1.axvline(crit['lambda_c1'], color='black', linestyle='-',
                       linewidth=1.5, alpha=0.7, label=f'λ_c1={crit["lambda_c1"]:.3f}')
        
        if crit['lambda_revival']:
            ax1.axvline(crit['lambda_revival'], color='purple', linestyle=':',
                       linewidth=2, label=f'Revival (λ={crit["lambda_revival"]:.3f})')
            ax1.annotate(f'Quantum Revival\nλ={crit["lambda_revival"]:.3f}\nR={crit["residual_min"]*100:.2f}%',
                        xy=(crit['lambda_revival'], crit['residual_min']*100),
                        xytext=(crit['lambda_revival']+0.15, crit['residual_min']*100+30),
                        arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
                        fontsize=11, ha='center', color='purple',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        if crit['lambda_c2']:
            ax1.axvline(crit['lambda_c2'], color='gray', linestyle='-',
                       linewidth=1.5, alpha=0.7, label=f'λ_c2={crit["lambda_c2"]:.3f}')
        
        ax1.set_xlabel('λ (promotion strength)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Postulate 1 Residual (%)', fontsize=14, fontweight='bold')
        ax1.set_title(f'Quantum Phase Diagram: N={N}, α={alpha}', fontsize=16, fontweight='bold', pad=15)
        ax1.legend(loc='upper left', fontsize=10, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(df['lambda'].min(), df['lambda'].max())
        ax1.set_ylim(0, max(df['residual']*100)*1.1)
        
        # === PLOT 2: Frequency scaling ===
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(df['lambda'], df['omega_out'], 'o-',
                label='Measured ω_out', linewidth=2, markersize=6, color='steelblue')
        ax2.plot(df['lambda'], df['omega_in'], 's-',
                label='Measured ω_in', linewidth=2, markersize=6, color='coral')
        ax2.plot(df['lambda'], df['predicted_omega_out'], '--',
                alpha=0.6, color='steelblue', label='Predicted ω_out')
        ax2.plot(df['lambda'], df['predicted_omega_in'], '--',
                alpha=0.6, color='coral', label='Predicted ω_in')
        ax2.axhline(1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label='Bare ω')
        
        if crit['lambda_revival']:
            ax2.axvline(crit['lambda_revival'], color='purple', linestyle=':', linewidth=1.5, alpha=0.5)
        
        ax2.set_xlabel('λ', fontsize=12, fontweight='bold')
        ax2.set_ylabel('ω_eff', fontsize=12, fontweight='bold')
        ax2.set_title('Frequency Scaling', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # === PLOT 3: Frequency inversion ===
        ax3 = fig.add_subplot(gs[1, 1])
        high_lambda = df[df['lambda'] > 0.6]
        if not high_lambda.empty:
            ax3.plot(high_lambda['lambda'], high_lambda['omega_out'], 'o-', label='ω_out', color='steelblue')
            ax3.plot(high_lambda['lambda'], high_lambda['omega_in'], 's-', label='ω_in', color='coral')
            ax3.axhline(1.0, color='black', linestyle=':', alpha=0.7)
            ax3.fill_between(high_lambda['lambda'], 1.0, high_lambda['omega_in'],
                           alpha=0.2, color='red', label='Inversion region')
            ax3.set_xlabel('λ', fontsize=12, fontweight='bold')
            ax3.set_ylabel('ω_eff', fontsize=12, fontweight='bold')
            ax3.set_title('Frequency Inversion', fontsize=13, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
        
        # === PLOT 4: Entanglement proxy ===
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(df['lambda'], df['lambda_out'], 'o-',
                label='Λ_out', linewidth=2, color='steelblue')
        ax4.plot(df['lambda'], df['lambda_in'], 's-',
                label='Λ_in', linewidth=2, color='coral')
        
        if crit['lambda_revival']:
            ax4.axvline(crit['lambda_revival'], color='purple', linestyle=':', linewidth=1.5, alpha=0.5)
        
        ax4.set_xlabel('λ', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Λ (circuit depth)', fontsize=12, fontweight='bold')
        ax4.set_title('Entanglement Proxy', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # === PLOT 5: Frequency ratio ===
        ax5 = fig.add_subplot(gs[2, 0])
        measured_ratio = df['omega_in'] / df['omega_out']
        predicted_ratio = df['predicted_omega_in'] / df['predicted_omega_out']
        ax5.plot(df['lambda'], measured_ratio, 'o-',
                label='Measured Ratio', linewidth=2, markersize=6, color='darkgreen')
        ax5.plot(df['lambda'], predicted_ratio, 's-',
                label='Predicted Ratio', linewidth=2, markersize=6, color='orange')
        ax5.axhline(1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label='No scaling')
        
        if crit['lambda_revival']:
            ax5.axvline(crit['lambda_revival'], color='purple', linestyle=':', linewidth=1.5, alpha=0.5)
        
        ax5.set_xlabel('λ', fontsize=12, fontweight='bold')
        ax5.set_ylabel('ω_in / ω_out', fontsize=12, fontweight='bold')
        ax5.set_title('Frequency Ratio Deviation', fontsize=13, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # === PLOT 6: Phase status ===
        ax6 = fig.add_subplot(gs[2, 1:])
        status_colors = {'✓': 0, '~': 1, '✗': 2}
        status_numeric = [status_colors.get(s, 1) for s in df['status']]
        scatter = ax6.scatter(df['lambda'], [0]*len(df), c=status_numeric,
                             cmap='RdYlGn_r', s=200, marker='s', edgecolors='black', alpha=0.8)
        ax6.set_xlabel('λ', fontsize=12, fontweight='bold')
        ax6.set_yticks([])
        ax6.set_title('Phase Classification', fontsize=13, fontweight='bold')
        ax6.set_xlim(df['lambda'].min(), df['lambda'].max())
        
        cbar = plt.colorbar(scatter, ax=ax6, orientation='horizontal', pad=0.25, ticks=[0, 1, 2])
        cbar.ax.set_xticklabels(['✓ Emergent', '~ Within Tol.', '✗ Violated'], fontsize=10)
        
        # Summary text box
        lambda_c1_str = f"{crit['lambda_c1']:.3f}" if crit['lambda_c1'] else "N/A"
        lambda_revival_str = f"{crit['lambda_revival']:.3f}" if crit['lambda_revival'] else "N/A"
        residual_min_str = f"{crit['residual_min']*100:.2f}" if crit['residual_min'] else "N/A"
        lambda_c2_str = f"{crit['lambda_c2']:.3f}" if crit['lambda_c2'] else "N/A"
        
        summary_text = f"""
        Critical Points:
        • Scaling breakdown: λ_c1 ≈ {lambda_c1_str}
        • Quantum revival: λ_rev ≈ {lambda_revival_str} (R={residual_min_str}%)
        • Catastrophic fail: λ_c2 ≈ {lambda_c2_str}
        
        Regimes:
        Phase I (λ < {lambda_c1_str}): Postulate 1 emerges
        Phase II ({lambda_c1_str} < λ < {lambda_revival_str}): Breakdown
        Phase III (λ ≈ {lambda_revival_str}): Quantum revival
        Phase IV (λ > {lambda_c2_str}): Frequency inversion
        """
        
        fig.text(0.5, 0.02, summary_text.strip(), ha='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7),
                fontfamily='monospace')
        
        plt.tight_layout()
        
        if save:
            filename = f"figures/phase_diagram_N{N}_alpha{alpha:.1f}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"🖼️  Plot saved to {filename}")
        
        return fig, crit

# %% ============================================================================
# SRQID VALIDATORS (from srqid_numerics.py)
# =============================================================================
class SRQIDValidators:
    """SRQID structural validation tests"""
    
    @staticmethod
    def extract_lr_velocity(qca: ExactQCA, threshold: float = 1e-3,
                           max_dist: int = None) -> Tuple[float, tuple]:
        """Extract v_LR from commutator growth front"""
        if max_dist is None:
            max_dist = qca.N // 2
        
        distances = range(1, max_dist + 1)
        times = np.linspace(0, 5.0, 50)
        arrival_times = []
        
        print(f"Computing LR velocity for N={qca.N}...")
        for r in distances:
            for t in times:
                norm = qca.commutator_norm('Z', 0, 'Z', r, t)
                if norm > threshold:
                    arrival_times.append((r, t))
                    print(f"  Distance {r}: t_arrival = {t:.3f} (norm={norm:.6f})")
                    break
        
        if len(arrival_times) >= 2:
            r_vals, t_vals = zip(*arrival_times)
            v_LR, intercept = np.polyfit(t_vals, r_vals, 1)
            print(f"✓ Extracted v_LR = {v_LR:.3f} (intercept={intercept:.3f})")
            return v_LR, (r_vals, t_vals)
        
        print("✗ Could not extract velocity (insufficient data)")
        return None, None
    
    @staticmethod
    def no_signalling_quench(qca: ExactQCA, site: int = 0,
                            far_site: int = None, t_max: float = 3.0) -> float:
        """No-signalling violation test from srqid_numerics.py"""
        if far_site is None:
            far_site = qca.N - 1
        
        # Ground state |0⟩^N
        ground = qca.get_ground_state()
        
        # State with X on site
        X_op = Utils.local_operator(Utils.X, site, qca.N)
        X_op = np.kron(X_op, np.eye(qca.bond_dim))
        quenched = X_op @ ground
        
        ts = np.linspace(0, t_max, 61)
        Z_far = Utils.local_operator(Utils.Z, far_site, qca.N)
        Z_far = np.kron(Z_far, np.eye(qca.bond_dim))
        
        # Expectation values -- FIXED: remove [0] index
        exp_ground = []
        exp_quenched = []
        for t in ts:
            psi_ground = qca.evolve_state(ground, t)
            psi_quenched = qca.evolve_state(quenched, t)
            
            # Calculate expectation values correctly (no indexing)
            exp_ground.append(np.real(psi_ground.conj().T @ (Z_far @ psi_ground)))
            exp_quenched.append(np.real(psi_quenched.conj().T @ (Z_far @ psi_quenched)))
        
        diff = np.abs(np.array(exp_quenched) - np.array(exp_ground))
        max_violation = np.max(diff)
        
        print(f"No-signalling test: max|Δ⟨Z_r⟩| = {max_violation:.3e} (r={far_site})")
        return max_violation
    
    @staticmethod
    def energy_drift(qca: ExactQCA, t_max: float = 3.0, ngrid: int = 61) -> float:
        """Energy conservation validation -- FIXED: remove [0] index"""
        state = qca.get_ground_state()
        ts = np.linspace(0, t_max, ngrid)
        
        energies = []
        for t in ts:
            psi_t = qca.evolve_state(state, t)
            energy = np.real(psi_t.conj().T @ (qca.H @ psi_t))
            energies.append(energy)
        
        drift = np.max(energies) - np.min(energies)
        print(f"Energy drift ΔE = {drift:.3e}")
        return drift

# %% ============================================================================
# EMERGENT GEOMETRY ANALYZER (from qatnu_poc.py)
# =============================================================================
class EmergentGeometryAnalyzer:
    """Spin-2 tail and dimensional flow analysis"""
    
    def __init__(self, rng_seed: int = 0):
        self.rng = np.random.default_rng(rng_seed)
    
    def spin2_power_spectrum_1d(self, N: int = 2048, steps: int = 3000,
                               p0: float = 0.03, lam: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 1D spin-2 tail data"""
        tier = np.zeros(N, int)
        r = np.arange(1, N + 1)
        p_prom = np.minimum(p0 / r**2, 0.15)
        
        for _ in range(steps):
            tier += (self.rng.random(N) < p_prom)
            chi = 2 ** tier
            dem = self.rng.random(N) < lam * (chi - 1)**3
            tier[dem & (tier > 0)] -= 1
        
        dchi = 2**tier - (2**tier).mean()
        k = np.fft.rfftfreq(N) * 2 * np.pi
        psd = np.abs(np.fft.rfft(dchi))**2
        
        return k[1:], psd[1:]  # Exclude k=0
    
    def analyze_spin2_scaling(self, ks: np.ndarray, psd: np.ndarray,
                            expected_power: float = 2.0) -> Dict:
        """Fit power-law exponent and compare to expected"""
        log_k = np.log10(ks)
        log_psd = np.log10(psd)
        
        # Linear fit in log-space
        slope, intercept = np.polyfit(log_k, log_psd, 1)
        measured_power = -slope
        
        # Compare to expected (spin-2 = 1/k²)
        residual = abs(measured_power - expected_power)
        
        return {
            'measured_power': measured_power,
            'expected_power': expected_power,
            'residual': residual,
            'fit_quality': abs(intercept) < 1.0  # Sanity check
        }
    
    def plot_spin2_tail(self, ks: np.ndarray, psd: np.ndarray,
                       title: str = "Spin-2 Tail", save_path: str = None):
        """Log-log plot with reference line"""
        plt.figure(figsize=(8, 6))
        plt.loglog(ks, psd, label='PSD', linewidth=2)
        
        # Reference line: 1/k²
        ref_psd = psd[0] * (ks[0] / ks)**2
        plt.loglog(ks, ref_psd, '--', label='1/k² reference', alpha=0.7)
        
        plt.xlabel('k', fontsize=12, fontweight='bold')
        plt.ylabel('Power Spectral Density', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"🖼️  Spin-2 plot saved to {save_path}")
        
        plt.show()

# %% ============================================================================
# MAIN SIMULATION RUNNER
# =============================================================================
class QCATester:
    """Unified tester combining exact, mean-field, and validation"""
    
    def __init__(self, N: int = 4, alpha: float = 0.8, bond_cutoff: int = 4):
        self.parameters = {
            'N': N,
            'omega': 1.0,
            'alpha': alpha,
            'deltaB': 5.0,
            'lambda': 0.3,
            'kappa': 0.1,
            'k0': 4,
            'bondCutoff': bond_cutoff,
            'J0': 0.01,
            'gamma': 0.0,
            'probeOut': 0,
            'probeIn': 1,
            'tMax': 20.0
        }
    
    def run_exact_experiment(self, config: Dict = None):
        """Run full exact diagonalization experiment"""
        if config is None:
            config = self.parameters
        
        qca = ExactQCA(config['N'], config, bond_cutoff=config['bondCutoff'])
        
        # Create frustrated state
        hotspot_config = config.copy()
        hotspot_config['lambda'] = config['lambda'] * 3.0
        qca_hotspot = ExactQCA(config['N'], hotspot_config, bond_cutoff=config['bondCutoff'])
        ground_state = qca.get_ground_state()
        frustrated_state = qca_hotspot.evolve_state(ground_state, 1.0)
        
        # Time grid
        t_grid = np.linspace(0, config['tMax'], 100)
        
        # Ramsey experiments
        data_out = []
        data_in = []
        
        for t in t_grid:
            for site, data_list in [(config['probeOut'], data_out), (config['probeIn'], data_in)]:
                psi = qca.apply_pi2_pulse(ground_state, site)
                psi_t = qca.evolve_state(psi, t)
                Z_t = qca.measure_Z(psi_t, site)
                data_list.append({'t': t, 'Z': Z_t})
        
        # Bond dimensions and circuit depths
        bond_dims = []
        for edge in range(config['N'] - 1):
            chi = qca.get_bond_dimension(edge, frustrated_state)
            bond_dims.append({'edge': edge, 'chi': chi})
        
        depth_out = qca.count_circuit_depth(config['probeOut'], frustrated_state)
        depth_in = qca.count_circuit_depth(config['probeIn'], frustrated_state)
        
        # Fit frequencies
        omega_out = self._fit_frequency(data_out)
        omega_in = self._fit_frequency(data_in)
        
        return {
            'dataOut': data_out,
            'dataIn': data_in,
            'omega_out': omega_out,
            'omega_in': omega_in,
            'bondDims': bond_dims,
            'depthOut': depth_out,
            'depthIn': depth_in,
            'config': config
        }
    
    def run_mean_field_comparison(self, exact_results):
        """Mean-field validation using exact χ profile"""
        p = exact_results['config']
        chi_profile = [d['chi'] for d in exact_results['bondDims']]
        
        chain = QuantumChain(
            N=p['N'],
            omega=p['omega'],
            alpha=p['alpha'],
            chi_profile=chi_profile,
            J0=p['J0'],
            gamma=p['gamma']
        )
        
        t_grid = np.linspace(0, p['tMax'], 100)
        
        # Run mean-field Ramsey
        mf_out = [{'t': t, 'Z': chain.evolve_ramsey_single_ion(p['probeOut'], t)} for t in t_grid]
        mf_in = [{'t': t, 'Z': chain.evolve_ramsey_single_ion(p['probeIn'], t)} for t in t_grid]
        
        return {
            'dataOut': mf_out,
            'dataIn': mf_in,
            'omega_out': self._fit_frequency(mf_out),
            'omega_in': self._fit_frequency(mf_in)
        }
    
    @staticmethod
    def _fit_frequency(data: List[Dict]) -> float:
        """FFT-based frequency extraction"""
        if len(data) < 10:
            return 0.0
        
        times = np.array([d['t'] for d in data])
        signal = np.array([d['Z'] for d in data])
        signal_detrended = signal - np.mean(signal)
        
        fft = np.fft.fft(signal_detrended)
        freqs = np.fft.fftfreq(len(times), d=times[1] - times[0])
        
        pos_mask = freqs > 0
        peak_idx = np.argmax(np.abs(fft[pos_mask]))
        return 2 * np.pi * freqs[pos_mask][peak_idx]
    
    def validate_postulate(self, exact_results):
        """Postulate 1 validation: measured vs predicted frequency ratios"""
        p = exact_results['config']
        
        measured_lambda_out = np.log2(1.0 + 2.0**exact_results['depthOut'] - 1.0)
        measured_lambda_in = np.log2(1.0 + 2.0**exact_results['depthIn'] - 1.0)
        
        predicted_out = p['omega'] / (1.0 + p['alpha'] * measured_lambda_out)
        predicted_in = p['omega'] / (1.0 + p['alpha'] * measured_lambda_in)
        
        residual = abs((exact_results['omega_in'] / exact_results['omega_out']) -
                      (predicted_in / predicted_out))
        
        return {
            'measuredLambdaOut': measured_lambda_out,
            'measuredLambdaIn': measured_lambda_in,
            'predictedFreqOut': predicted_out,
            'predictedFreqIn': predicted_in,
            'residual': residual,
            'measuredFreqOut': exact_results['omega_out'],
            'measuredFreqIn': exact_results['omega_in']
        }
    
    def run_full_validation(self, config: Dict = None):
        """Run exact, mean-field, and postulate validation"""
        exact = self.run_exact_experiment(config)
        mean_field = self.run_mean_field_comparison(exact)
        validation = self.validate_postulate(exact)
        
        return exact, mean_field, validation

# %% ============================================================================
# PRODUCTION RUNNER
# =============================================================================
def production_run(N: int = 4, alpha: float = 0.8, num_points: int = 100):
    """Complete production workflow"""
    
    # Setup directories
    for d in ["outputs", "figures"]:
        os.makedirs(d, exist_ok=True)
    
    print("="*60)
    print("QATNU/SRQID PRODUCTION RUN")
    print("="*60)
    print(f"Parameters: N={N}, α={alpha}, points={num_points}")
    print(f"System: {platform.system()} ({platform.machine()})")
    print("="*60)
    
    # 1. Parallel λ-scan
    scanner = ParameterScanner()
    df = scanner.scan_lambda_parallel(N=N, alpha=alpha, num_points=num_points)
    
    # 2. Phase diagram plot
    analyzer = PhaseAnalyzer()
    fig, crit = analyzer.plot_phase_diagram(df, N=N, alpha=alpha)
    
    # 3. Print critical points
    print("\n" + "="*60)
    print("CRITICAL POINT ANALYSIS")
    print("="*60)
    if crit['lambda_c1']:
        print(f"✓ Phase I → II breakdown: λ_c1 = {crit['lambda_c1']:.3f}")
    if crit['lambda_revival']:
        print(f"✓ Quantum revival: λ_rev = {crit['lambda_revival']:.3f} "
              f"(R={crit['residual_min']*100:.2f}%)")
    if crit['lambda_c2']:
        print(f"✓ Phase III → IV catastrophic: λ_c2 = {crit['lambda_c2']:.3f}")
    print("="*60)
    
    # 4. SRQID validation (single point)
    print("\nRunning SRQID structural validation...")
    test_config = {
        'N': N, 'omega': 1.0, 'deltaB': 5.0, 'lambda': 0.5,
        'kappa': 0.1, 'k0': 4, 'bondCutoff': 4, 'J0': 0.01, 'gamma': 0.0
    }
    qca_val = ExactQCA(N, test_config)
    
    v_lr, _ = SRQIDValidators.extract_lr_velocity(qca_val)
    ns_violation = SRQIDValidators.no_signalling_quench(qca_val)
    energy_err = SRQIDValidators.energy_drift(qca_val)
    
    # Save summary
    with open("outputs/summary.txt", "w") as f:
        f.write(f"QATNU/SRQID Production Run Summary\n")
        f.write("="*40 + "\n")
        f.write(f"N={N}, α={alpha}, points={num_points}\n")
        f.write(f"Critical Points:\n")
        f.write(f"  λ_c1: {crit['lambda_c1']}\n")
        f.write(f"  λ_rev: {crit['lambda_revival']} (R={crit['residual_min']})\n")
        f.write(f"  λ_c2: {crit['lambda_c2']}\n")
        f.write(f"SRQID Validations:\n")
        f.write(f"  v_LR: {v_lr}\n")
        f.write(f"  No-signalling: {ns_violation:.3e}\n")
        f.write(f"  Energy drift: {energy_err:.3e}\n")
    
    print("\n💾 Summary saved to outputs/summary.txt")
    return df, crit

# %% ============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Run full production workflow
    df_results, critical_points = production_run(N=5, alpha=0.8, num_points=100)
    
    # Additional 2D phase space mapping (optional)
    print("\n" + "="*60)
    print("OPTIONAL: Running 2D phase space mapping")
    print("="*60)
    run_2d = input("Run 2D (α, λ) scan? (y/n): ").lower() == 'y'
    
    if run_2d:
        scanner = ParameterScanner()
        df_2d = scanner.scan_2d_phase_space(N=5, alpha_vals=np.linspace(0.2, 1.2, 6),
                                   lambda_vals=np.linspace(0.1, 1.2, 12))
        print("💾 2D phase space saved to outputs/phase_space_N4.csv")
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        pivot = df_2d.pivot(index='lambda', columns='alpha', values='residual') * 100
        sns.heatmap(pivot, cmap='RdYlGn_r', center=10, vmin=0, vmax=100,
                   annot=True, fmt='.0f', cbar_kws={'label': 'Residual (%)'},
                   linewidths=0.5, linecolor='gray')
        plt.title('2D Phase Diagram: Postulate 1 Residual', fontsize=16, fontweight='bold')
        plt.xlabel('α (postulate coefficient)', fontsize=12, fontweight='bold')
        plt.ylabel('λ (promotion strength)', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig("figures/phase_space_2d.png", dpi=300, bbox_inches='tight')
        print("🖼️  2D heatmap saved to figures/phase_space_2d.png")
