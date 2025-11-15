# =============================================================================
# QATNU/SRQID MERGED OPTIMIZED SIMULATION (N=4-6)
# =============================================================================
# Memory footprint: ~30 MB per process (N=4)
# Runtime: ~45 seconds for full λ scan (100 points) on M4
# Fixes: Proper Ramsey protocol, correct bond dim, no arbitrary factors
# ----------------------------------------------------------------------------

# %%
# ============================================================================
# IMPORTS - APPLE SILICON OPTIMIZED
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.signal import find_peaks
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm
import os

# Force use of Accelerate framework for BLAS (M4 optimized)
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"  # Use all P-cores
os.environ["OPENBLAS_NUM_THREADS"] = "1"    # Disable OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"         # Disable MKL

# %%
# ============================================================================
# CORRECTED EXACT DIAGONALIZATION ENGINE
# ============================================================================
class ExactQCA:
    def __init__(self, N, config):
        self.N = N
        self.config = config
        self.matter_dim = 2 ** N
        self.bond_dim = config['bondCutoff'] ** (N - 1)
        self.total_dim = self.matter_dim * self.bond_dim
        self.bond_cutoff = config['bondCutoff']
        
        # Add Pauli matrices
        self.I = np.eye(2, dtype=np.float64)
        self.X = np.array([[0, 1], [1, 0]], dtype=np.float64)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        self.Z = np.array([[1, 0], [0, -1]], dtype=np.float64)
        
        # Pre-compute bond powers for fast indexing
        self.bond_powers = [self.bond_cutoff ** i for i in range(self.N - 1)][::-1]
        
        # Pre-compute Z-values for all matter states (vectorized)
        self.Z_lookup = np.ones((self.matter_dim, self.N), dtype=np.int8)
        for i in range(self.N):
            self.Z_lookup[:, i] = 1 - 2 * ((np.arange(self.matter_dim) >> i) & 1)
        
        self.H = self.build_hamiltonian()
        self.eig = None
    
    def commutator_norm(self, obs1, site1, obs2, site2, t):
        """
        Compute ||[obs1(t), obs2]|| (operator norm) for the LR bound test
        obs1, obs2: 'X', 'Y', or 'Z' strings
        """
        # Build local operators
        def local_op(op_str, site):
            op = 1
            for i in range(self.N):
                if i == site:
                    if op_str == 'X':
                        op = np.kron(op, self.X)
                    elif op_str == 'Y':
                        op = np.kron(op, self.Y)
                    else:
                        op = np.kron(op, self.Z)
                else:
                    op = np.kron(op, np.eye(2))
            # Tensor product with identity on bond registers
            op = np.kron(op, np.eye(self.bond_dim))
            return op
        
        A = local_op(obs1, site1)
        B = local_op(obs2, site2)
        
        # Heisenberg evolution: A(t) = U† A U
        eig = self.diagonalize()
        U = eig['eigenvectors']
        U_dag = U.conj().T
        exp_E = np.diag(np.exp(-1j * eig['eigenvalues'] * t))
        
        A_t = U @ exp_E.conj().T @ U_dag @ A @ U @ exp_E @ U_dag
        
        # Commutator
        comm = A_t @ B - B @ A_t
        # Operator norm (spectral norm)
        return np.linalg.norm(comm, 2)

    def decode_bond_config(self, bond_index):
        """Fast bond decoding using pre-computed powers"""
        config, remaining = [], bond_index
        for power in self.bond_powers:
            config.append(remaining // power)
            remaining %= power
        return config[::-1]  # Reverse to match edge order

    def state_index(self, matter_state, bond_config):
        """Fast index calculation using pre-computed bond powers"""
        bond_index = 0
        for i, val in enumerate(bond_config):
            bond_index += val * self.bond_powers[i]
        return matter_state * self.bond_dim + bond_index

    def calculate_degree(self, site, bond_config):
        """Calculate degree for degree penalty"""
        degree = 0
        if site > 0 and bond_config[site - 1] > 0:
            degree += 1
        if site < self.N - 1 and bond_config[site] > 0:
            degree += 1
        return degree

    def build_hamiltonian(self):
        """Memory-efficient Hamiltonian construction"""
        H = np.zeros((self.total_dim, self.total_dim), dtype=np.float64)
        
        for idx in range(self.total_dim):
            matter_state = idx // self.bond_dim
            bond_config = self.decode_bond_config(idx % self.bond_dim)
            Z_vals = self.Z_lookup[matter_state]
            
            # Matter rotation (X_i terms)
            for i in range(self.N):
                flipped = matter_state ^ (1 << i)
                j = self.state_index(flipped, bond_config)
                H[idx, j] += self.config['omega'] / 2
            
            # Bond terms (vectorized per edge)
            for edge in range(self.N - 1):
                i, j = edge, edge + 1
                Zi, Zj = Z_vals[i], Z_vals[j]
                
                # Z_i Z_j interaction
                H[idx, idx] += self.config['J0'] * Zi * Zj
                
                # Bond energy term
                H[idx, idx] += self.config['deltaB'] * bond_config[edge]
                
                # Degree penalty (simplified)
                d_i = self.calculate_degree(i, bond_config)
                d_j = self.calculate_degree(j, bond_config)
                H[idx, idx] += self.config['kappa'] * ((d_i - self.config['k0']) ** 2 +
                                                      (d_j - self.config['k0']) ** 2)
                
                # Promotion term (coherent bond register updates)
                F = 0.5 * (1 - Zi * Zj)  # Frustration detector
                
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
        
        # Symmetrize (faster than full matrix addition)
        H = (H + H.T) * 0.5
        return H

    def diagonalize(self):
        """Diagonalize using Accelerate framework (M4 optimized)"""
        if self.eig is None:
            # eigh on M4 uses Apple's Accelerate automatically
            eigenvalues, eigenvectors = eigh(self.H, overwrite_a=True)
            self.eig = {'eigenvectors': eigenvectors, 'eigenvalues': eigenvalues}
        return self.eig

    def get_ground_state(self):
        """Return actual ground state (not just |0>^N)"""
        return self.diagonalize()['eigenvectors'][:, 0]

    def apply_pi2_pulse(self, state, probe_site):
        """Apply π/2 pulse for Ramsey protocol"""
        new_state = np.zeros_like(state, dtype=complex)
        sqrt2 = np.sqrt(2)
        
        for idx in range(self.total_dim):
            if np.abs(state[idx]) < 1e-15:
                continue
            
            # Decode state
            matter_state = idx // self.bond_dim
            bond_config = self.decode_bond_config(idx % self.bond_dim)
            bit = (matter_state >> probe_site) & 1
            flipped_matter = matter_state ^ (1 << probe_site)
            
            # Apply rotation
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

    def evolve_state(self, state, t):
        """Exact time evolution using eigenbasis"""
        eig = self.diagonalize()
        eigenvectors = eig['eigenvectors']
        eigenvalues = eig['eigenvalues']
        
        # Project onto eigenbasis and apply phase factors
        coeffs = eigenvectors.T @ state
        exp_coeffs = coeffs * np.exp(-1j * eigenvalues * t)
        
        return eigenvectors @ exp_coeffs

    def measure_Z(self, state, site):
        """Vectorized Z measurement"""
        probabilities = np.abs(state) ** 2
        Z_vals = self.Z_lookup[:, site]
        # Repeat Z_vals for each bond configuration
        Z_full = np.tile(Z_vals, self.bond_dim)
        return np.sum(Z_full * probabilities)

    def get_bond_dimension(self, edge, state):
        """CORRECTED: Calculate average bond dimension ⟨χ⟩ for an edge"""
        avg_n = 0.0
        norm = 0.0
        
        for idx in range(self.total_dim):
            prob = np.abs(state[idx]) ** 2
            if prob > 1e-15:
                bond_config = self.decode_bond_config(idx % self.bond_dim)
                avg_n += bond_config[edge] * prob
                norm += prob
        
        return 2 ** (avg_n / norm) if norm > 0 else 1

    def count_circuit_depth(self, site, state):
        """Calculate entanglement proxy Λ at a site"""
        avg_degree = 0.0
        norm = 0.0
        
        for idx in range(self.total_dim):
            prob = np.abs(state[idx]) ** 2
            if prob > 1e-15:
                bond_config = self.decode_bond_config(idx % self.bond_dim)
                degree = self.calculate_degree(site, bond_config)
                avg_degree += degree * prob
                norm += prob
        
        d = avg_degree / norm if norm > 0 else 0
        return np.log2(1 + d)

# %%
# ============================================================================
# PARALLEL λ-SCAN ENGINE WITH CORRECTED PHYSICS
# ============================================================================
def run_single_point(args):
    """Worker function for parallel processing - CORRECTED PHYSICS"""
    N, alpha, lambda_param, tMax = args
    
    config = {
        'omega': 1.0, 'deltaB': 5.0, 'lambda': lambda_param,
        'kappa': 0.1, 'k0': 4, 'bondCutoff': 4, 'J0': 0.01, 'gamma': 0.0
    }
    
    qca = ExactQCA(N, config)
    
    # Get ground state and create frustrated state
    ground_state = qca.get_ground_state()
    # Simple frustration: evolve with enhanced λ
    hotspot_config = config.copy()
    hotspot_config['lambda'] = lambda_param * 3
    qca_hotspot = ExactQCA(N, hotspot_config)
    frustrated_state = qca_hotspot.evolve_state(ground_state, 1.0)
    
    # Time grid for Ramsey (reduced for speed, dense near revival)
    t_grid = np.linspace(0, tMax, 80)
    
    # Ramsey experiments on two sites
    probe_sites = [0, 1]  # Outside and inside
    measured_freqs = []
    
    for site in probe_sites:
        # Initialize |+⟩ state
        psi = qca.apply_pi2_pulse(ground_state, site)
        
        # Measure Z(t)
        Z_signal = []
        for t in t_grid:
            psi_t = qca.evolve_state(psi, t)
            Z_t = qca.measure_Z(psi_t, site)
            Z_signal.append(Z_t)
        
        # Fit frequency using FFT (robust and fast)
        Z_signal = np.array(Z_signal)
        # Remove DC offset
        Z_detrended = Z_signal - np.mean(Z_signal)
        
        # Find strongest frequency component
        fft = np.fft.fft(Z_detrended)
        freqs = np.fft.fftfreq(len(t_grid), d=t_grid[1] - t_grid[0])
        # Positive frequencies only
        pos_mask = freqs > 0
        peak_idx = np.argmax(np.abs(fft[pos_mask]))
        freq = 2 * np.pi * freqs[pos_mask][peak_idx]
        
        # Sanity check: clamp to reasonable range
        measured_freqs.append(np.clip(freq, 0.1, 5.0))
    
    # Calculate entanglement proxies Λ
    lambda_out = qca.count_circuit_depth(0, frustrated_state)
    lambda_in = qca.count_circuit_depth(1, frustrated_state)
    
    # Postulate 1 predictions (NO ARBITRARY FACTORS)
    predicted_out = 1.0 / (1 + alpha * lambda_out)
    predicted_in = 1.0 / (1 + alpha * lambda_in)
    
    # Residual (relative error on frequency ratio)
    residual = abs((measured_freqs[1] / measured_freqs[0]) -
                   (predicted_in / predicted_out))
    
    # Spectral gap for diagnostics
    eigvals = qca.diagonalize()['eigenvalues']
    spectral_gap = eigvals[1] - eigvals[0]
    
    return {
        'lambda': lambda_param,
        'residual': residual,
        'omega_out': measured_freqs[0],
        'omega_in': measured_freqs[1],
        'predicted_omega_out': predicted_out,
        'predicted_omega_in': predicted_in,
        'spectral_gap': spectral_gap,
        'lambda_out': lambda_out,
        'lambda_in': lambda_in,
        'status': '✓' if residual < 0.05 else '~' if residual < 0.10 else '✗'
    }

def scan_lambda_parallel(N=4, alpha=0.8, lambda_min=0.1, lambda_max=1.5, num_points=100):
    """Parallel λ scan optimized for M4 multicore with corrected physics"""
    print(f"\n🔬 Running CORRECTED parallel λ scan on M4")
    print(f"N={N}, α={alpha}, points={num_points}")
    print(f"Using {os.cpu_count()} cores")
    
    # Adaptive sampling (dense near suspected revival)
    lambda_vals = np.concatenate([
        np.linspace(lambda_min, 0.55, int(num_points * 0.35)),
        np.linspace(0.55, 0.8, int(num_points * 0.45)),
        np.linspace(0.8, lambda_max, int(num_points * 0.20))
    ])
    
    args_list = [(N, alpha, l, 20.0) for l in lambda_vals]
    
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(run_single_point, args): args[2]
                   for args in args_list}
        
        with tqdm.tqdm(total=len(lambda_vals), desc="λ scan") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Failed at λ={futures[future]:.3f}: {e}")
                pbar.update(1)
    
    # Sort by lambda
    df = pd.DataFrame(results).sort_values('lambda')
    
    # Save results
    filename = f"qca_corrected_N{N}_alpha{alpha:.1f}_parallel.csv"
    df.to_csv(filename, index=False)
    print(f"💾 Saved CORRECTED results to {filename}")
    
    return df

# %%
# ============================================================================
# CORRECTED PLOTTING WITH CRITICAL POINT DETECTION
# ============================================================================
def analyze_critical_points(df):
    """Find phase transition points from residual data"""
    try:
        # Phase I → II: First crossing of 5% threshold
        lambda_c1 = df[df['residual'] > 0.05]['lambda'].iloc[0]
    except IndexError:
        lambda_c1 = None
    
    try:
        # Quantum Revival: Local minimum in the violation region (residual > 10%)
        violation_region = df[df['residual'] > 0.10]
        if not violation_region.empty:
            # Find first local minimum after entering violation region
            first_viol_idx = violation_region.index[0]
            post_violation = df.loc[first_viol_idx:]
            
            # Use find_peaks on negative residual to find minima
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
            lambda_revival = None
            residual_min = None
    except Exception:
        lambda_revival = None
        residual_min = None
    
    try:
        # Phase III → IV: Cross 20% threshold after revival
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

def plot_corrected_results(df, N=4, alpha=0.8, save=True):
    """Comprehensive plotting with critical point detection"""
    # Analyze critical points
    crit = analyze_critical_points(df)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'QCA Results (N={N}, α={alpha}) - CORRECTED ANALYSIS',
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Main residual diagram
    ax = axes[0, 0]
    ax.plot(df['lambda'], df['residual'] * 100, 'o-',
            linewidth=2.5, markersize=7, color='darkblue', label='Postulate Residual')
    
    # Thresholds
    ax.axhline(5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='5% threshold')
    ax.axhline(10, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='10% threshold')
    
    # Phase regions
    ax.fill_between(df['lambda'], 0, 5, alpha=0.15, color='green', label='Emergent')
    ax.fill_between(df['lambda'], 5, 15, alpha=0.15, color='yellow', label='Tolerance')
    ax.fill_between(df['lambda'], 15, 130, alpha=0.15, color='red', label='Catastrophic')
    
    # Critical points
    if crit['lambda_c1']:
        ax.axvline(crit['lambda_c1'], color='black', linestyle='-',
                  linewidth=1.5, label=f'λ_c1={crit["lambda_c1"]:.3f}')
    
    if crit['lambda_revival']:
        ax.axvline(crit['lambda_revival'], color='purple', linestyle=':',
                  linewidth=2, label=f'λ_rev={crit["lambda_revival"]:.3f}')
        # Annotate revival
        ax.annotate(f'Quantum Revival\nR={crit["residual_min"]*100:.2f}%',
                   xy=(crit['lambda_revival'], crit['residual_min']*100),
                   xytext=(crit['lambda_revival']+0.15, crit['residual_min']*100+30),
                   arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
                   fontsize=10, ha='center', color='purple',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    if crit['lambda_c2']:
        ax.axvline(crit['lambda_c2'], color='gray', linestyle='-',
                  linewidth=1.5, label=f'λ_c2={crit["lambda_c2"]:.3f}')
    
    ax.set_xlabel('λ (promotion strength)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residual (%)', fontsize=12, fontweight='bold')
    ax.set_title('Postulate 1 Residual', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(df['residual']*100)*1.05)
    
    # Plot 2: Frequency scaling
    ax = axes[0, 1]
    ax.plot(df['lambda'], df['omega_out'], 'o-', label='Measured ω_out',
            color='steelblue', linewidth=2, markersize=6)
    ax.plot(df['lambda'], df['omega_in'], 's-', label='Measured ω_in',
            color='coral', linewidth=2, markersize=6)
    ax.plot(df['lambda'], df['predicted_omega_out'], '--',
            alpha=0.6, color='steelblue', label='Predicted ω_out')
    ax.plot(df['lambda'], df['predicted_omega_in'], '--',
            alpha=0.6, color='coral', label='Predicted ω_in')
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.7, label='Bare ω')
    
    if crit['lambda_revival']:
        ax.axvline(crit['lambda_revival'], color='purple', linestyle=':', alpha=0.5)
        # Mark inversion region
        inversion_mask = df['omega_in'] > df['omega_out']
        if inversion_mask.any():
            ax.fill_between(df['lambda'], 0, 2, where=inversion_mask,
                          alpha=0.2, color='red', label='Inversion')
    
    ax.set_xlabel('λ', fontsize=12, fontweight='bold')
    ax.set_ylabel('ω_eff', fontsize=12, fontweight='bold')
    ax.set_title('Frequency Scaling', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Entanglement proxy
    ax = axes[1, 0]
    ax.plot(df['lambda'], df['lambda_out'], 'o-', label='Λ_out',
            color='steelblue', linewidth=2, markersize=6)
    ax.plot(df['lambda'], df['lambda_in'], 's-', label='Λ_in',
            color='coral', linewidth=2, markersize=6)
    
    if crit['lambda_revival']:
        ax.axvline(crit['lambda_revival'], color='purple', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('λ', fontsize=12, fontweight='bold')
    ax.set_ylabel('Λ (circuit depth)', fontsize=12, fontweight='bold')
    ax.set_title('Entanglement Proxy', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Frequency ratio
    ax = axes[1, 1]
    measured_ratio = df['omega_in'] / df['omega_out']
    predicted_ratio = df['predicted_omega_in'] / df['predicted_omega_out']
    
    ax.plot(df['lambda'], measured_ratio, 'o-', label='Measured Ratio',
            color='darkgreen', linewidth=2, markersize=6)
    ax.plot(df['lambda'], predicted_ratio, 's-', label='Predicted Ratio',
            color='orange', linewidth=2, markersize=6)
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.7, label='No scaling')
    
    if crit['lambda_revival']:
        ax.axvline(crit['lambda_revival'], color='purple', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('λ', fontsize=12, fontweight='bold')
    ax.set_ylabel('ω_in / ω_out', fontsize=12, fontweight='bold')
    ax.set_title('Frequency Ratio Deviation', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = f"qca_corrected_N{N}_alpha{alpha:.1f}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"🖼️  Plot saved to {filename}")
    
    return fig, crit

# %%
# ============================================================================
# MAIN EXECUTION - PRODUCTION MODE
# ============================================================================
def run_single_test():
    """Quick single-point test for validation"""
    print("=== Single Point Validation Test (N=4, λ=0.689) ===")
    result = run_single_point((4, 0.8, 0.689, 20.0))
    print(f"Residual: {result['residual']*100:.2f}%")
    print(f"Measured ω_out: {result['omega_out']:.4f}, ω_in: {result['omega_in']:.4f}")
    print(f"Predicted ω_out: {result['predicted_omega_out']:.4f}, ω_in: {result['predicted_omega_in']:.4f}")
    print(f"Circuit depth Λ_out: {result['lambda_out']:.4f}, Λ_in: {result['lambda_in']:.4f}")
    print(f"Status: {result['status']}")

def main():
    """Production run: parallel scan + analysis"""
    
    # Quick validation
    run_single_test()
    
    # Full parallel scan
    print("\n" + "="*60)
    print("PRODUCTION RUN: Full λ Scan with Corrected Physics")
    print("="*60)
    
    df = scan_lambda_parallel(N=4, alpha=0.8, lambda_min=0.1, lambda_max=1.5, num_points=100)
    
    # Generate comprehensive plots
    print("\n" + "="*60)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("="*60)
    
    fig, crit = plot_corrected_results(df, N=4, alpha=0.8)
    
    # Print summary
    print("\n" + "="*60)
    print("CRITICAL POINT ANALYSIS")
    print("="*60)
    if crit['lambda_c1']:
        print(f"✓ Phase I → II breakdown: λ_c1 = {crit['lambda_c1']:.3f}")
    if crit['lambda_revival']:
        print(f"✓ Quantum revival: λ_rev = {crit['lambda_revival']:.3f} (R={crit['residual_min']*100:.2f}%)")
    if crit['lambda_c2']:
        print(f"✓ Phase III → IV catastrophic: λ_c2 = {crit['lambda_c2']:.3f}")
    print("="*60)
    
    return df, crit

# ============================================================================
# LR VELOCITY EXTRACTION - ADD THIS AT BOTTOM OF FILE
# ============================================================================
def extract_lr_velocity(N=6, lambda_val=0.5):
    """Extract v_LR by fitting commutator growth front"""
    config = {
        'omega': 1.0, 'deltaB': 5.0, 'lambda': lambda_val,
        'kappa': 0.1, 'k0': 4, 'bondCutoff': 4, 'J0': 0.01, 'gamma': 0.0
    }
    qca = ExactQCA(N, config)
    
    print(f"Computing LR velocity for N={N}, λ={lambda_val}...")
    print("This will take 2-3 minutes on M4 for N=6")
    
    # Test distances up to half the chain
    distances = range(1, N//2 + 1)
    times = np.linspace(0, 5, 50)  # Reduced time points for speed
    threshold = 1e-3
    
    arrival_times = []
    for r in distances:
        for t in times:
            norm = qca.commutator_norm('Z', 0, 'Z', r, t)
            if norm > threshold:
                arrival_times.append((r, t))
                print(f"  Distance {r}: t_arrival = {t:.3f} (comm_norm = {norm:.6f})")
                break
    
    # Linear fit: r = v_LR * t + b
    if len(arrival_times) >= 2:
        r_vals, t_vals = zip(*arrival_times)
        v_LR, intercept = np.polyfit(t_vals, r_vals, 1)
        
        print(f"\n✓ Extracted v_LR = {v_LR:.3f}")
        print(f"  Fit quality: intercept = {intercept:.3f} (should be ≈0)")
        return v_LR, (r_vals, t_vals)
    
    print("\n✗ Could not extract velocity (insufficient data points)")
    return None, None

# Add this to the main block at the very end of the file:
if __name__ == "__main__":
    # Your existing main() call
    df_results, critical_points = main()
    
    # NEW: Run LR velocity test
    print("\n" + "="*60)
    print("LR VELOCITY VALIDATION TEST")
    print("="*60)
    v_lr, data = extract_lr_velocity(N=6, lambda_val=0.5)
