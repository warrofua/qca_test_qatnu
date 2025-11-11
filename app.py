# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.linalg import eigh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# ============================================================================
# COMPLEX NUMBER ARITHMETIC HELPERS (Python has native complex, but we keep
# the structure for consistency)
# ============================================================================

class Complex:
    """Helper class for complex arithmetic (Python's built-in complex type is used)"""
    @staticmethod
    def from_complex(z: complex):
        return z
    
    @staticmethod
    def add(a: complex, b: complex) -> complex:
        return a + b
    
    @staticmethod
    def sub(a: complex, b: complex) -> complex:
        return a - b
    
    @staticmethod
    def mul(a: complex, b: complex) -> complex:
        return a * b
    
    @staticmethod
    def conj(a: complex) -> complex:
        return np.conj(a)
    
    @staticmethod
    def abs2(a: complex) -> float:
        return abs(a) ** 2
    
    @staticmethod
    def scale(c: float, a: complex) -> complex:
        return c * a
    
    @staticmethod
    def zero() -> complex:
        return 0.0 + 0.0j
    
    @staticmethod
    def one() -> complex:
        return 1.0 + 0.0j
    
    @staticmethod
    def i() -> complex:
        return 0.0 + 1.0j

# %%
# ============================================================================
# EXACT DIAGONALIZATION ENGINE (Option A) with Complex Arithmetic
# ============================================================================

# Add this to app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

def analyze_and_plot_qca_phase(alpha=0.8, N=6, lambda_min=0.1, lambda_max=1.5, num_points=20, save_prefix="qca_results"):
    """
    Complete analysis of QCA phase diagram with corrected revival detection.
    """
    print(f"\n🔬 Running QCA Phase Analysis: N={N}, α={alpha}")
    print("=" * 70)
    
    # Run scan
    results = []
    lambda_vals = np.linspace(lambda_min, lambda_max, num_points)
    
    for i, l in enumerate(lambda_vals):
        print(f"Progress: {i+1}/{num_points} (λ={l:.3f})...")
        
        exact_results, mf_results, validation = run_with_parameters(
            N=N, alpha=alpha, lambda_param=l, tMax=20
        )
        
        results.append({
            'lambda': l,
            'residual': validation['residual'],
            'omega_out': validation['measuredFreqOut'],
            'omega_in': validation['measuredFreqIn'],
            'predicted_omega_out': validation['predictedFreqOut'],
            'predicted_omega_in': validation['predictedFreqIn'],
            'lambda_out': validation['measuredLambdaOut'],
            'lambda_in': validation['measuredLambdaIn'],
            'status': '✓' if validation['residual'] < 0.05 else '~' if validation['residual'] < 0.10 else '✗'
        })
    
    df = pd.DataFrame(results)
    
    # Find critical points
    
    # 1. Phase I → II: First crossing of 5% threshold
    try:
        lambda_c1 = df[df['residual'] > 0.05]['lambda'].iloc[0]
    except:
        lambda_c1 = None
    
    # 2. Quantum Revival: Local minimum *after* entering violation region
    #    Filter for data points where residual first exceeds 10% then drops
    try:
        # Start from first violation
        first_violation_idx = df[df['residual'] > 0.10].index[0]
        post_violation = df.loc[first_violation_idx:]
        
        # Find local minima in the violation region
        from scipy.signal import find_peaks
        # Use negative residual to find minima
        minima_indices, _ = find_peaks(-post_violation['residual'].values, height=-0.15)
        
        if len(minima_indices) > 0:
            # Get the first local minimum (the revival)
            revival_idx = post_violation.index[minima_indices[0]]
            lambda_revival = df.loc[revival_idx, 'lambda']
            residual_min = df.loc[revival_idx, 'residual']
        else:
            # Fallback: global minimum in violation region
            violation_data = df[df['residual'] > 0.10]
            if not violation_data.empty:
                idx_revival = violation_data['residual'].idxmin()
                lambda_revival = df.loc[idx_revival, 'lambda']
                residual_min = df.loc[idx_revival, 'residual']
            else:
                lambda_revival = None
                residual_min = None
    except:
        lambda_revival = None
        residual_min = None
    
    # 3. Phase III → IV: Cross 20% threshold after revival
    try:
        if lambda_revival:
            post_revival = df[df['lambda'] > lambda_revival]
            lambda_c2 = post_revival[post_revival['residual'] > 0.20]['lambda'].iloc[0]
        else:
            lambda_c2 = None
    except:
        lambda_c2 = None
    
    # Save raw data
    data_filename = f"{save_prefix}_N{N}_alpha{alpha:.1f}_data.csv"
    df.to_csv(data_filename, index=False)
    print(f"\n💾 Raw data saved to: {data_filename}")
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)
    
    # === PLOT 1: Residual vs Lambda (Main phase diagram) ===
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['lambda'], df['residual'] * 100, 'o-', linewidth=2.5, markersize=8, 
             color='darkblue', label='Postulate 1 Residual')
    
    # Thresholds
    ax1.axhline(5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='5% threshold')
    ax1.axhline(10, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='10% threshold')
    
    # Phase regions
    ax1.fill_between(df['lambda'], 0, 5, alpha=0.15, color='green', label='Emergent (Phase I)')
    ax1.fill_between(df['lambda'], 5, 50, alpha=0.15, color='yellow', label='Breakdown (Phase II)')
    ax1.fill_between(df['lambda'], 50, 150, alpha=0.15, color='red', label='Catastrophic (Phase IV)')
    
    # Mark critical points
    if lambda_c1:
        ax1.axvline(lambda_c1, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label=f'λ_c1={lambda_c1:.3f}')
    if lambda_revival:
        ax1.axvline(lambda_revival, color='purple', linestyle=':', linewidth=2, 
                    label=f'Revival (λ={lambda_revival:.3f})')
        ax1.annotate(f'Quantum Revival\nλ={lambda_revival:.3f}\nR={residual_min*100:.2f}%', 
                     xy=(lambda_revival, residual_min*100), 
                     xytext=(lambda_revival+0.15, residual_min*100+30),
                     arrowprops=dict(arrowstyle='->', color='purple', lw=1.5),
                     fontsize=11, ha='center', color='purple', 
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    if lambda_c2:
        ax1.axvline(lambda_c2, color='gray', linestyle='-', linewidth=1.5, alpha=0.7, label=f'λ_c2={lambda_c2:.3f}')
    
    ax1.set_xlabel('λ (promotion strength)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Postulate 1 Residual (%)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Quantum Phase Diagram: N={N}, α={alpha}', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(lambda_min, lambda_max)
    ax1.set_ylim(0, max(df['residual']*100)*1.1)
    
    # === PLOT 2: Frequency Scaling ===
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df['lambda'], df['omega_out'], 'o-', label='Measured ω_out', linewidth=2, markersize=6, color='steelblue')
    ax2.plot(df['lambda'], df['omega_in'], 's-', label='Measured ω_in', linewidth=2, markersize=6, color='coral')
    ax2.plot(df['lambda'], df['predicted_omega_out'], '--', label='Predicted ω_out', alpha=0.6, color='steelblue')
    ax2.plot(df['lambda'], df['predicted_omega_in'], '--', label='Predicted ω_in', alpha=0.6, color='coral')
    ax2.axhline(1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label='Bare ω')
    if lambda_revival:
        ax2.axvline(lambda_revival, color='purple', linestyle=':', linewidth=1.5, alpha=0.5)
    ax2.set_xlabel('λ', fontsize=12, fontweight='bold')
    ax2.set_ylabel('ω_eff', fontsize=12, fontweight='bold')
    ax2.set_title('Frequency Scaling', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # === PLOT 3: Frequency Inversion ===
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
    
    # === PLOT 4: Entanglement Proxy ===
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(df['lambda'], df['lambda_out'], 'o-', label='Λ_out', linewidth=2, color='steelblue')
    ax4.plot(df['lambda'], df['lambda_in'], 's-', label='Λ_in', linewidth=2, color='coral')
    if lambda_revival:
        ax4.axvline(lambda_revival, color='purple', linestyle=':', linewidth=1.5, alpha=0.5)
    ax4.set_xlabel('λ', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Λ (circuit depth)', fontsize=12, fontweight='bold')
    ax4.set_title('Entanglement Proxy', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # === PLOT 5: Frequency Ratio ===
    ax5 = fig.add_subplot(gs[2, 0])
    measured_ratio = df['omega_in'] / df['omega_out']
    predicted_ratio = df['predicted_omega_in'] / df['predicted_omega_out']
    ax5.plot(df['lambda'], measured_ratio, 'o-', label='Measured Ratio', linewidth=2, markersize=6, color='darkgreen')
    ax5.plot(df['lambda'], predicted_ratio, 's-', label='Predicted Ratio', linewidth=2, markersize=6, color='orange')
    ax5.axhline(1.0, color='black', linestyle=':', linewidth=1.5, alpha=0.7, label='No scaling')
    if lambda_revival:
        ax5.axvline(lambda_revival, color='purple', linestyle=':', linewidth=1.5, alpha=0.5)
    ax5.set_xlabel('λ', fontsize=12, fontweight='bold')
    ax5.set_ylabel('ω_in / ω_out', fontsize=12, fontweight='bold')
    ax5.set_title('Frequency Ratio Deviation', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # === PLOT 6: Phase Status ===
    ax6 = fig.add_subplot(gs[2, 1:])
    status_colors = {'✓': 0, '~': 1, '✗': 2}
    status_numeric = [status_colors[s] for s in df['status']]
    scatter = ax6.scatter(df['lambda'], [0]*len(df), c=status_numeric, 
                         cmap='RdYlGn_r', s=200, marker='s', edgecolors='black', alpha=0.8)
    ax6.set_xlabel('λ', fontsize=12, fontweight='bold')
    ax6.set_yticks([])
    ax6.set_title('Phase Classification', fontsize=13, fontweight='bold')
    ax6.set_xlim(lambda_min, lambda_max)
    cbar = plt.colorbar(scatter, ax=ax6, orientation='horizontal', pad=0.25, ticks=[0, 1, 2])
    cbar.ax.set_xticklabels(['✓ Emergent', '~ Within Tol.', '✗ Violated'], fontsize=10)
    
    # Prepare formatted strings for summary (FIXED)
    lambda_c1_str = f"{lambda_c1:.3f}" if lambda_c1 is not None else "N/A"
    lambda_revival_str = f"{lambda_revival:.3f}" if lambda_revival is not None else "N/A"
    residual_min_str = f"{residual_min*100:.2f}" if residual_min is not None else "N/A"
    lambda_c2_str = f"{lambda_c2:.3f}" if lambda_c2 is not None else "N/A"
    
    # Add summary text box (FIXED)
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
    
    # Save figure
    plot_filename = f"{save_prefix}_N{N}_alpha{alpha:.1f}_phase_diagram.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"🖼️  Phase diagram saved to: {plot_filename}")
    
    # Print summary
    print("\n" + "="*60)
    print("CORRECTED CRITICAL POINT ANALYSIS")
    print("="*60)
    print("Using quantum revival detection (local minimum in violation region)")
    print("-"*60)
    if lambda_c1:
        print(f"✓ Phase I → II breakdown: λ_c1 = {lambda_c1:.3f}")
    else:
        print("✗ No breakdown point detected")
    
    if lambda_revival:
        print(f"✓ Quantum revival point: λ_rev = {lambda_revival:.3f} (R={residual_min*100:.2f}%)")
    else:
        print("✗ No quantum revival detected")
    
    if lambda_c2:
        print(f"✓ Phase III → IV catastrophic: λ_c2 = {lambda_c2:.3f}")
    else:
        print("✗ No catastrophic transition detected")
    
    print("="*60)
    
    return df, {'lambda_c1': lambda_c1, 'lambda_revival': lambda_revival, 'lambda_c2': lambda_c2}

def full_phase_diagram(N=4, save_prefix="qca_results"):
    """
    Generate full 2D phase diagram in (α, λ) space
    Shows when and where Postulate 1 breaks down
    """
    # Define parameter grid
    alphas = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    lambdas = np.linspace(0.1, 1.2, 12)  # Focus on relevant region
    
    data = []
    total = len(alphas) * len(lambdas)
    count = 0
    
    print(f"\n🗺️  Mapping (α, λ) phase space: {total} points...")
    print("="*60)
    
    for a in alphas:
        for l in lambdas:
            count += 1
            print(f"Progress: {count}/{total} (α={a:.1f}, λ={l:.2f})")
            try:
                # Run simulation
                exact_results, mf_results, validation = run_with_parameters(
                    N=N, alpha=a, lambda_param=l, tMax=20
                )
                # Determine regime
                residual = validation['residual']
                if residual < 0.05:
                    regime = 0  # Emergent
                elif residual < 0.15:
                    regime = 1  # Tolerance
                elif residual < 0.50:
                    regime = 2  # Breakdown
                else:
                    regime = 3  # Catastrophic
                
                data.append({
                    'alpha': a,
                    'lambda': l,
                    'residual': residual,
                    'regime': regime,
                    'omega_out': validation['measuredFreqOut'],
                    'omega_in': validation['measuredFreqIn'],
                    'regime_label': ['Emergent', 'Tolerance', 'Breakdown', 'Catastrophic'][regime]
                })
            except Exception as e:
                print(f"  ⚠️  Failed at α={a}, λ={l}: {e}")
                continue
    
    df = pd.DataFrame(data)
    
    # Create heatmap plot
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Residual heatmap
    plt.subplot(2, 1, 1)
    pivot_residual = df.pivot(index='lambda', columns='alpha', values='residual') * 100
    sns.heatmap(pivot_residual, cmap='RdYlGn_r', center=10, vmin=0, vmax=100,
                annot=True, fmt='.0f', cbar_kws={'label': 'Residual (%)'},
                linewidths=0.5, linecolor='gray')
    
    plt.title(f'Postulate 1 Residual: N={N}, Hilbert dim={2**N * 4**(N-1)}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('α (postulate coefficient)', fontsize=14, fontweight='bold')
    plt.ylabel('λ (promotion strength)', fontsize=14, fontweight='bold')
    
    # Plot 2: Regime classification
    plt.subplot(2, 1, 2)
    pivot_regime = df.pivot(index='lambda', columns='alpha', values='regime')
    
    # Custom colormap for regimes
    cmap_regime = plt.cm.colors.ListedColormap(['green', 'yellow', 'orange', 'red'])
    sns.heatmap(pivot_regime, cmap=cmap_regime, vmin=0, vmax=3,
                annot=False, cbar_kws={'label': 'Regime'},
                linewidths=0.5, linecolor='gray')
    
    # Add regime labels to colorbar
    cbar = plt.gcf().axes[-1]
    cbar.set_yticks([0.5, 1.5, 2.5, 3.5])
    cbar.set_yticklabels(['Emergent\n(R<5%)', 'Tolerance\n(5-15%)', 'Breakdown\n(15-50%)', 'Catastrophic\n(R>50%)'])
    
    plt.xlabel('α (postulate coefficient)', fontsize=14, fontweight='bold')
    plt.ylabel('λ (promotion strength)', fontsize=14, fontweight='bold')
    plt.title('Regime Classification', fontsize=16, fontweight='bold', pad=20)
    
    # Save
    filename = f"{save_prefix}_N{N}_2D_phase_diagram.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n🖼️  2D phase diagram saved to: {filename}")
    
    return df

class ExactQCA:
    def __init__(self, N, config):
        self.N = N
        self.config = config
        self.m = self.initialize_hilbert_space()
        self.H = self.build_hamiltonian()
        self.eig = None  # Cache eigen-decomposition

    def initialize_hilbert_space(self):
        self.bond_cutoff = self.config.get('bondCutoff', 4)
        self.matter_dim = 2 ** self.N
        self.bond_dim = self.bond_cutoff ** (self.N - 1)
        self.total_dim = self.matter_dim * self.bond_dim
        
        return {
            'matterDim': self.matter_dim,
            'bondDim': self.bond_dim,
            'totalDim': self.total_dim
        }

    def decode_bond_config(self, bond_index):
        """Decode bond configuration from integer index"""
        config = []
        remaining = bond_index
        for _ in range(self.N - 1):
            config.append(remaining % self.bond_cutoff)
            remaining //= self.bond_cutoff
        return config

    def calculate_degree(self, site, bond_config):
        """Calculate degree (number of active bonds) at a site"""
        degree = 0
        if site > 0 and bond_config[site - 1] > 0:
            degree += 1
        if site < self.N - 1 and bond_config[site] > 0:
            degree += 1
        return degree

    def state_index(self, matter_state, bond_config):
        """Get linear index from matter and bond configurations"""
        bond_index = 0
        mult = 1
        for i in range(self.N - 1):
            bond_index += bond_config[i] * mult
            mult *= self.bond_cutoff
        return matter_state * self.bond_dim + bond_index

    def build_hamiltonian(self):
        """Build the full Hamiltonian matrix"""
        H = np.zeros((self.total_dim, self.total_dim), dtype=complex)
        
        for idx in range(self.total_dim):
            matter_state = idx // self.bond_dim
            bond_config = self.decode_bond_config(idx % self.bond_dim)
            
            # 1. Matter rotation term: (ω/2) Σ X_i
            for i in range(self.N):
                flipped_matter = matter_state ^ (1 << i)
                j = self.state_index(flipped_matter, bond_config)
                H[idx, j] += self.config['omega'] / 2
            
            # 2-4. Bond energy, promotion, and degree penalty (all per edge)
            for edge in range(self.N - 1):
                i, j = edge, edge + 1
                Zi = 1 if ((matter_state >> i) & 1) else -1
                Zj = 1 if ((matter_state >> j) & 1) else -1
                
                # Z_i Z_j term
                H[idx, idx] += self.config['J0'] * Zi * Zj
                
                # Δ_b n_ij term
                H[idx, idx] += self.config['deltaB'] * bond_config[edge]
                
                # Degree penalty for this edge
                d_i = self.calculate_degree(i, bond_config)
                d_j = self.calculate_degree(j, bond_config)
                C = (d_i - self.config['k0'])**2 + (d_j - self.config['k0'])**2
                H[idx, idx] += self.config['kappa'] * C
                
                # Promotion term
                F = 0.5 * (1 - Zi * Zj)
                
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
        
        # Symmetrize H (ensure Hermitian)
        H = (H + H.conj().T) / 2
        
        # Since the Hamiltonian is actually real symmetric, return real part
        return np.real_if_close(H, tol=1e-12).real

    def diagonalize(self):
        """Diagonalize Hamiltonian using scipy's eigh"""
        if self.eig is None:
            eigenvalues, eigenvectors = eigh(self.H)
            self.eig = {
                'eigenvectors': eigenvectors,
                'eigenvalues': eigenvalues
            }
        return self.eig

    def evolve_state(self, state, t):
        """Time evolve a state using the eigenbasis"""
        eig = self.diagonalize()
        eigenvectors = eig['eigenvectors']
        eigenvalues = eig['eigenvalues']
        
        # Project onto eigenbasis
        coeffs = eigenvectors.T @ state
        
        # Apply time evolution
        exp_coeffs = coeffs * np.exp(-1j * eigenvalues * t)
        
        # Transform back to computational basis
        return eigenvectors @ exp_coeffs

    def get_ground_state(self):
        """Get the ground state (simple initial state)"""
        state = np.zeros(self.total_dim, dtype=complex)
        state[0] = 1.0 + 0.0j
        return state

    def apply_pi2_pulse(self, state, probe_site):
        """Apply π/2 pulse to a probe site"""
        new_state = np.zeros_like(state)
        sqrt2 = np.sqrt(2)
        
        for idx in range(self.total_dim):
            if np.abs(state[idx]) < 1e-15:
                continue
            
            matter_state = idx // self.bond_dim
            bond_config = self.decode_bond_config(idx % self.bond_dim)
            bit = (matter_state >> probe_site) & 1
            flipped_matter = matter_state ^ (1 << probe_site)
            
            j1 = self.state_index(matter_state, bond_config)
            j2 = self.state_index(flipped_matter, bond_config)
            amplitude = state[idx]
            
            if bit == 0:
                new_state[j1] += amplitude / sqrt2
                new_state[j2] += (1j * amplitude) / sqrt2
            else:
                new_state[j1] += amplitude / sqrt2
                new_state[j2] += (-1j * amplitude) / sqrt2
        
        return new_state

    def measure_Z(self, state, site):
        """Measure Z expectation value at a site"""
        expZ = 0.0
        for idx in range(self.total_dim):
            matter_state = idx // self.bond_dim
            Zi = 1 if ((matter_state >> site) & 1) else -1
            expZ += Zi * Complex.abs2(state[idx])
        return expZ

    def get_bond_dimension(self, bond_index, state):
        """Calculate average bond dimension (χ) for an edge"""
        avgN = 0.0
        norm = 0.0
        for idx in range(self.total_dim):
            prob = Complex.abs2(state[idx])
            if prob > 0:
                bond_config = self.decode_bond_config(idx % self.bond_dim)
                avgN += bond_config[bond_index] * prob
                norm += prob
        return max(1, 2 ** avgN) if norm > 0 else 1

    def count_circuit_depth(self, site, state):
        """Count circuit depth Λ at a site"""
        avg_degree = 0.0
        norm = 0.0
        for idx in range(self.total_dim):
            prob = Complex.abs2(state[idx])
            if prob > 0:
                bond_config = self.decode_bond_config(idx % self.bond_dim)
                degree = self.calculate_degree(site, bond_config)
                avg_degree += degree * prob
                norm += prob
        
        d = avg_degree / norm if norm > 0 else 0
        return np.log2(1 + d)

# %%
# ============================================================================
# MEAN-FIELD COMPARISON
# ============================================================================

class QuantumChain:
    def __init__(self, N, omega, alpha, chi_profile, J0, eta, gamma):
        self.N = N
        self.omega = omega
        self.alpha = alpha
        self.chi_profile = chi_profile
        self.J0 = J0
        self.eta = eta
        self.gamma = gamma
        self.compute_lambda()

    def compute_lambda(self):
        """Compute entanglement proxy Λ_i"""
        self.Lambda = np.zeros(self.N)
        for i in range(self.N):
            if i > 0:
                self.Lambda[i] += np.log2(max(1, self.chi_profile[i - 1]))
            if i < self.N - 1:
                self.Lambda[i] += np.log2(max(1, self.chi_profile[i]))

    def get_effective_frequency(self, i):
        """Get effective frequency according to Postulate 1"""
        return self.omega / (1 + self.alpha * self.Lambda[i])

    def evolve_ramsey_single_ion(self, probe, t):
        """Mean-field Ramsey evolution for a single ion"""
        omega_eff = self.get_effective_frequency(probe)
        dt = min(0.1, np.pi / (4 * omega_eff))
        steps = int(np.floor(t / dt))
        if steps <= 0:
            return 1.0

        # Initial Bell state (|00> + |11>)/sqrt(2)
        state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        
        for _ in range(steps):
            angle = omega_eff * dt / 2
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            new_state = np.array([
                cos_a * state[0] - sin_a * state[2],
                cos_a * state[1] - sin_a * state[3],
                cos_a * state[2] + sin_a * state[0],
                cos_a * state[3] + sin_a * state[1]
            ], dtype=complex)
            
            if self.gamma > 0:
                decay = np.exp(-self.gamma * dt / 2)
                new_state[1] *= decay
                new_state[3] *= decay
            
            # Normalize
            norm = np.sqrt(np.sum(np.abs(new_state) ** 2))
            state = new_state / norm
        
        # Final π/2 pulse
        sqrt2 = np.sqrt(2)
        final_state = np.array([
            (state[0] - state[3]) / sqrt2,
            (state[1] + state[2]) / sqrt2,
            (state[2] + state[1]) / sqrt2,
            (state[3] - state[0]) / sqrt2
        ], dtype=complex)
        
        prob_up = np.abs(final_state[0]) ** 2 + np.abs(final_state[1]) ** 2
        prob_down = np.abs(final_state[2]) ** 2 + np.abs(final_state[3]) ** 2
        return prob_up - prob_down

# %%
# ============================================================================
# RAMSEY FITTING
# ============================================================================

def fit_ramsey_frequency(data):
    """Fit Ramsey oscillation frequency from time series data"""
    if data is None or len(data) < 10:
        return {'frequency': 0, 'amplitude': 0, 'T2': 1000, 'phase': 0, 'fitQuality': 0}
    
    times = np.array([d['t'] for d in data])
    signal = np.array([d['Z'] for d in data])
    mean = np.mean(signal)
    signal_detrended = signal - mean
    
    # Find peaks and troughs
    peaks = []
    troughs = []
    for i in range(1, len(signal_detrended) - 1):
        if signal_detrended[i] > signal_detrended[i-1] and signal_detrended[i] > signal_detrended[i+1]:
            peaks.append(i)
        if signal_detrended[i] < signal_detrended[i-1] and signal_detrended[i] < signal_detrended[i+1]:
            troughs.append(i)
    
    # Estimate frequency from peak spacing
    frequency = 0.1  # default
    if len(peaks) >= 2:
        periods = []
        for i in range(1, len(peaks)):
            period = times[peaks[i]] - times[peaks[i-1]]
            if period > 0:
                periods.append(period)
        if periods:
            avg_period = np.mean(periods)
            frequency = 2 * np.pi / avg_period
    
    # Clamp frequency
    dt = times[1] - times[0]
    frequency = np.clip(frequency, 0.001, np.pi / dt)
    
    # Estimate amplitude
    amplitude = max(0.1, (np.max(signal) - np.min(signal)) / 2)
    
    return {
        'frequency': frequency,
        'amplitude': amplitude,
        'T2': 1000,
        'phase': 0,
        'fitQuality': 0.95
    }

# %%
# ============================================================================
# MAIN SIMULATION RUNNER
# ============================================================================

class QCATester:
    def __init__(self):
        self.parameters = {
            'N': 4,
            'omega': 1.0,
            'alpha': 0.5,
            'deltaB': 5.0,
            'lambda': 0.3,
            'kappa': 0.1,
            'k0': 4,
            'bondCutoff': 4,
            'J0': 0.01,
            'gamma': 0.0,
            'probeOut': 0,
            'probeIn': 1,
            'tMax': 20
        }
    
    def run_exact_experiment(self):
        """Run the full exact diagonalization experiment"""
        p = self.parameters
        
        config = {
            'omega': p['omega'],
            'deltaB': p['deltaB'],
            'lambda': p['lambda'],
            'kappa': p['kappa'],
            'k0': p['k0'],
            'bondCutoff': p['bondCutoff'],
            'J0': p['J0'],
            'gamma': p['gamma']
        }
        
        print(f"Running exact QCA with Hilbert space dim: {2**p['N'] * p['bondCutoff']**(p['N']-1)}")
        
        qca = ExactQCA(p['N'], config)
        
        # Create frustrated state
        hotspot_config = config.copy()
        hotspot_config['lambda'] = p['lambda'] * 3
        qca_hotspot = ExactQCA(p['N'], hotspot_config)
        ground_state = qca.get_ground_state()
        frustrated_state = qca_hotspot.evolve_state(ground_state, 1.0)
        
        # Time grids
        t_grid = np.linspace(0, p['tMax'], 100)
        
        # Ramsey experiments
        data_out = []
        data_in = []
        
        for t in t_grid:
            # Outside probe
            psi_out = qca.get_ground_state()
            psi_out = qca.apply_pi2_pulse(psi_out, p['probeOut'])
            psi_out = qca.evolve_state(psi_out, t)
            Z_out = qca.measure_Z(psi_out, p['probeOut'])
            data_out.append({'t': t, 'Z': Z_out})
            
            # Inside probe
            psi_in = qca.get_ground_state()
            psi_in = qca.apply_pi2_pulse(psi_in, p['probeIn'])
            psi_in = qca.evolve_state(psi_in, t)
            Z_in = qca.measure_Z(psi_in, p['probeIn'])
            data_in.append({'t': t, 'Z': Z_in})
        
        # Bond dimensions
        bond_dims = []
        for edge in range(p['N'] - 1):
            chi = qca.get_bond_dimension(edge, frustrated_state)
            bond_dims.append({'edge': edge, 'chi': chi})
        
        # Circuit depths
        depth_out = qca.count_circuit_depth(p['probeOut'], frustrated_state)
        depth_in = qca.count_circuit_depth(p['probeIn'], frustrated_state)
        
        # Fit frequencies
        fit_out = fit_ramsey_frequency(data_out)
        fit_in = fit_ramsey_frequency(data_in)
        
        print(f"Fit frequencies - Out: {fit_out['frequency']:.4f}, In: {fit_in['frequency']:.4f}")
        
        return {
            'dataOut': data_out,
            'dataIn': data_in,
            'fitOut': fit_out,
            'fitIn': fit_in,
            'bondDims': bond_dims,
            'depthOut': depth_out,
            'depthIn': depth_in,
            'config': config
        }

    def run_mean_field_comparison(self, exact_results):
        """Run mean-field comparison using exact results for χ profile"""
        p = self.parameters
        
        # Extract χ profile from exact results
        chi_profile = [d['chi'] for d in exact_results['bondDims']]
        
        chain = QuantumChain(
            N=p['N'],
            omega=p['omega'],
            alpha=p['alpha'],
            chi_profile=chi_profile,
            J0=p['J0'],
            eta=1.5,
            gamma=p['gamma']
        )
        
        t_grid = np.linspace(0, p['tMax'], 100)
        
        data_out = []
        data_in = []
        
        for t in t_grid:
            Z_out = chain.evolve_ramsey_single_ion(p['probeOut'], t)
            Z_in = chain.evolve_ramsey_single_ion(p['probeIn'], t)
            data_out.append({'t': t, 'Z': Z_out})
            data_in.append({'t': t, 'Z': Z_in})
        
        fit_out = fit_ramsey_frequency(data_out)
        fit_in = fit_ramsey_frequency(data_in)
        
        return {
            'dataOut': data_out,
            'dataIn': data_in,
            'fitOut': fit_out,
            'fitIn': fit_in
        }

    def validate_postulate(self, exact_results):
        """Validate Postulate 1 using measured vs predicted frequencies"""
        p = self.parameters
        
        # Measured Λ from circuit depth
        measured_lambda_out = np.log2(1 + 2 ** exact_results['depthOut'] - 1)
        measured_lambda_in = np.log2(1 + 2 ** exact_results['depthIn'] - 1)
        
        # Predicted frequencies
        predicted_freq_out = p['omega'] / (1 + p['alpha'] * measured_lambda_out)
        predicted_freq_in = p['omega'] / (1 + p['alpha'] * measured_lambda_in)
        
        # Measured frequencies
        measured_freq_out = exact_results['fitOut']['frequency']
        measured_freq_in = exact_results['fitIn']['frequency']
        
        # Residual
        residual = abs((measured_freq_in / measured_freq_out) - 
                      (predicted_freq_in / predicted_freq_out))
        
        return {
            'measuredLambdaOut': measured_lambda_out,
            'measuredLambdaIn': measured_lambda_in,
            'predictedFreqOut': predicted_freq_out,
            'predictedFreqIn': predicted_freq_in,
            'residual': residual,
            'measuredFreqOut': measured_freq_out,
            'measuredFreqIn': measured_freq_in
        }

    def create_plots(self, exact_results, mf_results, validation):
        """Create all diagnostic plots"""
        
        # 1. Exact vs Mean-Field comparison
        fig1 = go.Figure()
        
        fig1.add_trace(go.Scatter(
            x=[d['t'] for d in exact_results['dataOut']],
            y=[d['Z'] for d in exact_results['dataOut']],
            mode='markers',
            name=f"Exact (ion {self.parameters['probeOut']})",
            marker=dict(size=6, color='rgb(31, 119, 180)')
        ))
        
        fig1.add_trace(go.Scatter(
            x=[d['t'] for d in exact_results['dataIn']],
            y=[d['Z'] for d in exact_results['dataIn']],
            mode='markers',
            name=f"Exact (ion {self.parameters['probeIn']})",
            marker=dict(size=6, color='rgb(255, 127, 14)')
        ))
        
        fig1.add_trace(go.Scatter(
            x=[d['t'] for d in mf_results['dataOut']],
            y=[d['Z'] for d in mf_results['dataOut']],
            mode='lines',
            name=f"Mean-Field (ion {self.parameters['probeOut']})",
            line=dict(color='rgb(31, 119, 180)', dash='dash')
        ))
        
        fig1.add_trace(go.Scatter(
            x=[d['t'] for d in mf_results['dataIn']],
            y=[d['Z'] for d in mf_results['dataIn']],
            mode='lines',
            name=f"Mean-Field (ion {self.parameters['probeIn']})",
            line=dict(color='rgb(255, 127, 14)', dash='dash')
        ))
        
        fig1.update_layout(
            title='Exact vs Mean-Field Ramsey Fringes',
            xaxis_title='Time (ℏ/ω)',
            yaxis_title='⟨Z⟩',
            height=400,
            template='plotly_white'
        )
        
        
        # 2. Bond dimensions
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=[d['edge'] for d in exact_results['bondDims']],
            y=[np.log2(d['chi']) for d in exact_results['bondDims']],
            mode='markers',
            name='log₂⟨χ⟩ after frustration',
            marker=dict(size=10, color='rgb(44, 160, 44)')
        ))
        
        fig2.update_layout(
            title='Bond Register Excitations',
            xaxis_title='Edge Index',
            yaxis_title='log₂⟨χ⟩',
            height=300,
            template='plotly_white'
        )
        
        
        # 3. Postulate validation
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=[0, 1],
            y=[validation['measuredFreqOut'], validation['measuredFreqIn']],
            mode='markers',
            name='Measured ω',
            marker=dict(size=12, color='black')
        ))
        
        fig3.add_trace(go.Scatter(
            x=[0, 1],
            y=[validation['predictedFreqOut'], validation['predictedFreqIn']],
            mode='lines+markers',
            name='Postulate 1 Prediction',
            line=dict(color='red', dash='dash'),
            marker=dict(size=10, symbol='x')
        ))
        
        fig3.update_layout(
            title=f"Testing Postulate 1: Residual = {(validation['residual'] * 100):.2f}%",
            xaxis=dict(
                tickvals=[0, 1],
                ticktext=['Outside', 'Inside']
            ),
            yaxis_title='Effective Frequency ω_eff',
            height=300,
            template='plotly_white'
        )
        
        
        return fig1, fig2, fig3

    def print_summary(self, exact_results, validation):
        """Print validation summary"""
        p = self.parameters
        
        print("\n" + "="*60)
        print(f"POSTULATE 1 VALIDATION SUMMARY")
        print("="*60)
        print(f"Parameters: N={p['N']}, ω={p['omega']}, α={p['alpha']}")
        print(f"Probes: Outside ion {p['probeOut']}, Inside ion {p['probeIn']}")
        print("-"*60)
        print(f"Measured ω_out:  {validation['measuredFreqOut']:.6f}")
        print(f"Predicted ω_out: {validation['predictedFreqOut']:.6f}")
        print(f"Measured ω_in:   {validation['measuredFreqIn']:.6f}")
        print(f"Predicted ω_in:  {validation['predictedFreqIn']:.6f}")
        print("-"*60)
        print(f"Postulate 1 Residual: {(validation['residual']*100):.2f}%")
        
        if validation['residual'] < 0.05:
            print("✓ STATUS: Postulate 1 EMERGES from dynamics")
        elif validation['residual'] < 0.10:
            print("~ STATUS: Postulate 1 is within tolerance")
        else:
            print("✗ STATUS: Postulate 1 VIOLATED")
        
        print("="*60 + "\n")

# %%
# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main simulation runner"""
    # Initialize tester with parameters
    tester = QCATester()
    
    # Update parameters if desired
    tester.parameters.update({
        'N': 4,
        'omega': 1.0,
        'alpha': 0.5,
        'deltaB': 5.0,
        'lambda': 0.3,
        'kappa': 0.1,
        'k0': 4,
        'bondCutoff': 4,
        'J0': 0.01,
        'gamma': 0.0,
        'probeOut': 0,
        'probeIn': 1,
        'tMax': 20
    })
    
    # Run exact simulation
    print("Running exact diagonalization...")
    exact_results = tester.run_exact_experiment()
    
    # Run mean-field comparison
    print("Running mean-field comparison...")
    mf_results = tester.run_mean_field_comparison(exact_results)
    
    # Validate Postulate 1
    validation = tester.validate_postulate(exact_results)
    
    # Print summary
    tester.print_summary(exact_results, validation)
    
    # Create plots
    tester.create_plots(exact_results, mf_results, validation)
    
    return exact_results, mf_results, validation

def scan_lambda(alpha=0.8, N=4, lambda_min=0.1, lambda_max=1.5, num_points=15):
    """Systematic scan of lambda parameter with visualization"""
    results = []
    lambda_vals = np.linspace(lambda_min, lambda_max, num_points)
    
    print(f"\nScanning λ from {lambda_min} to {lambda_max} (N={N}, α={alpha})")
    print("-" * 60)
    
    for i, l in enumerate(lambda_vals):
        try:
            print(f"Running λ = {l:.3f} ({i+1}/{num_points})...")
            exact_results, mf_results, validation = run_with_parameters(
                N=N, alpha=alpha, lambda_param=l, tMax=20
            )
            results.append({
                'lambda': l,
                'residual': validation['residual'],
                'omega_in': validation['measuredFreqIn'],
                'omega_out': validation['measuredFreqOut'],
                'predicted_omega_in': validation['predictedFreqIn'],
                'predicted_omega_out': validation['predictedFreqOut'],
                'lambda_in': validation['measuredLambdaIn'],
                'lambda_out': validation['measuredLambdaOut']
            })
        except Exception as e:
            print(f"Failed at λ={l}: {e}")
            break
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'QCA Parameter Sweep: N={N}, α={alpha}', fontsize=16, fontweight='bold')
    
    # Plot 1: Residual vs Lambda
    ax1 = axes[0, 0]
    ax1.plot(df['lambda'], df['residual'] * 100, 'o-', linewidth=2, markersize=8, color='darkblue')
    ax1.axhline(5, color='red', linestyle='--', linewidth=2, label='5% threshold')
    ax1.axhline(10, color='orange', linestyle='--', linewidth=2, label='10% threshold')
    ax1.fill_between(df['lambda'], 0, 5, alpha=0.2, color='green', label='Emergent region')
    ax1.set_xlabel('λ (promotion strength)', fontsize=12)
    ax1.set_ylabel('Postulate 1 Residual (%)', fontsize=12)
    ax1.set_title('Postulate 1 Breakdown', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Frequency vs Lambda
    ax2 = axes[0, 1]
    ax2.plot(df['lambda'], df['omega_out'], 'o-', label='Measured ω_out', linewidth=2, markersize=6)
    ax2.plot(df['lambda'], df['omega_in'], 's-', label='Measured ω_in', linewidth=2, markersize=6)
    ax2.plot(df['lambda'], df['predicted_omega_out'], '--', label='Predicted ω_out', alpha=0.7)
    ax2.plot(df['lambda'], df['predicted_omega_in'], '--', label='Predicted ω_in', alpha=0.7)
    ax2.axhline(1.0, color='black', linestyle=':', label='Bare ω')
    ax2.set_xlabel('λ', fontsize=12)
    ax2.set_ylabel('Effective Frequency', fontsize=12)
    ax2.set_title('Frequency Scaling', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Lambda (circuit depth proxy) vs Lambda (parameter)
    ax3 = axes[1, 0]
    ax3.plot(df['lambda'], df['lambda_out'], 'o-', label='Λ_out', linewidth=2)
    ax3.plot(df['lambda'], df['lambda_in'], 's-', label='Λ_in', linewidth=2)
    ax3.set_xlabel('λ', fontsize=12)
    ax3.set_ylabel('Λ (circuit depth)', fontsize=12)
    ax3.set_title('Entanglement Proxy', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Frequency Ratio
    ax4 = axes[1, 1]
    measured_ratio = df['omega_in'] / df['omega_out']
    predicted_ratio = df['predicted_omega_in'] / df['predicted_omega_out']
    ax4.plot(df['lambda'], measured_ratio, 'o-', label='Measured Ratio', linewidth=2, markersize=6)
    ax4.plot(df['lambda'], predicted_ratio, 's-', label='Predicted Ratio', linewidth=2, markersize=6)
    ax4.axhline(1.0, color='black', linestyle=':', label='No scaling')
    ax4.set_xlabel('λ', fontsize=12)
    ax4.set_ylabel('ω_in / ω_out', fontsize=12)
    ax4.set_title('Frequency Ratio Deviation', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
def analyze_critical_points(df):
    """Find the λ values where phase transitions occur"""
    
    # Phase I → II transition (cross 5% threshold)
    lambda_c1 = df[df['residual'] > 0.05]['lambda'].iloc[0]
    
    # Phase II → III revival (minimum residual)
    lambda_revival = df.loc[df['residual'].idxmin(), 'lambda']
    
    # Phase III → IV (cross 20% threshold after revival)
    post_revival = df[df['lambda'] > lambda_revival]
    lambda_c2 = post_revival[post_revival['residual'] > 0.20]['lambda'].iloc[0]
    
    print(f"Critical Points:")
    print(f"  Scaling breakdown: λ_c1 = {lambda_c1:.3f}")
    print(f"  Quantum revival:   λ_rev = {lambda_revival:.3f}")
    print(f"  Catastrophic fail: λ_c2 = {lambda_c2:.3f}")
    
    return {'lambda_c1': lambda_c1, 'lambda_revival': lambda_revival, 'lambda_c2': lambda_c2}
    #return df

def phase_diagram(N=4, lambda_vals=[0.1, 0.5, 1.0], alpha_vals=[0.2, 0.5, 0.8]):
    """Map the full parameter space"""
    results = []
    for l in lambda_vals:
        for a in alpha_vals:
            try:
                print(f"λ={l}, α={a}")
                _, _, v = run_with_parameters(N=N, lambda_param=l, alpha=a, tMax=20)
                results.append({'lambda': l, 'alpha': a, 'residual': v['residual']})
            except:
                continue
    
    df = pd.DataFrame(results)
    df_pivot = df.pivot(index='lambda', columns='alpha', values='residual') * 100
    
    plt.figure(figsize=(10, 8))
    plt.imshow(df_pivot.values, cmap='RdYlGn_r', aspect='auto', 
               extent=[min(alpha_vals), max(alpha_vals), max(lambda_vals), min(lambda_vals)])
    plt.colorbar(label='Residual (%)')
    plt.xlabel('α (postulate coefficient)')
    plt.ylabel('λ (promotion strength)')
    plt.title('Postulate 1: Phase Diagram')

# %%
# Run the simulation
#if __name__ == "__main__":
    exact_results, mf_results, validation = main()

# %%
# Interactive parameter exploration for Jupyter notebooks
def run_with_parameters(**kwargs):
    """Run simulation with custom parameters"""
    # Handle lambda keyword gracefully
    if 'lambda' in kwargs:
        kwargs['lambda_param'] = kwargs.pop('lambda')
    
    tester = QCATester()
    # Map parameters correctly
    param_map = {
        'N': 'N',
        'alpha': 'alpha',
        'lambda_param': 'lambda',  # maps to the internal parameter name
        'tMax': 'tMax',
        # add other parameters as needed
    }

    for key, value in kwargs.items():
        if key in param_map:
            tester.parameters[param_map[key]] = value
    
    exact_results = tester.run_exact_experiment()
    mf_results = tester.run_mean_field_comparison(exact_results)
    validation = tester.validate_postulate(exact_results)
    
    tester.print_summary(exact_results, validation)
    tester.create_plots(exact_results, mf_results, validation)
    
    return exact_results, mf_results, validation
# %%
