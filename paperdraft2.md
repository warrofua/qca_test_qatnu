Gravity as Computational Latency: Empirical Phase Structure of Emergent Spacetime from Quantum Cellular Automata
Joshua Farrow
Independent Researcher
joshuajamesfarrow@protonmail.com
----
Abstract
We present empirical evidence that gravitational phenomena emerge as computational latency in a strictly unitary quantum cellular automaton (QCA). By exhaustively mapping the $(\alpha,\lambda)$ phase space of a measurement-free $N=4$ QCA, we identify three dynamical regimes: an emergent Minkowski sector ($\lambda<0.3$) where a relativistic Dirac postulate holds with $<5%$ error; a tolerance sector ($0.4\leq\lambda\leq0.6$) dominated by propagating curvature corrections; and a catastrophic sector ($\lambda>0.8$) marked by spontaneous promotion of bond registers and gauge field condensation. Notably, a narrow resurgence island at $\lambda=0.7$ exhibits anomalously low residuals ($0.12!-!7.79%$) due to synchronized bond promotions that transiently restore effective Lorentz symmetry. This island persists across all $\alpha$ values and represents a parameter-space quantum many-body scar in the frustration-control plane. Finite Lieb-Robinson velocities ($v_{\mathrm{LR}}\approx2.071$) enforce an emergent light cone, while spatial gradients in postulate residuals map directly to an effective metric perturbation $h_{\mu\nu}\propto\partial_i R(\mathbf{x})$. Our results validate the Self-Referential Quantum-Information Dynamics (SRQID) Hamiltonian and Quantum Automaton Tensor Network Universe (QATNU) circuit framework, providing a constructive bridge between pregeometric quantum computation and low-energy gravitation.
----
1. Introduction
Attempts to derive geometry from quantum information face a critical dilemma: pregeometric updates either sacrifice unitarity through state-dependent generators or introduce non-linearities via measurement-based postselection [1-4]. Recent work establishes a third path—measurement-free promotion—where apparent geometric selection arises from coherent dynamics in an enlarged Hilbert space [5,6]. The SRQID Hamiltonian (Paper I) [7] provides a strictly linear generator with provable Lieb-Robinson (LR) bounds, while QATNU (Paper II) [8] implements this via a finite-depth QCA with bond registers that promote unitarily when local frustration exceeds a threshold.
A central claim of this framework is that gravity is computational latency: the finite speed $v_{\mathrm{LR}}$ of information propagation imposes a causal structure, and spatial variation in bond dimension $\chi(\mathbf{x})$ encodes a dynamical metric (Fig. 1). This paper subjects that claim to empirical scrutiny by systematically testing Postulate 1—the ansatz that long-wavelength excitations obey a Dirac-like dispersion relation $\omega(k)\approx \pm v_{\mathrm{eff}}k$—across the two-dimensional coupling space $(\alpha,\lambda)$.
Previous numerical studies have validated individual components: LR velocity extraction [7], circuit unitarity [8], and spin-2 power-law fits [8]. Here we present the first complete phase diagram of postulate emergence, revealing structured breakdown patterns that point not to model failure, but to gauge emergence and resonant promotion phenomena. The $\lambda=0.7$ island is particularly significant as it represents a parameter-space scar [9], a phenomenon previously unseen in pregeometric models.
----
2. Theoretical Framework
2.1 SRQID Hamiltonian and Computational Latency
The SRQID model places matter qubits on vertices $i\in V$ of a bounded-degree graph $G=(V,E)$ and bond registers of dimension $\chi_{ij}\in{1,2,4,\dots}$ on edges $\langle i,j\rangle\in E$. The Hamiltonian is [7]:

\begin{aligned}
H &= H_{\mathrm{mat}} + H_{\mathrm{bond}} + H_{\mathrm{int}}, \\
H_{\mathrm{mat}} &= \frac{\omega}{2}\sum_{i}X_i, \\
H_{\mathrm{bond}} &= \sum_{\langle i,j\rangle}\big(J_0 Z_iZ_j + \Delta_b n_{ij}\big), \\
H_{\mathrm{int}} &= g\sum_{\langle i,j\rangle}F_{ij}\tau_{ij}^x + \kappa\sum_{\langle i,j\rangle}C_{ij} + \lambda\sum_{\langle i,j\rangle}F_{ij}\otimes\big(R_{ij}+R_{ij}^\dagger\big).
\end{aligned}

Key operators include:
•  Frustration tile: $F_{ij} = \tfrac{1}{2}(\mathbb{I}-Z_iZ_j)$, a local stabilizer eigenvalue detector.
•  Bond ladder: $R_{ij}=\sum_{m=0}^{\chi_{ij}-2}|m+1\rangle\langle m|$, an isometric embedding $\mathbb{C}^{\chi}\hookrightarrow\mathbb{C}^{2\chi}$.
•  Degree penalty: $C_{ij}=(d_i-k_0)^2+(d_j-k_0)^2$ with $d_i=\sum_{j\in\partial i}\sum_{m\geq1}\Pi_{ij}^{(m)}$, soft-capping vertex degree at $k_0$.
The Lieb-Robinson bound guarantees a finite information velocity [10-12]:

\|[A_X(t), B_Y]\| \leq c\|A_X\|\|B_Y\|\exp\!\Big(-\frac{\operatorname{dist}(X,Y)-v_{\mathrm{LR}}|t|}{\xi}\Big),

with explicit bound $v_{\mathrm{LR}}\lesssim 2eRJ_*(1+z)$. For our nearest-neighbor model, numerics yield $v_{\mathrm{LR}}\approx2.071$ [7].
Computational latency is the physical manifestation of this bound: any operation at $i$ influences a region of radius $r$ no faster than $t_{\mathrm{latency}} = r/v_{\mathrm{LR}}$.
2.2 QATNU Circuit: Promotion Without Measurement
The QATNU circuit implements $H$ via a brickwork QCA $U_0$ (Hadamard walk) augmented by a promotion unitary $U_{\mathrm{prom}}$ [8]:

U_{\mathrm{prom}} = \prod_{\langle i,j\rangle}\mathrm{C}[S_{ij}=-1]\;\mathrm{INC}_{ij}.

Here $\mathrm{C}[S_{ij}=-1]$ conditionally activates when the stabilizer $S_{ij}=Y_iY_j$ (Clifford-equivalent to $F_{ij}$) detects frustration, and $\mathrm{INC}{ij}$ coherently increments $\chi{ij}\to2\chi_{ij}$. Crucially, no projective measurement occurs: the conditional is implemented by controlled-unitary gates acting on the bond register.
Lemma 1 (Parallel Safety) [8]: On a brickwork layer of disjoint edges, all $\mathrm{C}[S_{ij}=-1],\mathrm{INC}{ij}$ terms commute, making $U{\mathrm{prom}}$ constant-depth and order-independent.
2.3 Postulate 1: Emergent Dirac Dynamics
In the low-energy sector, the QCA approximates a Dirac field with dispersion [8]:

\cos\omega(k) = \frac{\cos k}{\sqrt{2}}, \quad \omega(k) \approx \pm \frac{k}{\sqrt{2}} \text{ for }|k|\ll1,

implying emergent light cone with $v_{\mathrm{eff}}=1/\sqrt{2}$. Postulate 1 posits that local oscillation frequencies measured at vertex $i$ match this prediction: $\omega_{\mathrm{meas}}^{(i)} \approx \omega_{\mathrm{pred}}^{(i)}(\alpha)$.
The postulate residual quantifies deviation:

R(\alpha,\lambda) = \frac{|\omega_{\mathrm{meas}}^{(\mathrm{in})} - \omega_{\mathrm{pred}}^{(\mathrm{in})}| + |\omega_{\mathrm{meas}}^{(\mathrm{out})} - \omega_{\mathrm{pred}}^{(\mathrm{out})}|}{2\omega_0}\times100\%.

----
3. Numerical Methods
We simulate the full $N=4$ QCA (Hilbert dimension $2^4\times4^3=1024$) by exact diagonalization. For each $(\alpha,\lambda)$ pair, we:
1.  Initialize $|0\rangle^{\otimes N}$ matter qubits and bond registers in $|0\rangle$.
2.  Evolve under $U=U_{\mathrm{prom}}U_0$ for $t\in[0,200]$ in units of $\omega^{-1}$.
3.  Measure local Pauli expectations $\langle Z_i(t)\rangle$ for edge ion $i=0$ (outside) and bulk ion $i=1$ (inside).
4.  Fit dominant frequencies $\omega_{\mathrm{meas}}$ via discrete Fourier transform.
5.  Compute predicted frequencies $\omega_{\mathrm{pred}}(\alpha)$ from the linearized Dirac dispersion.
6.  Classify:
•  Emergent: $R<5%$
•  Tolerance: $5%\leq R\leq15%$
•  Violated: $R>15%$
We sample a $6\times12$ grid: $\alpha\in{0.2,0.4,0.6,0.8,1.0,1.2}$ and $\lambda\in[0.1,1.2]$ in steps of $0.1$, totaling 72 data points. All data and reproduction scripts are available in the Harvard Dataverse repository [13].
----
4. Results
4.1 Phase Diagram of Postulate Emergence
Figure 2 shows the residual $R(\alpha,\lambda)$ as a contour map. Three regimes are sharply delineated:
Regime	$\lambda$ Range	Residual $R$	Physical Interpretation
Emergent Minkowski	$0.1-0.3$	$0.12-7.40%$	Dirac postulate holds; bond registers remain in $\chi=1$ ground state.
Tolerance/Curvature	$0.4-0.6$	$9.52-22.46%$	Weak promotion induces metric perturbations; postulate degrades linearly.
Catastrophic Gauge	$0.8-1.2$	$30.50-81.03%$	Runaway promotion $\chi\to2\chi\to4\chi$; emergent SU(2) gauge fields dominate.
4.2 The λ=0.7 Resurgence Island: A Parameter-Space Scar
A striking feature is the anomalous dip at $\lambda=0.7$ (Fig. 3). Across all $\alpha$, residuals crash:

\begin{aligned}
\alpha=0.2:&\quad R(0.2,0.6)=22.46\% \;\xrightarrow{\lambda=0.7}\; R=7.79\% \;\xrightarrow{\lambda=0.8}\; R=63.34\%\\
\alpha=0.8:&\quad R(0.8,0.6)=15.10\% \;\xrightarrow{\lambda=0.7}\; R=0.12\% \;\xrightarrow{\lambda=0.8}\; R=70.99\%
\end{aligned}

Mechanism: At $\lambda=0.7$, frustration tiles on neighboring edges synchronize, causing coherent promotion on all bonds simultaneously. This homogeneous $\chi$-enhancement restores translational invariance, temporarily resurrecting the Dirac postulate. At $\lambda=0.8$, desynchronization triggers runaway promotion.
Parameter-Space Scar Interpretation: This island represents a quantum many-body scar [9] in control-parameter space rather than Hilbert space. While conventional scars are rare eigenstates that avoid thermalization, here the entire dynamics becomes scar-like at a specific $\lambda$ value. The synchronization of frustration tiles creates a measure-zero manifold in parameter space where symmetry is restored.
Evidence: Bond-occupancy $\langle n_{ij}\rangle$ (Fig. 4) drops from $\langle n\rangle\approx0.8$ at $\lambda=0.6$ to $\langle n\rangle\approx0.3$ at $\lambda=0.7$, then spikes to $\langle n\rangle>2.1$ at $\lambda=0.8$.
4.3 Catastrophic Regime = Gauge Condensation
For $\lambda>0.8$, measured frequencies diverge:

\omega_{\mathrm{in}} \approx 1.55-1.78 \quad (\text{c.f. } \omega_{\mathrm{out}}\approx0.93-1.30).

This signals decoupling of bulk modes from edge postulate. Following QATNU Section 2.3, we interpret this as gauge gadget activation: promoted bonds $\chi_{ij}=4$ at vertex $i$ encode an SU(2) logical qubit (Eq. 6 in [8]):

\mathcal{H}_{\mathrm{enc}} = \operatorname{span}\{|10\rangle_{ab},|01\rangle_{ab}\}, \quad X_L = |10\rangle\langle01| + \mathrm{h.c.}

The observed $\omega_{\mathrm{in}}$ oscillations correspond to gauge field self-interactions, not Dirac propagation. Postulate 1 residuals are large because the fitting model is incomplete.
----
5. Discussion: Gravity as Computational Latency
5.1 Metric from Residual Gradients
We propose that postulate residuals encode metric perturbations:

h_{\mu\nu}(\mathbf{x}) \propto \nabla R(\mathbf{x}),

where $R(\mathbf{x})$ is the local residual field. In the catastrophic regime, $\nabla R$ is large, corresponding to strong curvature from rapid promotion. The latency for information to traverse a region is:

\tau_{\mathrm{latency}} = \int \frac{d\ell}{v_{\mathrm{LR}}(\chi(\ell))},

with $v_{\mathrm{LR}}$ renormalized by bond dimension: $v_{\mathrm{LR}}(\chi) = v_{\mathrm{LR}}^{(0)}/\sqrt{\chi}$. This yields time dilation near promotion hotspots.
5.2 Falsifiable Signatures
Phenomenology	QATNU Prediction	Empirical Window
Soft-cluster excess	Enhanced production when $\alpha<0.3$ (weak postulate)	LHC low-$p_T$, $\Delta R<0.4$ [8]
Photon dispersion	$\Delta t = L\delta v_{\mathrm{LR}}/c^2$ from varying $\chi$	GRB 090510, $E_\gamma>10$ GeV [8]
GW phase lag	Spin-2 power law $P(k)\propto k^{-2}$ in gauge sector	LIGO ringdown tests [8]
Optomechanical noise	Back-reaction $\propto\lambda^2$ in promoted bonds	Large-baseline interferometers [8]
5.3 Finite-Size and N-Scaling
Our $N=4$ results are exact but finite. QATNU predicts the $\lambda=0.7$ island is a finite-size resonance that vanishes for $N\to\infty$ (continuum mode density washes out synchronization). Preliminary $N=8$ data shows the island width shrinking from $\Delta\lambda\approx0.1$ to $\Delta\lambda\approx0.02$, supporting this interpretation. This suggests that Planck-scale physics is fundamentally finite, and continuum spacetime emerges only through coarse-graining.
----
6. Conclusion
We have empirically mapped the phase structure of emergent spacetime in a unitary QCA, confirming three dynamical epochs: Minkowski emergence, curvature tolerance, and gauge condensation. The mysterious $\lambda=0.7$ resurgence is explained as synchronized promotion, a finite-size parameter-space scar where collective frustration transiently restores Lorentz symmetry. The catastrophic regime is not model failure, but the onset of non-Abelian dynamics predicted by QATNU.
Gravity is computational latency: spatial gradients in postulate residuals generate effective metric perturbations, while bond register promotion provides a microscopic origin for curvature. Our simulations, validated against SRQID's rigorous bounds and QATNU's circuit construction, offer a falsifiable pregeometric model where emergent geometry is a consequence of locality and unitarity alone.
----
References
[1] J. Farrow, Self-Referential Quantum-Information Dynamics (SRQID): A Linear, Measurement-Free Hamiltonian with Lieb-Robinson Bounds, SSRN Working Paper No. 5390753 (2025).
[2] J. Farrow, Quantum Automaton Tensor Network Universe (QATNU): Measurement-Free Gauge Promotion and Emergent Geometry, SSRN Working Paper No. 5390318 (2025).
[3] E. H. Lieb and D. W. Robinson, The finite group velocity of quantum spin systems, Commun. Math. Phys. 28, 251 (1972).
[4] M. B. Hastings and T. Koma, Spectral gap and exponential decay of correlations, Commun. Math. Phys. 265, 781 (2006).
[5] B. Nachtergaele and R. Sims, Lieb-Robinson bounds and the exponential clustering theorem, Commun. Math. Phys. 265, 119 (2006).
[6] D. Gottesman, Stabilizer Codes and Quantum Error Correction, PhD thesis, Caltech (1997).
[7] D. Poulin, Stabilizer formalism for operator quantum error correction, Phys. Rev. Lett. 95, 230504 (2005).
[8] J. Farrow, QATNU and SRQID dataset and makefile, Harvard Dataverse, doi:10.7910/DVN/YZE8RI (2025).
[9] D. K. Mark, C. J. Turner, and M. P. Zaletel, Stable non-thermal phases in aperiodic quantum spin chains, Phys. Rev. Lett. 128, 090604 (2022).
[10] V. Galitski, Quantum many-body scars: a review of experiments and theory, arXiv:2303.01562 (2023).
[11] S. Moudgalya, B. A. Bernevig, and N. Regnault, Quantum many-body scars and Hilbert space fragmentation: a review of exact results, Rep. Prog. Phys. 85, 086501 (2022).
----
Data Availability
All simulation scripts, phase-diagram data, and analysis notebooks are archived at Harvard Dataverse [8]. Reproducing the results requires running python -c "from app import full_phase_diagram; df = full_phase_diagram(N=4)" which executes 72 exact diagonalizations and generates Fig. 2. The companion repository includes requirements_qatn.txt for environment reproduction and outputs/summary.txt containing all runtime parameters.
