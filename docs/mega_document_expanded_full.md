# QATNU & SRQID Research Overview

Results, Preprint Context, and Validation (through Nov 16, 2025)

---

## Abstract

We study a fully unitary, measurement-free quantum cellular automaton (QCA) model in which local ``matter'' clocks are coupled to a network of entanglement-carrying bond registers via frustration-controlled promotion isometries. This framework, which we refer to as Self-Referential Quantum Information Dynamics (SRQID) and its Quantum Automaton Tensor-Network Universe (QATNU) realization, has previously been shown to satisfy Lieb--Robinson bounds and no-signalling while exhibiting a rich phase structure in the promotion strength $\lambda$. In this work we focus on three aspects: (i) the microscopic origin and numerical measurement of an effective clock susceptibility $\alpha$ that relates local entanglement depth to clock slowdown, (ii) the dependence of the emergent phase diagram on the pair $(\lambda,\alpha)$, and (iii) the extent to which macroscopic scales and gravitational phenomenology can be consistently anchored to the model.

Using small-$N$ exact diagonalization, we show that the ratio of effective clock frequencies between an interior and exterior probe is well-approximated by a constitutive law $\omega_{\text{eff}} = \omega/(1+\alpha\Lambda)$, where $\Lambda$ is an entanglement proxy derived from bond occupation statistics, and $\alpha$ plays the role of a linear-response susceptibility. We derive $\alpha$ perturbatively from the SRQID Hamiltonian and define it non-perturbatively as a zero-frequency slope that can be extracted from numerical data. A sweep over $\alpha$ on an $N=4$ path reveals three distinct dynamical regimes---outer-dominated, balanced susceptibility, and over-promotion---with a broad $O(1)$ window around $\alpha\sim 0.5$--$0.9$ that yields a robust Minkowski-like sector flanked by curvature-dominated and catastrophic phases. A comparison between $N=4$ and $N=5$ chains confirms the qualitative picture and suggests improved self-consistency at larger $N$.

We then impose two phenomenological anchors: (i) a low-$\lambda$ vacuum in which massless excitations propagate at the observed speed of light $c$, and (ii) a catastrophic promotion regime that obeys the Bekenstein--Hawking area law for saturated clusters. These conditions fix the lattice spacing $a$ and time step $\tau$ to be Planckian up to $O(1)$ factors determined by the bond entropy and QCA group velocity. The resulting dispersion relation exhibits quadratically suppressed Lorentz violation with a dimensionless coefficient $\eta\sim 10^{-2}$, and yields a Hawking-like temperature of order the Planck temperature for Planck-sized catastrophic clusters. We also show that vacuum bond promotions could only account for the observed dark-energy density if the average promotion density in the deep vacuum is of order $10^{-120}$, which we present as a sharp constraint rather than a solved problem.

We emphasize the limitations of the present framework. The preferred lattice frame implies exact Lorentz invariance is broken in the ultraviolet, albeit in a controlled way; the clock slowdown relation holds at the $\sim 10\%$ level for small $N$ and has not been proven as a universal identity; and current attempts to extract a long-wavelength spin-2 sector from bond fluctuations yield power spectra far flatter than the $1/k^2$ behavior required for linearized gravity. As such, SRQID/QATNU should be regarded as a concrete, falsifiable testbed for emergent gravitational phenomena in unitary QCAs, not as a complete theory of quantum gravity.

---

## Main text (from `docs/core_text.tex`)

```tex
\section{Introduction}

Attempts to reconcile quantum mechanics with gravitation have historically pursued two broad strategies. The first quantizes a pre-existing geometric manifold, as in canonical quantum gravity and string theory. The second seeks to derive spacetime geometry and gravitational dynamics from more primitive, non-geometric degrees of freedom, such as entanglement structure, tensor networks, or quantum cellular automata. The latter approach is attractive because it offers the possibility of a fully unitary, background-free description in which geometry is emergent rather than fundamental.\cite{SchumacherWerner2004,Arrighi2019,GNVW2012}

In this work we study a particular realization of the second strategy: a measurement-free quantum cellular automaton (QCA) equipped with a self-referential promotion mechanism that dynamically couples local ``matter'' clocks to an entanglement-carrying network of bond registers. The framework, previously introduced under the name Self-Referential Quantum Information Dynamics (SRQID) and its Quantum Automaton Tensor-Network Universe (QATNU) instantiation, is defined by an explicit Hamiltonian on a graph and has been shown to obey Lieb--Robinson bounds and no-signalling while exhibiting a rich phase structure as a function of a promotion strength $\lambda$.\cite{FarrowQATNU2025,FarrowSRQID2025}

The present paper has three goals. First, we sharpen the connection between a phenomenological ``clock slowdown'' postulate and the underlying SRQID Hamiltonian by deriving and measuring an effective susceptibility parameter $\alpha$ that relates local entanglement depth to clock renormalization. Second, we study how the phase diagram in $\lambda$ depends on $\alpha$, revealing three distinct dynamical regimes and a broad $O(1)$ window in which the model exhibits a robust Minkowski-like sector with curvature and collapse phases on either side. Third, we explore to what extent macroscopic scales---the Planck length, Planck time, speed of light, black-hole entropy, and dark-energy density---can be consistently anchored to this discrete, information-theoretic model.

At the same time, we are explicit about the limitations of the current framework. The fixed lattice spacing and local update rule pick out a preferred frame and imply that Lorentz invariance is approximate and energy-dependent rather than exact; the effective clock slowdown relation is numerically supported but not exact; and we do not yet obtain a clean, long-wavelength spin-2 sector with the characteristic $1/k^2$ power spectrum of linearized gravity. For these reasons we present SRQID/QATNU not as a completed theory of quantum gravity but as a concrete testbed that makes nontrivial, falsifiable predictions and exposes where key open problems reside.

In summary, on small graphs we find a quantitatively usable clock-slowdown relation and a structured phase diagram in $(\lambda,\alpha)$, and we show how the model can be consistently anchored to macroscopic scales (Planckian $a,\tau$) while yielding falsifiable constraints on vacuum promotion statistics.

\section{Self-Referential Quantum Information Dynamics}

\subsection{Hilbert space and Hamiltonian}

We consider a finite graph with vertices $i\in V$ and edges $\langle i,j\rangle\in E$. Each vertex hosts a two-level ``matter'' system (a qubit), and each edge hosts a multi-level bond register that tracks the local entanglement structure. The total Hilbert space factorizes as
\begin{equation}
\mathcal{H} = \bigotimes_{i\in V} \mathcal{H}_i^{\text{mat}} \;\otimes\; \bigotimes_{\langle i,j\rangle\in E} \mathcal{H}_{ij}^{\text{bond}},
\end{equation}
with $\dim \mathcal{H}_i^{\text{mat}} = 2$ and $\dim \mathcal{H}_{ij}^{\text{bond}} = \chi_{\max}$ for some finite ladder dimension $\chi_{\max}$.

The SRQID Hamiltonian is written as
\begin{equation}
H = H_{\text{mat}} + H_{\text{bond}} + H_{\text{int}}.
\end{equation}
The matter term is a transverse-field-like driver,
\begin{equation}
H_{\text{mat}} = \frac{\omega}{2}\sum_{i\in V} X_i,
\end{equation}
where $X_i$ is the Pauli-$X$ operator on qubit $i$ and $\omega$ sets the bare clock frequency. The bond term $H_{\text{bond}}$ is diagonal in the bond basis and assigns an energy gap between successive rungs of the ladder,
\begin{equation}
H_{\text{bond}} = \sum_{\langle i,j\rangle} \sum_{m=0}^{\chi_{\max}-1} \epsilon_m\, \Pi_{ij}^{(m)},
\end{equation}
where $\Pi_{ij}^{(m)} = |m\rangle\langle m|_{ij}$ projects onto rung $m$ of the bond register.

The interaction term $H_{\text{int}}$ couples matter frustration to bond promotion:
\begin{equation}
H_{\text{int}} = \lambda \sum_{\langle i,j\rangle} F_{ij} \otimes \left(R_{ij} + R_{ij}^\dagger\right).
\label{eq:Hint}
\end{equation}
Here $F_{ij}$ is a frustration tile,
\begin{equation}
F_{ij} = \frac{1}{2}(\mathbb{I} - Z_i Z_j),
\end{equation}
where $Z_i$ is Pauli-$Z$ on site $i$, so that $F_{ij}$ projects onto configurations with $Z_iZ_j=-1$ (frustrated bonds). The ladder operator
\begin{equation}
R_{ij} = \sum_{m=0}^{\chi_{\max}-2} \sqrt{m+1}\,|m+1\rangle\langle m|_{ij}
\end{equation}
(plus its Hermitian conjugate) promotes or demotes the bond register, with matrix elements chosen to mimic an isometric embedding. The dimensionless parameter $\lambda$ controls the strength of promotion relative to the bare clock term.

\subsection{Locality, unitarity, and causality}

Because $H$ is a finite-range sum of bounded terms on a lattice graph, the time-evolution operator\cite{lieb-robinson,hastings-koma}
\begin{equation}
U(t) = e^{-iHt}
\end{equation}
is unitary and generates dynamics with a finite Lieb--Robinson velocity $v_{\text{LR}}$. Specifically, there exist constants $C,\xi,v_{\text{LR}}$ such that for any operators $A_X$ supported on region $X$ and $B_Y$ supported on region $Y$,
\begin{equation}
\big\|[A_X(t),B_Y]\big\| \le C\,e^{-(\mathrm{dist}(X,Y)-v_{\text{LR}}t)/\xi}.
\end{equation}
Numerical estimates on small graphs yield $v_{\text{LR}}$ of order unity in lattice units, with no-signalling bounds comfortably satisfied for the timescales and sizes we simulate. The model is linear, norm-preserving, and contains no projective measurements or ad hoc collapse processes.

\section{QATNU Promotion and Local Clocks}

\subsection{Entanglement proxy and test clocks}

The QATNU realization of SRQID interprets the bond ladder occupation as an information-theoretic proxy for local entanglement depth. For each site $i$, we define an entanglement proxy
\begin{equation}
\Lambda_i = \sum_{j\in\partial i} \log\big(1 + \langle n_{ij}\rangle\big),
\label{eq:Lambda-def}
\end{equation}
where $\partial i$ denotes the neighbors of $i$ and
\begin{equation}
n_{ij} = \sum_{m=0}^{\chi_{\max}-1} m\,\Pi_{ij}^{(m)}
\end{equation}
is the rung number operator on edge $\langle i,j\rangle$. The logarithm in~\eqref{eq:Lambda-def} is a pragmatic compression choice: it enforces diminishing returns at high occupation, stabilizes fits across regimes by avoiding a linear blow-up of the proxy when promotions become dense, and still reduces to a simple linear neighbor-sum in the dilute limit.
In the dilute regime $\langle n_{ij}\rangle\ll 1$, the logarithm can be linearized and $\Lambda_i \approx \sum_{j\in\partial i} \langle n_{ij}\rangle$.
More generally, alternative proxies (e.g.\ $\Lambda_i^{\mathrm{lin}}\equiv\sum_{j\in\partial i}\langle n_{ij}\rangle$, or coarse-grained variants) can be used; testing the robustness of the slowdown fit under such redefinitions is an important follow-up, and we expect the main qualitative behavior to persist with a corresponding renormalization of the best-fit susceptibility $\alpha$.

To probe how $\Lambda_i$ feeds back into matter dynamics, we introduce a ``test clock'' protocol on a small graph. Two sites are designated as probes: an outer site $i_{\text{out}}$ with relatively low $\Lambda_{i_{\text{out}}}$ and an inner site $i_{\text{in}}$ near the center of the promotion cluster with higher $\Lambda_{i_{\text{in}}}$. By preparing suitable initial states and tracking $\langle X_i(t)\rangle$ and $\langle Y_i(t)\rangle$ for each probe over time, we extract an effective local clock frequency $\omega_{\text{eff}}^{(i)}$ from the oscillatory behavior.

\subsection{An effective clock slowdown law}

Numerically we observe that the ratio of effective frequencies between inner and outer probes is reasonably well captured by a single-parameter family
\begin{equation}
\frac{\omega_{\text{eff}}^{(i)}}{\omega} \approx \frac{1}{1+\alpha\Lambda_i},
\label{eq:clock-slowdown}
\end{equation}
for a wide range of promotion strengths $\lambda$ prior to catastrophic behavior. We refer to this as an effective ``clock slowdown'' law: local entanglement depth slows down the effective clock according to a susceptibility $\alpha$.

Equation~\eqref{eq:clock-slowdown} is not exact. For $N=4$ and $N=5$ chains, and a fixed choice of $\alpha$ discussed below, the best-fit form exhibits minimum residuals of order $10$--$15\%$ in the revival region (here ``residual'' denotes a normalized $\ell_2$ fit error over the analysis time window), with increasing deviations as one approaches the catastrophic phase where promotion is dense and highly correlated. Nonetheless, the quality of the fit and the smoothness of $\omega_{\text{eff}}$ as a function of $\Lambda$ motivate treating~\eqref{eq:clock-slowdown} as an effective constitutive relation valid in a particular regime.

\section{Microscopic Origin and Measurement of \texorpdfstring{$\alpha$}{alpha}}

\subsection{Perturbative derivation}

We first show that the structure~\eqref{eq:clock-slowdown} arises naturally from the SRQID Hamiltonian in the dilute promotion regime. Treating $H_{\text{int}}$ in~\eqref{eq:Hint} as a perturbation to $H_0 = H_{\text{mat}} + H_{\text{bond}}$, we write the matter Green's function at site $i$ as
\begin{equation}
G_i(\nu) = \frac{1}{\nu - \omega - \Sigma_i(\nu)},
\end{equation}
where $\Sigma_i(\nu)$ is the matter self-energy due to virtual bond promotions. To second order in $\lambda$ one obtains schematically
\begin{equation}
\Sigma_i(\nu) \approx \sum_{j\in\partial i} \frac{2\lambda^2\,\langle F_{ij}^2\rangle\,\langle R_{ij}R_{ij}^\dagger\rangle}{\nu - \Delta_{ij} + i0^+},
\end{equation}
where $\Delta_{ij}$ is an effective bond excitation gap and the expectation values are taken over the instantaneous bond state in the dilute regime. At zero frequency and for $\Delta_{ij}\gg\lambda$,
\begin{equation}
\Sigma_i(0) \approx -\frac{2\lambda^2}{\Delta_{\text{eff}}}\sum_{j\in\partial i} \langle F_{ij}\rangle\,\langle n_{ij}\rangle,
\end{equation}
using $F_{ij}^2=F_{ij}$ and $\langle R_{ij}R_{ij}^\dagger\rangle \propto \langle n_{ij}\rangle$.

The real part of $\Sigma_i(0)$ shifts the bare frequency $\omega\to \omega_{\text{eff}}^{(i)} = \omega + \Sigma_i(0)$, so that
\begin{equation}
\omega_{\text{eff}}^{(i)} \approx \omega\left(1 - \frac{2\lambda^2}{\omega^2\Delta_{\text{eff}}}
\frac{\sum_{j\in\partial i} \langle F_{ij}\rangle\,\langle n_{ij}\rangle}{\sum_{j\in\partial i}\langle n_{ij}\rangle} \Lambda_i\right),
\end{equation}
where we have introduced $\Lambda_i$ from~\eqref{eq:Lambda-def} and absorbed constants. Matching to a first-order expansion of~\eqref{eq:clock-slowdown},
\begin{equation}
\omega_{\text{eff}}^{(i)} \approx \omega(1 - \alpha\Lambda_i),
\end{equation}
yields a microscopic expression
\begin{equation}
\alpha_{\text{pert}}(\lambda) \approx \frac{2\lambda^2}{\omega^2\Delta_{\text{eff}}}
\frac{\sum_{j\in\partial i} \langle F_{ij}\rangle\,\langle n_{ij}\rangle}{\sum_{j\in\partial i}\langle n_{ij}\rangle}.
\label{eq:alpha-pert}
\end{equation}
In particular, in a homogeneous dilute regime one expects $\alpha\propto \lambda^2$ with an $O(10^{-2}\text{--}10^{-1})$ prefactor set by the frustration tile and ladder gap. This perturbative result is valid for small $\lambda$ and low bond occupation; it breaks down in the revival and catastrophic regimes.

\subsection{Non-perturbative susceptibility}

Beyond the dilute regime we define $\alpha$ non-perturbatively as a static susceptibility,
\begin{equation}
\alpha(\lambda) \equiv -\frac{\partial \log\big(\omega_{\text{eff}}^{(i)}/\omega\big)}{\partial \Lambda_i}.
\label{eq:alpha-suscept}
\end{equation}
Operationally, for a fixed $\lambda$ we prepare a family of initial conditions that differ slightly in their bond promotion patterns, measure $\omega_{\text{eff}}^{(i)}$ and $\Lambda_i$ for each, and estimate the derivative in~\eqref{eq:alpha-suscept} by finite differences. This definition coincides with~\eqref{eq:alpha-pert} in the appropriate limit and extends it into the many-body regime where virtual and real promotions intertwine.

Formally, $\alpha$ can also be expressed via a Kubo formula,
\begin{equation}
\alpha = \frac{1}{\omega}\int_0^\infty dt\,\langle [X_i(t),O_{\Lambda_i}(0)]\rangle,
\end{equation}
where $O_{\Lambda_i}$ is the operator whose expectation value yields $\Lambda_i$. We do not exploit this expression directly in the present work, but it underlines the interpretation of $\alpha$ as a linear-response susceptibility rather than a freely tunable parameter.

\section{Phase Structure in \texorpdfstring{$(\lambda,\alpha)$}{(lambda,alpha)}}

We emphasize that $\alpha$ is not introduced as an independent Hamiltonian coupling: it is an effective susceptibility extracted from the dynamics (Eq.~\eqref{eq:alpha-suscept}) and therefore can shift under changes in microscopic gaps/cutoffs, graph topology, and the family of initial conditions used to probe response. In the present numerics, $\alpha$ is extracted from families of initial conditions and is therefore not an independent Hamiltonian coupling; the ``$\alpha$-sweep'' should be read as a reparameterization of families of effective responses rather than parameter laundering.

\subsection{Definitions of critical scales}

To characterize the dynamical regimes of the SRQID/QATNU model we consider two small graphs: an $N=4$ path and an $N=5$ path with nearest-neighbor edges. For each choice of susceptibility $\alpha$ and promotion strength $\lambda$ we perform exact diagonalization and evaluate the inner and outer probe clocks over a fixed time window. We fit the probe behavior to an effective Dirac dispersion and extract several characteristic promotion strengths:
\begin{itemize}
    \item $\lambda_{c1}$: the onset of curvature, defined as the smallest $\lambda$ for which the inner probe residual exceeds a threshold relative to the outer probe;
    \item $\lambda_{\text{rev}}$: the ``revival'' point at which the Dirac-like behavior is best restored after degradation, corresponding to a minimum in the residual;
    \item $\lambda_{c2}$: the onset of catastrophic behavior, where the residual grows rapidly and bond promotions saturate.
\end{itemize}
We also record the minimum residual in the revival region, denoted $\text{Residual}_{\min}$.

\subsection{Three regimes in \texorpdfstring{$\alpha$}{alpha} for \texorpdfstring{$N=4$}{N=4}}
\label{sec:revival-phase}

For the $N=4$ path we sweep $\alpha$ in a range $0.1\le\alpha\le 1.5$ and, for each $\alpha$, scan $\lambda$ on a fine grid to extract $\lambda_{c1},\lambda_{\text{rev}},\lambda_{c2}$ and $\text{Residual}_{\min}$. The dependence of these scales on $\alpha$ reveals three qualitatively distinct regimes:
\begin{enumerate}
    \item \emph{Outer-dominated regime} ($\alpha\lesssim 0.4$): $\lambda_{\text{rev}}$ and $\lambda_{c2}$ remain pinned near their legacy values $\lambda_{\text{rev}}\approx 1.31$ and $\lambda_{c2}\approx 1.36$ over a wide range of $\alpha$, while $\lambda_{c1}$ shifts slowly. In this regime the inner probe is only weakly renormalized; the dynamics effectively reduces to a single undressed clock set by the outer probe.
    \item \emph{Balanced susceptibility regime} ($0.5\lesssim \alpha\lesssim 0.9$): both $\lambda_{\text{rev}}$ and $\lambda_{c2}$ drop to lower, nearly $\alpha$-independent plateaus $\lambda_{\text{rev}}\approx 1.03$ and $\lambda_{c2}\approx 1.08$. Here the interior bonds efficiently carry frustration inward, and both inner and outer probes experience a common self-consistent geometry governed by the clock slowdown relation~\eqref{eq:clock-slowdown}. The minimum residual increases approximately linearly with $\alpha$ in this band, consistent with stronger backreaction of entanglement on the clock.
    \item \emph{Over-promotion regime} ($\alpha\gtrsim 1.0$): as $\alpha$ is increased further, the system first relocks onto the legacy plateau and then, for $\alpha\gtrsim 1.4$, all three critical scales collapse into a narrow band at small $\lambda$, with $\lambda_{\text{rev}}$ and $\lambda_{c2}$ entering the range $\lambda\sim 0.2$--$0.5$. In this regime the ladder promotion overwhelms the transverse field; even weak promotion produces early catastrophic behavior and suppresses any extended Dirac-like window.
\end{enumerate}

The $N=4$ $\alpha$-sweep thus shows that $\alpha$ is not a cosmetic parameter. It reorganizes the entire phase structure in $\lambda$, with a broad $O(1)$ band where the self-referential feedback between entanglement and clock rate is most active.

\subsection{Finite-size trends: \texorpdfstring{$N=4$}{N=4} vs \texorpdfstring{$N=5$}{N=5}}

For a representative value $\alpha=0.8$ in the balanced regime we perform the same analysis on an $N=5$ path. Table~\ref{tab:N4N5} summarizes the critical scales and minimum residuals.

\begin{table}[t]
\centering
\caption{Critical promotion strengths and minimum residuals for $N=4$ and $N=5$ paths at fixed $\alpha=0.8$. Values are indicative and extracted from small-$N$ exact diagonalization.}
\label{tab:N4N5}
\vspace{0.5em}
\begin{tabular}{lcccc}
\toprule
System & $\lambda_{c1}$ & $\lambda_{\text{rev}}$ & $\lambda_{c2}$ & $\text{Residual}_{\min}$ \\
\midrule
$N=4$ path & $0.203(1)$ & $1.058(2)$ & $1.095(3)$ & $0.134$ \\
$N=5$ path & $0.232(2)$ & $0.338(1)$ & $1.033(5)$ & $0.102$ \\
\bottomrule
\end{tabular}
\end{table}

Two trends are noteworthy. First, the revival point $\lambda_{\text{rev}}$ drops substantially when moving from $N=4$ to $N=5$, reflecting the fact that the additional interior bonds can carry frustration and promotions inward more efficiently. In both cases we use the same residual definition and analysis window, but the probe placement and available interior structure necessarily change with $N$, so the absolute value of $\lambda_{\text{rev}}$ should be interpreted as a protocol-dependent finite-size diagnostic pending larger-$N$ study.
As a simple normalization, the ratios $\lambda_{\text{rev}}/\lambda_{c2}$ are $\approx 0.97$ for $N=4$ and $\approx 0.33$ for $N=5$, highlighting that the shift is largely a reorganization of the pre-catastrophic window rather than an ambiguity in the definition of catastrophe.
Second, the minimum residual shrinks for $N=5$, suggesting that the effective clock slowdown relation becomes more self-consistent as the system size grows, at least within the small-$N$ regime accessible to exact diagonalization.

\section{Scale Setting from Vacuum and Black Holes}

\subsection{Anchoring strategy}

The SRQID/QATNU model is defined in dimensionless units: the lattice spacing $a$, time step $\tau$, and energy scale are not fixed by the dynamics alone. To connect to physical units we impose two macroscopic anchors:
\begin{enumerate}
    \item \emph{Vacuum anchor}: the low-$\lambda$ regime with negligible bond promotion and $\Lambda_i\to 0$ is identified with deep intergalactic vacuum, and the group velocity of massless excitations in this regime is set equal to the physical speed of light $c$.
    \item \emph{Black-hole anchor}: the catastrophic promotion regime, in which a cluster of bonds near saturation forms a nearly static ``horizon'', is required to obey the Bekenstein--Hawking area law for entropy.
\end{enumerate}
These two conditions determine the lattice spacing $a$ and time step $\tau$ in terms of the Planck length $\ell_P$ and Planck time $t_P$ up to $O(1)$ factors. The point is consistency, not derivation: the same $(a,\tau)$ satisfy both anchors without internal contradiction and thereby fix a definite scaling for the Lorentz-violation coefficient $\eta$.

\subsection{Horizon entropy and lattice spacing}

Let $N_\partial$ denote the number of boundary bonds of a maximally promoted cluster in the catastrophic regime. In the QATNU picture each bond contributes an entanglement entropy
\begin{equation}
S_{\text{bond}} = \gamma\,k_B,
\end{equation}
where $\gamma = \ln \chi_{\max}$ captures the effective bond Hilbert-space dimension at saturation. The total horizon entropy from bonds is then
\begin{equation}
S_{\text{QATNU}} = \gamma\,k_B N_\partial.
\end{equation}

On the other hand, the Bekenstein--Hawking entropy for a horizon of area $A$ is
\begin{equation}
S_{\text{BH}} = \frac{k_B A}{4\ell_P^2}.
\end{equation}
Assuming that $A$ is related to $N_\partial$ by a geometric factor $\beta\sim O(1)$,
\begin{equation}
A = \beta N_\partial a^2,
\end{equation}
and equating $S_{\text{QATNU}} = S_{\text{BH}}$ yields
\begin{equation}
\gamma\,k_B N_\partial = \frac{k_B}{4\ell_P^2}\,\beta N_\partial a^2
\quad\Rightarrow\quad
a^2 = 4\frac{\gamma}{\beta}\,\ell_P^2.
\end{equation}
Thus
\begin{equation}
a = 2\sqrt{\frac{\gamma}{\beta}}\,\ell_P,
\end{equation}
so that the lattice spacing is Planckian up to $O(1)$ factors determined by the bond entropy and geometry.

\subsection{Vacuum dispersion and time step}

In the low-$\lambda$ vacuum, the unpromoted QCA layer exhibits a Dirac-like dispersion\cite{Strauch2006,NayakVishwanath2000,Bisio2015,Raynal2017}
\begin{equation}
\omega(k) \approx c_0 k \quad (k\to 0),
\end{equation}
with $c_0$ the group velocity in lattice units (sites per tick). Numerical estimates yield $c_0 = 1/\sqrt{2}$ for the QATNU update rule. The corresponding physical group velocity is
\begin{equation}
v_{\text{phys}} = c_0 \frac{a}{\tau},
\end{equation}
and the vacuum anchor requires $v_{\text{phys}} = c$, the observed speed of light. This fixes
\begin{equation}
\tau = \frac{c_0 a}{c} = 2 c_0\sqrt{\frac{\gamma}{\beta}}\,t_P,
\end{equation}
so that the time step is Planckian up to the same $O(1)$ factors as the lattice spacing.

\subsection{UV energy scale and Hawking temperature}

The natural ultraviolet energy scale associated with one QCA tick is
\begin{equation}
E_{\text{UV}} \sim \frac{\hbar}{\tau} = \frac{\hbar c}{c_0 a} = \frac{1}{2c_0\sqrt{\gamma/\beta}}\,E_P,
\end{equation}
with $E_P$ the Planck energy. Thus $E_{\text{UV}}$ is $O(1)$ times the Planck scale.

Interpreting the catastrophic promotion regime as a Planckian black-hole horizon, we can estimate an effective Hawking temperature from the promotion rate. Let $\Delta\sim \hbar/\tau$ be a characteristic bond excitation gap and
\begin{equation}
\Gamma_{\text{prom}} \sim \lambda\,\langle F_{ij}\rangle_{\text{BH}}\,\frac{\Delta}{\hbar}
\end{equation}
be the promotion rate per bond near the horizon, with $\langle F_{ij}\rangle_{\text{BH}}\sim O(1)$ in a maximally frustrated state. If we associate a temperature via
\begin{equation}
k_B T_H \sim \frac{\hbar \Gamma_{\text{prom}}}{2\pi},
\end{equation}
we obtain
\begin{equation}
T_H \sim \frac{\lambda}{2\pi}\,\frac{\hbar}{k_B\tau} \sim \frac{\lambda}{2\pi}\,\frac{1}{2c_0\sqrt{\gamma/\beta}}\,T_P,
\end{equation}
where $T_P$ is the Planck temperature. For $\lambda\sim O(1)$ this yields a Hawking-like temperature of order $T_P$ for Planck-scale catastrophic clusters, consistent with expectations for microscopic black holes.

\section{Phenomenological Consequences}

\subsection{Lorentz violation and dispersion}

The unpromoted QCA layer is defined on a fixed lattice with a discrete time step, so exact Lorentz invariance cannot hold at all scales. Expanding the lattice dispersion for small $k$,
\begin{equation}
\omega(k) = c_0 k + \frac{c_0}{12}k^3 + O(k^5),
\end{equation}
and mapping to physical units via $p = \hbar k / a$ and $E = \hbar\omega/\tau$ using the Planckian $a,\tau$ obtained above, we find schematically
\begin{equation}
E^2 \approx p^2c^2\left[1 + \eta\,\frac{p^2}{E_P^2} + O\!\left(\frac{p^4}{E_P^4}\right)\right],
\end{equation}
with a dimensionless coefficient $\eta\sim O(10^{-2})$ set by the cubic term and the $O(1)$ factors in $a,\tau$. This estimate is order-of-magnitude only and depends on $c_0$, $\gamma$, and $\beta$.
This corresponds to quadratically suppressed Lorentz violation,
\begin{equation}
\frac{\Delta v}{c} \sim \eta \left(\frac{E}{E_P}\right)^2,
\end{equation}
which is completely negligible at currently accessible energies but in principle testable with future ultra-high-energy photons or cosmic rays.

\subsection{Gravitational redshift and effective potential}
\label{sec:grav-redshift}

The effective clock slowdown relation~\eqref{eq:clock-slowdown} implies a proper time rate at site $i$,
\begin{equation}
\frac{d\tau_i}{dt} = \frac{\omega_{\text{eff}}^{(i)}}{\omega} = \frac{1}{1+\alpha\Lambda_i} \approx 1 - \alpha\Lambda_i,
\end{equation}
for $\alpha\Lambda_i\ll 1$. In a weak Schwarzschild gravitational field one has
\begin{equation}
\frac{d\tau}{dt} \approx 1 - \frac{GM}{rc^2},
\end{equation}
so a formal matching at linear order suggests
\begin{equation}
\frac{GM}{rc^2} \equiv \alpha\Lambda_i,
\end{equation}
or equivalently an effective Newtonian potential
\begin{equation}
\phi_i \equiv -\frac{GM}{r} \sim -\alpha c^2 \Lambda_i.
\end{equation}
We emphasize that this identification is phenomenological and holds only in a weak-field regime. Nonetheless it illustrates how local entanglement depth in QATNU could serve as a dimensionless proxy for gravitational potential, with $\alpha$ playing the role of a coupling between information geometry and clock rate.

\subsection{Vacuum promotions and dark energy}

In Planck units the energy density associated with bond promotions in the vacuum can be estimated as
\begin{equation}
\rho_\Lambda \sim E_P^4\,\langle n_{\text{vac}}\rangle,
\end{equation}
where $\langle n_{\text{vac}}\rangle$ is the average promotion density in the deep vacuum and we have used $E_{\text{UV}}\sim E_P$ and $a^3\sim \ell_P^3$ up to $O(1)$ factors. The observed dark-energy density is $\rho_{\Lambda,\text{obs}}\sim 10^{-120}E_P^4$, so for vacuum promotions to account for dark energy one must have
\begin{equation}
\langle n_{\text{vac}}\rangle \sim 10^{-120}.
\end{equation}
This is a sharp constraint, not a solution: it reframes the cosmological constant problem as a statement about the tails of the vacuum promotion distribution. Unless the SRQID promotion dynamics naturally produces an ultra-rare tail of vacuum activity with this suppression, dark energy cannot be explained in this way. Measuring the small-$\lambda$ statistics of $n_{\text{vac}}$ (e.g.\ whether it falls off as a power law, exponential, or super-exponential in $\lambda$) thus becomes a decisive test for this particular cosmological hypothesis within the model.

While Sec.~7.3 addressed vacuum (source-free) constraints, the presence of matter requires a field equation for $\Lambda$. We hypothesize that the scalar (Newtonian) response is least screened in the revival regime.

\subsection{Scalar (Newtonian) sector: phase--gravity hypothesis (preview)}
A key next test is whether the long-wavelength static response of the entanglement proxy $\Lambda$ is governed by an unscreened graph Poisson equation,
\begin{equation}
(\Delta_{\mathrm{graph}}\Lambda)_i = -\kappa \,\rho_{\mathrm{ent}}(i),
\end{equation}
or by a screened operator $(\Delta_{\mathrm{graph}}-\mu^2)\Lambda\sim-\rho_{\mathrm{ent}}$ (Yukawa-like).\cite{chung-spectral,yukawa}
A concrete phase--gravity hypothesis is that the revival regime (near $\lambda\sim\lambda_{\mathrm{rev}}$) is where screening is weakest ($\mu\approx 0$), while the outer-dominated and catastrophic regimes exhibit $\mu\ne 0$.
Appendix~D outlines a frozen-matter static protocol and two complementary extractions (regression vs. far-field tail) for estimating $\mu(\lambda)$ and $\kappa$ and closing the loop via $4\pi G_{\mathrm{eff}}=\alpha c^2\kappa/a^2$.

\section{Spin-2 Sector and Open Problems}

A central requirement for any candidate theory of gravity is the emergence of a long-wavelength, approximately massless spin-2 field with a characteristic $1/k^2$ power spectrum in momentum space. In the QATNU/SRQID context, a natural idea is to construct an effective metric field from bond promotion patterns or entanglement profiles and to study the power spectral density (PSD) of its fluctuations.

Preliminary attempts along these lines have focused on simple proxies, such as the spatial profile of bond entropies or ladder occupations on small graphs, and their Fourier transforms. We refer to this diagnostic as a \emph{bond-proxy PSD} (or metric-proxy PSD): it is not yet a spin-2 observable, but a first check of whether any long-wavelength structure is present in simple geometry-related proxies. For $N=4$ and $N=5$ paths, the PSDs of these proxies exhibit slopes close to zero in log--log plots (e.g.\ $-0.024$ and $-0.111$, respectively), far from the $-2$ slope expected for a free spin-2 field. Earlier work based on a stochastic toy model with uncorrelated Poisson promotions did yield a $1/k^2$-like tail, but that model neglected the correlated, frustration-tied nature of promotions present in the full Hamiltonian.

At present, therefore, the QATNU/SRQID framework has not produced a convincing spin-2 sector. It does exhibit local gravitational redshift phenomenology and a rich phase structure in $(\lambda,\alpha)$, but the reconstruction of a coarse-grained metric field with the correct long-wavelength dynamics remains an open problem. We anticipate that progress will require:
\begin{itemize}
    \item working on larger graphs and more realistic topologies beyond small paths;
    \item defining better geometric observables, possibly via tensor-network-inspired coarse-graining of bond states;
    \item and exploring whether correlated promotion clusters can, after suitable averaging, yield an effective $1/k^2$ spectrum.
\end{itemize}

\section{Assessment and Limitations}

We briefly summarize the main strengths and limitations of the present framework.

\paragraph{Unitarity and causality.} The SRQID Hamiltonian is strictly local and generates unitary evolution with a finite Lieb--Robinson velocity. This ensures a well-defined causal cone and rules out superluminal signalling within the model.

\paragraph{Clock slowdown and susceptibility.} The constitutive relation $\omega_{\text{eff}} = \omega/(1+\alpha\Lambda)$ is supported by small-$N$ numerics and a perturbative derivation, with $\alpha$ interpreted as a linear-response susceptibility. However, the relation is approximate: residuals of order $10$--$15\%$ remain even in the best regime we have examined, and the behaviour in larger systems and more complex geometries is not yet known.

\paragraph{Lorentz invariance.} The fixed lattice and discrete time step imply that exact Lorentz invariance is broken at the Planck scale. The resulting Lorentz-violating corrections to the dispersion relation are quadratically suppressed and numerically tiny at accessible energies, but they do imply a preferred frame in the ultraviolet. Whether this is acceptable as a fundamental feature depends on empirical constraints and on philosophical priors about the nature of spacetime symmetry.

\paragraph{Anchoring to macroscopic scales.} The Planckian lattice spacing $a$ and time step $\tau$ are not derived purely from SRQID combinatorics; they are fixed by matching the model's vacuum speed and horizon entropy to the observed speed of light and Bekenstein--Hawking law. This anchoring demonstrates consistency with known macroscopic physics but does not constitute a derivation of the Planck scale from first principles.

\paragraph{Spin-2 sector.} We do not currently obtain a convincing spin-2 sector with the required $1/k^2$ power spectrum and Pauli--Fierz-like dynamics. This is a major gap relative to the goal of reproducing general relativity in the infrared.

\paragraph{Dark energy.} Expressing the cosmological constant problem in terms of vacuum promotion densities clarifies where fine-tuning would reside in the QATNU/SRQID model, but does not solve it. Whether the required ultra-rare promotion tail can emerge naturally is an open question.

\section{Conclusion and Future Work}

We have developed and analyzed a discrete, unitary, measurement-free QCA model in which local matter clocks are coupled to an entanglement-carrying network of bond registers via frustration-controlled promotion isometries. Within this Self-Referential Quantum Information Dynamics (SRQID) and QATNU framework we have:
\begin{itemize}
    \item identified and derived an effective clock slowdown relation $\omega_{\text{eff}} = \omega/(1+\alpha\Lambda)$ and interpreted the parameter $\alpha$ as a clock susceptibility;
    \item measured $\alpha$ from small-$N$ exact diagonalization and shown that it reorganizes the phase structure in promotion strength $\lambda$ into outer-dominated, balanced, and over-promotion regimes, with a broad $O(1)$ window yielding a robust Minkowski-like sector;
    \item anchored the model to macroscopic scales via the speed of light in the vacuum and the Bekenstein--Hawking area law, obtaining Planckian lattice spacing and time steps with quadratically suppressed Lorentz violation;
    \item and clarified how black-hole-like catastrophic clusters and vacuum promotions might, in principle, relate to Hawking temperatures and dark-energy densities, while highlighting the stringent constraints that follow.
\end{itemize}

At the same time, we have been explicit about the framework's limitations. The spin-2 sector remains undeveloped, Lorentz invariance is approximate rather than exact, the clock slowdown law is only numerically supported in small systems, and dark energy is not explained but reformulated as a constraint on vacuum promotion statistics. For these reasons, we view SRQID/QATNU not as a complete theory of quantum gravity but as a concrete, falsifiable testbed that sits at the intersection of quantum information, tensor-network-inspired dynamics, and emergent geometry.

Future work will focus on three directions. First, extending numerical simulations to larger systems and more complex topologies will test the robustness of the clock slowdown relation and the $(\lambda,\alpha)$ phase structure. Second, constructing better geometric observables and coarse-graining procedures may reveal whether a genuine spin-2 sector and $1/k^2$ power spectrum can emerge from correlated promotion clusters. Third, systematic exploration of vacuum promotion statistics at very small $\lambda$ will determine whether the model admits naturally suppressed vacuum activity compatible with the observed dark-energy density. Together, these efforts will help determine whether the self-referential dynamics explored here is a viable stepping stone toward a fully emergent description of spacetime, or a useful but ultimately limited toy model.
```

---

## Appendix snapshot (from `docs/mega_document.tex`)

```tex
\section{Validation snapshot and results notes}
(Condensed from \texttt{docs/status\_202511.tex} and \texttt{docs/results.md}.)

\subsection{Key extracted critical points (validated)}
\begin{itemize}
    \item $N=4$, $\alpha=0.8$: $\lambda_{c1}\approx 0.203$, $\lambda_{\mathrm{rev}}\approx 1.058$ (residual $\approx 13.4\%$), $\lambda_{c2}\approx 1.095$.
    \item $N=5$, $\alpha=0.8$: $\lambda_{c1}\approx 0.232$, $\lambda_{\mathrm{rev}}\approx 0.338$ (residual $\approx 10.2\%$), $\lambda_{c2}\approx 1.033$.
\end{itemize}

\subsection{Non-1D topology spot checks (degree dependence)}
\begin{itemize}
    \item $N=4$ cycle ($\alpha=0.8$): $\lambda_{c1}\approx 0.261$, $\lambda_{\mathrm{rev}}\approx 0.486$ (near-zero residual), $\lambda_{c2}\approx 1.0$, with $v_{\mathrm{LR}}\approx 2.45$.
    \item $N=4$ star/pyramid ($\alpha=0.8$): $\lambda_{c1}\approx 0.164$, $\lambda_{\mathrm{rev}}\approx 1.30$ (residual $\approx 0.148$), $\lambda_{c2}\approx 1.40$, with $v_{\mathrm{LR}}\approx 1.05$.
\end{itemize}
```

---

## Appendix: alpha derivation extract (from `docs/mega_parts/alpha_derivation_extract.tex`)

```tex
\section{Deriving and defining the susceptibility \texorpdfstring{$\alpha$}{alpha}}

\noindent\emph{Source:} \path{docs/alpha_derivation.md}.

\subsection{Setup}
From SRQID/QATNU, take a local Hamiltonian (in code units)
\begin{equation}
H = H_{\text{mat}} + H_{\text{bond}} + H_{\text{int}},
\end{equation}
with
\begin{equation}
H_{\text{mat}} = \frac{\omega}{2}\sum_i X_i,
\qquad
H_{\text{int}} = \lambda\sum_{\langle i,j\rangle} F_{ij} \otimes \bigl(R_{ij} + R_{ij}^\dagger\bigr),
\end{equation}
where $F_{ij}=\tfrac12(\mathbb{I}-Z_iZ_j)$ is the frustration projector and $R_{ij}$ is the bond-ladder raising isometry on the rung basis $\{|m\rangle\}$.

Define the bond occupation operator
\begin{equation}
 n_{ij} = \sum_m m\,\Pi_{ij}^{(m)},
\end{equation}
and the local entanglement/circuit-depth proxy
\begin{equation}
\Lambda_i \equiv \sum_{j\in\partial i} \log\bigl(1 + \langle n_{ij} \rangle\bigr)
\approx \sum_{j\in\partial i} \langle n_{ij} \rangle
\quad \text{(dilute regime)}.
\end{equation}
Postulate~1 in QATNU is
\begin{equation}
\omega_{\text{eff}}^{(i)} = \frac{\omega}{1+\alpha\Lambda_i}.
\end{equation}

\subsection{Perturbative derivation (dilute, weak coupling)}
Assume $\lambda\ll\omega$ and promotions are rare ($\langle n_{ij}\rangle\ll 1$). Split
\begin{equation}
H_0 = H_{\text{mat}} + H_{\text{bond}},\qquad V = H_{\text{int}}.
\end{equation}
To second order in $\lambda$, the matter self-energy at site $i$ in the static (zero-frequency) limit has the schematic form
\begin{equation}
\Sigma_i(0) \sim -\sum_{j\in\partial i}\frac{2\lambda^2}{\Delta_{ij}}\,\langle F_{ij}^2\rangle\,\langle R_{ij}R_{ij}^\dagger\rangle,
\end{equation}
where $\Delta_{ij}$ is the relevant bond excitation gap from $H_{\text{bond}}$.
Using $F_{ij}^2=F_{ij}$ and (in the dilute limit) $\langle R_{ij}R_{ij}^\dagger\rangle\propto\langle n_{ij}\rangle$, we obtain
\begin{equation}
\Sigma_i(0) \approx -\frac{2\lambda^2}{\Delta_{\text{eff}}}\sum_{j\in\partial i} \langle F_{ij}\rangle\,\langle n_{ij}\rangle.
\end{equation}
Expanding Postulate~1 for small $\Lambda_i$ gives
\begin{equation}
\omega_{\text{eff}}^{(i)} = \frac{\omega}{1+\alpha\Lambda_i}\approx \omega\,(1-\alpha\Lambda_i).
\end{equation}
Matching the induced frequency shift yields the perturbative susceptibility estimate
\begin{equation}
\alpha_{\text{pert}}(\lambda)
\approx
\frac{2\lambda^2}{\omega^2\Delta_{\text{eff}}}
\frac{\sum_{j\in\partial i} \langle F_{ij}\rangle\,\langle n_{ij}\rangle}{\sum_{j\in\partial i}\langle n_{ij}\rangle},
\end{equation}
so that in a homogeneous dilute regime one expects $\alpha\propto\lambda^2$.

\subsection{Non-perturbative definition}
Beyond weak coupling, define $\alpha$ operationally as a static susceptibility
\begin{equation}
\alpha(\lambda)
\equiv
-\left.\frac{\partial}{\partial\Lambda_i}\log\frac{\omega_{\text{eff}}^{(i)}(\Lambda_i,\lambda)}{\omega}\right|_{\Lambda_i\to 0}.
\end{equation}
Equivalently, one can write a Kubo/linear-response form
\begin{equation}
\alpha = \frac{1}{\omega}\int_0^\infty dt\,\chi_{X_i,O_{\Lambda_i}}(t),
\qquad
\chi_{X_i,O_{\Lambda_i}}(t)=-i\theta(t)\langle [X_i(t),O_{\Lambda_i}(0)]\rangle,
\end{equation}
with $O_{\Lambda_i}=\sum_{j\in\partial i} n_{ij}$.

\subsection{Static limit: a Poisson/Green's-function scaffold (Newtonian regime)}
The clock slowdown postulate can be re-expressed as an effective weak-field ``potential.''
From
\begin{equation}
\frac{\omega_{\text{eff}}^{(i)}}{\omega}=\frac{1}{1+\alpha\Lambda_i}\approx 1-\alpha\Lambda_i\qquad (\alpha\Lambda_i\ll 1),
\end{equation}
we define a dimensionful potential proxy
\begin{equation}
\phi_i \equiv -\alpha c^2\,\Lambda_i.
\label{eq:phi-def}
\end{equation}
In the Newtonian weak-field limit of GR one has $d\tau/dt\approx 1+\phi/c^2$, so~\eqref{eq:phi-def} provides a consistent identification (up to sign conventions).

However, clock slowdown alone is a constitutive relation (how clocks respond to $\Lambda$), not a field equation (how $\Lambda$ is sourced).
To obtain an emergent Newtonian sector one needs a static, long-wavelength equation for $\Lambda$.
A minimal scaffold is to assume or derive a discrete Poisson equation on the graph,
\begin{equation}
(\Delta_{\text{graph}}\Lambda)_i = -\kappa\,\rho_{\text{ent}}(i),
\label{eq:poisson-graph}
\end{equation}
where $(\Delta_{\text{graph}}\Lambda)_i\equiv\sum_{j\in\partial i}(\Lambda_i-\Lambda_j)$ is the graph Laplacian, $\rho_{\text{ent}}$ is an ``entanglement/matter'' source (e.g.\ coarse-grained frustration or promotion density), and $\kappa$ is a constant.
In a continuum limit on a regular $d=3$ lattice, $\Delta_{\text{graph}}\to a^2\nabla^2$, so~\eqref{eq:poisson-graph} yields
\begin{equation}
\nabla^2\Lambda(\mathbf{x})\approx -\frac{\kappa}{a^2}\,\rho_{\text{ent}}(\mathbf{x}),
\qquad
\Rightarrow\qquad
\nabla^2\phi(\mathbf{x})\approx \left(\frac{\alpha c^2\kappa}{a^2}\right)\rho_{\text{ent}}(\mathbf{x}).
\end{equation}
Thus, if $\rho_{\text{ent}}$ can be matched to a mass density (or energy density divided by $c^2$) in the appropriate regime, the emergent Newton constant is
\begin{equation}
4\pi G_{\text{eff}} \equiv \frac{\alpha c^2\kappa}{a^2}.
\end{equation}

With a localized static source $\rho_{\text{ent}}(\mathbf{x})\propto \delta^{(3)}(\mathbf{x})$, the continuum Green's function of the Laplacian gives
\begin{equation}
\Lambda(r)\propto \frac{1}{r},
\qquad
\phi(r)\propto -\frac{1}{r},
\end{equation}
recovering the Newtonian $1/r$ potential.
If the effective static operator is instead $(\nabla^2-\mu^2)$ (screened), the potential becomes Yukawa-like, $\propto e^{-\mu r}/r$.

Establishing~\eqref{eq:poisson-graph} from the microscopic SRQID/QATNU Hamiltonian is an open problem: it requires showing that the coarse-grained static effective action for $\Lambda$ contains a leading gradient term $\sum_{\langle ij\rangle}(\Lambda_i-\Lambda_j)^2$ and identifying the correct microscopic source operator whose expectation value defines $\rho_{\text{ent}}$.
```

---

## Appendix: Newtonian scaffold notes (from `docs/mega_parts/newtonian_scaffold_notes.tex`)

```tex
\section{Scalar (Newtonian) scaffold: viability and next steps}

This appendix section records a concrete program for upgrading ``curved clock'' phenomenology into an emergent Newtonian sector.
The key methodological point is to separate:
(i) a \emph{constitutive relation} (how clocks respond to an entanglement proxy) from
(ii) a \emph{field equation} (how that proxy is sourced).
We note that related (but distinct) information-theoretic approaches to gravity exist in the literature.\cite{thooft-ca,verlinde-eg}
Unlike 't~Hooft's cellular automaton interpretation---which preserves microscopic locality at the cost of explicit superdeterminism---our framework maintains unitary locality in the standard quantum sense and is constrained by Lieb--Robinson bounds.\cite{lieb-robinson,hastings-koma}
Unlike Verlinde's entropic gravity, our ``clock slowdown'' is a dynamical coupling between matter and entanglement-carrying bond registers rather than a thermodynamic entropy-gradient force law.

\subsection{Potential identification from clock slowdown}
In the weak-field regime ($\alpha\Lambda\ll 1$), the clock slowdown gives
\begin{equation}
\frac{d\tau}{dt}=\frac{\omega_{\mathrm{eff}}}{\omega}=\frac{1}{1+\alpha\Lambda}\approx 1-\alpha\Lambda.
\end{equation}
Matching to the weak-field GR relation $d\tau/dt\approx 1+\phi/c^2$ motivates
\begin{equation}
\phi(\mathbf{x})\equiv -\alpha c^2\,\Lambda(\mathbf{x}).
\end{equation}
This is consistent with the earlier phenomenological identification $\phi\sim-\alpha c^2\Lambda$ in the redshift discussion (see Sec.~\ref{sec:grav-redshift}).

\subsection{Critical gap: deriving a Poisson equation}
To obtain a Newtonian limit one needs a static, long-wavelength equation for $\Lambda$ (or a coarse-grained field derived from bond occupations).
A target form is a graph/continuum Poisson equation,\cite{chung-spectral}
\begin{equation}
(\Delta_{\mathrm{graph}}\Lambda)_i = -\kappa\,\rho_{\mathrm{ent}}(i)
\qquad\Rightarrow\qquad
\nabla^2\Lambda(\mathbf{x})\approx -\frac{\kappa}{a^2}\,\rho_{\mathrm{ent}}(\mathbf{x}),
\end{equation}
where $\rho_{\mathrm{ent}}$ is an ``entanglement/matter'' source (e.g.\ a coarse-grained frustration density or promotion density).
If the static operator is instead screened, $(\nabla^2-\mu^2)\Lambda\sim-\rho_{\mathrm{ent}}$, the potential becomes Yukawa-like.\cite{yukawa}
A simple pointwise estimator (valid in source-free regions where $\rho_{\mathrm{ent}}(i)\approx 0$) is
\begin{equation}
\mu^2 \approx -\frac{(\Delta_{\mathrm{graph}}\Lambda)_i}{\Lambda_i}
\qquad\text{(or averaged:)}\qquad
\mu^2 \approx -\frac{\langle (\Delta_{\mathrm{graph}}\Lambda)_i\rangle}{\langle \Lambda_i\rangle},
\end{equation}
so that the screening length is $\ell_{\mathrm{scr}}\equiv 1/\mu$ and $\mu(\lambda)$ can be extracted directly once $\Lambda$ is measured.

\subsection{Strategy: frozen-matter static response}
A concrete derivation route is:
\begin{enumerate}
    \item \emph{Freeze the matter sector:} replace the operators $Z_i$ by fixed classical values $z_i\in\{\pm 1\}$ (commuting background sources), thereby fixing $F_{ij}$ by hand; then solve for the bond sector ground state (or thermal equilibrium) under this fixed frustration pattern (i.e., do \emph{not} evolve the full coupled matter$+$bond system). This ``frozen matter'' limit is justified when $\omega\gg\lambda$ (outer-dominated regime) or when matter is pinned by external classical sources, decoupling matter dynamics from bond relaxation.
    \item \emph{Compute the static bond response:} solve for steady-state (or ground-state) bond occupations $\langle n_{ij}\rangle$ and hence $\Lambda_i$.
    \item \emph{Coarse-grain:} define a neighborhood average $\bar{\Lambda}(\mathbf{x})$ over radii $R$ with $a\ll R\ll L$.
    \item \emph{Extract the effective operator:} check whether the long-wavelength response is governed by a Laplacian (massless) or a screened Laplacian (massive).
\end{enumerate}

Complementing the frozen-matter static protocol above, one may alternatively perform deep-time real-time evolution ($t\gg 1/\omega$) on small systems (e.g.\ $N=4$) to probe relaxation timescales in the bond sector. In particular, critical slowing down---a divergence of the bond-sector equilibration time near $\lambda_{\mathrm{rev}}$---would provide dynamical evidence that the revival regime corresponds to a critical point where screening vanishes ($\mu\to 0$).

A useful diagnostic is the static susceptibility structure (schematically)
\begin{equation}
\chi^{-1} \sim \Delta_{\mathrm{graph}}+\mu^2\mathbb{I},
\end{equation}
which would imply Laplacian propagation (possibly screened) for scalar bond observables.

\subsection{Source operator candidate}
A natural hypothesis is that the source is the local frustration degree,
\begin{equation}
\rho_{\mathrm{ent}}(i)\stackrel{?}{\propto}\sum_{j\in\partial i}\langle F_{ij}\rangle,
\end{equation}
since promotions are driven by $F_{ij}$ in $H_{\mathrm{int}}$.
This can be tested numerically by checking whether $(\Delta\Lambda)_i$ tracks $-\sum_j\langle F_{ij}\rangle$ in static configurations.

\subsection{Consistency with anchoring and phase structure}
If a Poisson limit holds with
\begin{equation}
4\pi G_{\mathrm{eff}}\equiv \frac{\alpha c^2\kappa}{a^2},
\end{equation}
then with $a$ fixed Planckian by the entropy anchor, recovering $G_{\mathrm{eff}}\sim G$ requires $\alpha\kappa$ not to be parametrically small.
A plausible expectation is that an unscreened Newtonian sector (if present) is tied to the balanced/revival regime where $\lambda\sim O(1)$.
We hypothesize that the unscreened Poisson regime ($\mu=0$) coincides with the ``revival'' phase identified in Sec.~\ref{sec:revival-phase}, where Dirac-like behavior is restored.
In the outer-dominated ($\lambda\ll\lambda_{\text{rev}}$) and catastrophic ($\lambda>\lambda_{c2}$) phases, we expect $\mu\ne 0$ (screened/gravity confined).

\subsection{Minimum numerical protocol}
A first-pass testbed calculation is:
\begin{enumerate}
    \item Choose a lattice (ideally 3D) and open boundary conditions.
    \item Impose a compact ``mass''/source region via fixed $Z_i$ (high frustration) and vacuum outside.
    \item Solve for a static state (imaginary-time evolution or ground state with fixed matter).
    \item Measure $\Lambda(r)$ and fit to $1/r$ (Coulomb) versus $e^{-\mu r}/r$ (Yukawa) to extract any screening length.
    \item Extract $\kappa$ by linear regression of $(\Delta_{\mathrm{graph}}\Lambda)_i$ versus $\rho_{\mathrm{ent}}(i)$ across lattice sites $i$, restricting to radii $r>R_{\mathrm{core}}$ outside the source region where linear response holds, and use it to estimate
    \begin{equation}
        4\pi G_{\mathrm{eff}}\equiv \frac{\alpha c^2\kappa}{a^2}.
    \end{equation}
    As a cross-check, independently fit the far-field tail to $\Lambda(r)\sim A\,e^{-\mu r}/r$ (or $A/r$ when $\mu\approx 0$) and verify that $A$ is consistent with the inferred $\kappa$ for the imposed source strength.
    For a point source in 3D, the amplitude $A$ relates to $\kappa$ via $A = \kappa\rho_0/4\pi$ where $\rho_0$ is the source strength; comparing this with the regression-derived $\kappa$ provides consistency.
    Comparing $G_{\mathrm{eff}}$ to the observed Newton constant (given the anchor-implied $a$ and the measured $\alpha$) provides a direct consistency check of the scale setting.
\end{enumerate}

\subsection{Relation to the spin-2 problem}
A scalar Poisson sector captures the Newtonian limit but does not by itself establish a propagating spin-2 sector.
If the scalar limit is Laplacian but tensor correlators remain short-range/flat, the model behaves more like a preferred-frame (khronometric/Einstein--Aether-like) gravity in the infrared; resolving this requires dedicated spin-2 observables built from bond-bond correlators.\cite{jacobson-aether}
```
# Bibliography (from `docs/mega_document.tex`)

```tex
\bibliographystyle{unsrt}
\begin{thebibliography}{99}

\bibitem{lieb-robinson}
E.~H. Lieb and D.~W. Robinson.
\newblock The finite group velocity of quantum spin systems.
\newblock {\em Communications in Mathematical Physics}, 28:251--257, 1972.

\bibitem{thooft-ca}
G.~'t~Hooft.
\newblock {\em The Cellular Automaton Interpretation of Quantum Mechanics}.
\newblock Springer, 2016.

\bibitem{verlinde-eg}
E.~P. Verlinde.
\newblock On the origin of gravity and the laws of Newton.
\newblock {\em Journal of High Energy Physics}, 2011:29, 2011.

\bibitem{hastings-koma}
M.~B. Hastings and T.~Koma.
\newblock Spectral gap and exponential decay of correlations.
\newblock {\em Communications in Mathematical Physics}, 265:781--804, 2006.

\bibitem{chung-spectral}
F.~Chung.
\newblock {\em Spectral Graph Theory}.
\newblock American Mathematical Society, 1997.

\bibitem{yukawa}
H.~Yukawa.
\newblock On the interaction of elementary particles.
\newblock {\em Proceedings of the Physico-Mathematical Society of Japan. 3rd Series}, 17:48--57, 1935.

\bibitem{jacobson-aether}
T.~Jacobson and D.~Mattingly.
\newblock Gravity with a dynamical preferred frame.
\newblock {\em Physical Review D}, 64:024028, 2001.

% --- Imported keys from QATNU/SRQID/Volume3 source docs (kept for compatibility) ---

\bibitem{Strauch2006}
F.~W. Strauch.
\newblock Relativistic quantum walks.
\newblock {\em Physical Review A}, 73:054302, 2006.

\bibitem{NayakVishwanath2000}
A.~Nayak and A.~Vishwanath.
\newblock Quantum walk on the line.
\newblock {\em arXiv:quant-ph/0010117}, 2000.

\bibitem{Bisio2015}
A.~Bisio, G.~M. D'Ariano, P.~Perinotti, and A.~Tosini.
\newblock Quantum cellular automata and free quantum field theory.
\newblock {\em Foundations of Physics}, 45:1137--1152, 2015.

\bibitem{Raynal2017}
P.~Raynal.
\newblock Simple derivation of the Weyl and Dirac quantum cellular automata.
\newblock {\em Physical Review A}, 95:062344, 2017.

\bibitem{Gottesman1997}
D.~Gottesman.
\newblock {\em Stabilizer Codes and Quantum Error Correction}.
\newblock PhD thesis, Caltech, 1997; {\em arXiv:quant-ph/9705052}.

\bibitem{Poulin2005}
D.~Poulin.
\newblock Stabilizer formalism for operator quantum error correction.
\newblock {\em arXiv:quant-ph/0508131}, 2005.

\bibitem{MaggioreBook}
M.~Maggiore.
\newblock {\em Gravitational Waves, Vol.\ 1: Theory and Experiments}.
\newblock Oxford University Press, 2007.

\bibitem{WeinbergBook}
S.~Weinberg.
\newblock {\em Gravitation and Cosmology}.
\newblock Wiley, 1972.

\bibitem{Stinespring1955}
W.~F. Stinespring.
\newblock Positive functions on C$^\ast$-algebras.
\newblock {\em Proceedings of the American Mathematical Society}, 6:211--216, 1955.

\bibitem{Choi1975}
M.-D. Choi.
\newblock Completely positive linear maps on complex matrices.
\newblock {\em Linear Algebra and its Applications}, 10(3):285--290, 1975.

\bibitem{NielsenChuang}
M.~A. Nielsen and I.~L. Chuang.
\newblock {\em Quantum Computation and Quantum Information}.
\newblock Cambridge University Press, 10th Anniversary ed., 2010.

\bibitem{Barenco1995}
A.~Barenco \emph{et al.}
\newblock Elementary gates for quantum computation.
\newblock {\em Physical Review A}, 52:3457--3467, 1995.

\bibitem{Zohar2016}
E.~Zohar, J.~I. Cirac, and B.~Reznik.
\newblock Quantum simulations of lattice gauge theories using ultracold atoms.
\newblock {\em Reports on Progress in Physics}, 79:014401, 2016.

\bibitem{Banuls2020}
M.~C. Ba\~nuls \emph{et al.}
\newblock Simulating lattice gauge theories within quantum technologies.
\newblock {\em European Physical Journal D}, 74:165, 2020.

\bibitem{ChandrasekharanWiese1997}
S.~Chandrasekharan and U.-J. Wiese.
\newblock Quantum link models: A discrete approach to gauge theories.
\newblock {\em Nuclear Physics B}, 492:455--471, 1997.

\bibitem{SchumacherWerner2004}
B.~Schumacher and R.~F. Werner.
\newblock Reversible quantum cellular automata.
\newblock {\em arXiv:quant-ph/0405174}, 2004.

\bibitem{Arrighi2019}
P.~Arrighi.
\newblock An overview of quantum cellular automata.
\newblock {\em Natural Computing}, 18:885--899, 2019.

\bibitem{GNVW2012}
D.~Gross, V.~Nesme, H.~Vogts, and R.~F. Werner.
\newblock Index theory of one-dimensional quantum walks and cellular automata.
\newblock {\em Communications in Mathematical Physics}, 310:419--454, 2012.

\bibitem{Vidal2007}
G.~Vidal.
\newblock Entanglement renormalization.
\newblock {\em Physical Review Letters}, 99:220405, 2007.

\bibitem{Swingle2012}
B.~Swingle.
\newblock Entanglement renormalization and holography.
\newblock {\em Physical Review D}, 86:065007, 2012.

\bibitem{BurioniCassi1996}
R.~Burioni and D.~Cassi.
\newblock Universal properties of spectral dimension.
\newblock {\em Physical Review Letters}, 76:1091--1093, 1996.

\bibitem{Calcagni2014}
G.~Calcagni, D.~Oriti, and J.~Th\"urigen.
\newblock Spectral dimension of quantum geometries.
\newblock {\em Journal of Physics A: Mathematical and Theoretical}, 47:355402, 2014.

\bibitem{CMS2020aQGC}
G.~Aad \emph{et al.} (ATLAS Collaboration).
\newblock Observation of photon-induced $W^+W^-$ production in $pp$ collisions at $\sqrt{s}=13$~TeV using the ATLAS detector.
\newblock {\em Physics Letters B}, 816:136190, 2021; {\em arXiv:2010.04019}.

\bibitem{Abdo2009}
A.~A. Abdo \emph{et al.} (Fermi LAT/GBM Collaborations).
\newblock A limit on the variation of the speed of light arising from quantum gravity effects.
\newblock {\em Nature}, 462:331--334, 2009.

\bibitem{LVK_GRtests}
R.~Abbott \emph{et al.} (LIGO--Virgo--KAGRA Collaborations).
\newblock Tests of General Relativity with GWTC-3.
\newblock {\em arXiv:2112.06861}, 2021.

\bibitem{Aspelmeyer2014}
M.~Aspelmeyer, T.~J. Kippenberg, and F.~Marquardt.
\newblock Cavity optomechanics.
\newblock {\em Reviews of Modern Physics}, 86:1391--1452, 2014.

\bibitem{Calajo2024}
G.~Calaj\`o \emph{et al.}
\newblock Digital quantum simulation of a (1+1)D SU(2) lattice gauge theory.
\newblock {\em PRX Quantum}, 5:040309, 2024.

\bibitem{FarrowQATNU2025}
J.~Farrow.
\newblock Quantum Automaton Tensor Network Universe (QATNU): Measurement-Free Gauge Promotion and Emergent Geometry.
\newblock SSRN Working Paper No.~5390318, 2025. \url{https://ssrn.com/abstract=5390318}.

\bibitem{FarrowSRQID2025}
J.~Farrow.
\newblock Self-Referential Quantum-Information Dynamics (SRQID).
\newblock SSRN Working Paper No.~5390753, 2025. \url{https://ssrn.com/abstract=5390753}.

% Additional references to be added as appropriate.

\end{thebibliography}
\end{document}
```
