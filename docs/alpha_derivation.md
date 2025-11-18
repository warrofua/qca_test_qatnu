### Setup

From SRQID/QATNU, take the local Hamiltonian (in code units):

[
H = H_{\text{mat}} + H_{\text{bond}} + H_{\text{int}}
]

with

[
H_{\text{mat}} = \frac{\omega}{2}\sum_i X_i,
]

[
H_{\text{int}} = \lambda\sum_{\langle i,j\rangle} F_{ij} \otimes \bigl(R_{ij} + R_{ij}^\dagger\bigr),
]

where

* (F_{ij} = \tfrac12(\mathbb{I} - Z_iZ_j)) is the frustration projector,
* (R_{ij} = \sum_m |m+1\rangle\langle m|) is the bond-ladder raising isometry on the rung basis ({|m\rangle}).

Define the **bond occupation operator**

[
n_{ij} = \sum_m m,\Pi_{ij}^{(m)},
]

and the **local entanglement/circuit-depth proxy**

[
\Lambda_i \equiv \sum_{j\in\partial i} \log(1 + \langle n_{ij} \rangle);\approx;\sum_{j\in\partial i} \langle n_{ij} \rangle
]

in the dilute regime.

Postulate 1 in QATNU is

[
\omega_{\text{eff}}^{(i)} = \frac{\omega}{1+\alpha\Lambda_i}
]

for the local effective clock at site (i).

---

## 1. Perturbative derivation of (\alpha)

Work in the **dilute, weak-coupling regime**:

* (\lambda \ll \omega),
* bond registers mostly in the ground sector (|0\rangle_{ij}),
* promotions are rare: (\langle n_{ij}\rangle \ll 1).

Split

[
H_0 = H_{\text{mat}} + H_{\text{bond}},\qquad V = H_{\text{int}}.
]

In the interaction picture, to second order in (\lambda), the effective Hamiltonian on the matter sector is

[
H_{\text{eff}} \approx H_{\text{mat}} + \Sigma,
]

where the static self-energy contribution at site (i) (zero-frequency limit) is

[
\Sigma_i(0) \sim -,\sum_{j\in\partial i} \frac{2\lambda^2}{\Delta_{ij}},\langle F_{ij}^2 \rangle,\langle R_{ij}R_{ij}^\dagger\rangle.
]

Here (\Delta_{ij}) is the relevant bond excitation gap from (H_{\text{bond}}), and the expectation values are taken in the (nearly) unperturbed bond state.

Use:

* (F_{ij}) is a projector, so (F_{ij}^2 = F_{ij}) ⇒ (\langle F_{ij}^2 \rangle = \langle F_{ij}\rangle).
* In the dilute limit, the ladder operator satisfies (schematically)
  [
  \langle R_{ij}R_{ij}^\dagger\rangle \sim \langle n_{ij}\rangle
  ]
  up to order-1 factors, because (R_{ij}) connects (|m\rangle\leftrightarrow|m+1\rangle) with weight (\sqrt{m+1}) and the occupation distribution is sharply peaked at low m.

Thus

[
\Sigma_i(0);\approx;-\frac{2\lambda^2}{\Delta_{\text{eff}}}\sum_{j\in\partial i} \langle F_{ij}\rangle,\langle n_{ij}\rangle,
]

where (\Delta_{\text{eff}}) is an effective averaged gap for the bond ladder.

On the other hand, expanding Postulate 1 for small (\Lambda_i):

[
\omega_{\text{eff}}^{(i)}
= \frac{\omega}{1+\alpha\Lambda_i}
\approx \omega,(1 - \alpha\Lambda_i).
]

Interpreting the self-energy as a shift of the local frequency,

[
\omega_{\text{eff}}^{(i)} = \omega + \delta\omega_i,\qquad
\delta\omega_i = \frac{1}{\omega}\Sigma_i(0),
]

we match the linear terms:

[
\delta\omega_i \approx -\omega\alpha\Lambda_i
\quad\Rightarrow\quad
\Sigma_i(0) \approx -\omega^2 \alpha\Lambda_i.
]

Equate the two expressions for (\Sigma_i(0)):

[
-\omega^2 \alpha\Lambda_i
;\approx;
-\frac{2\lambda^2}{\Delta_{\text{eff}}}\sum_{j\in\partial i} \langle F_{ij}\rangle,\langle n_{ij}\rangle.
]

Using (\Lambda_i \approx \sum_{j\in\partial i}\langle n_{ij}\rangle) in the dilute regime, we obtain

[
\alpha;\Lambda_i
;\approx;
\frac{2\lambda^2}{\omega^2\Delta_{\text{eff}}},\sum_{j\in\partial i} \langle F_{ij}\rangle,\langle n_{ij}\rangle.
]

Divide both sides by (\Lambda_i):

[
\boxed{
\displaystyle
\alpha_{\text{pert}}(\lambda)
;\approx;
\frac{2\lambda^2}{\omega^2\Delta_{\text{eff}}};
\frac{\sum_{j\in\partial i} \langle F_{ij}\rangle,\langle n_{ij}\rangle}
{\sum_{j\in\partial i} \langle n_{ij}\rangle}
}
]

In a homogeneous dilute phase where (\langle F_{ij}\rangle) and (\Delta_{ij}) are approximately edge-independent, this simplifies to

[
\boxed{
\displaystyle
\alpha_{\text{pert}}(\lambda)
;\approx;
\frac{2\lambda^2}{\omega^2\Delta_{\text{eff}}},\langle F\rangle_0
}
]

with (\langle F\rangle_0) the frustration expectation in the unperturbed vacuum. This is the **perturbative** small-(\lambda) expression: (\alpha \propto \lambda^2) and tends to 0 as (\lambda\to 0).

---

## 2. Non-perturbative definition of (\alpha)

Beyond the dilute, weak-coupling regime, the second-order expansion breaks down. In the full interacting theory, (\alpha) is best defined **operationally** as a susceptibility:

1. For a given (\lambda), prepare many eigenstates / thermal states with slightly different local (\Lambda_i) (by varying bond promotion patterns).
2. Measure the corresponding local effective clock frequency (\omega_{\text{eff}}^{(i)}(\Lambda_i,\lambda)) from Ramsey fringes or spectral gaps.
3. Fit the dependence of (\omega_{\text{eff}}^{(i)}) on (\Lambda_i) near (\Lambda_i=0).

Formally, define

[
\boxed{
\displaystyle
\alpha(\lambda)
;\equiv;
-,\left.
\frac{\partial}{\partial\Lambda_i}
\log\frac{\omega_{\text{eff}}^{(i)}(\Lambda_i,\lambda)}{\omega}
\right|_{\Lambda_i\to 0}
}
]

This reproduces Postulate 1, because if (\omega_{\text{eff}}^{(i)} = \omega/(1+\alpha\Lambda_i)), then

[
\log\frac{\omega_{\text{eff}}^{(i)}}{\omega}
= -\log(1+\alpha\Lambda_i)
\approx -\alpha\Lambda_i,
]

so the derivative at (\Lambda_i\to 0) gives precisely (-\alpha).

Equivalently, in Kubo-linear-response form, you can write

[
\alpha(\lambda)
= \frac{1}{\omega}\int_0^\infty dt;\chi_{X_i,O_{\Lambda_i}}(t),
]

where

* (X_i) is the local clock operator from (H_{\text{mat}}),
* (O_{\Lambda_i} = \sum_{j\in\partial i} n_{ij}) is the operator whose expectation defines (\Lambda_i),
* (\chi) is the retarded response function
  [
  \chi_{X_i,O_{\Lambda_i}}(t)
  = -i,\theta(t),\langle [X_i(t), O_{\Lambda_i}(0)]\rangle.
  ]

In practice (ED / numerics):

* Extract (\omega_{\text{eff}}^{(i)}) for a set of states with slightly different (\Lambda_i),
* Fit the slope of (\log(\omega_{\text{eff}}^{(i)}/\omega)) vs. (\Lambda_i),
* That slope (with a minus sign) is the **non-perturbative** (\alpha(\lambda)).

Thus:

* **Perturbative regime:** (\alpha_{\text{pert}}(\lambda)) computed from microscopic matrix elements and gaps, (\propto \lambda^2).
* **Full theory:** (\alpha(\lambda)) defined as the **zero-frequency clock susceptibility** to changes in local bond occupation, measured directly from the spectrum/dynamics of the full Hamiltonian.
