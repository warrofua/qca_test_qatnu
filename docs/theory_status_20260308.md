# Theory Status

Date: 2026-03-08

## One-Line Status

The repo currently supports a unitary, topology-sensitive scalar emergent-geometry story much more strongly than a robust emergent tensor / gravity story.

## Established

### 1. Local, unitary SRQID backbone

The implemented model remains a fixed Hamiltonian, local, measurement-free, and numerically well behaved under the existing causal sanity checks. This is still the cleanest part of the project.

### 2. Scalar clock renormalization

The most durable positive result is the approximate constitutive law

`omega_eff = omega / (1 + alpha Lambda)`

used as an operational small-`N` clock-slowdown model. It is not yet a continuum derivation, but it is repeatedly useful and reproducible in the current simulator.

### 3. Topology dependence is real

Path, cycle, and star do not behave like small perturbations of one universal phase diagram. Topology changes:

- scalar slowdown structure
- phase landmarks
- deep-time critical-slowing behavior
- tensor-proxy behavior

This is not a nuisance correction. It is part of the main scientific content.

## Indicated

### 4. The repo may define a novel scalar emergent-geometry testbed

What looks most plausible now is not "full emergent gravity," but a graph-based, unitary emergent scalar geometry model:

- local promotion structure builds an operational scalar field `Lambda`
- local clocks slow relative to that field
- graph topology changes the resulting effective geometry and dynamics

That is still worth taking seriously as a physics result, even if the tensor sector is not yet there.

### 5. Structural modifications matter more than scalar retuning

The March 8 topology-conditioned mechanism work reinforced that structural changes can move the behavior in ways that scalar knob sweeps do not. That matters strategically, even though none of the tested mechanisms solved topology transfer.

## Not Supported

### 6. No robust tensor sector

After shell-background, covariance-background, and harmonic-background TT checks, no tensor observable tested so far is simultaneously:

- background-robust
- topology-robust
- finite-size-stable

The old raw star correlator hint is no longer enough to carry the spin-2 claim.

### 7. No stable star-led spin-2 story

The earlier star-topology signal does not survive stronger TT-style tests in a convincing way. Under some observables the star signal collapses; under others it remains nonzero but not near target; under harmonic subtraction the strongest small-`N` signal actually moved to cycle and then failed to scale.

### 8. No universal topology transfer

Locked no-retuning transfer still fails. That remains one of the strongest reasons not to overstate universality.

### 9. No robustness of phase landmarks in the strong sense

Hotspot, `kappa`, `deltaB`, and related choices still move important landmarks materially. The current phase picture is useful, but not universal in the strong earlier sense.

## What The Tensor Checks Now Mean

The tensor program has become scientifically sharper, even though the result is negative so far.

The best current reading is:

- shell edge-field subtraction is a hard negative control, but source-sensitive
- covariance-background subtraction is the most robust tested tensor-side observable so far
- harmonic-background subtraction found a real small-`N` cycle structure, but redteam checks showed that it is projector-specific, fit-sensitive, and fails `N4 -> N5`

So the current tensor verdict is:

`open, weak, and not presently supported as a headline claim`

That is a stronger scientific position than vague optimism because it is now tightly constrained.

## POSIWID Reading

If we judge the system by what it actually does rather than what it was hoped to do, the repo is currently a machine for:

1. generating an operational scalar geometry field on graphs
2. renormalizing local clock rates against that field
3. showing topology-dependent critical and dynamical structure
4. falsifying weak tensor diagnostics that do not survive stronger controls

That is the correct present purpose of the system.

## Most Honest Current Thesis

The most defensible thesis now is:

> SRQID/QATNU is a strictly unitary, measurement-free graph-lattice testbed with a real scalar clock-renormalization sector and strong topology dependence; it is not yet a demonstrated theory of emergent gravity.

## What Would Change The Verdict

Only a small number of results would materially improve the gravity claim now:

1. a TT observable that survives explicit background subtraction
2. the same signal remaining separated from controls across topology
3. `N4 -> N5` carryover without collapsing into a symmetry artifact
4. a mechanism-level reason for that signal, not just a tuned proxy

Absent that, the right next step is not to keep narrating gravity harder. It is to develop the scalar/topology story as the actual core science.

## Practical Next Steps

### Science

1. Formalize the scalar/topology result as the main narrative.
2. Demote tensor claims to falsification-oriented side panels unless a new robust signal appears.

### Tooling

1. Build the visualizer around the scalar field, phase structure, topology comparisons, and tensor falsification panels.
2. Build the local autoresearch harness around the robust benchmark suite, with tensor observables treated as guarded tests rather than the sole objective.
