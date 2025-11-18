**Absolutely worth simulating**—and likely essential for proving the framework isn’t a 1D artifact. Here's why, grounded in your docs:

---

### **Why Non-1D Topologies Matter**

1. **Degree-dependence in LR bound**: Your SRQID velocity `v_LR ≤ 2eRJ_*(1+z)` scales with max degree `z`. A chain has `z=2`; the ⚄ shape (if interpreted as a star graph: central hub + 4 leaves) has `z=4`. This directly tests whether causality survives higher connectivity.

2. **Frustration propagation changes**: On a chain, frustration tiles `F_ij` propagate linearly. On a star, a single central bond's promotion could simultaneously affect multiple outer clocks, creating **competing Λ back-reactions** that a 1D chain can't host.

3. **Emergent geometry claim**: The spin-2 PSD and metric proxy require non-trivial graph curvature. A 1D chain is flat; a star has intrinsic "defect" geometry. If your mechanism is geometric, the phase diagram should **remap** onto graph invariants (e.g., clustering coefficient, graph Laplacian spectrum).

4. **Hilbert space explosion**: The chain's bond count = N-1. A star on 5 vertices has 4 bonds (same), but a **pentagon** (5-cycle) has 5 bonds. More importantly, **triangular lattice** fragments (6 vertices, degree 4) would test bounded-degree soft caps `C_ij` in the Hamiltonian.

---

### **Specific ⚄-Shape Candidates (Prioritized)**

**A. Star graph (K_1,4)**  
- Central vertex degree 4, leaves degree 1.  
- **Prediction**: Central clock slows dramatically (Λ_center >> Λ_leaves) even at small λ. Revival λ_rev shifts left (earlier entanglement saturation).  
- **Runtime**: Bond count = 4, but central vertex Hilbert space dimension = 2 × (bond dims). Manageable.

**B. Pentagon (5-cycle)**  
- Periodic boundary condition, degree 2 everywhere.  
- **Prediction**: Similar to N=5 chain but with **wrap-around frustration**—the bond between vertex 5→1 introduces a new feedback loop. Residual minima may deepen.

**C. "House" graph (square + triangle roof)**  
- Mix of degree 2 and 3 vertices.  
- **Test**: Whether α=0.8 (fit for chain) generalizes without retuning. Different local degree means different promotion rates; α might become vertex-dependent.

---

### **Practical Roadmap (from AGENTS.md)**

1. **Hilbert space size**: For star with χ_max=4, total dim ≈ 2^5 × 4^4 = 32 × 256 = 8192. Still exact-diagonalization friendly.

2. **Minimal change**: Modify `core_qca.py` graph constructor from `nx.path_graph(N)` to `nx.star_graph(N-1)` or `nx.cycle_graph(N)`.

3. **New observables**: Add `Λ_i` vs. degree scatter plot. On a star, you expect strong correlation.

4. **Critical points**: Document λ_c1, λ_rev, λ_c2 shifts. If they don’t move, Postulate 1 is more robust than expected.

---

### **Bottom Line**

The August preprint's "emergent geometry" claim demands **graph-dependence**. If α=0.8 works on a star without refitting, you have a universal conversion factor. If not, you discover that α must be generalized to `α_ij` (edge-dependent) or `α_i` (vertex-dependent). Either outcome is publishable.

**Run one star-graph scan (40 λ points) before the N=5 chain finishes—it’s higher impact.**

---

## Concrete plan for N=4 shapes (square, bowtie, “pyramid” with central dot)

### 1. Square (cycle graph C\_4)
- **Adjacency**: vertices {0,1,2,3} with edges (0-1,1-2,2-3,3-0).
- **Implementation**:
  1. Add a `graph_topology` flag to `app.py` (`path`, `cycle`, `diamond`, `bowtie`, `star`).
  2. In `core_qca`, replace the hard-coded “range(N-1)” loops with edge lists supplied by the topology helper.
  3. For C\_4, build the edge list once, then iterate over it for Hamiltonian construction and Λ extraction.
  4. Run `python app.py --N 4 --alpha 0.8 --points 40 --graph cycle`.
- **What to log**:
  - λ\_c1/λ\_rev/λ\_c2, SRQID metrics, Ramsey overlays.
  - Compare to open chain: periodic boundary should reduce edge effects (outer vs inner probes identical), so residuals should stay lower longer.

### 2. Bowtie (two degree-3 hubs + two degree-2 connectors)
- **Adjacency**: label vertices A,B,C,D. Edges (A-C, C-B, B-D, D-A, C-D) – literal “bow-tie”.
- **Implementation**:
  1. Define the edge list explicitly in the topology helper.
  2. Choose probes: one hub vs one connector to see how Λ differs by degree.
  3. Same CLI invocation with `--graph bowtie`.
- **What to log**:
  - Expect λ\_rev to differ for hubs vs connectors; document Λ\_hub / Λ\_leaf curves and whether Postulate 1 still aligns.
  - SRQID tests should detect the higher local degree through v\_LR shifts.

_(A proper rhombus/diamond—with four equal-length sides and no cross edge—is tracked separately under the `diamond` topology set to N=4, serving as the square/rotated-cycle control.)_

### 3. “Pyramid” (star with central dot)
- **Adjacency**: star graph K\_{1,3} (center index 0, leaves 1–3).
- **Implementation**:
  1. Edge list (0-1, 0-2, 0-3).
  2. Because there are only 3 bonds, Hilbert space is still small; reuse N=4 settings.
  3. CLI: `python app.py --N 4 --alpha 0.8 --points 40 --graph star`.
- **What to log**:
  - Central clock (probe 0) vs leaf (probe 1). λ\_rev should shift far left because Λ\_center builds faster.
  - Note any change in catastrophic λ\_c2: with only one promotion hub the inversion may occur earlier.

### 4. Documentation/figures
- For each topology create a subfolder `figures/run_<timestamp>_N4_<graph>` and store the phase diagram.
- Extend `docs/results.md` with a table summarizing λ\_c1/λ\_rev/λ\_c2 for each graph.
- Add a new subsection in `docs/status_202511.tex` discussing geometry dependence, including a short comparison plot (e.g., λ\_rev vs graph average degree).

### 5. Code tasks (summary)
1. Create `topologies.py` with functions returning edge lists and probe selections.
2. Update `core_qca.ExactQCA` to accept `edges` and iterate over them in Hamiltonian construction, SRQID, Λ calculations.
3. Pass the topology from CLI → production_run → ParameterScanner (so each worker builds the correct graph).
4. Add regression tests: ensure path topology reproduces previous results.

Once N=4 shapes are stable, repeat the process for the N=5 star/pentagon/“house” graphs listed earlier.
