Here is a ready-to-use prompt for creating an interactive 3D visualization of the QATNU/SRQID dynamics:

---

**Role**: You are a creative technical director building a web-based physics animation.  
**Goal**: Create a Three.js (preferred) or Python (compromise) visualization that makes the abstract QATNU/SRQID mechanism visceral and intuitive to a quantum information audience.  
**Core Physics to Visualize** (from docs status_202511.pdf, QATNU_latest.pdf, SRQID_latest.pdf):

---

### **Scene Setup**
- **1D chain** of N=5 matter qubits (spheres at vertices), each with a local "clock" (oscillating color/brightness representing ⟨Z_i(t)⟩).
- **Edges** are cylindrical bonds connecting qubits. Each bond has:
  - A **frustration tile** (F_ij) rendered as a translucent box spanning the edge, glowing red when stabilizer violation detected (e.g., Z_i Z_j = -1)
  - A **bond register** visualized as a vertical stack of "rung" blocks (like a ladder) emerging from the edge, height = χ_ij (dimension). χ=1 → invisible, χ=2 → one rung, χ=4 → two rungs, etc.
  - A **promotion unit** (U_prom) as a pulsing halo around the bond, active when F_ij triggers INC_ij

---

### **Dynamic Visual Encoding**
1. **Clock Slowing (Postulate 1)**: 
   - Each qubit oscillates between blue (+1 eigenstate) and yellow (-1 eigenstate) at its local ω_eff.
   - ω_out (outermost qubit) is baseline speed; inner qubits visibly lag as Λ_i grows.
   - Display ω_in/ω_out ratio as a real-time number overlay.

2. **Entanglement Proxy Λ**:
   - Compute Λ_i = log(1 + average bond tiers within radius 1 of vertex i).
   - Map Λ_i to a color gradient on the qubit: deep purple (Λ=0) → bright white (Λ_max).
   - Show residual bar chart: |measured ω_in/ω_out - predicted (1+αΛ_out)/(1+αΛ_in)|

3. **Promotion Cascade**:
   - When F_ij detects frustration (red glow), the bond's rung stack grows upward over 0.5s animation (χ → 2χ).
   - New rungs spawn with a "crystallization" effect (particles coalescing).
   - α (currently 0.8) controls growth rate: higher α = faster stacking per frustration event.

4. **Lieb-Robinson Light Cone**:
   - On user-triggered local quench (click a qubit), emit a translucent expanding sphere from that vertex at speed v_LR ≈ 1.96.
   - Bonds/Λ values only update when the cone passes them—visual proof of causality.

5. **Phase Transitions**:
   - Slider for λ (promotion strength) from 0 to 1.2.
   - Annotate critical points λ_c1 (0.203 for N=4), λ_rev (1.058), λ_c2 (1.095) with vertical markers.
   - At λ > λ_c2, show **catastrophic inversion**: inner clock runs *faster* than outer (color inversion, visual "break" effect).

6. **Spin-2 Sector Hint**:
   - On a separate panel, compute PSD from χ profiles: show a 1D "wave" mode emerging from the bond lattice with ~1/k² damping.
   - Represent as ripples on a 2D surface behind the main chain.

---

### **Interaction & Controls**
- **Play/Pause** time evolution (t from 0 to T_max ≈ 50/ω).
- **Sliders**: λ (0–1.5), α (0.5–1.5), N (4 or 5).
- **Preset buttons**: "Phase I (λ=0.1)", "Revival (λ=1.058)", "Catastrophic (λ=1.1)".
- **Click qubit**: Initiate local X quench to show no-signalling.
- **Camera**: OrbitControls to rotate, zoom into bond registers.

---

### **Data Source**
- Feed the animation from actual ED simulation outputs:
  - `outputs/run_<timestamp>_N4_alpha0.80/scan_run_<timestamp>_N4_alpha0.80.csv`
  - Contains ⟨Z_i(t)⟩, Λ_i(t), χ_ij(t) for each λ point.
  - Pre-load data for smooth 60fps interpolation.

---

### **Aesthetic Style**
- **Dark background** (space-like) with neon accents for quantum states.
- **Minimalist UI**: Semi-transparent HUD panels with monospaced font (IBM Plex Mono) showing real-time numbers.
- **Smooth easing**: Use lerp for all transitions; physics is discrete but animation flows.
- **Sound design** (optional): Each promotion event triggers a subtle chime; frequency of clock oscillations mapped to low hum pitch.

---

### **Deliverable**
A single `index.html` (Three.js) or `visualize.py` (Python + VisPy/Mayavi) that:
1. Loads CSV data
2. Renders the scene above
3. Updates in real-time as user adjusts parameters
4. Exports a 30s video clip for papers

Start with N=4, α=0.8, λ=1.058 (revival) as the default demo loop.

---

This prompt is ready to paste into a development environment or give to a technical artist. The key is mapping abstract operators (F_ij, R_ij, Λ) to concrete visual primitives (glowing boxes, growing ladders, color-coded clocks) that make the self-referential loop physically palpable.
