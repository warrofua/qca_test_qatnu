
# SRQID numerics: Lieb–Robinson velocity, no-signalling quench, energy drift
# Produces CSVs in ./outputs and PDFs in ./figures
import os, csv
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# --------- Pauli and helpers ---------
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
I = np.eye(2, dtype=complex)

def kronN(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def local(op, i, N):
    return kronN([op if k==i else I for k in range(N)])

def unitary_from_H(H, t):
    evals, evecs = np.linalg.eigh(H)
    phase = np.exp(-1j*evals*t)
    return evecs @ np.diag(phase) @ evecs.conj().T

def heisenberg(U, A):
    return U.conj().T @ A @ U

def comm_norm(A, B):
    C = A @ B - B @ A
    s = np.linalg.svd(C, compute_uv=False)
    return float(np.real(s[0]))

# --------- Model parameters (toy chain) ---------
N      = int(os.environ.get("SRQID_N", 8))         # sites
omega  = float(os.environ.get("SRQID_OMEGA", 1.0)) # transverse X field
J      = float(os.environ.get("SRQID_J", 1.0))     # ZZ coupling
t_max  = float(os.environ.get("SRQID_TMAX", 3.0))
ngrid  = int(os.environ.get("SRQID_NGRID", 61))
eps    = float(os.environ.get("SRQID_EPS", 1e-3))

# Build H = sum_i (omega/2) X_i  + sum_<i,i+1> J Z_i Z_{i+1}
dim = 2**N
H = np.zeros((dim, dim), dtype=complex)
for i in range(N):
    H += (omega/2.0) * local(X, i, N)
for i in range(N-1):
    H += J * (local(Z, i, N) @ local(Z, i+1, N))

# --------- 1) Lieb–Robinson velocity via commutator-growth threshold ---------
A0 = local(Z, 0, N)
ts = np.linspace(0, t_max, ngrid)
front = []
for r in range(1, N):
    Br = local(Z, r, N)
    t_hit = np.nan
    for t in ts:
        U = unitary_from_H(H, t)
        At = heisenberg(U, A0)
        cn = comm_norm(At, Br)
        if cn > eps:
            t_hit = t; break
    front.append((r, t_hit))

front = np.array(front, dtype=float)
mask = ~np.isnan(front[:,1])
if np.any(mask):
    coef = np.polyfit(front[mask,1], front[mask,0], 1)
    vLR, intercept = float(coef[0]), float(coef[1])
else:
    vLR, intercept = float("nan"), float("nan")

# Save CSV and plot
with open("outputs/lr_fit.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["r","t_hit"])
    for r,t in front: w.writerow([int(r), t])
with open("outputs/summary.txt", "w") as f:
    f.write(f"v_LR_est ~ {vLR:.6f} (eps={eps})\n")

plt.figure()
plt.scatter(front[:,1], front[:,0], label="threshold hits")
if not np.isnan(vLR):
    tfit = np.linspace(0, np.nanmax(front[:,1])*1.05, 100)
    plt.plot(tfit, vLR*tfit + intercept, label=f"fit v≈{vLR:.3f}")
plt.xlabel("time t"); plt.ylabel("distance r"); plt.legend()
plt.title("Commutator-growth threshold")
plt.tight_layout()
plt.savefig("figures/lr_front.pdf")
plt.close()

# --------- 2) No-signalling local quench at far site ---------
def evolve_states(H, psi0, ts):
    evals, evecs = np.linalg.eigh(H)
    coeffs = evecs.conj().T @ psi0
    out = []
    for t in ts:
        phase = np.exp(-1j*evals*t)
        psi_t = evecs @ (phase * coeffs)
        out.append(psi_t)
    return out

zero = np.array([[1],[0]], dtype=complex)
psi0 = kronN([zero[:,0] for _ in range(N)]).reshape(-1,1)
psi_quench = local(X,0,N) @ psi0
traj0 = evolve_states(H, psi0, ts)
trajQ = evolve_states(H, psi_quench, ts)

r_far = N-1; Zr = local(Z, r_far, N)
def expval(psi, O): return float(np.real((psi.conj().T @ (O @ psi))[0,0]))
diff = np.array([expval(trajQ[i], Zr) - expval(traj0[i], Zr) for i in range(len(ts))], dtype=float)

with open("outputs/quench.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["t","delta_Zr"])
    for i,t in enumerate(ts): w.writerow([t, diff[i]])

plt.figure()
plt.plot(ts, diff)
plt.axhline(0, ls='--')
plt.xlabel("t"); plt.ylabel(r"$\Delta \langle Z_r(t)\rangle$")
plt.title(f"No-signalling local quench (r={r_far})")
plt.tight_layout()
plt.savefig("figures/quench.pdf")
plt.close()

# --------- 3) Energy drift ---------
def energy(psi): return float(np.real((psi.conj().T @ (H @ psi))[0,0]))
E = [energy(traj0[i]) for i in range(len(ts))]
dE = float(np.max(E) - np.min(E))

with open("outputs/energy.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["t","E"])
    for i,t in enumerate(ts): w.writerow([t, E[i]])

with open("outputs/summary.txt", "a") as f:
    f.write(f"energy_drift ~ {dE:.6e}\n")

plt.figure()
plt.plot(ts, E)
plt.xlabel("t"); plt.ylabel("⟨H⟩")
plt.title("Energy conservation")
plt.tight_layout()
plt.savefig("figures/energy.pdf")
plt.close()

print(f"[SRQID] Estimated v_LR ≈ {vLR:.3f}  (eps={eps})")
print(f"[SRQID] Max |Δ⟨Z_r(t)⟩| at r={r_far}: {np.max(np.abs(diff)):.3e}")
print(f"[SRQID] Energy drift ΔE = {dE:.3e}")
