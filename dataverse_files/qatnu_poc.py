
# QATNU Proof-of-Concept numerics: local unitarity, causal cone, spin-2 tail, MERA, back-reaction
import os, csv
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
try:
    from scipy.stats import pearsonr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

os.makedirs("outputs", exist_ok=True)
os.makedirs("figures", exist_ok=True)

def save_txt(path, text):
    with open(path, "a") as f: f.write(text + "\n")

# -------------------- 1) Local unitarity (8-qubit brickwork) --------------------
H = (1/np.sqrt(2))*np.array([[1, 1],
                             [1,-1]], dtype=complex)
CZ = np.diag([1,1,1,-1]).astype(complex)

def kronN(ops):
    out = ops[0]
    for g in ops[1:]:
        out = np.kron(out, g)
    return out

n   = 8
dim = 2**n
U1 = kronN([H]*n)

# Apply CZ on pairs (0,1), (2,3), ... (6,7)
U2 = np.eye(dim, dtype=complex)
def CZ_on_pair(a, b, n):
    I2 = np.eye(2, dtype=complex)
    ops = [I2]*n
    # Build via tensor insertion trick: act with CZ on (a,b)
    # We'll construct by iterating computational basis—efficient enough for n=8
    U = np.zeros((2**n, 2**n), dtype=complex)
    for idx in range(2**n):
        bits = [(idx>>k)&1 for k in range(n)][::-1]
        phase = -1 if (bits[a]==1 and bits[b]==1) else 1
        U[idx,idx] = phase
    return U

for a in range(0, n, 2):
    U2 = CZ_on_pair(a, a+1, n) @ U2

U = U2 @ U1
err = norm(U.conj().T @ U - np.eye(dim))
save_txt("outputs/qatnu_summary.txt", f"unitarity_err = {err:.2e}")

# -------------------- 2) 1-D Hadamard QCA: causal cone + dispersion -------------
N = 121; steps = 40
psi = np.zeros((N,2), complex); psi[N//2,1]=1
def step(state):
    s = state@H.T
    new=np.zeros_like(s)
    new[1:,0]=s[:-1,0]; new[:-1,1]=s[1:,1]
    return new
hist=[psi]
for _ in range(steps): hist.append(step(hist[-1]))
prob = np.sum(np.abs(hist)**2,2)
plt.imshow(prob,aspect='auto',origin='lower',
           extent=[-steps,steps,0,steps],cmap='inferno')
plt.plot(np.arange(-steps,steps+1), np.arange(-steps,steps+1),'w--')
plt.plot(np.arange(-steps,steps+1),-np.arange(-steps,steps+1),'w--')
plt.title('1-D causal light-cone'); plt.xlabel('x'); plt.ylabel('t')
plt.tight_layout(); plt.savefig("figures/qca_lightcone.pdf"); plt.close()

Ndk = 256
k = np.linspace(-np.pi, np.pi, Ndk)
omega = np.arccos(np.cos(k)/np.sqrt(2))
plt.plot(k, omega, label='+ω(k)')
plt.plot(k,-omega, label='-ω(k)')
plt.xlabel('k'); plt.ylabel('ω'); plt.title('Dirac-like dispersion'); plt.legend()
plt.tight_layout(); plt.savefig("figures/dispersion.pdf"); plt.close()

# -------------------- 3) Spin-2 tail: 1/k^2 in 1-D/2-D/3-D ---------------------
rng = np.random.default_rng(0)

def psd_1d(N=4096, steps=4000, p0=0.05, lam=1e-5):
    tier=np.zeros(N,int); r=np.arange(1,N+1)
    p_prom=np.minimum(p0/r**2,0.15)
    for _ in range(steps):
        tier+= (rng.random(N)<p_prom)
        chi=2**tier
        dem = rng.random(N)<lam*(chi-1)**3
        tier[dem&(tier>0)]-=1
    dchi=2**tier - (2**tier).mean()
    k=np.fft.rfftfreq(N)*2*np.pi
    psd=np.abs(np.fft.rfft(dchi))**2
    return k, psd

def plot_psd(k, psd, fname, title):
    plt.loglog(k[1:], psd[1:], label='PSD')
    plt.loglog(k[1:], psd[1]*(k[1]/k[1:])**2, '--', label='1/k²')
    plt.legend(); plt.title(title); plt.xlabel('k'); plt.ylabel('Power')
    plt.tight_layout(); plt.savefig(fname); plt.close()

k1, p1 = psd_1d(N=2048, steps=3000)
plot_psd(k1, p1, "figures/graviton_psd_1d.pdf", "1-D spin-2 tail")

# 2-D small (128x128)
NX=NY=int(os.environ.get("QATNU_SIZE_2D", 128)); steps2=3000; p0=0.05; lam=1e-5
X,Y=np.meshgrid(np.arange(NX),np.arange(NY),indexing='ij')
r2=(X-NX//2)**2+(Y-NY//2)**2+1
tier=np.zeros((NX,NY),int); p=np.minimum(p0/r2,0.12)
for _ in range(steps2):
    tier+=(rng.random((NX,NY))<p)
    chi=2**tier
    dem=rng.random((NX,NY))<lam*(chi-1)**3
    tier[dem&(tier>0)]-=1
chi=2**tier
dchi=chi-chi.mean()
FT=np.fft.fftshift(np.fft.fft2(dchi)); P=np.abs(FT)**2
kx=np.fft.fftshift(np.fft.fftfreq(NX))*NX
ky=np.fft.fftshift(np.fft.fftfreq(NY))*NY
KX,KY=np.meshgrid(kx,ky,indexing='ij'); kr=np.sqrt(KX**2+KY**2)
bins=np.arange(1,int(kr.max()))
ps=np.array([P[(kr>=k-0.5)&(kr<k+0.5)].mean() for k in bins])
plt.loglog(bins[1:],ps[1:],label='radial PSD')
plt.loglog(bins[1:],ps[1]*(bins[1]/bins[1:])**2,'--',label='1/k²')
plt.legend(); plt.title('2-D spin-2 tail'); plt.tight_layout()
plt.savefig("figures/graviton_psd_2d.pdf"); plt.close()

# 3-D tiny (32^3) for speed
NZ=int(os.environ.get("QATNU_SIZE_3D", 32)); NX=NY=NZ; steps3=2000
tier=np.zeros((NX,NX,NZ),int)
X,Y,Z=np.indices((NX,NX,NZ))
r2=(X-NX//2)**2+(Y-NY//2)**2+(Z-NZ//2)**2+1
p=np.minimum(0.04/r2,0.12)
for _ in range(steps3):
    tier+=(rng.random((NX,NX,NZ))<p)
    chi=2**tier
    dem=rng.random((NX,NX,NZ))<8e-6*(chi-1)**3
    tier[dem&(tier>0)]-=1
chi=2**tier
dchi=chi-chi.mean()
P=np.abs(np.fft.fftshift(np.fft.fftn(dchi)))**2
k=np.fft.fftshift(np.fft.fftfreq(NX))*NX
KX,KY,KZ=np.meshgrid(k,k,k,indexing='ij'); kr=np.sqrt(KX**2+KY**2+KZ**2)
bins=np.arange(1,int(kr.max()))
ps=np.array([P[(kr>=kv-0.5)&(kr<kv+0.5)].mean() for kv in bins])
plt.loglog(bins[1:],ps[1:],label='3-D PSD')
plt.loglog(bins[1:],ps[1]*(bins[1]/bins[1:])**2,'--',label='1/k²')
plt.legend(); plt.title('3-D spin-2 tail'); plt.tight_layout()
plt.savefig("figures/graviton_psd_3d.pdf"); plt.close()

# -------------------- 4) MERA coarse-graining: Hausdorff flow -------------------
def D_H_1d(field):
    L=1; scale=[]; cnt=[]
    while L<=len(field)//2:
        scale.append(L)
        cnt.append(sum(np.any(np.abs(field[i:i+L]-1)>0.01) for i in range(0,len(field),L)))
        L*=2
    return np.polyfit(np.log10(1/np.array(scale)), np.log10(cnt),1)[0]

tier_1d = np.zeros(2048,int)
r = np.arange(1,2049); p_prom=np.minimum(0.03/r**2,0.15)
for _ in range(2500):
    tier_1d += (rng.random(2048)<p_prom)
    chi1 = 2**tier_1d
    dem = rng.random(2048) < 1e-5*(chi1-1)**3
    tier_1d[dem&(tier_1d>0)]-=1
chi1 = 2**tier_1d
theta=np.pi/4; U2=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
def coarse_1d(arr): return 0.5*((U2@(np.vstack([arr[0::2],arr[1::2]]))).mean(0))
dims=[]; cur=chi1.copy()
for _ in range(6):
    dims.append(D_H_1d(cur)); cur=coarse_1d(cur)
plt.plot(dims,marker='o'); plt.title('Hausdorff flow 1-D'); plt.xlabel('layer'); plt.ylabel('D_H')
plt.tight_layout(); plt.savefig("figures/mera_dim_flow_1d.pdf"); plt.close()

# -------------------- 5) Back-reaction: δχ vs T correlation --------------------
N=2048; steps=3000; p0,lam,alpha = 0.02,1e-5,0.8
r=np.arange(1,N+1); T=1.0/r**2; tier=np.zeros(N,int)
for _ in range(steps):
    p_prom=np.minimum(p0/r**2 + alpha*T,0.15)
    tier += (rng.random(N)<p_prom)
    chi=2**tier
    dem=rng.random(N) < lam*(chi-1)**3
    tier[dem&(tier>0)]-=1
dchi = 2**tier - (2**tier).mean()

if HAVE_SCIPY:
    rval,pval = pearsonr(dchi,T)
else:
    xm=(dchi-dchi.mean())/dchi.std(); ym=(T-T.mean())/T.std()
    rval = float(np.mean(xm*ym)); pval = float("nan")
plt.scatter(T,dchi,s=3,alpha=0.4)
plt.xlabel("Local stress T(r)"); plt.ylabel("δχ")
plt.title(f"δχ vs T (Pearson r ≈ {rval:.2f})")
plt.tight_layout(); plt.savefig("figures/dchi_vs_T_scatter.pdf"); plt.close()
save_txt("outputs/qatnu_summary.txt", f"delta_chi_vs_T_pearson_r = {rval:.4f}, p = {pval}")

print("QATNU PoC complete. See figures/ and outputs/qatnu_summary.txt")
