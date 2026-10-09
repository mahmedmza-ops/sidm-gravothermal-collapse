"""
Scratch diagnostic for gravothermal_collapse.py Stage 3. Throwaway: it does NOT import
from or modify gravothermal_collapse.py, and changes no solver. It only asks whether the
density-reconstruction ("rebalance") step used by Stage 3 is a fixed point on the verified
Stage 1-2 initial state. If the reconstruction were correct, calling it on the unchanged
initial v^2 should return the initial density, so rho_new[0]/rho[0] would be ~1.

Stage 1-2 setup below is copied line-for-line from gravothermal_collapse.py.
The two reconstruction functions are faithful copies of:
  - rebalance_head : hydrostatic_rebalance() in the committed version (HEAD, exp form)
  - rebalance_wt   : hydrostatic_from_v2()   in the working tree (Newton/LM, (1+x) form)
Run:  python diagnose_stage3.py
"""

import numpy as np

G = 1.0

# ---------- Stage 1-2 initial state (copied unchanged) ----------
N = 200
r = np.logspace(-2, 2, N)
alpha = 2.19
rho = 1.0 / (1.0 + r**alpha)
dr = np.gradient(r)
M = np.cumsum(4 * np.pi * r**2 * rho * dr)

gravity_term = G * M * rho / r**2
p = np.zeros_like(r)
for i in range(N - 2, -1, -1):
    p[i] = p[i+1] + gravity_term[i] * (r[i+1] - r[i])
v2 = p / rho

M_total = 4.0 * np.pi * np.trapezoid(r**2 * rho, r)
v2_clamped = np.maximum(v2, 1e-8)   # what Stage 3 actually starts from (line 150)


def enclosed_mass(r, rho):
    """Same quantity in both committed and working-tree versions."""
    d = np.gradient(r)
    shell_mass = 4.0 * np.pi * r**2 * rho * d
    return np.concatenate(([0.0], np.cumsum(shell_mass[1:])))


def rebalance_head(r, rho, v2, M_total):
    """Committed version: P[i] = P[i+1] * exp(G M dr / (r^2 v2)); rho = P/v2; rescale to mass."""
    Menc = enclosed_mass(r, rho)
    P = np.zeros_like(r)
    P[-1] = rho[-1] * v2[-1]
    for i in range(len(r) - 2, -1, -1):
        dr_i = r[i + 1] - r[i]
        coeff = G * Menc[i] / (r[i]**2 * (v2[i] + 1e-12))
        P[i] = P[i + 1] * np.exp(coeff * dr_i)
    rho_raw = np.clip(P / (v2 + 1e-12), 1e-12, 1e6)
    mass = 4.0 * np.pi * np.trapezoid(r**2 * rho_raw, r)
    scale = M_total / mass
    return P, rho_raw, scale, rho_raw * scale


def rebalance_wt(r, rho_ref, v2, M_total):
    """Working-tree version: explicit P[i] = P[i+1] + G M rho_guess dr / r^2; rescale; clip."""
    P = np.empty_like(v2)
    Menc = enclosed_mass(r, rho_ref)
    P[-1] = rho_ref[-1] * v2[-1]
    for i in range(len(r) - 2, -1, -1):
        dr_i = r[i + 1] - r[i]
        rho_guess = P[i + 1] / (v2[i] + 1e-12)
        dP_dr = G * Menc[i] * rho_guess / (r[i]**2 + 1e-12)
        P[i] = P[i + 1] + dP_dr * dr_i
    rho_raw = np.clip(P / (v2 + 1e-12), 1e-12, 1e6)
    mass = 4.0 * np.pi * np.trapezoid(r**2 * rho_raw, r)
    scale = M_total / mass
    rho_final = np.clip(rho_raw * scale, 1e-12, 1e6)
    return P, rho_raw, scale, rho_final


def report(name, fn):
    P, rho_raw, scale, rho_new = fn(r, rho, v2_clamped, M_total)
    mass_after = 4.0 * np.pi * np.trapezoid(r**2 * rho_new, r)
    print(f"--- {name} ---")
    print(f"  rho[0]  initial            = {rho[0]:.6e}")
    print(f"  rho_new[0] after one call  = {rho_new[0]:.6e}")
    print(f"  RATIO rho_new[0]/rho[0]    = {rho_new[0] / rho[0]:.6e}   (1.0 if initial state were a fixed point)")
    print(f"  P[-1] (anchor, = rho[-1]*v2[-1]) = {P[-1]:.6e}   true Stage-2 p[-2] = {p[-2]:.6e}")
    print(f"  rho_raw[-2]/rho_raw[-1] (before mass rescale) = {rho_raw[-2] / rho_raw[-1]:.6e}   initial rho[-2]/rho[-1] = {rho[-2] / rho[-1]:.6e}")
    print(f"  mass rescale factor M_total/mass = {scale:.6e}")
    print(f"  rho_new[-1]/rho[-1] = {rho_new[-1] / rho[-1]:.6e}   rho_new[-2]/rho[-2] = {rho_new[-2] / rho[-2]:.6e}")
    print(f"  mass after = {mass_after:.6e}  vs M_total = {M_total:.6e}")
    # how much of the (trapezoid) mass sits in the outermost interval
    edge_mass = 0.5 * 4.0 * np.pi * (r[-1]**2 * rho_new[-1] + r[-2]**2 * rho_new[-2]) * (r[-1] - r[-2])
    print(f"  fraction of total mass in outermost interval = {edge_mass / mass_after:.4f}")


print("=" * 72)
print("A. BOUNDARY: the Stage 1-2 initial state at the outer edge")
print("=" * 72)
print(f"  v2[-3:]            (Stage 2, unclamped) = {v2[-3:]}")
print(f"  v2_clamped[-3:]    (what Stage 3 uses)  = {v2_clamped[-3:]}")
print(f"  p[-3:]             = {p[-3:]}")
print(f"  p[-1] = {p[-1]:.3e}   rho[-1] = {rho[-1]:.6e}   rho[0] = {rho[0]:.6e}")
print(f"  v2[-2]/v2_clamped[-1] (size of the temperature cliff) = {v2[-2] / v2_clamped[-1]:.3e}")
print(f"  M_total (trapezoid) = {M_total:.6e}")
print()

print("=" * 72)
print("B. FIXED-POINT TEST: reconstruct density from the UNCHANGED initial v^2")
print("=" * 72)
report("committed version (exp form)", rebalance_head)
print()
report("working-tree version ((1+x) form)", rebalance_wt)
print()

print("=" * 72)
print("C. PRESSURE-RATIO FACTOR: reconstruction factor vs the correct one")
print("   Stage 2 builds p[i] = p[i+1] + g*rho[i]*dr, i.e. p[i]/p[i+1] = 1/(1-x),")
print("   with x = g*dr / v2[i]  (Stage-2 definition, v2[i] = p[i]/rho[i]).")
print("=" * 72)
Menc = enclosed_mass(r, rho)
dri = np.empty(N - 1)
dri[:] = r[1:] - r[:-1]
x = G * Menc[:-1] * dri / (r[:-1]**2 * v2[:-1])      # i = 0 .. N-2, unclamped v2
true_ratio = np.full(N - 1, np.inf)
true_ratio[:-1] = p[:-2] / p[1:-1]                   # i = 0 .. N-3 (p[N-1]=0 -> inf at i=N-2)
print(f"  {'i':>4} {'r[i]':>10} {'x':>10} {'exp(x)':>10} {'1+x':>10} {'1/(1-x)':>10} {'true p[i]/p[i+1]':>18}")
for i in [0, 50, 100, 150, 170, 185, 190, 195, 197, N - 2]:
    inv = 1.0 / (1.0 - x[i]) if x[i] < 1.0 else np.inf
    print(f"  {i:4d} {r[i]:10.3e} {x[i]:10.4f} {np.exp(x[i]):10.4f} {1 + x[i]:10.4f} {inv:10.4f} {true_ratio[i]:18.4f}")

# cumulative pressure growth from cell N-2 up to the centre, excluding the edge cell (p[-1]=0)
idx = np.arange(0, N - 2)
cum_true = p[0] / p[N - 2]
cum_exp = np.prod(np.exp(x[idx]))
cum_lin = np.prod(1.0 + x[idx])
cum_inv = np.prod(1.0 / (1.0 - x[idx])) if np.all(x[idx] < 1.0) else np.nan
print()
print(f"  cumulative P[0]/P[N-2] over cells 0..N-3:")
print(f"    true (Stage 2)       = {cum_true:.6e}")
print(f"    prod exp(x)          = {cum_exp:.6e}   ratio to true = {cum_exp / cum_true:.6e}")
print(f"    prod (1+x)           = {cum_lin:.6e}   ratio to true = {cum_lin / cum_true:.6e}")
print(f"    prod 1/(1-x)         = {cum_inv:.6e}   ratio to true = {cum_inv / cum_true:.6e}   (should be ~1 if Stage-2 definition is self-consistent here)")
print(f"  max x over cells 0..N-3 = {x[idx].max():.4f}  (x>=1 would make 1/(1-x) singular)")
print()

print("=" * 72)
print("D. CONTROL: separate the boundary cliff from the fixed-point error")
print("   Anchor the pressure at the true Stage-2 value p[k] (not rho*1e-8), use the")
print("   UNCLAMPED v2 for cells i<k, keep original rho for cells i>=k. Diagnostic only.")
print("=" * 72)
k = N - 2
for name, form in [("exp(x)", "exp"), ("(1+x)", "lin")]:
    P = p.copy()
    for i in range(k - 1, -1, -1):
        dr_i = r[i + 1] - r[i]
        xi = G * Menc[i] * dr_i / (r[i]**2 * v2[i])
        P[i] = P[i + 1] * (np.exp(xi) if form == "exp" else (1.0 + xi))
    rho_c = rho.copy()
    rho_c[:k] = P[:k] / v2[:k]
    mass = 4.0 * np.pi * np.trapezoid(r**2 * rho_c, r)
    print(f"  anchor k={k}, {name:6s}: rho_rec[0]/rho[0] (no rescale) = {rho_c[0] / rho[0]:.6e}"
          f"   mass ratio = {mass / M_total:.6f}   after mass rescale: {rho_c[0] * M_total / mass / rho[0]:.6e}")
print()
print("=" * 72)
print("E. DEFECT 3: evolve the Stage-3 conduction equation alone, density FIXED, no PdV work")
print("   dv2/dt = (1/r^2) d/dr [ r^2 kappa dv2/dr ],  kappa = 1e-2 * rho * r^2 / sqrt(v2)")
print("   (same equation and kappa as Stage 3). Backward Euler, kappa lagged one substep,")
print("   conservative finite volume, zero-flux at both ends (the BC the comments intended).")
print("   Initial v2 is the clamped Stage-2 profile. Question: can the central v2 ever RISE?")
print("=" * 72)


def conduction_operator(r, rho, v2):
    """Return L (dv/dt = -L v) and cell weights w; w^T L = 0 so sum(w*v2) is conserved."""
    n = len(r)
    kappa = 1.0e-2 * rho * r**2 / (np.sqrt(np.maximum(v2, 1e-12)) + 1e-12)
    rh = 0.5 * (r[1:] + r[:-1])
    kh = 0.5 * (kappa[1:] + kappa[:-1])
    cond = rh**2 * kh / (r[1:] - r[:-1])
    Delta = np.empty(n)
    Delta[1:-1] = 0.5 * (r[2:] - r[:-2])
    Delta[0] = 0.5 * (r[1] - r[0])
    Delta[-1] = 0.5 * (r[-1] - r[-2])
    w = r**2 * Delta
    A = np.zeros((n, n))
    for j in range(n - 1):
        c = cond[j]
        A[j, j] += c
        A[j, j + 1] -= c
        A[j + 1, j + 1] += c
        A[j + 1, j] -= c
    return A / w[:, None], w


def evolve(v_init, t_out, n_sub):
    v = v_init.copy()
    t_prev = 0.0
    traj = []
    for t_next in t_out:
        dt = (t_next - t_prev) / n_sub
        for _ in range(n_sub):
            L, w = conduction_operator(r, rho, v)
            v = np.linalg.solve(np.eye(len(r)) + dt * L, v)
        traj.append(v.copy())
        t_prev = t_next
    return np.array(traj), w


t_out = np.logspace(-1, 5, 25)
traj1, w = evolve(v2_clamped, t_out, 1)
traj8, _ = evolve(v2_clamped, t_out, 8)
mean_v2 = np.sum(w * v2_clamped) / np.sum(w)
print(f"  initial v2[0] (centre) = {v2_clamped[0]:.6f}   initial max(v2) = {v2_clamped.max():.6f}"
      f"   centre is global max: {v2_clamped[0] == v2_clamped.max()}")
print(f"  weighted mean v2 (conserved) = {mean_v2:.6f}")
print(f"  {'t':>10} {'v2[0] 1 sub':>14} {'v2[0] 8 sub':>14} {'v2[i=100]':>12} {'v2[-2]':>12} {'v2[-1]':>12}")
print(f"  {0.0:10.3e} {v2_clamped[0]:14.6f} {v2_clamped[0]:14.6f} {v2_clamped[100]:12.6f} {v2_clamped[-2]:12.6f} {v2_clamped[-1]:12.3e}")
for k_, t_ in enumerate(t_out):
    print(f"  {t_:10.3e} {traj1[k_, 0]:14.6f} {traj8[k_, 0]:14.6f} {traj8[k_, 100]:12.6f} {traj8[k_, -2]:12.6f} {traj8[k_, -1]:12.3e}")

for label, traj in [("1 substep", traj1), ("8 substeps", traj8)]:
    c = np.concatenate(([v2_clamped[0]], traj[:, 0]))
    dc = np.diff(c)
    print(f"  [{label}] central v2: number of output intervals where it ROSE = {int(np.sum(dc > 0))} of {len(dc)};"
          f" largest single rise = {dc.max():.3e};  final/initial = {c[-1] / c[0]:.6f}")
    print(f"  [{label}] max over all cells and times of v2 minus initial max = {traj.max() - v2_clamped.max():.3e}")
    drift = np.abs(np.sum(w * traj, axis=1) - np.sum(w * v2_clamped)) / np.sum(w * v2_clamped)
    print(f"  [{label}] relative drift of conserved sum(w*v2) = {drift.max():.3e}")
print()
print("RECORDED OUTCOME OF E (owner's run): the original prediction 'central v2 never rises' was FALSIFIED.")
print("  The Stage-2 profile is not centrally peaked (centre 2.89 < max 4.04 off-centre), so the centre")
print("  warms by inflow from a hotter neighbouring shell (15 of 25 intervals, ~+37%), then everything")
print("  relaxes toward the mean. What E does show: max(v2) never exceeds its initial value and")
print("  sum(w*v2) is conserved to ~1e-16, so pure conduction only redistributes heat toward uniformity")
print("  and cannot give the unbounded central heating of gravothermal runaway. That needs PdV work.")
print()
print("Interpretation guide (numbers above are the evidence, not this text):")
print("  A/B: a huge cliff v2[-2]/v2[-1] and rho_new[0]/rho[0] far from 1 point to the boundary anchor.")
print("  C/D: cumulative ratios far from 1, and D ratios far from 1 with a clean anchor, point to the")
print("       fixed-point (discretisation) error independent of the boundary.")
