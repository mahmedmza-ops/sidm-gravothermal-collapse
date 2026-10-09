"""
Scratch diagnostic: (1) is Stage 2's initial v^2 off by ~7-9% from discretisation?
                    (2) a non-trivial conservation check of the Stage 3 conduction step.
Changes nothing. Imports gravothermal_collapse.py read-only to use its Stage 3 functions;
the import runs that file's setup checks (no time evolution).

(1) Compare against a CONTINUUM reference: the exact enclosed mass and hydrostatic pressure
of the Stage 1 profile, computed on a very fine grid. Then measure the error of
  - Stage 2 as written (copied below, N parametrised),
  - the Stage 3 shell construction with p = 0 at the edge (same problem as Stage 2),
  - the Stage 3 shell construction with P_ext at the edge (what the scaffold uses),
at N = 200, 400, 800, 1600. If an error halves each time N doubles, it is a first-order
discretisation error that vanishes with resolution; if it quarters, second order.
Run:  python diagnose_stage2_resolution.py
"""

import numpy as np
from scipy.integrate import quad
import gravothermal_collapse as gc

G = 1.0
alpha = 2.19
print()
print("=" * 78)


def rho_f(x):
    return 1.0 / (1.0 + x**alpha)


# ---------- continuum reference ----------
rf = np.geomspace(1e-6, 100.0, 400001)
f = 4.0 * np.pi * rf**2 * rho_f(rf)
M_f = 4.0 / 3.0 * np.pi * rf[0]**3 + np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(rf))))
g = G * M_f * rho_f(rf) / rf**2
seg = 0.5 * (g[1:] + g[:-1]) * np.diff(rf)
p0_f = np.concatenate((np.cumsum(seg[::-1])[::-1], [0.0]))   # p(100) = 0, Stage 2's boundary


def M_ref(x):
    return np.interp(np.log(x), np.log(rf), M_f)


def v2_ref(x, P_edge):
    return (np.interp(np.log(x), np.log(rf), p0_f) + P_edge) / rho_f(x)


# accuracy of the reference itself, checked independently with quad
M1_quad = quad(lambda x: 4.0 * np.pi * x**2 * rho_f(x), 0.0, 1.0, epsabs=1e-13, epsrel=1e-12)[0]
p1_quad = quad(lambda x: G * quad(lambda y: 4.0 * np.pi * y**2 * rho_f(y), 0.0, x, epsrel=1e-11)[0]
               * rho_f(x) / x**2, 1.0, 100.0, epsrel=1e-10, limit=200)[0]
print("REFERENCE ACCURACY (fine-grid reference vs independent quad at r = 1):")
print(f"  M(1): {M_ref(1.0):.10e} vs {M1_quad:.10e}   rel diff {abs(M_ref(1.0) / M1_quad - 1):.2e}")
print(f"  p(1): {np.interp(0.0, np.log(rf), p0_f):.10e} vs {p1_quad:.10e}"
      f"   rel diff {abs(np.interp(0.0, np.log(rf), p0_f) / p1_quad - 1):.2e}")
print()


# ---------- the three discrete constructions ----------
def stage2(N):
    """Stage 2 exactly as in gravothermal_collapse.py, with N as a parameter."""
    r = np.logspace(-2, 2, N)
    rho = 1.0 / (1.0 + r**alpha)
    dr = np.gradient(r)
    M = np.cumsum(4 * np.pi * r**2 * rho * dr)
    gravity_term = G * M * rho / r**2
    p = np.zeros_like(r)
    for i in range(N - 2, -1, -1):
        p[i] = p[i+1] + gravity_term[i] * (r[i+1] - r[i])
    return r, M, p / rho


def shells(N, use_P_ext):
    """Stage 3 scaffold construction (its own functions), with N as a parameter."""
    r_edge = np.concatenate(([0.0], np.logspace(-2, 2, N)))
    dm = np.array([quad(lambda x: 4.0 * np.pi * x**2 * gc.rho_stage1(x), r_edge[i], r_edge[i + 1])[0]
                   for i in range(N)])
    M_edge = np.concatenate(([0.0], np.cumsum(dm)))
    P_ext = gc.envelope_pressure(r_edge[-1], M_edge[-1]) if use_P_ext else 0.0
    P = gc.pressure_from_hse(r_edge, dm, P_ext)
    return gc.shell_centres(r_edge), P / gc.shell_density(r_edge, dm), P_ext


samples = np.array([0.0102, 0.1, 1.0, 10.0, 50.0])
Ns = [200, 400, 800, 1600]
P_ext_val = gc.P_ext
hdr = "  " + f"{'N':>10}" + "".join(f"{'r=' + format(s, 'g'):>12}" for s in samples)

print("1a. STAGE 2 ENCLOSED MASS: relative error M_stage2(r) / M_exact(r) - 1")
print(hdr)
for N in Ns:
    r_, M_, _ = stage2(N)
    err = np.interp(np.log(samples), np.log(r_), M_) / M_ref(samples) - 1.0
    print("  " + f"{N:10d}" + "".join(f"{e:12.4e}" for e in err))
print()

results = {}
for label, kind in [("1b. STAGE 2 v^2 (p = 0 edge) vs continuum with p = 0 edge", "stage2"),
                    ("1c. SHELLS v^2, p = 0 edge   vs continuum with p = 0 edge", "shells0"),
                    ("1d. SHELLS v^2, P_ext edge   vs continuum with P_ext edge", "shellsE")]:
    print(label + ":  relative error v2 / v2_exact - 1")
    print(hdr)
    rows = []
    for N in Ns:
        if kind == "stage2":
            x_, _, v_ = stage2(N)
            P_edge = 0.0
        else:
            x_, v_, P_edge = shells(N, kind == "shellsE")
        err = np.interp(np.log(samples), np.log(x_), v_) / v2_ref(samples, P_edge) - 1.0
        rows.append(err)
        print("  " + f"{N:10d}" + "".join(f"{e:12.4e}" for e in err))
    rows = np.array(rows)
    ratio = rows[:-1] / rows[1:]
    print("  error ratio err(N)/err(2N)  (~2: first order, ~4: second order)")
    for k in range(len(Ns) - 1):
        print("  " + f"{Ns[k]}/{Ns[k + 1]}".rjust(10) + "".join(f"{q:12.3f}" for q in ratio[k]))
    results[kind] = rows
    print()

print(f"  continuum v2 at the samples, p = 0 edge : {np.round(v2_ref(samples, 0.0), 6)}")
print(f"  continuum v2 at the samples, P_ext edge : {np.round(v2_ref(samples, P_ext_val), 6)}   (P_ext = {P_ext_val:.6e})")
print()

# ---------- (2) conservation check of the actual Stage 3 conduction step ----------
print("=" * 78)
print("2. CONDUCTION STEP CONSERVATION (gc.conduction_step on the scaffold's initial state)")
print("   'moved' = sum(1.5 dm |dv2|) / E_th: how much heat the step actually redistributed.")
print("   A meaningful check needs 'moved' to be large while 'drift' stays at round-off.")
print("=" * 78)
v2_0 = gc.v2_s
E0 = np.sum(1.5 * gc.dm * v2_0)
print(f"  {'dt':>8} {'max|dv2|':>12} {'max|dv2|/v2':>12} {'moved':>12} {'drift':>12} {'min v2':>10} {'v2[0]':>10}")
for dt in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
    v1 = gc.conduction_step(gc.r_edge, gc.dm, v2_0, dt)
    dv = v1 - v2_0
    E1 = np.sum(1.5 * gc.dm * v1)
    moved = np.sum(1.5 * gc.dm * np.abs(dv)) / E0
    print(f"  {dt:8.0e} {np.abs(dv).max():12.4e} {np.max(np.abs(dv) / v2_0):12.4e} {moved:12.4e}"
          f" {(E1 - E0) / E0:12.4e} {v1.min():10.6f} {v1[0]:10.6f}")
