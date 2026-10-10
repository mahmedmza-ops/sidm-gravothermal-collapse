"""
Check of the NFW initial halo (gravothermal_collapse.build_nfw_halo). No evolution is run.
Changes nothing; imports gravothermal_collapse.py (its cored-halo setup checks print first).

Predictions stated BEFORE running (two, they disagree; the numbers decide):
  advisor's brief : NFW v^2 rises toward the centre ("hot cusp"), no inversion; the centre is
                    self-gravitating (|W|/3p_sV not stuck well below 1).
  Claude (memory, unsourced this session): isotropic hydrostatic NFW has v^2 -> 0 at the centre
                    (~ r ln(1/r)), peaking near r ~ 0.76 r_s, i.e. a STRONGER inversion than the
                    cored profile; |W|/3p_sV < 1 at small R, falling slowly inward (~ 1/ln(1/R)).
Run:  python check_nfw_initial.py
"""

import numpy as np
from scipy.integrate import quad
import gravothermal_collapse as gc

G = 1.0
h = gc.build_nfw_halo()
r_e, dm, P_ext, rho, v2 = h["r_edge"], h["dm"], h["P_ext"], h["rho"], h["v2"]
n = len(dm)
rc = gc.shell_centres(r_e)

print()
print("=" * 96)
print(f"NFW INITIAL HALO: {n} shells, r from {r_e[1]:g} to {r_e[-1]:g} r_s (shell 0 = sphere inside {r_e[1]:g})")
print("=" * 96)

# A. masses
M_closed = gc.mass_nfw(r_e[-1])
print("A. MASS")
print(f"  sum of shell masses {np.sum(dm):.12e}   closed-form M(30) {M_closed:.12e}"
      f"   rel diff {abs(np.sum(dm) / M_closed - 1):.2e}")
worst = 0.0
for i in [0, 1, 50, 100, 150, n - 1]:
    q = quad(lambda x: 4.0 * np.pi * x**2 * gc.rho_nfw(x), r_e[i], r_e[i + 1], epsrel=1e-13, epsabs=0)[0]
    worst = max(worst, abs(dm[i] / q - 1))
print(f"  closed-form shell masses vs quad for shells 0,1,50,100,150,{n - 1}: worst rel diff {worst:.2e}")
print()

# B. envelope pressure
P_quad, P_err = quad(lambda x: gc.rho_nfw(x) * G * gc.mass_nfw(x) / x**2, r_e[-1], np.inf, epsrel=1e-12)
print("B. ENVELOPE PRESSURE at r = 30 (weight of the NFW halo beyond the grid)")
print(f"  P_ext (log-grid trapezoid, as used) {P_ext:.10e}   quad with closed-form M {P_quad:.10e}"
      f" (quad error estimate {P_err:.1e})   rel diff {abs(P_ext / P_quad - 1):.2e}")
print()

# C. fixed point
res = gc.hydrostatic_residual(r_e, dm, h["P"], P_ext)
r1, rho1, v21, info = gc.relax_to_hydrostatic(r_e, dm, h["s"], P_ext)
print("C. FIXED POINT of the relaxation solver")
print(f"  max |hydrostatic residual| (relative to gravity term) = {np.max(np.abs(res)):.3e}")
print(f"  relax_to_hydrostatic: Newton updates {info['iterations']}, worst row / threshold"
      f" {info['max_row_ratio']:.3f}, max|dr/r| {np.max(np.abs(r1[1:] / r_e[1:] - 1)):.3e},"
      f" max|dv2/v2| {np.max(np.abs(v21 / v2 - 1)):.3e}")
print()

# D. v^2 profile, against the continuum hydrostatic solution
def p_cont(x):
    return P_quad + quad(lambda y: gc.rho_nfw(y) * G * gc.mass_nfw(y) / y**2, x, r_e[-1],
                         epsrel=1e-12, limit=200)[0]


h2 = gc.build_nfw_halo(n_shell=400)
rc2 = gc.shell_centres(h2["r_edge"])
print("D. v^2 PROFILE: shells (200 and 400) vs continuum p(r)/rho(r) from quad")
print(f"  {'r':>9} {'v2 shells':>11} {'v2 continuum':>13} {'200 vs cont':>12} {'400 vs cont':>12}")
for target in [0.01, 0.03, 0.1, 0.3, 0.76, 1.0, 3.0, 10.0, 28.0]:
    k = int(np.argmin(np.abs(np.log(rc / target))))
    vc = p_cont(rc[k]) / gc.rho_nfw(rc[k])
    v400 = np.exp(np.interp(np.log(rc[k]), np.log(rc2), np.log(h2["v2"])))
    print(f"  {rc[k]:9.4f} {v2[k]:11.6f} {vc:13.6f} {v2[k] / vc - 1:+12.3e} {v400 / vc - 1:+12.3e}")
kmax = int(np.argmax(v2))
print(f"  centre (shell 0, r < {r_e[1]:g}): v2 = {v2[0]:.6f};  max v2 = {v2[kmax]:.6f} at r = {rc[kmax]:.4f}")
print(f"  -> {'CENTRALLY PEAKED' if kmax == 0 else 'INVERTED: v2 peaks off-centre'};"
      f"  max/centre = {v2[kmax] / v2[0]:.4f}")
L = gc.luminosity(r_e, dm, v2)
print(f"  edges conducting heat INWARD (L < 0) at t = 0: {np.sum(L[1:-1] < 0)} of {n - 1}"
      + (f"; outermost at r = {r_e[1:-1][L[1:-1] < 0].max():.4f}" if np.any(L[1:-1] < 0) else ""))
print()

# E. self-gravity of inner regions
print("E. SELF-GRAVITY of regions r < R:  2K + W = 3 p_s V;  |W|/3p_sV > 1 means self-gravitating")
print(f"  {'R used':>9} {'shells':>6} {'mass frac':>10} {'virial resid':>13} {'|W|/3psV':>9}")
for R in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
    J = int(np.sum(r_e[1:] <= R * (1 + 1e-12)))
    twoK, W, PS = gc.region_virial(r_e, dm, v2, J, P_ext)
    print(f"  {r_e[J]:9.4f} {J:6d} {np.sum(dm[:J]) / np.sum(dm):10.3e} {(twoK + W - PS) / twoK:13.3e} {abs(W) / PS:9.4f}")
print()
print("No evolution was run. The NFW halo is not used by any other script yet.")
