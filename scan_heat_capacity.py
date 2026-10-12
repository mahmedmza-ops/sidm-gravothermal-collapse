"""
EXPLORATORY DIAGNOSTIC -- cannot overturn the t=0 gate, which stands as FAIL (commit 24c571b).
Question: does negative heat capacity exist anywhere in this halo at t=0, and at what scale?
Changes nothing; imports gravothermal_collapse.py (setup checks print first; no evolution).

Choices fixed BEFORE running (advisor ruling):
  regions r < R for R = 0.3, 1, 3, 10, 30, 100 (100 = the whole halo); each region is the
  shells whose outer edge is <= R. Heat removed: 1% of each region shell's thermal energy
  (same as the gate). Effective heat capacity C = dU_region / dE, where dE < 0 is the energy
  removed and dU_region is the region's net thermal-energy change after relaxation.
  C < 0 means negative heat capacity. R = 1 is the gate's region, so it must reproduce +0.3232.

Static virial check (halo version of virial_theorem.py Part 1), on the initial state, per R.
Summing the discrete hydrostatic equations times 4 pi r_j^3 over edges 1..J gives exactly
    2K + W = 3 p_s V,    K = (3/2) sum dm v^2,  W = -sum_{j<=J} G M_j dmbar_j / r_j,
with p_s the pressure of the first shell outside the region (P_ext for the whole halo) and
V = (4 pi/3) r_J^3. (Discretisation detail: edge J's dmbar includes half the next shell.)
This identity holds whenever the state is in discrete equilibrium, so its residual checks the
code. The physics is in the split: a region is gravity-dominated where |W| > 3 p_s V, and
pressure-confined where the surface term dominates. virial_theorem.py is the case with no
surface term, where 2K + U = 0 and E = -K.
Whole halo only: both states are in equilibrium, so H = U + W + P_ext V = -K + 4 P_ext V
exactly, and dH = -dK + 4 P_ext dV is checked too.
Run:  python scan_heat_capacity.py
"""

import numpy as np
import gravothermal_collapse as gc

dm = gc.dm
P_ext = gc.P_ext
r0 = gc.r_edge.copy()
rho0 = gc.rho_s.copy()
v2_0 = gc.v2_s.copy()
s0 = gc.s_s.copy()
n = len(dm)
Rs = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def virial_terms(r_e, v2_s, J):
    """2K, W and 3 p_s V for the region made of shells 0..J-1 (surface at edge J)."""
    rho_e = gc.shell_density(r_e, dm)
    gtil = gc.gravity_terms(r_e, dm) * 4.0 * np.pi * r_e[1:]**4      # G M_j dmbar_j, edges 1..N
    W = -np.sum(gtil[:J] / r_e[1:J + 1])
    twoK = 3.0 * np.sum(dm[:J] * v2_s[:J])
    p_s = P_ext if J == n else rho_e[J] * v2_s[J]
    return twoK, W, 3.0 * p_s * (4.0 / 3.0) * np.pi * r_e[J]**3


print()
print("=" * 96)
print("A. STATIC VIRIAL CHECK on the initial state:  2K + W = 3 p_s V  for each region r < R")
print("=" * 96)
print(f"  {'R used':>9} {'shells':>6} {'mass frac':>10} {'2K':>12} {'W':>12} {'3 p_s V':>12}"
      f" {'(2K+W-3psV)/2K':>15} {'|W| / 3psV':>11}")
for R in Rs:
    J = int(np.sum(r0[1:] <= R * (1 + 1e-12)))
    twoK, W, PS = virial_terms(r0, v2_0, J)
    print(f"  {r0[J]:9.4f} {J:6d} {np.sum(dm[:J]) / np.sum(dm):10.4e} {twoK:12.5e} {W:12.5e} {PS:12.5e}"
          f" {(twoK + W - PS) / twoK:15.3e} {abs(W) / PS:11.4f}")
print("  |W| / 3psV > 1: self-gravity dominates the region's balance; < 1: the surrounding pressure does.")
print()

print("=" * 96)
print("B. EFFECTIVE HEAT CAPACITY: remove 1% of the region's thermal energy, relax, C = dU_region / dE")
print("   (diagnostic only; the t=0 gate verdict, FAIL, is not affected by anything here)")
print("=" * 96)
print(f"  {'R used':>9} {'shells':>6} {'E removed':>11} {'dU region':>12} {'C':>9}"
      f" {'v2 region: before':>18} {'after':>11} {'dr_J/r_J':>11} {'updates':>7} {'resid/floor':>11}")
for R in Rs:
    J = int(np.sum(r0[1:] <= R * (1 + 1e-12)))
    v2_loss = v2_0.copy()
    v2_loss[:J] *= 0.99
    s = s0.copy()
    s[:J] = v2_loss[:J]**1.5 / rho0[:J]
    r1, rho1, v21, info = gc.relax_to_hydrostatic(r0, dm, s, P_ext)
    E_removed = 1.5 * np.sum(dm[:J] * (v2_0[:J] - v2_loss[:J]))
    dU = 1.5 * np.sum(dm[:J] * (v21[:J] - v2_0[:J]))
    C = dU / (-E_removed)
    mw0 = np.sum(dm[:J] * v2_0[:J]) / np.sum(dm[:J])
    mw1 = np.sum(dm[:J] * v21[:J]) / np.sum(dm[:J])
    print(f"  {r0[J]:9.4f} {J:6d} {E_removed:11.4e} {dU:+12.4e} {C:+9.4f} {mw0:18.8f} {mw1:11.8f}"
          f" {(r1[J] - r0[J]) / r0[J]:+11.3e} {info['iterations']:7d}"
          f" {info['residual'] / info['roundoff_floor']:11.3f}")
    twoK1, W1, PS1 = virial_terms(r1, v21, J)
    print(f"  {'':9} after relaxation: virial residual {(twoK1 + W1 - PS1) / twoK1:.3e},"
          f"  |W| / 3psV = {abs(W1) / PS1:.4f};  worst row / threshold {info['max_row_ratio']:.3f};"
          f"  final step max|dr/r| {info['step']:.3e} (bound {info['step_bound']:.3e})")
    if J == n:
        U0, W0, PV0 = gc.discrete_energy(r0, dm, v2_0, P_ext)
        U1, Wd1, PV1 = gc.discrete_energy(r1, dm, v21, P_ext)
        dH = (U1 + Wd1 + PV1) - (U0 + W0 + PV0)
        rhs = -(U1 - U0) + 4.0 * (PV1 - PV0)
        print(f"  {'':9} whole halo: dH = {dH:+.6e},  -dK + 4 P_ext dV = {rhs:+.6e}"
              f"  (identity between equilibria; relative difference {abs(dH - rhs) / abs(dH):.2e})")
print()
print("C < 0 at some R: negative heat capacity exists in this halo at t=0 on that scale.")
print("C > 0 at every R: it does not exist anywhere in this halo at t=0.")
