"""
Tests for the Stage 3 relaxation step in gravothermal_collapse.py (formulation approved by
the advisor before coding). Changes nothing; imports gravothermal_collapse.py, which prints
its setup checks (no evolution).

  python test_relaxation.py      -> test 1 (fixed point) and test 2 (single-shell response)
  python test_relaxation.py 3    -> test 3, the GATE: no collapse evolution is run unless it passes

Choices fixed BEFORE any run (not to be tuned after seeing results):
  test 2: the shell whose centre is nearest r = 0.1 (inside the core); heat removed: 1% and
          10% of that shell's thermal energy (v2 multiplied by 0.99 and 0.90).
  test 3: "core" = the shells inside r = 1, the half-density radius of rho = 1/(1 + r^2.19)
          (rho(1) = rho_c / 2). Heat removed: 1% (primary, decides the gate) and 10%
          (secondary, a non-linearity check) of each core shell's thermal energy.
          GATE criterion: the mass-weighted core v2 after relaxation exceeds its value before
          the heat was removed. If it does not, the gate FAILS and that is reported as is.
"""

import sys
import numpy as np
import gravothermal_collapse as gc

dm = gc.dm
P_ext = gc.P_ext
r0 = gc.r_edge.copy()
rho0 = gc.rho_s.copy()
v2_0 = gc.v2_s.copy()
s0 = gc.s_s.copy()
rc0 = gc.shell_centres(r0)
n = len(dm)


def energy(r_edge, v2_s):
    U, W, PV = gc.discrete_energy(r_edge, dm, v2_s, P_ext)
    return U, W, PV, U + W + PV


def jacobian_fd_error(r_edge, s):
    """Worst row-relative difference between the analytic tridiagonal Jacobian and central finite differences."""
    sysm = gc.relaxation_system(r_edge, dm, s, P_ext)
    J = np.diag(sysm["diag"]) + np.diag(sysm["lower"], -1) + np.diag(sysm["upper"], 1)
    J_fd = np.zeros((n, n))
    for k in range(n):
        h = 1e-6 * (r_edge[k + 1] - r_edge[k])
        rp, rm = r_edge.copy(), r_edge.copy()
        rp[k + 1] += h
        rm[k + 1] -= h
        J_fd[:, k] = (gc.relaxation_system(rp, dm, s, P_ext)["F"]
                      - gc.relaxation_system(rm, dm, s, P_ext)["F"]) / (2.0 * h)
    return np.max(np.max(np.abs(J - J_fd), axis=1) / np.max(np.abs(J), axis=1))


def solver_line(info):
    return (f"Newton updates {info['iterations']}, halvings {info['halvings']}, "
            f"final max|F/g| {info['residual']:.3e} (round-off floor {info['roundoff_floor']:.3e}),\n"
            f"          Thomas vs dense {info['thomas_vs_dense']:.3e}, "
            f"min diag dominance |b|/(|a|+|c|) {info['min_diag_dominance']:.4f}\n"
            + gc.history_text(info))


print()
print("=" * 78)

if len(sys.argv) == 1:
    # ---------------- TEST 1: fixed point ----------------
    print("TEST 1 - FIXED POINT: relax the unperturbed initial state")
    print("=" * 78)
    sysm = gc.relaxation_system(r0, dm, s0, P_ext)
    g = sysm["g"]
    print(f"  initial max|F/g| = {np.max(np.abs(sysm['F'] / g)):.3e}  (tolerance 1e-10)")
    dx = gc.thomas_solve(sysm["lower"] / g[1:], sysm["diag"] / g, sysm["upper"] / g[:-1], -sysm["F"] / g)
    print(f"  one forced Newton step anyway: max|dr/r| = {np.max(np.abs(dx) / r0[1:]):.3e}  (should be round-off)")
    r1, rho1, v21, info = gc.relax_to_hydrostatic(r0, dm, s0, P_ext)
    print(f"  relax_to_hydrostatic: iterations {info['iterations']}, max|dr/r| = {np.max(np.abs(r1[1:] - r0[1:]) / r0[1:]):.3e},"
          f" max|dv2/v2| = {np.max(np.abs(v21 / v2_0 - 1)):.3e}")
    print(f"  analytic Jacobian vs finite differences (initial state): worst row-relative error {jacobian_fd_error(r0, s0):.3e}")
    print()

    # ---------------- TEST 2: single-shell response ----------------
    print("=" * 78)
    k = int(np.argmin(np.abs(rc0 - 0.1)))
    print(f"TEST 2 - SINGLE-SHELL RESPONSE: remove heat from shell {k} only (centre r = {rc0[k]:.4f})")
    print("  expected by the formulation: shell compressed (outer edge in, rho up);")
    print("  its v2 RISES during relaxation (compression heating); point 5(b) predicts it still ends")
    print("  BELOW its original v2 (one shell alone does not show negative heat capacity).")
    print("=" * 78)
    H0 = energy(r0, v2_0)
    for f in [0.01, 0.10]:
        v2_loss = v2_0.copy()
        v2_loss[k] *= (1.0 - f)
        s = s0.copy()
        s[k] = v2_loss[k]**1.5 / rho0[k]
        Hl = energy(r0, v2_loss)
        r1, rho1, v21, info = gc.relax_to_hydrostatic(r0, dm, s, P_ext)
        H1 = energy(r1, v21)
        dr_rel = (r1[1:] - r0[1:]) / r0[1:]
        print(f"  --- heat removed: {100 * f:.0f}% of shell {k}'s thermal energy ---")
        print(f"  solver: {solver_line(info)}")
        print(f"  inner edge r_{k}:   dr/r = {(r1[k] - r0[k]) / r0[k]:+.4e}   (predicted > 0 before the first run;"
              f" observed < 0: everything outside sinks, raising the weight on the inner core)")
        print(f"  outer edge r_{k + 1}: dr/r = {(r1[k + 1] - r0[k + 1]) / r0[k + 1]:+.4e}   (expect < 0)")
        print(f"  rho_k: after/before = {rho1[k] / rho0[k]:.6f}   (expect > 1)")
        print(f"  v2_k: original {v2_0[k]:.6f}  after loss {v2_loss[k]:.6f}  after relax {v21[k]:.6f}")
        print(f"        relaxation heating v2(relax) - v2(loss) = {v21[k] - v2_loss[k]:+.4e}   (expect > 0)")
        print(f"        net v2(relax) - v2(original)        = {v21[k] - v2_0[k]:+.4e}   (point 5b predicts < 0)")
        print(f"  neighbours v2 after/original, shells {k - 2}..{k + 2}: "
              + "  ".join(f"{v21[i] / v2_0[i]:.6f}" for i in range(k - 2, k + 3)))
        print(f"  entropy held: max|v2^1.5/rho / s - 1| = {np.max(np.abs(v21**1.5 / rho1 / s - 1)):.3e};"
              f"  shell masses are fixed by construction (dm is not an unknown)")
        print(f"  edges moved in: {np.sum(dr_rel < 0)}, out: {np.sum(dr_rel > 0)};"
              f" largest |dr/r| = {np.max(np.abs(dr_rel)):.3e} at r = {r0[1:][np.argmax(np.abs(dr_rel))]:.4f}")
        print(f"  energy U+W+PV: original {H0[3]:.10e}, after loss {Hl[3]:.10e}, after relax {H1[3]:.10e}")
        print(f"        removed by hand {H0[3] - Hl[3]:.4e};  change during relaxation {H1[3] - Hl[3]:+.4e}"
              f"  (ratio {(H1[3] - Hl[3]) / (H0[3] - Hl[3]):+.3e})")
        if f == 0.01:
            print(f"  analytic Jacobian vs finite differences (after loss, before relax): "
                  f"{jacobian_fd_error(r0, s):.3e}")
        print()
    print("Tests 1 and 2 done. Test 3 (the gate) is run separately: python test_relaxation.py 3")

elif sys.argv[1] == "3":
    # ---------------- TEST 3: core energy test (GATE) ----------------
    core = r0[1:] <= 1.0 + 1e-12          # shells whose outer edge is inside r = 1
    n_core = int(np.sum(core))
    M_core = np.sum(dm[core])
    print(f"TEST 3 - CORE ENERGY TEST (GATE): remove heat from the {n_core} shells inside r = {r0[n_core]:.4f}")
    print(f"  core mass {M_core:.6e} of {np.sum(dm):.6e}. Analogue of virial_theorem.py Part 2:")
    print("  does the core end HOTTER after losing energy (negative heat capacity)?")
    print("  GATE (fixed before running): mass-weighted core v2 after relaxation > before the loss, at 1%.")
    print("=" * 78)

    def core_v2(v2_s):
        return np.sum(dm[core] * v2_s[core]) / M_core

    H0 = energy(r0, v2_0)
    verdict = None
    for f in [0.01, 0.10]:
        v2_loss = v2_0.copy()
        v2_loss[core] *= (1.0 - f)
        s = s0.copy()
        s[core] = v2_loss[core]**1.5 / rho0[core]
        Hl = energy(r0, v2_loss)
        r1, rho1, v21, info = gc.relax_to_hydrostatic(r0, dm, s, P_ext)
        H1 = energy(r1, v21)
        E_removed = H0[3] - Hl[3]
        dU_core_net = np.sum(1.5 * dm[core] * (v21[core] - v2_0[core]))
        role = "PRIMARY (decides the gate)" if f == 0.01 else "secondary (non-linearity check)"
        print(f"  --- heat removed: {100 * f:.0f}% of each core shell's thermal energy; {role} ---")
        print(f"  solver: {solver_line(info)}")
        print(f"  core mass-weighted v2: before loss {core_v2(v2_0):.8f}  after loss {core_v2(v2_loss):.8f}"
              f"  after relax {core_v2(v21):.8f}")
        print(f"  central shell v2:      before loss {v2_0[0]:.8f}  after loss {v2_loss[0]:.8f}"
              f"  after relax {v21[0]:.8f}")
        print(f"  core radius {r0[n_core]:.6f} -> {r1[n_core]:.6f}   (dr/r = {(r1[n_core] - r0[n_core]) / r0[n_core]:+.4e});"
              f"  core mean density x{(r0[n_core] / r1[n_core])**3:.6f}")
        print(f"  energy removed by hand {E_removed:.6e};  net core thermal energy change after relax"
              f" {dU_core_net:+.6e}")
        print(f"  effective core heat capacity dU_core / dE = {dU_core_net / (-E_removed):+.4f}"
              f"   (negative = negative heat capacity)")
        print(f"  relaxation changes: dU {H1[0] - Hl[0]:+.4e}  dW {H1[1] - Hl[1]:+.4e}  d(PV) {H1[2] - Hl[2]:+.4e}"
              f"  d(U+W+PV) {H1[3] - Hl[3]:+.4e}  (ratio to removed {(H1[3] - Hl[3]) / E_removed:+.3e})")
        print(f"  outer edge moved: dr_N/r_N = {(r1[-1] - r0[-1]) / r0[-1]:+.4e}")
        if f == 0.01:
            verdict = core_v2(v21) > core_v2(v2_0)
        print()
    print("=" * 78)
    print(f"GATE RESULT: {'PASS' if verdict else 'FAIL'} - core mass-weighted v2 after relaxation is "
          f"{'HIGHER' if verdict else 'NOT higher'} than before the 1% heat loss.")
    if not verdict:
        print("  No collapse evolution is to be run. Report these numbers; do not work around the gate.")
else:
    print("usage: python test_relaxation.py   (tests 1-2)   |   python test_relaxation.py 3   (gate)")
