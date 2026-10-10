"""
Capstone Stage 4: the pre-registered long gravothermal evolution of the NFW halo.

  python run_collapse.py          -> THE PRE-REGISTERED RUN (caps below; writes the record files)
  python run_collapse.py test 50  -> optional short check of the loop on the NFW halo (50 steps);
                                     the criterion is NOT evaluated and no record files are written
Ctrl+C stops the run cleanly: the record so far is kept and evaluated, stop reason "interrupted".

EVERYTHING BELOW WAS FIXED BEFORE THE RUN (advisor rulings). It is not to be changed after results.
Initial halo : NFW (gravothermal_collapse.build_nfw_halo(): 200 shells, r = 0.01-30 r_s).
Unchanged    : relaxation solver and per-row criterion, conduction step, evolve() loop
               (predictor, dt * min(1.2, 0.02 / max|drho/rho|), reject-and-retry). dt0 = 1e-3.
               Near-floor rejections are accepted as a speed cost (option c); carrying shell
               volumes as state (option b) is held in reserve.
Central density : density of shell 0 (the sphere inside r = 0.01 at t = 0, followed as the same
                  mass); rho_c0 = 148.02.
Binding central region : the shells inside r = 0.3 at t = 0 (85 shells, 1.2% of the mass,
                  |W|/3p_sV = 0.55 at t = 0), followed as the same shells.
SUCCESS CRITERION (binding). Gravothermal collapse is demonstrated if and only if ALL hold:
  1. rho_c / rho_c0 >= 10 at some recorded step (relative to the t = 0 value -- the stricter
     reading, since core formation first LOWERS rho_c; the rise from the minimum is printed but
     does not count);
  2. the binding region's |W|/3p_sV, < 1 at t = 0, exceeds 1 at some recorded step;
  3. total mass conserved every step (relative error <= 1e-12);
  4. the solver converged on every accepted step, and criteria 1 and 2 were met before any
     breakdown.
Otherwise: "NO COLLAPSE / INCOMPLETE COLLAPSE", with the stop reason and whatever was reached.
Caps : 20,000 steps; t = 5,000 code units; stop early once rho_c >= 100 rho_c0; stop on breakdown
       (dt < 1e-16). A cap reached is reported as "incomplete", not as a physics failure.
NON-BINDING diagnostics (cannot alter the verdict): |W|/3p_sV for the shells inside r = 0.1 at
  t = 0, for the whole halo, for the innermost 0.1% and 1% of the mass (fixed shells, as every
  mass-based region is in this Lagrangian scheme), and for the CURRENT core: the shells from the
  centre out to where rho first falls below rho_c / 2 (its mass changes with time).
Record : stage4_collapse_history.csv, one row per accepted step, flushed as written;
         stage4_collapse_snapshots.npz, profiles (r, rho, v2) every 500 steps and at the end.
Expected (stated before running): an initial core-formation phase -- heat flows inward (108 of
  199 edges at t = 0), rho_c FALLS -- before any collapse. The time unit is uncalibrated
  (C_LMFP_PLACEHOLDER); the collapse time estimate is ~1e2-1e3 code units plus core formation.
"""

import csv
import sys
import time
import numpy as np
import gravothermal_collapse as gc

TEST = len(sys.argv) > 1 and sys.argv[1] == "test"
STEP_CAP = int(sys.argv[2]) if TEST and len(sys.argv) > 2 else (50 if TEST else 20000)
T_CAP, RHO_STOP, DT0 = 5000.0, 100.0, 1e-3
HIST, SNAP = "stage4_collapse_history.csv", "stage4_collapse_snapshots.npz"

h = gc.build_nfw_halo()
r0, dm, P_ext, v2_0, rho0 = h["r_edge"], h["dm"], h["P_ext"], h["v2"], h["rho"]
n = len(dm)
M_tot = np.sum(dm)
cum = np.cumsum(dm)
J_bind = int(np.sum(r0[1:] <= 0.3 * (1 + 1e-12)))
J_r01 = int(np.sum(r0[1:] <= 0.1 * (1 + 1e-12)))
J_m01 = int(np.searchsorted(cum, 0.001 * M_tot)) + 1
J_m1 = int(np.searchsorted(cum, 0.01 * M_tot)) + 1
rho_c0 = rho0[0]
H0 = sum(gc.discrete_energy(r0, dm, v2_0, P_ext))


def ratio(r, v2, J):
    twoK, W, PS = gc.region_virial(r, dm, v2, max(J, 1), P_ext)
    return abs(W) / PS


def core_J(rho):
    below = np.nonzero(rho < 0.5 * rho[0])[0]
    return int(below[0]) if below.size else n


COLS = ["step", "t", "dt", "rho_c_over_rho_c0", "v2_c", "ratio_bind_r03", "ratio_r01", "ratio_whole",
        "r_m01", "ratio_m01", "r_m1", "ratio_m1", "core_shells", "core_mass_frac", "ratio_core",
        "max_drho", "newton", "worst_row", "rejections", "energy_drift", "mass_err"]

print()
print("=" * 100)
print(f"STAGE 4 {'SHORT CHECK (NFW, criterion NOT evaluated, no files written)' if TEST else 'PRE-REGISTERED RUN'}:"
      f" step cap {STEP_CAP}, time cap {T_CAP:g}, stop at rho_c >= {RHO_STOP:g} rho_c0")
print(f"  t = 0: rho_c0 = {rho_c0:.4f}, v2_c = {v2_0[0]:.6f}; binding region {J_bind} shells,"
      f" ratio {ratio(r0, v2_0, J_bind):.4f}; 0.1% mass = {J_m01} shells (r < {r0[J_m01]:.4f}),"
      f" 1% mass = {J_m1} shells (r < {r0[J_m1]:.4f})")
print("=" * 100)

fh = None if TEST else open(HIST, "w", newline="")
writer = None if TEST else csv.writer(fh)
if writer:
    writer.writerow(COLS)
    fh.flush()
snaps = {"step": [0], "t": [0.0], "r": [r0.copy()], "rho": [rho0.copy()], "v2": [v2_0.copy()]}
rows, stop_reason, t_wall = [], None, time.time()
last = None
try:
    for rec in gc.evolve(r0, v2_0, dm, P_ext, DT0, STEP_CAP):
        r, v2, rho, info = rec["r"], rec["v2"], rec["rho"], rec["info"]
        Jc = core_J(rho)
        row = [rec["step"], rec["t"], rec["dt_used"], rho[0] / rho_c0, v2[0], ratio(r, v2, J_bind),
               ratio(r, v2, J_r01), ratio(r, v2, n), r[J_m01], ratio(r, v2, J_m01), r[J_m1],
               ratio(r, v2, J_m1), Jc, np.sum(dm[:max(Jc, 1)]) / M_tot, ratio(r, v2, Jc),
               rec["max_drho"], info["iterations"], info["max_row_ratio"], rec["rejections"],
               (sum(gc.discrete_energy(r, dm, v2, P_ext)) - H0) / abs(H0), rec["mass_err"]]
        rows.append(row)
        last = (r, rho, v2, rec["step"], rec["t"])
        if writer:
            writer.writerow([f"{x:.10e}" if isinstance(x, float) else x for x in row])
            fh.flush()
        if rec["step"] % (10 if TEST else 100) == 0:
            print(f"  step {rec['step']:6d}  t {rec['t']:11.4e}  dt {rec['dt_used']:9.3e}  rho_c/rho_c0 {row[3]:10.5f}"
                  f"  v2_c {row[4]:8.5f}  ratio(bind) {row[5]:7.4f}  core shells {Jc:3d}"
                  f"  rejects so far {sum(q[18] for q in rows):5d}  wall {time.time() - t_wall:7.1f}s")
        if not TEST and rec["step"] % 500 == 0:
            for key, val in zip(["step", "t", "r", "rho", "v2"], [rec["step"], rec["t"], r, rho, v2]):
                snaps[key].append(val.copy() if isinstance(val, np.ndarray) else val)
            np.savez(SNAP, **{k: np.array(v) for k, v in snaps.items()})
        if rec["t"] >= T_CAP:
            stop_reason = f"time cap t >= {T_CAP:g} reached"
            break
        if row[3] >= RHO_STOP:
            stop_reason = f"rho_c >= {RHO_STOP:g} rho_c0 reached"
            break
    else:
        stop_reason = f"step cap {STEP_CAP} reached"
except RuntimeError as e:
    stop_reason = f"BREAKDOWN: {e}"
except KeyboardInterrupt:
    stop_reason = "interrupted by the owner (Ctrl+C)"
finally:
    if fh:
        fh.close()
    if not TEST and last is not None:
        r, rho, v2, step, t = last
        for key, val in zip(["step", "t", "r", "rho", "v2"], [step, t, r, rho, v2]):
            snaps[key].append(val.copy() if isinstance(val, np.ndarray) else val)
        np.savez(SNAP, **{k: np.array(v) for k, v in snaps.items()})

print("=" * 100)
print(f"STOPPED after {len(rows)} accepted steps: {stop_reason}   (wall time {time.time() - t_wall:.1f} s)")
if not rows:
    sys.exit(0)
A = np.array(rows, dtype=float)
k_max, k_min = int(np.argmax(A[:, 3])), int(np.argmin(A[:, 3]))
print(f"  t reached {A[-1, 1]:.6e};  dt min {A[:, 2].min():.3e}, max {A[:, 2].max():.3e}, last {A[-1, 2]:.3e}")
print(f"  rho_c/rho_c0: min {A[k_min, 3]:.5f} (t = {A[k_min, 1]:.4e}), max {A[k_max, 3]:.5f}"
      f" (t = {A[k_max, 1]:.4e}), final {A[-1, 3]:.5f};  rise after the minimum (non-binding)"
      f" x{A[k_min:, 3].max() / A[k_min, 3]:.4f}")
print(f"  v2_c: initial {v2_0[0]:.5f}, final {A[-1, 4]:.5f}, max {A[:, 4].max():.5f}")
print(f"  binding ratio (r < 0.3 at t = 0): initial {ratio(r0, v2_0, J_bind):.4f}, min {A[:, 5].min():.4f},"
      f" max {A[:, 5].max():.4f}, final {A[-1, 5]:.4f}")
print(f"  non-binding ratios, final: r<0.1 {A[-1, 6]:.4f}, whole {A[-1, 7]:.4f}, 0.1% mass {A[-1, 9]:.4f},"
      f" 1% mass {A[-1, 11]:.4f}, current core {A[-1, 14]:.4f} ({int(A[-1, 12])} shells, mass frac {A[-1, 13]:.3e})")
print(f"  rejections {int(A[:, 18].sum())} on {int(np.sum(A[:, 18] > 0))} of {len(A)} steps;"
      f" worst row/threshold {A[:, 17].max():.3f}; energy drift final {A[-1, 19]:+.3e}; max mass error {A[:, 20].max():.2e}")
if TEST:
    print("  SHORT CHECK: the success criterion is not evaluated.")
    sys.exit(0)

c1 = bool(np.any(A[:, 3] >= 10.0))
c2 = bool(np.any(A[:, 5] > 1.0))
c3 = bool(np.all(A[:, 20] <= 1e-12))
c4 = not stop_reason.startswith("BREAKDOWN") or (c1 and c2)
print("=" * 100)
print("PRE-REGISTERED CRITERION, evaluated from the record:")
print(f"  1. rho_c/rho_c0 >= 10 at some step            : {'MET' if c1 else 'NOT MET'} (max {A[:, 3].max():.4f})")
print(f"  2. binding |W|/3p_sV exceeds 1 at some step    : {'MET' if c2 else 'NOT MET'} (max {A[:, 5].max():.4f})")
print(f"  3. mass conserved every step (<= 1e-12)        : {'MET' if c3 else 'NOT MET'} (max {A[:, 20].max():.2e})")
print(f"  4. converged every accepted step; 1-2 before any breakdown : {'MET' if c4 else 'NOT MET'}")
verdict = c1 and c2 and c3 and c4
print(f"VERDICT: {'GRAVOTHERMAL COLLAPSE DEMONSTRATED' if verdict else 'NO COLLAPSE / INCOMPLETE COLLAPSE'}"
      f"  (stop reason: {stop_reason})")
print(f"Record: {HIST} ({len(A)} rows), {SNAP} ({len(snaps['step'])} snapshots)")
