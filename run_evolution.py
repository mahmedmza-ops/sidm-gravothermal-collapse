"""
Capstone Stage 4: gravothermal time evolution (gravothermal_collapse.evolve). Pre-registered.

  python run_evolution.py        -> SHORT TEST, 50 steps (default). The success criterion is
                                    NOT evaluated on a short test; its quantities are printed.
  python run_evolution.py 30     -> short test of 30 steps
The long run is a separate decision, after the short test has been reviewed.

PRE-REGISTERED SUCCESS CRITERION (advisor, fixed before any run; not to be changed after results).
Gravothermal collapse is demonstrated if and only if ALL hold:
  1. central density rises by >= 10x over the run;
  2. the central region's |W| / 3 p_s V crosses from < 1 to > 1 during the evolution;
  3. total mass is conserved every step;
  4. the solver stays converged every step (reject-and-retry on failure; if it cannot proceed,
     stop and report).
Otherwise: "no collapse / incomplete collapse", reported with whatever was reached.
Definitions fixed here, before any run:
  "central density"  = density of the innermost shell (shell 0, r < 0.01 at t = 0).
  "central region"   = the innermost 100 shells (the gate's core, mass 2.598, |W|/3p_sV = 0.274
                       at t = 0), followed as a Lagrangian region (same shells, same mass).
Numerical choices (not physics): dt0 = 1e-3 (starting value only; at t = 0 a step of dt = 1
changed the centre's v2 by 7.6%), xi = 0.02, growth cap 1.2, dt floor 1e-16.
Expected early behaviour, stated before running: the cored start has a temperature inversion
(76 edges conduct heat INWARD), so the centre should first warm and expand -- central density
FALLING at first is physical for this start and is not, by itself, a loop failure.
"""

import sys
import numpy as np
import gravothermal_collapse as gc

n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
DT0 = 1e-3
CORE = 100

dm, P_ext = gc.dm, gc.P_ext
r0, v2_0, rho0 = gc.r_edge.copy(), gc.v2_s.copy(), gc.rho_s.copy()
M0 = np.sum(dm)
H0 = sum(gc.discrete_energy(r0, dm, v2_0, P_ext))
twoK, W, PS = gc.region_virial(r0, dm, v2_0, CORE, P_ext)
ratio0 = abs(W) / PS


def inward_edges(r_e, v2_s):
    L = gc.luminosity(r_e, dm, v2_s)
    return int(np.sum(L[1:-1] < 0))


print()
print("=" * 118)
print(f"STAGE 4 SHORT TEST: {n_steps} steps. Success criterion NOT evaluated on a short test.")
print(f"  t = 0: rho_c = {rho0[0]:.6f}, v2_c = {v2_0[0]:.6f}, core |W|/3psV = {ratio0:.4f},"
      f" inward-conducting edges = {inward_edges(r0, v2_0)}, mass = {M0:.12e}")
print("=" * 118)
print(f"  {'step':>4} {'t':>11} {'dt used':>10} {'rho_c/rho_c0':>13} {'v2_c':>10} {'core |W|/3psV':>14}"
      f" {'max drho/rho':>12} {'inward L':>8} {'(H-H0)/|H0|':>12} {'Newton':>6} {'worst row':>9}"
      f" {'rejects':>7} {'pred':>4}")

rows = []
breakdown = None
try:
    for rec in gc.evolve(r0, v2_0, dm, P_ext, DT0, n_steps):
        r, v2, rho, info = rec["r"], rec["v2"], rec["rho"], rec["info"]
        twoK, W, PS = gc.region_virial(r, dm, v2, CORE, P_ext)
        H = sum(gc.discrete_energy(r, dm, v2, P_ext))
        row = dict(step=rec["step"], t=rec["t"], dt=rec["dt_used"], rc=rho[0] / rho0[0], v2c=v2[0],
                   ratio=abs(W) / PS, drho=rec["max_drho"], inward=inward_edges(r, v2),
                   dH=(H - H0) / abs(H0), newton=info["iterations"], worst=info["max_row_ratio"],
                   rej=rec["rejections"], pred=rec["predicted"], mass_err=rec["mass_err"],
                   reasons=rec["reasons"])
        rows.append(row)
        print(f"  {row['step']:4d} {row['t']:11.4e} {row['dt']:10.3e} {row['rc']:13.8f} {row['v2c']:10.6f}"
              f" {row['ratio']:14.6f} {row['drho']:12.3e} {row['inward']:8d} {row['dH']:+12.3e}"
              f" {row['newton']:6d} {row['worst']:9.3f} {row['rej']:7d} {'yes' if row['pred'] else 'no':>4}")
        for reason in row["reasons"]:
            print(f"         rejected: {reason}")
except RuntimeError as e:
    breakdown = str(e)

print("=" * 118)
print(f"SUMMARY of the short test ({len(rows)} of {n_steps} steps completed)")
if breakdown:
    print(f"  BREAKDOWN: {breakdown}")
if rows:
    rc = np.array([q["rc"] for q in rows])
    dts = np.array([q["dt"] for q in rows])
    print(f"  time reached t = {rows[-1]['t']:.6e}")
    print(f"  central density rho_c/rho_c0: final {rc[-1]:.8f}, min {rc.min():.8f}, max {rc.max():.8f}"
          f"  -> {'ROSE' if rc[-1] > 1 else 'FELL' if rc[-1] < 1 else 'unchanged'} over the test")
    print(f"  dt: first {dts[0]:.3e}, last {dts[-1]:.3e}, min {dts.min():.3e}, max {dts.max():.3e};"
          f" steps where growth was limited by xi (not the 1.2 cap): "
          f"{sum(1 for q in rows if q['drho'] > 0.02 / 1.2)}")
    print(f"  rejections: {sum(q['rej'] for q in rows)} in total; predictor used on "
          f"{sum(q['pred'] for q in rows)} of {len(rows)} steps")
    print(f"  solver: converged every accepted step by construction; worst row/threshold "
          f"{max(q['worst'] for q in rows):.3f}; Newton updates per step {min(q['newton'] for q in rows)}"
          f"-{max(q['newton'] for q in rows)}")
    print(f"  mass: max relative error {max(q['mass_err'] for q in rows):.2e} (fixed by construction)")
    print(f"  energy U+W+PV: final drift {rows[-1]['dH']:+.3e} relative (conduction conserves it exactly;"
          f" any drift comes from relaxation)")
    print(f"  core |W|/3psV: start {ratio0:.6f}, final {rows[-1]['ratio']:.6f},"
          f" max {max(q['ratio'] for q in rows):.6f}")
    print(f"  inward-conducting edges: start {inward_edges(r0, v2_0)}, final {rows[-1]['inward']}")
print("  Success criterion: not evaluated (short test). The long run is a separate decision.")
