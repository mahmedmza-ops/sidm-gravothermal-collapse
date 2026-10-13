"""
Re-run of the pre-registered Stage 4 collapse with EVERY step's profile saved, for animation,
plus a reproducibility check against the committed record (7175181). run_collapse.py itself is
not modified and the committed record is not touched: output goes to new files.

Settings identical to run_collapse.py (pre-registered in 9c2b84c): build_nfw_halo() (200 shells,
r = 0.01-30 r_s), evolve() with dt0 = 1e-3, xi = 0.02, growth cap 1.2; stop at rho_c >= 100 rho_c0,
20,000 steps or t = 5,000. Recorded columns are computed exactly as in run_collapse.py /
run_convergence.py and written with the same 11-significant-digit format.

Reproducibility check, printed at the end:
  - the four convergence-test quantities (step count, t10, exp1, exp2) plus final rho_c and t,
    computed by run_convergence.metrics() for both the committed record and this re-run;
  - every recorded step compared digit-for-digit with the committed record;
  - the three committed profile snapshots (steps 0, 500, 639) compared bit-for-bit.
Run:  python rerun_dense.py   -> stage4_dense_rerun.csv, stage4_dense_rerun.npz
"""

import csv
import numpy as np
import gravothermal_collapse as gc
import run_convergence as rcv

STEP_CAP, T_CAP, RHO_STOP, DT0 = 20000, 5000.0, 100.0, 1e-3     # as in run_collapse.py
BASE_CSV, BASE_NPZ = "stage4_collapse_history.csv", "stage4_collapse_snapshots.npz"
OUT_CSV, OUT_NPZ = "stage4_dense_rerun.csv", "stage4_dense_rerun.npz"


def fmt(x):
    return f"{x:.10e}" if isinstance(x, float) else str(x)


h = gc.build_nfw_halo()
r0, dm, P_ext, v2_0, rho0 = h["r_edge"], h["dm"], h["P_ext"], h["v2"], h["rho"]
n, M_tot = len(dm), np.sum(dm)
J_bind = int(np.sum(r0[1:] <= 0.3 * (1 + 1e-12)))
H0 = sum(gc.discrete_energy(r0, dm, v2_0, P_ext))


def ratio(r, v2, J):
    twoK, W, PS = gc.region_virial(r, dm, v2, max(J, 1), P_ext)
    return abs(W) / PS


prof = {"step": [0], "t": [0.0], "r": [r0.copy()], "rho": [rho0.copy()], "v2": [v2_0.copy()]}
stop = None
with open(OUT_CSV, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(rcv.COLS)
    try:
        for rec in gc.evolve(r0, v2_0, dm, P_ext, DT0, STEP_CAP):
            r, v2, rho = rec["r"], rec["v2"], rec["rho"]
            below = np.nonzero(rho < 0.5 * rho[0])[0]
            Jc = int(below[0]) if below.size else n
            row = [rec["step"], rec["t"], rec["dt_used"], rho[0] / rho0[0], v2[0], ratio(r, v2, J_bind), Jc,
                   np.sum(dm[:max(Jc, 1)]) / M_tot, rec["max_drho"], rec["rejections"],
                   (sum(gc.discrete_energy(r, dm, v2, P_ext)) - H0) / abs(H0), rec["mass_err"]]
            w.writerow([fmt(x) for x in row])
            for key, val in zip(["step", "t", "r", "rho", "v2"], [rec["step"], rec["t"], r, rho, v2]):
                prof[key].append(val.copy() if isinstance(val, np.ndarray) else val)
            if rec["t"] >= T_CAP:
                stop = "time cap"
                break
            if row[3] >= RHO_STOP:
                stop = "rho_c >= 100 rho_c0"
                break
        else:
            stop = "step cap"
    except RuntimeError as e:
        stop = f"BREAKDOWN: {str(e).splitlines()[0]}"
np.savez(OUT_NPZ, **{k: np.array(v) for k, v in prof.items()})
print(f"\nRe-run stopped after {len(prof['step']) - 1} steps: {stop}. Saved {OUT_CSV} and {OUT_NPZ}"
      f" ({len(prof['step'])} profiles).")

# ---------------- reproducibility check ----------------
print("=" * 96)
print("REPRODUCIBILITY CHECK against the committed record (7175181)")
mb = rcv.metrics(BASE_CSV, "rho_c >= 100 rho_c0")
mn = rcv.metrics(OUT_CSV, stop)
with open(BASE_CSV) as fh:
    base_rows = list(csv.DictReader(fh))
with open(OUT_CSV) as fh:
    new_rows = list(csv.DictReader(fh))
fin_b = float(base_rows[-1]["rho_c_over_rho_c0"]), float(base_rows[-1]["t"])
fin_n = float(new_rows[-1]["rho_c_over_rho_c0"]), float(new_rows[-1]["t"])
print(f"  {'':16} {'steps':>6} {'t10':>14} {'exp1':>12} {'exp2':>12} {'final rho_c/rho_c0':>19} {'final t':>14}")
print(f"  {'committed record':16} {mb['steps']:6d} {mb['t10']:14.8e} {mb['exp1']:12.8f} {mb['exp2']:12.8f}"
      f" {fin_b[0]:19.10e} {fin_b[1]:14.8e}")
print(f"  {'re-run':16} {mn['steps']:6d} {mn['t10']:14.8e} {mn['exp1']:12.8f} {mn['exp2']:12.8f}"
      f" {fin_n[0]:19.10e} {fin_n[1]:14.8e}")

common = [c for c in rcv.COLS if c in base_rows[0]]
n_cmp = min(len(base_rows), len(new_rows))
mismatch = {c: sum(base_rows[i][c] != new_rows[i][c] for i in range(n_cmp)) for c in common}
first_bad = next((i for i in range(n_cmp) if any(base_rows[i][c] != new_rows[i][c] for c in common)), None)
print(f"  every recorded step, {len(common)} columns, as stored (11 significant digits): "
      + ("IDENTICAL" if all(v == 0 for v in mismatch.values()) and len(base_rows) == len(new_rows)
         else f"DIFFERENT in {sum(v > 0 for v in mismatch.values())} columns, first at step "
              f"{None if first_bad is None else base_rows[first_bad]['step']}; row counts {len(base_rows)} vs {len(new_rows)}"))
base_snap = np.load(BASE_NPZ)
steps_new = list(prof["step"])
for k, s in enumerate(base_snap["step"]):
    if int(s) in steps_new:
        j = steps_new.index(int(s))
        same = all(np.array_equal(base_snap[q][k], prof[q][j]) for q in ("r", "rho", "v2"))
        diff = max(np.max(np.abs(base_snap[q][k] / prof[q][j] - 1)) for q in ("rho", "v2"))
        print(f"  committed snapshot at step {int(s):4d}: {'BIT-IDENTICAL' if same else f'differs, max rel diff {diff:.2e}'}")
    else:
        print(f"  committed snapshot at step {int(s):4d}: not reached by the re-run")
reproduced = (mb["steps"] == mn["steps"] and all(v == 0 for v in mismatch.values()))
print("RESULT: " + ("REPRODUCED - the re-run is identical to the committed record; an animation from it "
                    "visualises the recorded run." if reproduced else
                    "NOT IDENTICAL - see the differences above; decide before animating."))
