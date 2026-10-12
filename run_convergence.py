"""
Convergence test of the Stage 4 collapse (baseline: run_collapse.py, record committed in 7175181).
Question: did the scheme's energy leak (an artificial heat sink, drift -1.43e-3) drive the collapse?

  python run_convergence.py A        -> variant A: xi = 0.01, 200 shells   (writes stage4_conv_A.csv)
  python run_convergence.py B        -> variant B: xi = 0.02, 400 shells   (writes stage4_conv_B.csv)
  python run_convergence.py analyze  -> evaluates the pass conditions from the three records

EVERYTHING BELOW WAS FIXED BEFORE ANY VARIANT WAS RUN (advisor ruling; not to be changed after).
Both variants: NFW halo (r = 0.01-30 r_s), dt0 = 1e-3, growth cap 1.2, same solver and loop,
same stop condition (rho_c >= 100 rho_c0) and caps (20,000 steps, t = 5,000). rho_c = shell 0
(the same sphere r < 0.01 at both resolutions, rho_c0 = 148.02); binding region = shells inside
r = 0.3 at t = 0; core = shells out to where rho first falls below rho_c / 2.
Measurements, computed IDENTICALLY for baseline, A and B by metrics() below:
  t10  : first recorded t with rho_c / rho_c0 >= 10;
  exp1 : least-squares slope of ln v2_c against ln rho_c over all recorded steps with
         2 <= rho_c/rho_c0 <= 100 (baseline hand estimate 0.090; BSI self-similar 0.0867);
  exp2 : least-squares slope of ln rho_c against ln M_core over the same steps
         (baseline hand estimate -2.85; BSI -2.704);
  drift: total-energy drift (U + W + P_ext V) at the stop.
The +-10% bands are applied around the BASELINE's exp1 and exp2 as computed by metrics(), so
all three runs are measured the same way (the hand estimates above were approximate).
ROBUST if ALL hold:
  R1. the collapse criterion is met in both variants (rho_c >= 10 rho_c0; binding ratio > 1;
      mass error <= 1e-12; converged every step, 1-2 before any breakdown);
  R2. |drift_A| < |drift_baseline|; the ratio drift_baseline/drift_A is reported (~2 expected
      if the leak is a first-order step-size error);
  R3. exp1 and exp2 within +-10% of the baseline values, in both variants;
  R4. |t10_variant - t10_baseline| / t10_baseline < 25%, for both variants.
If t10 moves substantially as the drift shrinks, the heat sink was contributing: reported as such.
"""

import csv
import sys
import time
import numpy as np
import gravothermal_collapse as gc

BASE = "stage4_collapse_history.csv"
VARIANTS = {"A": dict(xi=0.01, n_shell=200, out="stage4_conv_A.csv"),
            "B": dict(xi=0.02, n_shell=400, out="stage4_conv_B.csv")}
STEP_CAP, T_CAP, RHO_STOP, DT0 = 20000, 5000.0, 100.0, 1e-3
COLS = ["step", "t", "dt", "rho_c_over_rho_c0", "v2_c", "ratio_bind_r03", "core_shells",
        "core_mass_frac", "max_drho", "rejections", "energy_drift", "mass_err"]


def run_variant(name):
    cfg = VARIANTS[name]
    h = gc.build_nfw_halo(n_shell=cfg["n_shell"])
    r0, dm, P_ext, v2_0, rho0 = h["r_edge"], h["dm"], h["P_ext"], h["v2"], h["rho"]
    n, M_tot = len(dm), np.sum(dm)
    J_bind = int(np.sum(r0[1:] <= 0.3 * (1 + 1e-12)))
    H0 = sum(gc.discrete_energy(r0, dm, v2_0, P_ext))

    def ratio(r, v2, J):
        twoK, W, PS = gc.region_virial(r, dm, v2, max(J, 1), P_ext)
        return abs(W) / PS

    print(f"VARIANT {name}: xi = {cfg['xi']}, {n} shells; rho_c0 = {rho0[0]:.4f}; binding region {J_bind} shells,"
          f" ratio {ratio(r0, v2_0, J_bind):.4f}")
    stop, t0, k = None, time.time(), 0
    with open(cfg["out"], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        try:
            for rec in gc.evolve(r0, v2_0, dm, P_ext, DT0, STEP_CAP, xi=cfg["xi"]):
                r, v2, rho = rec["r"], rec["v2"], rec["rho"]
                below = np.nonzero(rho < 0.5 * rho[0])[0]
                Jc = int(below[0]) if below.size else n
                row = [rec["step"], rec["t"], rec["dt_used"], rho[0] / rho0[0], v2[0], ratio(r, v2, J_bind), Jc,
                       np.sum(dm[:max(Jc, 1)]) / M_tot, rec["max_drho"], rec["rejections"],
                       (sum(gc.discrete_energy(r, dm, v2, P_ext)) - H0) / abs(H0), rec["mass_err"]]
                w.writerow([f"{x:.10e}" if isinstance(x, float) else x for x in row])
                fh.flush()
                k = rec["step"]
                if k % 100 == 0:
                    print(f"  step {k:6d}  t {rec['t']:11.4e}  dt {rec['dt_used']:9.3e}  rho_c/rho_c0 {row[3]:10.5f}"
                          f"  v2_c {row[4]:8.5f}  ratio(bind) {row[5]:7.4f}  wall {time.time() - t0:7.1f}s")
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
        except KeyboardInterrupt:
            stop = "interrupted by the owner"
    with open(cfg["out"].replace(".csv", "_stop.txt"), "w") as fh:
        fh.write(stop + "\n")
    print(f"VARIANT {name} STOPPED after {k} steps: {stop}  (wall {time.time() - t0:.1f} s); record {cfg['out']}")


def metrics(path, stop):
    with open(path) as fh:
        rd = csv.DictReader(fh)
        D = {c: [] for c in rd.fieldnames}
        for row in rd:
            for c in rd.fieldnames:
                D[c].append(float(row[c]))
    D = {c: np.array(v) for c, v in D.items()}
    rc, v2c, M = D["rho_c_over_rho_c0"], D["v2_c"], D["core_mass_frac"]
    win = (rc >= 2.0) & (rc <= 100.0)
    hit = np.nonzero(rc >= 10.0)[0]
    c1, c2, c3 = bool(hit.size), bool(np.any(D["ratio_bind_r03"] > 1.0)), bool(np.all(D["mass_err"] <= 1e-12))
    c4 = not stop.startswith("BREAKDOWN") or (c1 and c2)
    return {"steps": len(rc), "stop": stop, "collapse": c1 and c2 and c3 and c4,
            "t10": D["t"][hit[0]] if hit.size else float("nan"),
            "exp1": np.polyfit(np.log(rc[win]), np.log(v2c[win]), 1)[0] if win.sum() > 2 else float("nan"),
            "exp2": np.polyfit(np.log(M[win]), np.log(rc[win]), 1)[0] if win.sum() > 2 else float("nan"),
            "drift": D["energy_drift"][-1], "n_win": int(win.sum()),
            "rejections": int(D["rejections"].sum())}


def analyze():
    m = {"baseline": metrics(BASE, "rho_c >= 100 rho_c0")}
    for name, cfg in VARIANTS.items():
        with open(cfg["out"].replace(".csv", "_stop.txt")) as fh:
            m[name] = metrics(cfg["out"], fh.read().strip())
    b = m["baseline"]
    print("=" * 100)
    print("CONVERGENCE TEST (conditions fixed before running, run_convergence.py docstring)")
    print(f"  {'run':>9} {'steps':>6} {'collapse':>9} {'t10':>12} {'exp1':>9} {'exp2':>9} {'drift':>12}"
          f" {'win rows':>8} {'rejects':>7}  stop")
    for key in ["baseline", "A", "B"]:
        q = m[key]
        print(f"  {key:>9} {q['steps']:6d} {'yes' if q['collapse'] else 'NO':>9} {q['t10']:12.5e} {q['exp1']:9.5f}"
              f" {q['exp2']:9.4f} {q['drift']:+12.4e} {q['n_win']:8d} {q['rejections']:7d}  {q['stop']}")
    A, B = m["A"], m["B"]
    R1 = A["collapse"] and B["collapse"]
    R2 = abs(A["drift"]) < abs(b["drift"])
    R3 = all(abs(q[e] / b[e] - 1.0) <= 0.10 for q in (A, B) for e in ("exp1", "exp2"))
    R4 = all(abs(q["t10"] / b["t10"] - 1.0) < 0.25 for q in (A, B))
    print("=" * 100)
    print(f"  R1 collapse in both variants                : {'MET' if R1 else 'NOT MET'}")
    print(f"  R2 |drift_A| < |drift_baseline|             : {'MET' if R2 else 'NOT MET'}"
          f"   ratio drift_baseline/drift_A = {b['drift'] / A['drift']:.3f} (~2 expected if first order)")
    for key in ("A", "B"):
        q = m[key]
        print(f"  R3 [{key}] exp1 {100 * (q['exp1'] / b['exp1'] - 1):+6.2f}%, exp2 {100 * (q['exp2'] / b['exp2'] - 1):+6.2f}%"
              f"   (limit +-10%)      R4 [{key}] t10 {100 * (q['t10'] / b['t10'] - 1):+6.2f}%   (limit +-25%)")
    print(f"  R3 exponents within +-10% in both          : {'MET' if R3 else 'NOT MET'}")
    print(f"  R4 t10 within 25% in both                  : {'MET' if R4 else 'NOT MET'}")
    robust = R1 and R2 and R3 and R4
    print(f"RESULT: {'ROBUST - the energy leak did not drive the collapse (by the pre-registered test)' if robust else 'NOT ROBUST by the pre-registered test - see which condition failed'}")
    print(f"  for reference, BSI self-similar: exp1 = 0.0867, exp2 = -2.704")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in VARIANTS:
        run_variant(arg)
    elif arg == "analyze":
        analyze()
    else:
        print("usage: python run_convergence.py A | B | analyze")
