"""
Figure of the pre-registered Stage 4 collapse run, drawn ONLY from its committed record
(commit 7175181): stage4_collapse_snapshots.npz and stage4_collapse_history.csv.
Nothing is re-run; this script does not import the solver.

Left : density profiles from the saved snapshots (the run saved three: t = 0, step 500, final).
       The core-formation minimum (t ~ 18.5) was not saved as a profile, so it is not shown.
Right: central density against time, every recorded step, with the pre-registered 10x level.
Units are dimensionless: G = rho_s = r_s = 1 (NFW scale density and radius).
Run:  python plot_collapse_result.py   -> gravothermal_collapse_result.png
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

snap = np.load("stage4_collapse_snapshots.npz")
with open("stage4_collapse_history.csv") as fh:
    rows = list(csv.DictReader(fh))
t = np.array([float(q["t"]) for q in rows])
rc = np.array([float(q["rho_c_over_rho_c0"]) for q in rows])
rho_c0 = snap["rho"][0][0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: snapshot profiles, density against shell centre (volume midpoint of each recorded shell)
colours = ["black", "tab:orange", "tab:red"]
for k in range(len(snap["step"])):
    r_e, rho = snap["r"][k], snap["rho"][k]
    rc_shell = (0.5 * (r_e[:-1]**3 + r_e[1:]**3))**(1.0 / 3.0)
    if snap["step"][k] == 0:
        label, style = "t = 0: initial NFW halo", "--"
    else:
        label, style = (f"step {snap['step'][k]}, t = {snap['t'][k]:.2f}:"
                        f"  ρ_c = {rho[0] / rho_c0:.0f} ρ_c0"), "-"
    ax1.loglog(rc_shell, rho, style, color=colours[k % len(colours)], linewidth=2, label=label)
ax1.set_xlabel("radius  r / r_s")
ax1.set_ylabel("density  ρ / ρ_s")
ax1.set_title("Density profiles (all snapshots saved by the run)")
ax1.legend(fontsize=9)
ax1.grid(True, which="both", linestyle="--", alpha=0.3)

# Right: central density against time, every recorded step
ax2.semilogy(t, rc, color="crimson", linewidth=2)
ax2.axhline(1.0, color="grey", linewidth=1, linestyle=":")
ax2.axhline(10.0, color="black", linewidth=1, linestyle="--", label="pre-registered criterion: 10 ρ_c0")
k_min = int(np.argmin(rc))
ax2.plot(t[k_min], rc[k_min], "o", color="navy", label=f"minimum {rc[k_min]:.4f} ρ_c0 at t = {t[k_min]:.1f}")
ax2.set_xlabel("time  (code units, uncalibrated)")
ax2.set_ylabel("central density  ρ_c / ρ_c0")
ax2.set_title("Central density, every recorded step")
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(True, which="both", linestyle="--", alpha=0.3)

ins = ax2.inset_axes([0.47, 0.50, 0.40, 0.33])   # above the curve, so the main panel's data stays visible
w = t >= t[-1] - 10.0
ins.semilogy(t[w], rc[w], color="crimson", linewidth=2)
ins.axhline(10.0, color="black", linewidth=1, linestyle="--")
ins.set_title("last 10 time units", fontsize=8)
ins.tick_params(labelsize=7)
ins.grid(True, which="both", linestyle="--", alpha=0.3)

fig.suptitle("Gravothermal core collapse from an NFW halo (pre-registered run, record in commit 7175181)",
             fontsize=12)
fig.text(0.5, 0.005, "Long-mean-free-path conduction only. Dimensionless units, G = ρ_s = r_s = 1. "
         "Plotted from the committed record; no re-run.", ha="center", fontsize=8, color="dimgrey")
plt.tight_layout(rect=(0, 0.03, 1, 0.95))
plt.savefig("gravothermal_collapse_result.png", dpi=150)
print("Saved gravothermal_collapse_result.png from the committed record "
      f"({len(snap['step'])} snapshots, {len(t)} recorded steps).")
