"""
Animation of the Stage 4 gravothermal collapse -> gravothermal_collapse.gif

Source: stage4_dense_rerun.npz, written by rerun_dense.py. That re-run uses the settings of the
pre-registered run (run_collapse.py, 9c2b84c) and was checked against the committed record
(7175181): all 639 recorded steps identical to the stored 11 significant digits, and the three
committed profile snapshots bit-identical. So the frames show the recorded run itself.
Nothing is re-run here; run rerun_dense.py first if stage4_dense_rerun.npz is missing.

Pacing: frames are spaced by STEP, not by time. The time step is adaptive (each step changes any
shell's density by at most ~2%), so equal steps carry comparable amounts of change: the slow
middle phase (t ~ 18-150) takes few steps and the runaway gets its share of the frames.
Units are dimensionless: G = rho_s = r_s = 1.
Run:  python animate_collapse.py         -> gravothermal_collapse.gif (full version, in the repository)
      python animate_collapse.py light   -> gravothermal_collapse_light.gif (lighter, for sharing:
                                            every 7th step, smaller frame, target ~2 MB or under)
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

d = np.load("stage4_dense_rerun.npz")
steps, t, r_all, rho_all = d["step"], d["t"], d["r"], d["rho"]
rho_c0 = rho_all[0][0]
rc = rho_all[:, 0] / rho_c0
centres = (0.5 * (r_all[:, :-1]**3 + r_all[:, 1:]**3))**(1.0 / 3.0)

LIGHT = len(sys.argv) > 1 and sys.argv[1] == "light"
EVERY, HOLD = (7, 15) if LIGHT else (3, 25)      # step spacing between frames; hold on the final frame
DPI, FPS = (60, 15) if LIGHT else (80, 20)
OUT = "gravothermal_collapse_light.gif" if LIGHT else "gravothermal_collapse.gif"
frames = list(range(0, len(steps), EVERY))
if frames[-1] != len(steps) - 1:
    frames.append(len(steps) - 1)
frames += [len(steps) - 1] * HOLD

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
ax1.loglog(centres[0], rho_all[0], "--", color="black", linewidth=1.5, label="t = 0: initial NFW halo")
line, = ax1.loglog([], [], color="crimson", linewidth=2.2, label="current profile")
ax1.set_xlim(1e-3, 40)
ax1.set_ylim(1e-5, 3e4)
ax1.set_xlabel("radius  r / r_s")
ax1.set_ylabel("density  ρ / ρ_s")
ax1.set_title("Density profile")
ax1.legend(fontsize=8, loc="lower left")
ax1.grid(True, which="both", linestyle="--", alpha=0.3)
clock = ax1.text(0.97, 0.95, "", transform=ax1.transAxes, ha="right", va="top", fontsize=9,
                 family="monospace", bbox=dict(facecolor="white", edgecolor="grey", alpha=0.85))

ax2.semilogy(t, rc, color="lightgrey", linewidth=1.5)
trace, = ax2.semilogy([], [], color="crimson", linewidth=2)
dot, = ax2.semilogy([], [], "o", color="crimson", markersize=7)
ax2.axhline(10.0, color="black", linewidth=1, linestyle="--", label="pre-registered criterion: 10 ρ_c0")
ax2.set_xlim(-3, t[-1] + 5)
ax2.set_ylim(0.8 * rc.min(), 1.5 * rc.max())
ax2.set_xlabel("time  (code units, uncalibrated)")
ax2.set_ylabel("central density  ρ_c / ρ_c0")
ax2.set_title("Central density")
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(True, which="both", linestyle="--", alpha=0.3)

fig.suptitle("Gravothermal core collapse from an NFW halo (pre-registered run; frames reproduce the record in 7175181)",
             fontsize=10)
fig.text(0.5, 0.01, "Long-mean-free-path conduction only. Dimensionless units, G = ρ_s = r_s = 1.",
         ha="center", fontsize=8, color="dimgrey")
plt.tight_layout(rect=(0, 0.04, 1, 0.94))


def update(k):
    line.set_data(centres[k], rho_all[k])
    trace.set_data(t[:k + 1], rc[:k + 1])
    dot.set_data([t[k]], [rc[k]])
    clock.set_text(f"step {steps[k]:3d} / {steps[-1]}\nt = {t[k]:9.4f}\nρ_c = {rc[k]:8.4f} ρ_c0")
    return line, trace, dot, clock


ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
ani.save(OUT, writer=PillowWriter(fps=FPS), dpi=DPI)
print(f"Saved {OUT}: {len(frames)} frames ({len(frames) - HOLD} distinct, every {EVERY} steps,"
      f" plus a {HOLD}-frame hold on the final step), from stage4_dense_rerun.npz ({len(steps)} profiles).")
