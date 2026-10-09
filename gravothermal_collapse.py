"""
Gravothermal Collapse of a Self-Interacting Dark Matter Halo
Reproduction of Balberg, Shapiro & Inagaki (2002), cited in my BSc dissertation.

STAGE 1: radial grid and initial density (r^-2.19). [done]
STAGE 2: hydrostatic equilibrium imposed (mean residual ~0.05). [done]
STAGE 3: let heat conduct core -> halo and evolve in time. The core should
         contract and heat (negative heat capacity), reproducing the
         core-contraction of the paper's Figure 2. Long-mean-free-path
         conduction only; a simplified reproduction of the self-similar regime.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import root

G = 1.0

# ---------- grid and initial profile (Stages 1-2) ----------
N = 200
r = np.logspace(-2, 2, N)
alpha = 2.19
rho = 1.0 / (1.0 + r**alpha)
dr = np.gradient(r)
M = np.cumsum(4 * np.pi * r**2 * rho * dr)

# equilibrium temperature (velocity dispersion squared), from Stage 2
gravity_term = G * M * rho / r**2
p = np.zeros_like(r)
for i in range(N - 2, -1, -1):
    p[i] = p[i+1] + gravity_term[i] * (r[i+1] - r[i])
v2 = p / rho

# ---------- Stage 3: coupled hydrostatic + conductive energy update ----------
# STATUS: NON-WORKING AND SUPERSEDED. Kept only so the history is truthful.
# Three defects were diagnosed and shown numerically in diagnose_stage3.py:
#   1. Boundary cliff: Stage 2 gives v2[-1] = 0 (p[-1] = 0); clamped to 1e-8 it sits 3e7 below
#      v2[-2], anchors P[-1] ~ rho*1e-8, and the global mass rescale then puts ~100% of the
#      mass in the outermost interval (one call: rho_c falls to ~1e-6 of its start).
#   2. Fixed-point error: the pressure reconstruction uses exp(x) or (1+x) where Stage 2 has
#      1/(1-x), so even with a clean anchor one call changes rho_c by a factor 0.39 / 0.20.
#   3. No PdV work term: pure conduction conserves sum(w*v2) and never raises max(v2); it
#      relaxes toward uniform. It can warm the centre transiently (the Stage-2 profile peaks
#      off-centre), but it cannot give the unbounded central heating of gravothermal runaway.
#      That needs compression heating, i.e. the negative heat capacity of simulation 3.
# The comments below, which call this scheme "the correct approach" or what the literature
# uses, were written before the diagnosis and are not supported by the results. Also known:
# the dt-halving loop has no floor, and the plot/summary block is duplicated at the end.
# The previous Stage-3 implementation solved conduction and hydrostatic balance in
# sequence. That split is unphysical because the thermal support changes first and the
# density is then adjusted afterward, which inverts the gravothermal feedback: the core
# cools and its density collapses instead of contracting under the coupling of heat
# transport and hydrostatic support. The correct approach in the quasi-static SIDM
# gravothermal limit is to solve the coupled energy equation and hydrostatic equation
# simultaneously at the new time level.
#
# We therefore solve for the new velocity-dispersion profile v^2_{n+1} and the density
# field rho_{n+1} by Newton iteration. The residual is built from the implicit energy
# equation at backwards Euler time level and the hydrostatic balance at the same time
# level; each trial v^2_{n+1} is converted into a hydrostatic density field before the
# energy residual is evaluated. This is the coupled fully-implicit form that the
# SIDM-GTF/gravothermal literature uses, rather than the sequential update-then-rebalance
# scheme that produced the wrong sign of the core response.
#
# We keep the conduction-only, no-hyperbolic-flow limit exactly as requested: this is
# not a finite-volume hydrodynamic solver, just the diffusion of thermal energy in a
# quasi-static hydrostatic halo.


def enclosed_mass(r, rho):
    """Compute the enclosed-mass profile from a density field."""
    dr = np.gradient(r)
    shell_mass = 4.0 * np.pi * r**2 * rho * dr
    return np.concatenate(([0.0], np.cumsum(shell_mass[1:])))


def hydrostatic_from_v2(r, rho_ref, v2_trial, M_total):
    """Return rho consistent with hydrostatic equilibrium for a given v^2 profile."""
    # For a trial new-time temperature profile, we integrate the hydrostatic pressure
    # equation from the outside in: dP/dr = -G M(r) rho / r^2. Using P = rho v^2 and
    # rho = P / v^2, the cell pressure is adjusted to satisfy the same hydrostatic
    # support at the new time. This keeps the density response locked to the thermal
    # support, which is the actual physical coupling that the sequential method missed.
    P = np.empty_like(v2_trial)
    M_env = enclosed_mass(r, rho_ref)
    P[-1] = rho_ref[-1] * v2_trial[-1]
    for i in range(len(r) - 2, -1, -1):
        dr_i = r[i + 1] - r[i]
        rho_guess = P[i + 1] / (v2_trial[i] + 1e-12)
        dP_dr = G * M_env[i] * rho_guess / (r[i]**2 + 1e-12)
        P[i] = P[i + 1] + dP_dr * dr_i
    rho_new = np.clip(P / (v2_trial + 1e-12), 1e-12, 1e6)
    mass = 4.0 * np.pi * np.trapezoid(r**2 * rho_new, r)
    if mass > 0.0:
        rho_new *= M_total / mass
    return np.clip(rho_new, 1e-12, 1e6)


def conduction_residual(r, rho, v2_new, v2_old, dt):
    """Backward-Euler conduction residual at the new time level."""
    # The energy equation is the conduction diffusion equation with a local heat
    # capacity proportional to the thermal support: d(v^2)/dt = (1/r^2) d/dr[r^2 kappa
    # d(v^2)/dr], with kappa ~ rho r^2 / sqrt(v^2). This is the conduction-only
    # gravothermal limit in which the unstable negative heat-capacity response emerges
    # from the coupled thermal and hydrostatic update rather than from an ad hoc density
    # renormalisation.
    eps = 1e-12
    v2_safe = np.maximum(v2_new, eps)
    kappa = 1.0e-2 * rho * r**2 / (np.sqrt(v2_safe) + eps)
    grad_v2 = np.gradient(v2_new, r, edge_order=2)
    flux = -kappa * grad_v2
    div_flux = np.gradient(r**2 * flux, r, edge_order=2) / (r**2 + eps)
    return (v2_new - v2_old) / dt - div_flux


def coupled_implicit_step(r, rho, v2, dt, M_total, x0=None):
    """Solve the coupled hydrostatic-energy system with a Newton-type root solve."""
    # We use scipy.optimize.root here because the fully coupled residual is nonlinear
    # in both the conductive flux and the hydrostatic density reconstruction. A full
    # analytical Jacobian is possible in principle, but for a 80-point Stage-3 profile
    # the finite-difference Jacobian is much simpler and robust. method='lm' is used
    # here because the residual is stiff and the previous-step solution is a good warm
    # start; the Levenberg-Marquardt step is more forgiving than 'hybr' on ill-scaled
    # nonlinear systems while preserving the same implicit Newton-style correction.
    x0 = np.asarray(v2 if x0 is None else x0, dtype=float)

    def residual(x):
        rho_h = hydrostatic_from_v2(r, rho, x, M_total)
        return conduction_residual(r, rho_h, x, v2, dt)

    sol = root(
        residual,
        x0,
        method="lm",
        options={"xtol": 1e-7, "ftol": 1e-7, "gtol": 1e-7, "maxfev": 5000},
    )
    if not sol.success:
        return None, None, sol.message, np.max(np.abs(residual(sol.x))) if sol.x is not None else np.inf

    x_new = np.maximum(sol.x, 1e-12)
    rho_new = hydrostatic_from_v2(r, rho, x_new, M_total)
    return x_new, rho_new, "ok", np.max(np.abs(residual(x_new)))


# Stage 3 time evolution: keep the verified Stages 1-2 on the original grid, but
# evolve the gravothermal coupling on a smaller Stage-3 grid so the Newton solve runs
# in plain Python without a prohibitively expensive dense Jacobian.
rho0_initial = rho[0]
M_total = 4.0 * np.pi * np.trapezoid(r**2 * rho, r)

# Reduced Stage-3 grid to keep the coupled Newton solve finite and stable.
stage3_r = np.geomspace(r.min(), r.max(), 80)
rho_stage3 = np.interp(stage3_r, r, rho)
v2_stage3 = np.interp(stage3_r, r, v2)

central_density_history = []
time_history = []
snapshots = []
snapshot_triggers = [2.0, 5.0, 10.0, 30.0, 100.0]
next_trigger = 0

rho = rho_stage3.copy()
v2 = np.maximum(v2_stage3, 1e-8)

# We warm-start the implicit solve using the previous time level. This is standard in
# time-dependent implicit methods because the solution changes smoothly from one step to
# the next, and it greatly improves convergence without altering the underlying physics.
t = 0.0
step = 0
dt = 1e-5
max_steps = 3000
max_frac_change = 3e-3

while step < max_steps:
    v2_new, rho_new, solve_status, residual_norm = coupled_implicit_step(
        stage3_r, rho, v2, dt, M_total, x0=v2
    )

    if solve_status != "ok":
        dt *= 0.5
        if dt < 1e-12:
            raise RuntimeError(
                f"Coupled solve failed to converge even with reduced dt: residual={residual_norm:.3e}, message={solve_status}"
            )
        continue

    frac_change = abs(rho_new[0] - rho[0]) / (rho[0] + 1e-12)
    if frac_change > max_frac_change:
        dt *= 0.5
        continue

    mass_now = 4.0 * np.pi * np.trapezoid(stage3_r**2 * rho_new, stage3_r)
    mass_error = abs(mass_now - M_total) / M_total
    if not np.isfinite(mass_now) or mass_error > 1e-8:
        raise RuntimeError(f"Mass drift in stage 3: {mass_now:.12e} vs {M_total:.12e}")

    rho = rho_new
    v2 = v2_new
    t += dt
    step += 1
    central_density_history.append(rho[0])
    time_history.append(t)

    if next_trigger < len(snapshot_triggers) and rho[0] >= snapshot_triggers[next_trigger] * rho0_initial:
        snapshots.append((rho[0] / rho0_initial, rho.copy()))
        next_trigger += 1

    if rho[0] > 100.0 * rho0_initial or not np.isfinite(rho[0]):
        break

    dt = min(1.0e-4, dt * 1.1)
    if step % 25 == 0:
        print(f"step={step:5d}  t={t:.6e}  rho_c={rho[0]:.6e}  mass={mass_now:.12e}  dt={dt:.3e}  residual={residual_norm:.3e}")

    if next_trigger == len(snapshot_triggers):
        break

# ---------- plot ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

ax1.loglog(stage3_r, rho_stage3, "k--", linewidth=1.5, label=f"initial (rho0={rho0_initial:.3e})")
for factor, rho_snap in snapshots:
    ax1.loglog(stage3_r, rho_snap, linewidth=2, label=f"core x{factor:.0f}")
ax1.set_xlabel("radius")
ax1.set_ylabel("density")
ax1.set_title("Density profile snapshots (non-working Stage 3 attempt)")
ax1.legend()
ax1.grid(True, which="both", linestyle="--", alpha=0.4)

ax2.semilogy(time_history, central_density_history, color="crimson", linewidth=2)
ax2.set_xlabel("time")
ax2.set_ylabel("central density")
ax2.set_title("Central density vs time (as computed)")
ax2.grid(True, which="both", linestyle="--", alpha=0.4)

fig.suptitle("Stage 3: non-working Newton/LM attempt, superseded (no result claimed)", fontsize=13)
plt.tight_layout()
plt.savefig("gravothermal_stage3.png", dpi=150)
plt.close()

summary = [
    "STAGE 3 - coupled hydrostatic/conduction solve",
    f"  steps taken: {step}",
    f"  final central density: {rho[0]:.6e}  (started at {rho0_initial:.6e})",
    f"  contraction factor: {rho[0] / rho0_initial:.3f}",
    f"  snapshots captured: {len(snapshots)}",
    f"  total mass: {M_total:.12e}",
    "  Saved gravothermal_stage3.png",
]
Path("gravothermal_stage3_summary.txt").write_text("\n".join(summary), encoding="utf-8")
print("STAGE 3 - run ended (non-working attempt, no result claimed)")
print(f"steps taken: {step}")
print(f"final central density: {rho[0]:.6e} (started at {rho0_initial:.6e})")
print(f"contraction factor: {rho[0] / rho0_initial:.3f}")
print(f"snapshots captured: {len(snapshots)}")
print(f"mass conserved: {M_total:.12e}")

# ---------- plot ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

ax1.loglog(r, 1.0 / (1.0 + r**alpha), "k--", linewidth=1.5, label="initial")
for factor, rho_snap in snapshots:
    ax1.loglog(r, rho_snap, linewidth=2, label=f"core x{factor:.0f}")
ax1.set_xlabel("radius")
ax1.set_ylabel("density")
ax1.set_title("Density profile snapshots (non-working Stage 3 attempt)")
ax1.legend()
ax1.grid(True, which="both", linestyle="--", alpha=0.4)

ax2.semilogy(time_history, central_density_history, color="crimson", linewidth=2)
ax2.set_xlabel("time")
ax2.set_ylabel("central density")
ax2.set_title("Central density vs time (as computed)")
ax2.grid(True, which="both", linestyle="--", alpha=0.4)

fig.suptitle("Stage 3: non-working Newton/LM attempt, superseded (no result claimed)", fontsize=13)
plt.tight_layout()
plt.savefig("gravothermal_stage3.png", dpi=150)
plt.close()

summary = [
    "STAGE 3 - run ended (non-working attempt, no result claimed)",
    f"  steps taken: {step}",
    f"  final central density: {rho[0]:.2f}  (started at {rho0_initial:.2f})",
    f"  contraction factor: {rho[0] / rho0_initial:.1f}",
    f"  snapshots captured: {len(snapshots)}",
    f"  total mass: {M_total:.12e}",
    "  Saved gravothermal_stage3.png",
]
Path("gravothermal_stage3_summary.txt").write_text("\n".join(summary), encoding="utf-8")
print("STAGE 3 - run ended (non-working attempt, no result claimed)")
print(f"steps taken: {step}")
print(f"final central density: {rho[0]:.6e} (started at {rho0_initial:.6e})")
print(f"contraction factor: {rho[0] / rho0_initial:.3f}")
print(f"snapshots captured: {len(snapshots)}")
print(f"mass conserved: {M_total:.12e}")
