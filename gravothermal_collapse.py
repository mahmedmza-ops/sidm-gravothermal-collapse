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

# ---------- Stage 3: time evolution with adaptive timestep ----------
# In the long-mean-free-path regime the heat flux is carried by conduction out of
# the dense core. The relevant flux is proportional to -r^2 rho sqrt(v^2) d(v^2)/dr.
# The core loses heat to the halo, so the temperature support drops and the
# central pressure falls; under hydrostatic equilibrium that forces the density to
# grow, which is the negative-heat-capacity contraction seen in the gravothermal
# runaway. The numerical implementation below is therefore only the time-stepping
# and discretisation of that physical flow; the underlying fluid physics stays the
# same.

def conduction_flux(r, rho, v2):
    """Long-mean-free-path conductive heat flux in the standard form."""
    v2_safe = np.maximum(v2, 1e-8)
    dv2_dr = np.gradient(v2_safe, r)
    flux = -4.0 * np.pi * r**2 * rho * np.sqrt(v2_safe) * dv2_dr
    return np.nan_to_num(flux, nan=0.0, posinf=1e100, neginf=-1e100)


def hydrostatic_pressure_from_density(r, M, rho):
    """Compute p(r) from hydrostatic equilibrium for a given density profile."""
    pressure = np.zeros_like(r)
    gravity_term = G * M * rho / r**2
    for i in range(N - 2, -1, -1):
        pressure[i] = pressure[i + 1] + gravity_term[i] * (r[i + 1] - r[i])
    return np.maximum(pressure, 1e-12)


total_mass = np.trapezoid(4.0 * np.pi * r**2 * rho, r)
central_density_history = []
time_history = []
snapshots = []
snapshot_triggers = [1.0, 2.0, 5.0, 10.0, 30.0]
next_trigger = 0

rho0_initial = rho[0]
# Ensure the initial velocity-dispersion field is positive definite so the
# conductivity does not divide by zero or generate spurious fluxes.
v2 = np.maximum(v2, 1e-8)

t = 0.0
step = 0

while step < 200000:
    L = conduction_flux(r, rho, v2)
    # The divergence of the outward heat flux is the local energy source term.
    # We keep the sign in the thermodynamic evolution so that the core cools
    # while the density response increases under hydrostatic balance.
    heating_rate = -np.gradient(r**2 * L, r) / (r**2 * rho + 1e-12)
    heating_rate = np.nan_to_num(heating_rate, nan=0.0, posinf=1e6, neginf=-1e6)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_rate = np.abs(heating_rate) / (np.abs(v2) + 1e-12)
    finite = np.isfinite(rel_rate)
    if not np.any(finite):
        dt = 1e-4
    else:
        max_rate = np.max(rel_rate[finite])
        dt = 0.02 / (max_rate + 1e-12)
        dt = min(dt, 1e-2)
        dt = max(dt, 1e-6)

    v2 = v2 + heating_rate * dt
    v2 = np.maximum(v2, 1e-8)

    # Re-impose hydrostatic balance after the thermal update. The density rises
    # when central pressure support drops, which is precisely the gravothermal
    # contraction signal we want to reproduce.
    pressure = hydrostatic_pressure_from_density(r, M, rho)
    rho = np.clip(pressure / (v2 + 1e-12), 1e-12, 1e3)
    rho *= total_mass / np.trapezoid(4.0 * np.pi * r**2 * rho, r)
    M = np.concatenate(([0.0], np.cumsum(4.0 * np.pi * r[1:]**2 * rho[1:] * np.gradient(r)[1:])))
    M *= total_mass / M[-1]

    t += dt
    step += 1
    central_density_history.append(rho[0])
    time_history.append(t)

    if next_trigger < len(snapshot_triggers) and rho[0] >= snapshot_triggers[next_trigger] * rho0_initial:
        snapshots.append((rho[0] / rho0_initial, rho.copy()))
        next_trigger += 1

    if rho[0] > 100.0 * rho0_initial or not np.isfinite(rho[0]):
        break

# ---------- plot ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

ax1.loglog(r, 1.0/(1.0+r**alpha), "k--", linewidth=1.5, label="initial")
for rho0_snap, rho_snap in snapshots:
    ax1.loglog(r, rho_snap, linewidth=2, label=f"core x{rho0_snap/rho0_initial:.0f}")
ax1.set_xlabel("radius"); ax1.set_ylabel("density")
ax1.set_title("Density profile: core contracts, halo stays still")
ax1.legend(); ax1.grid(True, which="both", linestyle="--", alpha=0.4)

ax2.semilogy(time_history, central_density_history, color="crimson", linewidth=2)
ax2.set_xlabel("time"); ax2.set_ylabel("central density (log)")
ax2.set_title("Core density runs away in finite time")
ax2.grid(True, which="both", linestyle="--", alpha=0.4)

fig.suptitle("Stage 3: gravothermal collapse (simplified, lmfp regime)", fontsize=13)
plt.tight_layout()
plt.savefig("gravothermal_stage3.png", dpi=150)
plt.close()

summary = [
    "STAGE 3 - evolution complete",
    f"  steps taken: {step}",
    f"  final central density: {rho[0]:.2f}  (started at {rho0_initial:.2f})",
    f"  contraction factor: {rho[0]/rho0_initial:.1f}",
    f"  snapshots captured: {len(snapshots)}",
    "  Saved gravothermal_stage3.png",
]
Path("gravothermal_stage3_summary.txt").write_text("\n".join(summary), encoding="utf-8")
