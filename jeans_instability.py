"""
Jeans Instability
Part (a): the Jeans mass threshold, derived in my BSc dissertation 'Galactic Black Holes'.
Part (b): a SIMPLIFIED dynamical illustration of a uniform-sphere cloud above and
          below that threshold. This is a teaching model (uniform density, Newtonian
          energy balance), not a full hydrodynamic simulation.
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants (SI)
k_B = 1.381e-23      # Boltzmann constant, J/K
G = 6.674e-11        # gravitational constant
m_H = 1.673e-27      # hydrogen mass, kg
mu = 2.3             # mean molecular weight (molecular cloud, dimensionless)
M_sun = 1.989e30     # solar mass, kg
pc = 3.086e16        # parsec, m

# ---------- PART (a): the Jeans mass threshold ----------
def jeans_mass(T, rho):
    """Jeans mass in kg, for temperature T (K) and mass density rho (kg/m^3)."""
    term1 = (5 * k_B * T) / (G * mu * m_H)
    term2 = 3 / (4 * np.pi * rho)
    return term1**1.5 * term2**0.5

# Jeans mass vs density, at a few fixed temperatures
densities = np.logspace(-21, -17, 400)        # kg/m^3, typical molecular-cloud range
temperatures = [10, 20, 50]                   # kelvin

plt.figure(figsize=(8, 6))
for T in temperatures:
    MJ = jeans_mass(T, densities) / M_sun
    plt.loglog(densities, MJ, linewidth=2, label=f"T = {T} K")
plt.xlabel("Density  (kg / m^3)")
plt.ylabel("Jeans mass  (solar masses)")
plt.title("Jeans Mass Threshold")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("jeans_threshold.png", dpi=150)
plt.close()

# ---------- PART (b): simplified collapse-or-hold, two clouds ----------
# Uniform-sphere model: evolve radius R under gravity vs internal pressure.
# Energy-based: gravity always pulls in; thermal pressure resists.
# A cloud above the Jeans mass collapses; one below it oscillates/holds.

def evolve_cloud(mass, T, rho0, steps=2000, dt=3e11):
    """Evolve a uniform-sphere cloud's radius over time. Returns time and radius arrays."""
    # initial radius from mass and density, assuming a uniform sphere
    R = (3 * mass / (4 * np.pi * rho0))**(1/3)
    v = 0.0  # initial radial velocity
    N = mass / (mu * m_H)  # number of particles

    times, radii = [], []
    for i in range(steps):
        # gravitational acceleration at the surface (inward)
        a_grav = -G * mass / R**2
        # thermal pressure support, modelled as an outward acceleration
        # from the internal energy of an ideal gas (simplified)
        a_press = (5 * k_B * T * N) / (mu * m_H * mass * R)
        a = a_grav + a_press
        v += a * dt
        R += v * dt
        if R <= 0:          # collapsed to a point in this toy model
            times.append(i * dt); radii.append(0)
            break
        times.append(i * dt); radii.append(R)
    return np.array(times), np.array(radii)

# Set conditions, then build one cloud above and one below the Jeans mass
T_cloud = 10
rho_cloud = 1e-18
MJ = jeans_mass(T_cloud, rho_cloud)

mass_above = 3 * MJ    # should collapse
mass_below = 0.3 * MJ  # should hold

t1, R1 = evolve_cloud(mass_above, T_cloud, rho_cloud)
t2, R2 = evolve_cloud(mass_below, T_cloud, rho_cloud)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
ax1.plot(t1 / (3.15e13), R1 / pc, color="crimson", linewidth=2)
ax1.set_title(f"Above Jeans mass ({mass_above/M_sun:.0f} M_sun) - collapses")
ax1.set_xlabel("Time  (Myr)"); ax1.set_ylabel("Radius  (parsec)")
ax1.grid(True, linestyle="--", alpha=0.4)

ax2.plot(t2 / (3.15e13), R2 / pc, color="navy", linewidth=2)
ax2.set_title(f"Below Jeans mass ({mass_below/M_sun:.1f} M_sun) - holds")
ax2.set_xlabel("Time  (Myr)")
ax2.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Jeans Instability: collapse vs support (simplified uniform-sphere model)")
plt.tight_layout()
plt.savefig("jeans_evolution.png", dpi=150)
plt.close()

print(f"Jeans mass at T={T_cloud} K, rho={rho_cloud:.1e}: {MJ/M_sun:.2f} solar masses")
print("Saved jeans_threshold.png and jeans_evolution.png")
