"""
Virial Theorem and Negative Heat Capacity
Reproduced from the derivation in my BSc dissertation 'Galactic Black Holes'.

PART 1 (exact): verify the virial relation 2K + U = 0 for a self-gravitating
                sphere in equilibrium, and show E = -K.
PART 2 (simplified illustration): a self-gravitating sphere slowly radiates
                energy and, counter-intuitively, CONTRACTS and HEATS. This is
                the engine of gravothermal collapse. Simplified uniform-sphere
                model, shown evolving in time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Constants (SI)
G     = 6.674e-11
k_B   = 1.381e-23
m_H   = 1.673e-27
M_sun = 1.989e30
pc    = 3.086e16

# A self-gravitating sphere: mass M, radius R, N particles
M = 100 * M_sun
N = M / m_H

def grav_potential_energy(M, R):
    """U for a uniform sphere: U = -(3/5) G M^2 / R  (negative, bound)."""
    return -(3/5) * G * M**2 / R

def kinetic_energy(T):
    """Total kinetic energy of N particles at temperature T: K = (3/2) N kB T."""
    return 1.5 * N * k_B * T

def temperature_from_virial(M, R):
    """
    In virial equilibrium, 2K + U = 0, so K = -U/2.
    Solve for T:  (3/2) N kB T = (1/2)(3/5) G M^2 / R
    """
    U = grav_potential_energy(M, R)
    K = -U / 2
    return K / (1.5 * N * k_B)

# ---------- PART 1: verify the exact law ----------
R_test = 1.0 * pc
T_test = temperature_from_virial(M, R_test)
U = grav_potential_energy(M, R_test)
K = kinetic_energy(T_test)
E = K + U

print("PART 1 - Virial check for a 100 M_sun sphere at R = 1 pc")
print(f"  U (potential)   = {U:.3e} J")
print(f"  K (kinetic)     = {K:.3e} J")
print(f"  2K + U          = {2*K + U:.3e} J   (should be ~0)")
print(f"  E = K + U       = {E:.3e} J")
print(f"  -K              = {-K:.3e} J   (E should equal -K)")
print(f"  ratio E / (-K)  = {E / (-K):.4f}   (should be 1.0)")
print()

# ---------- PART 2: radiate energy, watch it contract AND heat ----------
# The system loses a fixed fraction of energy each step (radiation).
# At each step we find the new equilibrium radius for its reduced energy,
# then read off the new temperature. It contracts and heats. Simplified model.

steps = 200
R = 2.0 * pc
radii, temps, energies = [], [], []

for i in range(steps):
    T = temperature_from_virial(M, R)
    U = grav_potential_energy(M, R)
    E = -kinetic_energy(T)        # E = -K, the virial result
    radii.append(R); temps.append(T); energies.append(E)
    # radiate: remove a small fraction of |E|, system becomes MORE bound (E more negative)
    E_new = E - 0.01 * abs(E)
    # for a virial sphere, E = -(3/10) G M^2 / R, so R = -(3/10) G M^2 / E
    R = -(3/10) * G * M**2 / E_new

radii = np.array(radii) / pc
temps = np.array(temps)

# ---------- animation: sphere shrinks, colour shows temperature ----------
fig, (ax_cloud, ax_plot) = plt.subplots(1, 2, figsize=(12, 5.5))

lim = radii[0] * 1.2
ax_cloud.set_xlim(-lim, lim); ax_cloud.set_ylim(-lim, lim)
ax_cloud.set_aspect("equal"); ax_cloud.axis("off")
ax_cloud.set_title("Self-gravitating sphere (simplified model)")

Tmin, Tmax = temps.min(), temps.max()
def temp_colour(T):
    f = (T - Tmin) / (Tmax - Tmin)        # 0 (cool) to 1 (hot)
    return (0.3 + 0.7*f, 0.3*(1-f), 0.9*(1-f))   # blue -> red as it heats

circle = plt.Circle((0, 0), radii[0], color=temp_colour(temps[0]))
ax_cloud.add_patch(circle)
readout = ax_cloud.text(0, -lim*0.85, "", ha="center", fontsize=10)

ax_plot.plot(radii, temps, color="grey", alpha=0.4)
ax_plot.set_xlabel("Radius  (parsec)"); ax_plot.set_ylabel("Temperature  (K)")
ax_plot.set_title("Temperature RISES as it contracts")
ax_plot.invert_xaxis()        # contraction goes left-to-right in time
marker, = ax_plot.plot([], [], "o", color="crimson", markersize=9)
ax_plot.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Virial Theorem: energy radiated away, yet the system HEATS", fontsize=13)

def update(frame):
    circle.set_radius(radii[frame])
    circle.set_color(temp_colour(temps[frame]))
    readout.set_text(f"R = {radii[frame]:.2f} pc    T = {temps[frame]:.0f} K")
    marker.set_data([radii[frame]], [temps[frame]])
    return circle, readout, marker

ani = FuncAnimation(fig, update, frames=steps, interval=60, blit=False)
ani.save("virial_collapse.gif", writer=PillowWriter(fps=20))
print("PART 2 - Saved virial_collapse.gif")