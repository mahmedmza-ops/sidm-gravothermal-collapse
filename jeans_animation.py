"""
Jeans Instability - Animation (corrected)
Two clouds: one above the Jeans mass (collapses), one below it (holds).
Pressure now scales with compression, so a sub-Jeans cloud genuinely resists.
Simplified uniform-sphere model. Saves to jeans_collapse.gif.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

k_B = 1.381e-23
G   = 6.674e-11
m_H = 1.673e-27
mu  = 2.3
M_sun = 1.989e30
pc    = 3.086e16

def jeans_mass(T, rho):
    term1 = (5 * k_B * T) / (G * mu * m_H)
    term2 = 3 / (4 * np.pi * rho)
    return term1**1.5 * term2**0.5

def evolve_cloud(mass, T, rho0, steps=300, dt=2e11):
    """
    Uniform-sphere cloud. Gravity pulls in (~1/R^2).
    Pressure pushes out and RISES as the cloud compresses (~1/R^3 energy density),
    so a cloud with enough pressure settles instead of collapsing.
    """
    R = (3 * mass / (4 * np.pi * rho0))**(1/3)
    R0 = R
    v = 0.0
    N = mass / (mu * m_H)
    # pressure-support constant, set from the initial thermal energy
    K = 3 * N * k_B * T
    radii = []
    for i in range(steps):
        a_grav  = -G * mass / R**2
        a_press =  K * R / (mass * R0**2) * (R0 / R)**3   # grows as R shrinks
        v += (a_grav + a_press) * dt
        v *= 0.999                     # tiny damping, so a stable cloud settles
        R += v * dt
        R = max(R, 1e-6 * R0)          # floor, avoids divide-by-zero on collapse
        radii.append(R)
    return np.array(radii)

T_cloud, rho_cloud = 10, 1e-18
MJ = jeans_mass(T_cloud, rho_cloud)

R_above = evolve_cloud(3 * MJ,   T_cloud, rho_cloud) / pc
R_below = evolve_cloud(0.3 * MJ, T_cloud, rho_cloud) / pc

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
for ax, start, title, colour in [
    (ax1, R_above[0], f"Above Jeans mass ({3*MJ/M_sun:.0f} M_sun)",   "crimson"),
    (ax2, R_below[0], f"Below Jeans mass ({0.3*MJ/M_sun:.1f} M_sun)", "navy"),
]:
    lim = max(R_above[0], R_below[0]) * 1.2
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, color=colour)

circle1 = plt.Circle((0, 0), R_above[0], color="crimson", alpha=0.6)
circle2 = plt.Circle((0, 0), R_below[0], color="navy",    alpha=0.6)
ax1.add_patch(circle1); ax2.add_patch(circle2)

time_text = fig.text(0.5, 0.04, "", ha="center", fontsize=11)
fig.suptitle("Jeans Instability (simplified uniform-sphere model)", fontsize=13)

def update(frame):
    circle1.set_radius(max(R_above[frame], 0))
    circle2.set_radius(max(R_below[frame], 0))
    myr = frame * 2e11 / 3.15e13
    time_text.set_text(f"t = {myr:.1f} Myr")
    return circle1, circle2, time_text

ani = FuncAnimation(fig, update, frames=len(R_above), interval=60, blit=False)
ani.save("jeans_collapse.gif", writer=PillowWriter(fps=15))
print("Saved jeans_collapse.gif  (", len(R_above), "frames )")