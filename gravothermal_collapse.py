"""
Gravothermal Collapse of a Self-Interacting Dark Matter Halo
Reproduction of the gravothermal fluid model of Balberg, Shapiro & Inagaki (2002),
the source cited in my BSc dissertation 'Galactic Black Holes' for the
supermassive black hole seed mechanism.

STAGE 1 of 4: set up the radial grid (concentric shells) and an initial halo
profile. Nothing evolves yet. We build the halo and confirm it looks right.

Dimensionless units throughout (as the paper does): we scale by the initial
central density, central velocity dispersion, and a reference radius. This keeps
numbers near 1 and avoids the huge/tiny values that cause numerical trouble.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- the radial grid: concentric shells ----------
# We space shells logarithmically, because the core needs fine resolution
# (that's where everything happens) and the halo can be coarse.
N_shells = 200
r_min, r_max = 1e-2, 1e2          # dimensionless radius, core to far halo
r = np.logspace(np.log10(r_min), np.log10(r_max), N_shells)

# ---------- initial profile ----------
# A simple, standard cored halo: flat density in the centre, falling as the
# gravothermal solution's r^-2.19 far out. This is our starting shape; Stage 2
# will relax it to true hydrostatic equilibrium.
alpha = 2.19                       # the self-similar halo slope from the paper
rho = 1.0 / (1.0 + (r)**alpha)     # ~1 in the core, ~r^-2.19 in the halo

# velocity dispersion squared (the "temperature"): start nearly isothermal,
# as the paper's initial core is, then let it taper gently in the outskirts
v2 = 1.0 / (1.0 + (r/10)**2)**0.5

# enclosed mass: integrate density over spherical shells, M(r) = integral 4 pi r^2 rho dr
dr = np.gradient(r)
dM = 4 * np.pi * r**2 * rho * dr
M_enclosed = np.cumsum(dM)

# ---------- check it by eye: plot the three starting profiles ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].loglog(r, rho, color="darkblue", linewidth=2)
axes[0].set_xlabel("radius (dimensionless)"); axes[0].set_ylabel("density")
axes[0].set_title("Initial density profile")
axes[0].grid(True, which="both", linestyle="--", alpha=0.4)
# mark the r^-2.19 slope for reference
axes[0].loglog(r[r>1], 2*r[r>1]**(-alpha), "--", color="grey", alpha=0.7,
               label=f"r^-{alpha} slope")
axes[0].legend()

axes[1].semilogx(r, v2, color="crimson", linewidth=2)
axes[1].set_xlabel("radius (dimensionless)"); axes[1].set_ylabel("velocity dispersion squared")
axes[1].set_title("Initial 'temperature' profile")
axes[1].grid(True, which="both", linestyle="--", alpha=0.4)

axes[2].loglog(r, M_enclosed, color="darkgreen", linewidth=2)
axes[2].set_xlabel("radius (dimensionless)"); axes[2].set_ylabel("enclosed mass")
axes[2].set_title("Enclosed mass")
axes[2].grid(True, which="both", linestyle="--", alpha=0.4)

fig.suptitle("Stage 1: initial SIDM halo profile (not yet evolving)", fontsize=13)
plt.tight_layout()
plt.savefig("gravothermal_stage1.png", dpi=150)
plt.close()

# ---------- print sanity numbers ----------
print("STAGE 1 - initial halo set up")
print(f"  shells: {N_shells}, radius range {r_min} to {r_max}")
print(f"  central density: {rho[0]:.3f}  (should be ~1)")
print(f"  outer density:   {rho[-1]:.2e}  (should be tiny)")
print(f"  total enclosed mass: {M_enclosed[-1]:.3f}")
print(f"  density slope far out: {np.log(rho[-1]/rho[-10]) / np.log(r[-1]/r[-10]):.2f}  (should be about -{alpha})")
print("  Saved gravothermal_stage1.png")