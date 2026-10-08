"""
Gravothermal Collapse of a Self-Interacting Dark Matter Halo
Reproduction of Balberg, Shapiro & Inagaki (2002), cited in my BSc dissertation.

STAGE 1: radial grid and initial density profile (r^-2.19 slope). [done]
STAGE 2: impose hydrostatic equilibrium. Given the density profile, solve for the
         velocity dispersion (temperature) that balances gravity at every shell,
         so the halo sits still before we evolve it. Then TEST the balance.

Dimensionless units (G = 1), as the paper does.
"""

import numpy as np
import matplotlib.pyplot as plt

G = 1.0   # dimensionless units

# ---------- Stage 1: grid and density ----------
N_shells = 200
r = np.logspace(-2, 2, N_shells)
alpha = 2.19
rho = 1.0 / (1.0 + r**alpha)

# enclosed mass M(r) = integral 4 pi r^2 rho dr
dr = np.gradient(r)
M_enclosed = np.cumsum(4 * np.pi * r**2 * rho * dr)

# ---------- Stage 2: solve for equilibrium temperature ----------
# Pressure p = rho * v2. Hydrostatic equilibrium: dp/dr = -G M rho / r^2.
# We integrate this INWARD from the outer edge (where pressure ~ 0) to get p(r),
# then recover v2 = p / rho.
#
# Integrating inward: p(r) = integral from r to infinity of G M rho / r^2 dr
gravity_term = G * M_enclosed * rho / r**2          # the right-hand side magnitude
# cumulative integral from the outside inward:
p = np.zeros_like(r)
for i in range(N_shells - 2, -1, -1):
    p[i] = p[i+1] + gravity_term[i] * (r[i+1] - r[i])

v2 = p / rho                                        # velocity dispersion squared

# ---------- TEST: is it actually in equilibrium? ----------
# Compute dp/dr numerically and compare to -G M rho / r^2. They should match.
dp_dr = np.gradient(p, r)
required = -G * M_enclosed * rho / r**2
# relative residual, where pressure is non-negligible
mask = p > 1e-6 * p.max()
residual = np.abs(dp_dr[mask] - required[mask]) / np.abs(required[mask])
max_resid = residual.max()
mean_resid = residual.mean()

worst = np.argmax(residual)
r_worst = r[mask][worst]
print(f"  worst balance at radius r = {r_worst:.3e}  (grid runs {r[0]:.2e} to {r[-1]:.2e})")

# ---------- plot the equilibrium profiles and the balance check ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].loglog(r, rho, color="darkblue", linewidth=2, label="density")
axes[0].loglog(r, p, color="purple", linewidth=2, label="pressure")
axes[0].set_xlabel("radius"); axes[0].set_title("Density and pressure")
axes[0].legend(); axes[0].grid(True, which="both", linestyle="--", alpha=0.4)

axes[1].semilogx(r, v2, color="crimson", linewidth=2)
axes[1].set_xlabel("radius"); axes[1].set_ylabel("velocity dispersion squared")
axes[1].set_title("Equilibrium 'temperature'")
axes[1].grid(True, which="both", linestyle="--", alpha=0.4)

axes[2].loglog(r[mask], np.abs(dp_dr[mask]), color="black", linewidth=2, label="|dp/dr| actual")
axes[2].loglog(r[mask], np.abs(required[mask]), "--", color="orange", linewidth=2, label="required (gravity)")
axes[2].set_xlabel("radius"); axes[2].set_title("Hydrostatic balance check")
axes[2].legend(); axes[2].grid(True, which="both", linestyle="--", alpha=0.4)

fig.suptitle("Stage 2: halo in hydrostatic equilibrium", fontsize=13)
plt.tight_layout()
plt.savefig("gravothermal_stage2.png", dpi=150)
plt.close()

print("STAGE 2 - hydrostatic equilibrium imposed")
print(f"  central v2 (temperature): {v2[0]:.3f}")
print(f"  max relative residual in balance:  {max_resid:.2e}  (want < 0.1)")
print(f"  mean relative residual in balance: {mean_resid:.2e}  (want small)")
print("  Saved gravothermal_stage2.png")