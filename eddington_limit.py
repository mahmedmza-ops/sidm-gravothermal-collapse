"""
Eddington Luminosity Limit
Reproduced from the derivation in my BSc dissertation, 'Galactic Black Holes'.
The Eddington limit is the maximum luminosity at which radiation pressure
balances gravity. Above it, an object cannot accrete or hold its outer layers.
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical constants (SI units)
G = 6.674e-11        # gravitational constant, m^3 kg^-1 s^-2
m_p = 1.673e-27      # proton mass, kg
c = 2.998e8          # speed of light, m/s
sigma_T = 6.652e-29  # Thomson scattering cross-section, m^2
M_sun = 1.989e30     # solar mass, kg
L_sun = 3.828e26     # solar luminosity, W (for context only)


def eddington_luminosity(mass_kg):
    """Return the Eddington luminosity in watts for a given mass in kg."""
    return (4 * np.pi * G * mass_kg * m_p * c) / sigma_T


# A range of masses, from one Sun to one billion Suns (spanning star to SMBH)
masses_solar = np.logspace(0, 9, 500)      # 10^0 to 10^9 solar masses
masses_kg = masses_solar * M_sun

luminosities = eddington_luminosity(masses_kg)
luminosities_solar = luminosities / L_sun  # express in solar luminosities

# Plot
plt.figure(figsize=(8, 6))
plt.loglog(masses_solar, luminosities_solar, color="black", linewidth=2)
plt.xlabel("Mass  (solar masses)")
plt.ylabel("Eddington luminosity  (solar luminosities)")
plt.title("The Eddington Limit")
plt.grid(True, which="both", linestyle="--", alpha=0.4)

# Mark one reference point: a 10^8 solar-mass black hole
M_ref = 1e8
L_ref = eddington_luminosity(M_ref * M_sun) / L_sun
plt.scatter([M_ref], [L_ref], color="red", zorder=5)
plt.annotate(f"10^8 M_sun\n{L_ref:.1e} L_sun",
             (M_ref, L_ref), textcoords="offset points", xytext=(-80, 10))

plt.tight_layout()
plt.savefig("eddington_limit.png", dpi=150)
plt.show()
