"""
Gravothermal Collapse of a Self-Interacting Dark Matter Halo
Reproduction of Balberg, Shapiro & Inagaki (2002), cited in my BSc dissertation.

STAGE 1: radial grid and initial density (r^-2.19). [done]
STAGE 2: hydrostatic equilibrium. [done; method replaced]
         Canonical: Lagrangian-shell construction with exact shell masses and the envelope
         pressure P_ext at the outer edge. Second order; v^2 within 1.1e-3 of the continuum
         solution at N = 200 (diagnose_stage2_resolution.py).
         The original Stage 2 integration (mean residual ~0.05) is kept below as a superseded
         record: it is first order, its v^2 is 7.6-10% too high at N = 200, and its enclosed
         mass leaves out the sphere inside r = 0.01 and overcounts by about half a cell.
STAGE 3: BSI (2002) "conduction-then-relaxation" in Lagrangian mass shells.
         [SCAFFOLDING ONLY: shells, boundary conditions and the conduction step are
         built; the relaxation step is a stub and no evolution is run.]
         The previous non-working Stage 3 (Newton/LM) is kept in git commit f0e90af;
         its three defects are shown in diagnose_stage3.py.
Long-mean-free-path conduction only; a simplified reproduction of the self-similar regime.
"""

import numpy as np
from scipy.integrate import quad

G = 1.0

# ---------- Stage 1: grid and initial density profile ----------
N = 200
r = np.logspace(-2, 2, N)
alpha = 2.19


def rho_stage1(x):
    """The Stage 1 density profile, as a function of radius."""
    return 1.0 / (1.0 + x**alpha)


rho = rho_stage1(r)


# ---------- shell geometry and hydrostatics (used by Stages 2 and 3) ----------
def shell_density(r_edge, dm):
    """Mean density of each shell: fixed mass over current volume."""
    return dm / (4.0 / 3.0 * np.pi * (r_edge[1:]**3 - r_edge[:-1]**3))


def shell_centres(r_edge):
    """Radius that splits each shell's volume in half; used for gradients between shells."""
    return (0.5 * (r_edge[:-1]**3 + r_edge[1:]**3))**(1.0 / 3.0)


def envelope_pressure(r_out, M_out, r_far=1e10, n=20001):
    """Pressure the halo beyond r_out exerts at r_out: integral of G M rho / r^2 to infinity.

    The Stage 1 profile continues as rho ∝ r^-2.19 beyond the grid (BSI: the extended halo
    keeps this power law and barely evolves). r_out is a computational edge, not the edge of
    the halo, so the pressure there is the weight of the envelope outside it -- not zero.
    Integrand falls as r^-3.38, so stopping at r_far = 1e10 is negligible.
    """
    rr = np.geomspace(r_out, r_far, n)
    rho_t = rho_stage1(rr)
    f = 4.0 * np.pi * rr**2 * rho_t
    M_t = M_out + np.concatenate(([0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(rr))))
    return np.trapezoid(G * M_t * rho_t / rr**2, rr)


def pressure_from_hse(r_edge, dm, P_ext):
    """Shell pressures in exact discrete hydrostatic equilibrium, integrated inward from P_ext.

    Discretisation of dp/dm = -G m / (4 pi r^4), evaluated at the edges:
      interior edge j (between shells j-1 and j): p[j-1] - p[j] = G M_j (dm[j-1]+dm[j])/2 / (4 pi r_j^4)
      outer edge N:                               p[N-1] - P_ext = G M_N (dm[N-1]/2)     / (4 pi r_N^4)
    hydrostatic_residual() uses the same formulas, so this state is its exact fixed point
    (the property the old reconstruction lacked: defect 2).
    """
    M_e = np.concatenate(([0.0], np.cumsum(dm)))
    n = len(dm)
    P = np.empty(n)
    P[-1] = P_ext + G * M_e[-1] * (0.5 * dm[-1]) / (4.0 * np.pi * r_edge[-1]**4)
    for j in range(n - 1, 0, -1):
        P[j - 1] = P[j] + G * M_e[j] * 0.5 * (dm[j - 1] + dm[j]) / (4.0 * np.pi * r_edge[j]**4)
    return P


def hydrostatic_residual(r_edge, dm, P, P_ext):
    """Residual of the discrete hydrostatic equations at edges 1..N, relative to the gravity term."""
    M_e = np.concatenate(([0.0], np.cumsum(dm)))
    grav = np.empty(len(dm))
    grav[:-1] = G * M_e[1:-1] * 0.5 * (dm[:-1] + dm[1:]) / (4.0 * np.pi * r_edge[1:-1]**4)
    grav[-1] = G * M_e[-1] * 0.5 * dm[-1] / (4.0 * np.pi * r_edge[-1]**4)
    dP = np.empty(len(dm))
    dP[:-1] = P[:-1] - P[1:]
    dP[-1] = P[-1] - P_ext
    return (dP - grav) / grav


# ---------- Stage 2 (canonical): hydrostatic equilibrium on Lagrangian shells ----------
# Shell edges are the Stage 1 grid points plus the centre, so shell 0 is the sphere inside
# r[0]. Each shell's mass is the exact integral of the Stage 1 density over it (this is the
# corrected enclosed mass: it includes the inner sphere and has no half-cell overcount), and
# is then fixed for all time. M_edge[k] is the exact mass inside r_edge[k].
r_edge = np.concatenate(([0.0], r))
n_shell = len(r_edge) - 1
dm = np.array([quad(lambda x: 4.0 * np.pi * x**2 * rho_stage1(x), r_edge[i], r_edge[i + 1])[0]
               for i in range(n_shell)])
M_edge = np.concatenate(([0.0], np.cumsum(dm)))

# --- boundary conditions ---
# Inner (r = 0): r_0 = 0, M_0 = 0 and L_0 = 0 by spherical symmetry; edge 0 never moves.
# Outer (r_N = 100): the pressure is held at P_ext, the weight of the r^-2.19 envelope beyond
#   the grid (envelope_pressure). This replaces the original Stage 2's p = 0 edge, which forced
#   v^2 = 0 in the last cell and created the 3e7 temperature cliff (defect 1). P_ext is computed
#   once from the Stage 1 profile -- no free constant -- and held fixed because BSI's extended
#   halo barely evolves. The outer radius may move during relaxation.
#   Thermally the outer edge is insulating, L_N = 0: an isolated halo, as in BSI. Heat that
#   reaches the last shell stays there; with r_N = 100 core radii this is a stated
#   approximation, to be checked once evolution runs.
P_ext = envelope_pressure(r_edge[-1], M_edge[-1])

# --- initial thermal state ---
# Integrate hydrostatic equilibrium inward from P_ext, then v^2 = p / rho, using the
# discretisation the relaxation step will enforce.
#
# Temperature inversion -- a property of this cored halo, not a numerical artefact. The
# exact continuum solution for rho = 1/(1 + r^2.19) has v^2 = 2.69 at the centre rising to
# 3.61 near r = 10 (diagnose_stage2_resolution.py), because v^2 = p/rho rises wherever the
# density steepens faster than the pressure falls: d ln v^2/d ln r = d ln p/d ln r - d ln rho/d ln r.
# Starting from this cored halo, the evolution therefore has an initial phase in which heat
# flows INWARD and the core mildly expands, before any gravothermal runaway. That early phase
# is physical for this start and must not be read as the collapse, or as the old bug; the
# setup checks below count the inward-conducting edges so it stays visible. (BSI start from
# their self-similar profile instead; its ODEs are not available to us in a verifiable form,
# so we do not reconstruct it.)
rho_s = shell_density(r_edge, dm)
P_s = pressure_from_hse(r_edge, dm, P_ext)
v2_s = P_s / rho_s
s_s = v2_s**1.5 / rho_s

# ---------- Stage 2 (original, SUPERSEDED): kept verbatim as a record, not used ----------
# This was the first Stage 2 and carried a "verified" label on its mean residual of ~0.05.
# Measured against the continuum solution (diagnose_stage2_resolution.py):
#   - its v^2 is 7.6% (centre) to 10.2% (r = 10-50) too high at N = 200, a first-order error
#     that halves when N doubles: the enclosed mass is overcounted by about half a cell
#     (+5.5% at r = 1), and the pressure is integrated with gravity from the inner end of each
#     interval;
#   - its enclosed mass leaves out the sphere inside r = 0.01 (M is 80% low at r = 0.01, and
#     the error grows with N) -- the "rough inner-boundary shell" noted in commit 935c3f9;
#   - p = 0 at r = 100 gives v^2 = 0 in the last cell (defect 1 of the old Stage 3).
# The "verified" label applies to the canonical shell construction above, not to this.
dr_s2 = np.gradient(r)
M_s2 = np.cumsum(4 * np.pi * r**2 * rho * dr_s2)
gravity_term_s2 = G * M_s2 * rho / r**2
p_s2 = np.zeros_like(r)
for i in range(N - 2, -1, -1):
    p_s2[i] = p_s2[i+1] + gravity_term_s2[i] * (r[i+1] - r[i])
v2_s2 = p_s2 / rho

# ---------- Stage 3: gravothermal evolution in Lagrangian mass shells ----------
# This is the dissertation's conclusion made to run, driven by the same engine as
# virial_theorem.py (simulation 3). There, a virialised sphere has E = -K, so when it
# loses energy it contracts and its temperature RISES: negative heat capacity. Here the
# same mechanism is meant to act shell by shell inside a real halo, in two sub-steps:
#   1. conduction: heat flows down the temperature gradient. At fixed shell volume a shell
#      that loses heat has its v^2 fall -- ordinary, positive-heat-capacity loss. On its own
#      this can only relax the profile toward uniform (diagnose_stage3.py, section E).
#   2. relaxation: a shell that lost heat no longer has the pressure to carry the weight
#      above it, so the shells move back into hydrostatic equilibrium. The move is
#      adiabatic (each shell keeps its entropy), and compression at fixed entropy heats the
#      shell: v^2 ∝ rho^(2/3). This PdV work is the term the earlier attempts lacked.
# For a self-gravitating core the compression heating outweighs the conductive loss -- that
# is what negative heat capacity means -- so the core ends each cycle denser and hotter,
# conducts faster, and runs away: the gravothermal catastrophe of BSI (2002), the route to
# a seed black hole argued in the dissertation. That outcome is the thing to be SHOWN once
# the relaxation step exists; nothing here assumes or claims it.
#
# Equations (BSI 2002, eqs. 4-7; time derivatives Lagrangian), in mass coordinate m:
#   dr/dm   = 1 / (4 pi r^2 rho)                       mass
#   dp/dm   = -G m / (4 pi r^4),   p = rho v^2         hydrostatic equilibrium
#   L       = -4 pi r^2 kappa dv^2/dr                  conduction (flux down the gradient)
#   (3/2) Dv^2/Dt - (v^2/rho) Drho/Dt = -dL/dm         first law: heat flux + PdV work
# The first law is split exactly along the two sub-steps: conduction carries -dL/dm at fixed
# rho; relaxation carries the PdV term at fixed entropy s = v^3/rho (BSI's effective entropy).

# --- physical inputs for conduction (long-mean-free-path limit) ---
# lmfp flux, BSI eq. (12): F = L/(4 pi r^2) ∝ -rho (H^2 / t_r) dv^2/dr, with the gravitational
# scale height H^2 = v^2 / (4 pi G rho) and the SIDM relaxation time t_r ∝ 1/(rho sigma v)
# (paper text: "for SIDM t_r ∝ (rho v)^-1"). Hence kappa ∝ sigma rho v^3 / (4 pi G).
# The paper's combined bracket [a sigma^2 + 4 pi G (rho v^2)^-1]^-1 (text below eq. 14)
# reduces to rho v^2/(4 pi G) in this limit, consistent with the v^3 dependence.
# The advisor's first brief wrote v to the first power; that was checked against the paper
# text above and corrected to v^3 by the advisor's ruling.
SIGMA_HAT = 1.0
# Cross-section per unit mass in code units. In the pure lmfp limit it only sets the time
# unit (t ∝ 1/sigma); times will be reported in units of the initial central t_r.
C_LMFP_PLACEHOLDER = 1.0
# NAMED PLACEHOLDER, NOT A FITTED VALUE: the overall numerical coefficient of BSI eq. (14)
# (it involves b ≈ 1.002 and a = sqrt(16/pi)) is in the PDF's equation images, which this
# session could not read; the advisor will supply it. In the lmfp limit it only rescales
# time; it must be set before any time is compared with BSI's t_coll ≈ 290 t_r.


def thomas_solve(lower, diag, upper, rhs):
    """Solve a tridiagonal linear system with the Thomas algorithm (from commit 2ec9193)."""
    n = len(diag)
    c_prime = np.zeros_like(upper)
    d_prime = np.zeros_like(rhs)
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom

    x = np.zeros_like(rhs)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def conductivity_lmfp(rho_s, v2_s):
    """kappa in F = -kappa dv^2/dr, lmfp limit: C * sigma * rho * v^3 / (4 pi G). See notes above."""
    return C_LMFP_PLACEHOLDER * SIGMA_HAT * rho_s * v2_s**1.5 / (4.0 * np.pi * G)


def edge_conductance(r_edge, dm, v2_s):
    """C_j = 4 pi r_j^2 kappa_j / (rc_j - rc_{j-1}) at interior edges, so that L_j = -C_j (v2_j - v2_{j-1})."""
    kappa = conductivity_lmfp(shell_density(r_edge, dm), v2_s)
    rc = shell_centres(r_edge)
    return 4.0 * np.pi * r_edge[1:-1]**2 * 0.5 * (kappa[:-1] + kappa[1:]) / (rc[1:] - rc[:-1])


def luminosity(r_edge, dm, v2_s):
    """L at all N+1 edges. L = 0 at the centre (symmetry) and at the outer edge (isolated halo)."""
    L = np.zeros(len(dm) + 1)
    L[1:-1] = -edge_conductance(r_edge, dm, v2_s) * (v2_s[1:] - v2_s[:-1])
    return L


def conduction_step(r_edge, dm, v2_s, dt):
    """Implicit conduction sub-step at fixed shell radii (so fixed rho): returns new v^2.

    Per shell, (3/2) dm_i (v2_i' - v2_i) / dt = L_i - L_{i+1}, i.e. (3/2) Dv^2/Dt = -dL/dm.
    In the mass coordinate the spherical geometry sits in L = 4 pi r^2 F, so the divergence
    is just a difference of L across a shell. This settles the 1/r^2 question from the
    diagnosis: the working-tree operator (1/r^2) d/dr(r^2 kappa dv^2/dr) had the right
    spherical divergence but no heat capacity per volume, (3/2) rho; the committed operator
    d/dr(r^2 kappa dv^2/dr) without 1/r^2 was not a spherical divergence at all. Written per
    shell, sum((3/2) dm v^2) is conserved exactly when L = 0 at both ends.
    Backward Euler with kappa evaluated at the old v^2 (lagged), giving a tridiagonal system.
    """
    Cj = edge_conductance(r_edge, dm, v2_s)
    cap = 1.5 * dm
    diag = cap.copy()
    diag[:-1] += dt * Cj
    diag[1:] += dt * Cj
    off = -dt * Cj
    return thomas_solve(off, diag, off, cap * v2_s)


def relax_to_hydrostatic(r_edge, dm, s, P_ext):
    """STUB -- NOT IMPLEMENTED. Adiabatic return to hydrostatic equilibrium (BSI 2002, sec. 4:
    "hydrostatic equilibrium is solved by keeping the entropy of each Lagrangian zone fixed,
    while its position and other thermodynamic properties are modified").

    This is where the PdV heating -- the negative heat capacity of virial_theorem.py -- lives.
    Plan for the formulation check:
      held fixed : shell masses dm_i and shell entropies s_i = v_i^3 / rho_i
      unknowns   : edge radii r_1..r_N (r_0 = 0 at the centre stays fixed)
      per shell  : rho_i = dm_i / ((4 pi/3)(r_{i+1}^3 - r_i^3))
                   v2_i  = (s_i rho_i)^(2/3)       <- v^2 ∝ rho^(2/3): compression heats
                   p_i   = rho_i v2_i = s_i^(2/3) rho_i^(5/3)   (gamma = 5/3, monatomic)
      equations  : hydrostatic_residual(...) = 0 at edges 1..N, outer edge against P_ext.
      Edge j's equation involves p_{j-1} (depends on r_{j-1}, r_j), p_j (r_j, r_{j+1}) and r_j,
      so linearising in the edge displacements gives the tridiagonal system
          a_j dr_{j-1} + b_j dr_j + c_j dr_{j+1} = d_j     (d_j = -residual_j),
      solved with thomas_solve and iterated (Newton) until the residual is small.
      Returns new r_edge and v2; the caller recomputes s only after conduction steps.
    """
    raise NotImplementedError("Stage 3 relaxation step not built yet (awaiting formulation check).")


# ---------- setup checks (no time evolution is run) ----------
rc = shell_centres(r_edge)
M_trap = 4.0 * np.pi * np.trapezoid(r**2 * rho, r)
res = hydrostatic_residual(r_edge, dm, P_s, P_ext)
print("STAGE 3 SCAFFOLD - setup checks only; no evolution run")
print(f"  shells: {n_shell}   total shell mass = {M_edge[-1]:.6e}   (trapezoid mass on Stage 1 grid {M_trap:.6e})")
print(f"  P_ext (envelope beyond r = {r_edge[-1]:.0f}) = {P_ext:.6e}")
print(f"  v2 in last 3 shells = {v2_s[-3:]}   (superseded Stage 2 edge had v2 = {v2_s2[-1]:.1f})")
print(f"  max |hydrostatic residual| = {np.max(np.abs(res)):.3e}   (zero by construction:"
      f" confirms the initial state is the relaxation step's fixed point)")
print(f"  centre v2 = {v2_s[0]:.6f}   max v2 = {v2_s.max():.6f} at r = {rc[np.argmax(v2_s)]:.3f}")
print("  canonical shell v2 vs superseded Stage 2 v2 at the same radius:")
for target in [0.01, 0.1, 1.0, 10.0, 50.0, 90.0]:
    k = np.argmin(np.abs(rc - target))
    v2_old = np.interp(rc[k], r, v2_s2)
    print(f"    r = {rc[k]:9.4f}   shells {v2_s[k]:.6f}   superseded {v2_old:.6f}   ratio {v2_s[k] / v2_old:.6f}")

L0 = luminosity(r_edge, dm, v2_s)
print(f"  initial L: {np.sum(L0[1:-1] < 0)} of {n_shell - 1} interior edges carry heat INWARD (L < 0);"
      f" outermost inward edge at r = {r_edge[1:-1][L0[1:-1] < 0].max() if np.any(L0[1:-1] < 0) else float('nan'):.3f}")

# Operator check of the conduction sub-step (one step at fixed rho; not an evolution).
# dt is arbitrary here: conservation must hold for any dt (a scan of dt is in
# diagnose_stage2_resolution.py, part 2).
E_th0 = np.sum(1.5 * dm * v2_s)
v2_one = conduction_step(r_edge, dm, v2_s, dt=1.0)
E_th1 = np.sum(1.5 * dm * v2_one)
print(f"  one conduction step (dt = 1): thermal energy change {abs(E_th1 - E_th0) / E_th0:.3e} (relative),"
      f" max |dv2| = {np.abs(v2_one - v2_s).max():.4e}")
print("  relaxation step: STUB (relax_to_hydrostatic raises NotImplementedError)")
