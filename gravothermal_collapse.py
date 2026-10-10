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
         [Shells, boundary conditions, conduction step and relaxation step are built.
         The relaxation step is UNDER TEST (test_relaxation.py; test 3 is the gate for
         any evolution). No evolution is run.]
STAGE 4: time evolution loop, evolve() (conduction, entropy update, relaxation with a predictor
         guess, adaptive dt, reject-and-retry). Pre-registered criterion in run_evolution.py.
         [SHORT TEST STAGE: no long run yet.]
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


def envelope_pressure(r_out, M_out, r_far=1e10, n=20001, rho_func=None):
    """Pressure the halo beyond r_out exerts at r_out: integral of G M rho / r^2 to infinity.

    The Stage 1 profile continues as rho ∝ r^-2.19 beyond the grid (BSI: the extended halo
    keeps this power law and barely evolves). r_out is a computational edge, not the edge of
    the halo, so the pressure there is the weight of the envelope outside it -- not zero.
    Integrand falls as r^-3.38, so stopping at r_far = 1e10 is negligible.
    rho_func selects the profile (default: the Stage 1 cored profile).
    """
    rr = np.geomspace(r_out, r_far, n)
    rho_t = (rho_stage1 if rho_func is None else rho_func)(rr)
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


def gravity_terms(r_edge, dm):
    """g_j = G M_j dmbar_j / (4 pi r_j^4) at edges 1..N; dmbar_j = (dm[j-1]+dm[j])/2, and dm[N-1]/2 at the edge."""
    M_e = np.concatenate(([0.0], np.cumsum(dm)))
    grav = np.empty(len(dm))
    grav[:-1] = G * M_e[1:-1] * 0.5 * (dm[:-1] + dm[1:]) / (4.0 * np.pi * r_edge[1:-1]**4)
    grav[-1] = G * M_e[-1] * 0.5 * dm[-1] / (4.0 * np.pi * r_edge[-1]**4)
    return grav


def hydrostatic_residual(r_edge, dm, P, P_ext):
    """Residual of the discrete hydrostatic equations at edges 1..N, relative to the gravity term."""
    grav = gravity_terms(r_edge, dm)
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

# ---------- alternative initial halo: NFW (Navarro, Frenk & White 1996) -- UNDER CHECK ----------
# The cored profile above is kept unchanged (all committed results use it). The NFW halo is the
# standard cuspy starting point of the SIDM gravothermal literature:
#   rho(r) = rho_s / [(r/r_s)(1 + r/r_s)^2],   M(r) = 4 pi rho_s r_s^3 [ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s)]
# with rho_s = r_s = G = 1. Shell masses come from the closed-form M(r), so they are exact. The
# pressure is the same discrete hydrostatic integration as the canonical cored setup, inward from
# the weight of the NFW envelope beyond the grid (no v^2 = 0 cliff), so the state is a fixed point
# of the relaxation solver by construction.
def rho_nfw(x):
    """NFW density, rho_s = r_s = 1."""
    return 1.0 / (x * (1.0 + x)**2)


def mass_nfw(x):
    """NFW enclosed mass, closed form, rho_s = r_s = 1 (log1p for accuracy at small x)."""
    return 4.0 * np.pi * (np.log1p(x) - x / (1.0 + x))


def build_nfw_halo(n_shell=200, r_min=1e-2, r_max=30.0):
    """Lagrangian shells for an NFW halo in discrete hydrostatic equilibrium.

    Edges: 0, then n_shell log-spaced radii from r_min to r_max (shell 0 is the sphere inside r_min).
    Returns r_edge, dm, P_ext, rho (shell means), v2, s, P.
    """
    r_e = np.concatenate(([0.0], np.logspace(np.log10(r_min), np.log10(r_max), n_shell)))
    M_e = mass_nfw(r_e)
    dm_n = np.diff(M_e)
    P_out = envelope_pressure(r_e[-1], M_e[-1], rho_func=rho_nfw)
    rho_n = shell_density(r_e, dm_n)
    P_n = pressure_from_hse(r_e, dm_n, P_out)
    v2_n = P_n / rho_n
    return {"r_edge": r_e, "dm": dm_n, "P_ext": P_out, "rho": rho_n, "v2": v2_n,
            "s": v2_n**1.5 / rho_n, "P": P_n}


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


def discrete_energy(r_edge, dm, v2_s, P_ext):
    """Return (U, W, P_ext V): thermal, gravitational and boundary-work energy of the shells.

    U = sum (3/2) dm v^2;  W = -sum_j G M_j dmbar_j / r_j (edges 1..N, same dmbar as gravity_terms);
    P_ext V = P_ext (4 pi/3) r_N^3. At fixed shell entropies, d(U + W + P_ext V)/dr_j = -4 pi r_j^2 F_j,
    with F_j the hydrostatic residual of relaxation_system. So the discrete hydrostatic equations
    are exactly the stationarity conditions of this discrete energy, and this W is the
    gravitational energy consistent with them (it is a quadrature of -G m dm / r).
    """
    gtil = gravity_terms(r_edge, dm) * 4.0 * np.pi * r_edge[1:]**4      # G M_j dmbar_j
    U = np.sum(1.5 * dm * v2_s)
    W = -np.sum(gtil / r_edge[1:])
    return U, W, P_ext * 4.0 / 3.0 * np.pi * r_edge[-1]**3


def relaxation_system(r_edge, dm, s, P_ext):
    """State at fixed shell entropies, its hydrostatic residual F, and the tridiagonal Jacobian.

    Per shell: rho_i = dm_i / V_i,  v2_i = (s_i rho_i)^(2/3),  p_i = rho_i v2_i = s_i^(2/3) rho_i^(5/3).
    Residual at edge j:  F_j = p_{j-1} - p_j - g_j  (j < N),   F_N = p_{N-1} - P_ext - g_N.
    With W_i = r_{i+1}^3 - r_i^3:
      dp_i/dr_{i+1} = -5 p_i r_{i+1}^2 / W_i,  dp_i/dr_i = +5 p_i r_i^2 / W_i,  dg_j/dr_j = -4 g_j / r_j
    so  a_j = +5 p_{j-1} r_{j-1}^2 / W_{j-1}
        b_j = -5 p_{j-1} r_j^2 / W_{j-1} - 5 p_j r_j^2 / W_j + 4 g_j / r_j   (no p_j term at j = N)
        c_j = +5 p_j r_{j+1}^2 / W_j                                       (c_N = 0)
    Unknowns dr_1..dr_N (dr_0 = 0). Arrays follow thomas_solve's layout, unscaled.
    """
    rho_r = shell_density(r_edge, dm)
    v2_r = (s * rho_r)**(2.0 / 3.0)
    p_r = rho_r * v2_r
    g = gravity_terms(r_edge, dm)
    W = r_edge[1:]**3 - r_edge[:-1]**3
    F = np.empty(len(dm))
    F[:-1] = p_r[:-1] - p_r[1:] - g[:-1]
    F[-1] = p_r[-1] - P_ext - g[-1]
    inner = 5.0 * p_r[1:] * r_edge[1:-1]**2 / W[1:]          # 5 p_j r_j^2 / W_j, j = 1..N-1
    diag = -5.0 * p_r * r_edge[1:]**2 / W + 4.0 * g / r_edge[1:]
    diag[:-1] -= inner
    lower = inner.copy()
    upper = 5.0 * p_r[1:] * r_edge[2:]**2 / W[1:]
    return {"rho": rho_r, "v2": v2_r, "p": p_r, "g": g, "F": F,
            "lower": lower, "diag": diag, "upper": upper}


def relax_to_hydrostatic(r_edge, dm, s, P_ext, tol=1e-10, max_iter=50, dense_check=True):
    """Adiabatic return to hydrostatic equilibrium (BSI 2002, sec. 4: "hydrostatic equilibrium is
    solved by keeping the entropy of each Lagrangian zone fixed, while its position and other
    thermodynamic properties are modified"). Formulation approved by the advisor before coding.

    Held fixed : shell masses dm_i (so the edge masses M_j) and shell entropies s_i = v_i^3 / rho_i;
                 r_0 = 0; P_ext. Entropies change only in the conduction step, never here.
    Unknowns   : edge radii r_1..r_N.
    Compression law: S = c_v ln(T^(3/2)/rho) + const with k_B T = m v^2, so at fixed s
                 v^2 = (s rho)^(2/3) ∝ rho^(2/3) and p ∝ rho^(5/3) (gamma = 5/3). It is the first
                 law with no heat flow: d ln v^2 = (2/3) d ln rho. Compressing a shell heats it:
                 this is the PdV work, and the shell-level form of virial_theorem.py's E = -K.
    Equations  : the discrete form of dp/dm = -G m/(4 pi r^4) (identical to d(rho v^2)/dr =
                 -G M rho / r^2, with M fixed per edge), linearised into the tridiagonal system
                 a_j dr_{j-1} + b_j dr_j + c_j dr_{j+1} = -F_j  (see relaxation_system), rows
                 scaled by g_j.
    Iteration  : Newton loop of Thomas solves, judged ROW BY ROW (advisor rulings). Each edge j
                 has its own threshold t_j = max(tol, f_j), where f_j is the double-precision floor
                 of |F_j/g_j| computed from the current state (W_i cancellation, p ∝ W^(-5/3)).
                 Converged when |F_j/g_j| <= t_j for EVERY j AND every component of the next Newton
                 step is within |J^-1| t (|J^-1|: elementwise absolute value of the dense inverse of
                 the scaled Jacobian). If every row passes, |dr| = |J^-1 F/g| <= |J^-1| t by the
                 triangle inequality, so the step test then only guards the Thomas solve's own
                 round-off; the bound's size is reported as a conditioning warning.
                 History of this criterion, all from measured failures: 1e-10 alone is below the
                 floor (~1.35e-9, test 2); a fixed step bound of 1e-13 was not safe (the claim that
                 it sat "600x" above round-off rested on one case and was wrong); and comparing
                 max|F/g| against the LARGEST floor let a row near r ~ 2 sit 14x above its own floor
                 unnoticed (scan, R = 10), because floors differ between rows by ~4 decades.
                 Step halving: a step is halved if it would make any edge non-monotonic or would
                 not reduce the weighted residual ||(F/g)/t||, so round-off noise in one row cannot
                 hide a real error in another; this changes the path, not the solution. Failure
                 (no acceptable step, or max_iter reached) raises RuntimeError with the iteration
                 history: the state is never accepted unconverged. Every solve returns its residual,
                 floor, worst row-to-threshold ratio, final step, step bound and history.
    Conditioning: the +4g/r gravity term reduces |b_j|, so diagonal dominance (which Thomas
                 without pivoting relies on) is not guaranteed. With dense_check, every Thomas
                 solve is compared with numpy.linalg.solve and the worst relative difference is
                 reported, with the smallest |b_j| / (|a_j| + |c_j|).
    Sign of the feedback, stated precisely (point 5 of the formulation): a shell that lost heat
                 has lower s, so lower p; relaxation compresses it and its v^2 RISES during this
                 step. That is guaranteed by gamma = 5/3. It does NOT guarantee that the shell ends
                 hotter than before it lost heat: alone, it does not. Net heating -- negative heat
                 capacity -- needs the whole core to contract, so that gravity (∝ M/r^4) raises
                 the required pressure faster than the density (v^2 ~ G M_c / r_c). That is an
                 outcome to be demonstrated (test_relaxation.py, test 3), not an assumption.
    Returns (r_edge_new, rho, v2, info).
    """
    def roundoff_floor(r_e, sm):
        # Per-edge floor of |F_j/g_j| from double precision: W_i = r_{i+1}^3 - r_i^3 loses
        # (r_{i+1}^3 + r_i^3)/W_i of relative precision, and p ∝ W^(-5/3).
        Wv = r_e[1:]**3 - r_e[:-1]**3
        dp = (5.0 / 3.0) * np.finfo(float).eps * (r_e[1:]**3 + r_e[:-1]**3) / Wv * sm["p"]
        fl = np.empty(len(dp))
        fl[:-1] = dp[:-1] + dp[1:]
        fl[-1] = dp[-1]
        return fl / sm["g"]

    r_cur = r_edge.copy()
    info = {"iterations": 0, "halvings": 0, "thomas_vs_dense": float("nan"),
            "min_diag_dominance": float("nan"), "history": []}
    sysm = relaxation_system(r_cur, dm, s, P_ext)
    err = np.max(np.abs(sysm["F"] / sysm["g"]))
    converged = False
    for it in range(max_iter):
        g = sysm["g"]
        lo, di, up = sysm["lower"] / g[1:], sysm["diag"] / g, sysm["upper"] / g[:-1]
        rhs = -sysm["F"] / g
        dx = thomas_solve(lo, di, up, rhs)
        worst = int(np.argmax(np.abs(sysm["F"] / g)))
        step = np.max(np.abs(dx) / r_cur[1:])
        fl_vec = roundoff_floor(r_cur, sysm)
        floor = np.max(fl_vec)
        t = np.maximum(tol, fl_vec)                          # per-row threshold t_j = max(tol, f_j)
        row_ratio = np.abs(sysm["F"] / g) / t
        worst_row = int(np.argmax(row_ratio))
        info.setdefault("row_history", []).append((row_ratio[worst_row], r_cur[worst_row + 1]))
        bound_rel = float("nan")
        if np.all(row_ratio <= 1.0):
            # every row at its threshold: the step can then be at most |J^-1| t
            A = np.diag(di) + np.diag(lo, -1) + np.diag(up, 1)
            bound = np.abs(np.linalg.inv(A)) @ t
            bound_rel = np.max(bound / r_cur[1:])
            if np.all(np.abs(dx) <= bound):
                info["history"].append((it, err, r_cur[worst + 1], step, floor, bound_rel))
                converged = True
                info.update(residual=err, roundoff_floor=floor, step=step, step_bound=bound_rel,
                            max_row_ratio=row_ratio[worst_row])
                break
        info["history"].append((it, err, r_cur[worst + 1], step, floor, bound_rel))
        dom = np.abs(di) / (np.concatenate(([0.0], np.abs(lo))) + np.concatenate((np.abs(up), [0.0])))
        info["min_diag_dominance"] = np.nanmin([info["min_diag_dominance"], dom.min()])
        if dense_check:
            A = np.diag(di) + np.diag(lo, -1) + np.diag(up, 1)
            dx_dense = np.linalg.solve(A, rhs)
            diff = np.max(np.abs(dx - dx_dense)) / np.max(np.abs(dx_dense))
            info["thomas_vs_dense"] = np.nanmax([info["thomas_vs_dense"], diff])
        norm0 = np.linalg.norm(sysm["F"] / g / t)            # weighted: each row against its threshold
        lam = 1.0
        while True:
            r_try = r_cur.copy()
            r_try[1:] += lam * dx
            if np.all(np.diff(r_try) > 0.0):
                sys_try = relaxation_system(r_try, dm, s, P_ext)
                if np.linalg.norm(sys_try["F"] / sys_try["g"] / t) < norm0:
                    break
            lam *= 0.5
            info["halvings"] += 1
            if lam < 1e-8:
                raise RuntimeError(f"relaxation: no acceptable Newton step at iteration {it}, "
                                   f"max relative residual {err:.3e}\n" + history_text(info))
        r_cur, sysm = r_try, sys_try
        err = np.max(np.abs(sysm["F"] / sysm["g"]))
        info["iterations"] = it + 1
    if not converged:
        raise RuntimeError(f"relaxation did not converge in {max_iter} iterations: "
                           f"max relative residual {err:.3e}\n" + history_text(info))
    return r_cur, sysm["rho"], sysm["v2"], info


def region_virial(r_e, dm, v2_s, J, P_ext):
    """2K, W and 3 p_s V for the region made of shells 0..J-1 (surface at edge J).

    Summing the discrete hydrostatic equations times 4 pi r_j^3 over edges 1..J gives exactly
    2K + W = 3 p_s V; |W| / 3 p_s V > 1 means the region is held up by its own gravity.
    (Same quantities as scan_heat_capacity.py.)
    """
    rho_e = shell_density(r_e, dm)
    gtil = gravity_terms(r_e, dm) * 4.0 * np.pi * r_e[1:]**4      # G M_j dmbar_j, edges 1..N
    W = -np.sum(gtil[:J] / r_e[1:J + 1])
    twoK = 3.0 * np.sum(dm[:J] * v2_s[:J])
    p_s = P_ext if J == len(dm) else rho_e[J] * v2_s[J]
    return twoK, W, 3.0 * p_s * (4.0 / 3.0) * np.pi * r_e[J]**3


def evolve(r_start, v2_start, dm, P_ext, dt0, n_steps, xi=0.02, growth_cap=1.2, dt_min=1e-16):
    """Gravothermal time evolution (Stage 4): yields one record per accepted step.

    Each step, as pre-registered by the advisor:
      1. conduction_step (implicit, density fixed), then each shell's entropy s = v^3/rho is
         updated from the new v^2 -- the only place s changes;
      2. relax_to_hydrostatic (unchanged, per-row criterion), started from a PREDICTOR guess:
         r_guess = r(t) + (r(t) - r(t - dt_old)) / dt_old * dt_new. The guess only changes where
         Newton starts, not the solution; if it would make an edge non-monotonic, r(t) is used;
      3. on success, the next dt = dt * min(growth_cap, xi / max_i |drho_i / rho_i|)
         (xi = 0.02: no shell's density should change by more than ~2% per step);
      4. reject-and-retry: if relaxation raises, conduction gives a non-finite or non-positive
         v^2, or the shells' total mass differs from sum(dm) by more than 1e-12 (relative), the
         state is left unchanged and dt is halved. Below dt_min the evolution stops with
         RuntimeError ("breakdown"). Shell masses are fixed by construction in this Lagrangian
         scheme, so the mass check is expected to pass always; it is kept as asked.
    dt0 is a numerical starting value only; the controller then sets dt.
    """
    V = lambda r_e: 4.0 / 3.0 * np.pi * (r_e[1:]**3 - r_e[:-1]**3)
    r = r_start.copy()
    v2 = v2_start.copy()
    rho = shell_density(r, dm)
    r_prev, dt_prev = None, None
    t, dt = 0.0, dt0
    M_tot = np.sum(dm)
    for step in range(1, n_steps + 1):
        rejections, reasons = 0, []
        while True:
            if dt < dt_min:
                raise RuntimeError(f"breakdown at step {step}, t = {t:.6e}: dt fell below {dt_min:g} "
                                   f"after {rejections} rejections; last reasons: {reasons[-3:]}")
            v2_c = conduction_step(r, dm, v2, dt)
            if not np.all(np.isfinite(v2_c)) or np.any(v2_c <= 0.0):
                rejections += 1
                reasons.append("conduction gave non-finite or non-positive v2")
                dt *= 0.5
                continue
            s_new = v2_c**1.5 / rho
            guess, predicted = r, False
            if r_prev is not None:
                g_try = r + (r - r_prev) / dt_prev * dt
                if np.all(np.diff(g_try) > 0.0):
                    guess, predicted = g_try, True
            try:
                r_new, rho_new, v2_new, info = relax_to_hydrostatic(guess, dm, s_new, P_ext)
            except RuntimeError as e:
                rejections += 1
                reasons.append("relaxation: " + str(e).splitlines()[0])
                dt *= 0.5
                continue
            mass_err = abs(np.sum(rho_new * V(r_new)) - M_tot) / M_tot
            if mass_err > 1e-12:
                rejections += 1
                reasons.append(f"mass error {mass_err:.2e}")
                dt *= 0.5
                continue
            break
        max_drho = np.max(np.abs(rho_new / rho - 1.0))
        dt_used = dt
        r_prev, dt_prev = r, dt
        r, v2, rho = r_new, v2_new, rho_new
        t += dt_used
        dt = dt_used * (growth_cap if max_drho == 0.0 else min(growth_cap, xi / max_drho))
        yield {"step": step, "t": t, "dt_used": dt_used, "dt_next": dt, "r": r, "v2": v2, "rho": rho,
               "max_drho": max_drho, "rejections": rejections, "reasons": reasons,
               "predicted": predicted, "info": info, "mass_err": mass_err}


def history_text(info):
    """Per-iteration record of a relaxation solve, for error reports."""
    lines = ["  iter   max|F/g|   at edge r    max|dr/r|   floor |F/g|   max_j |F_j/g_j|/t_j  at edge r"
             "    step bound max(|J^-1| t / r)"]
    for (it, e, rw, dxr, fl, bd), (rr, rr_r) in zip(info["history"], info.get("row_history", [])):
        bd_txt = "  (a row above its threshold: not computed)" if np.isnan(bd) else f"  {bd:10.3e}"
        lines.append(f"  {it:4d}  {e:10.3e}  {rw:10.4e}  {dxr:10.3e}  {fl:10.3e}"
                     f"  {rr:14.3e}       {rr_r:10.4e}" + bd_txt)
    return "\n".join(lines)


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
print("  relaxation step: implemented, UNDER TEST (test_relaxation.py; test 3 gates any evolution)")
