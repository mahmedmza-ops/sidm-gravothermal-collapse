# Capstone: gravothermal collapse of a self-interacting dark matter halo — findings

This is the fourth and last simulation in a project that takes the mathematics of my BSc
dissertation, *Galactic Black Holes*, to working code. The dissertation followed Balberg,
Shapiro & Inagaki (2002, "BSI") in arguing that supermassive black hole seeds can form through
the gravothermal collapse of self-interacting dark matter (SIDM) halos. The engine of that
collapse is negative heat capacity: a self-gravitating system that loses energy contracts and
gets hotter. Simulation 3 (`virial_theorem.py`) demonstrates that for a uniform sphere, where
the virial theorem gives E = −K. This capstone asks whether the same mechanism is present, and
measurable, in a real halo, and whether it drives a collapse.

The document is written as the sequence the work actually followed, failures and corrections
included:

1. solver built and verified;
2. t = 0 gate failed;
3. negative heat capacity located by a scan;
4. evolution engine built;
5. the cored halo expanded in a short test;
6. the NFW halo collapsed, against a criterion fixed beforehand;
7. a convergence test, also fixed beforehand, confirmed the result.

Part I was written before any time evolution was run. It is kept as it was, with notes where
later work superseded a statement.

## Summary of the result

Starting from an NFW halo, this long-mean-free-path gravothermal model produced core formation
followed by gravothermal core collapse. The central density first fell by a factor of 58, then
rose to 100 times its starting value while the core heated, contracted and shed mass.

- **The success criterion was met, and fixed in advance.** It was written and committed to git
  before the run.
- **A convergence test confirmed the result.** It was also committed before it ran, and the
  collapse time and two collapse exponents stayed stable to within 3.3%.
- **The evidence positively rules out the main numerical suspect.** The scheme leaks a little
  energy, which acts like an artificial heat sink and could in principle speed a collapse up. But
  halving the per-step density limit halved that leak and made the collapse come **earlier**
  (t = 149.6 against 154.7), not later. That is the opposite of what a leak driving the collapse
  would produce.
- **The collapse exponents match BSI, with nothing tuned.** Extrapolated to zero step size, the
  two late-collapse exponents are 0.0871 and −2.765. BSI's self-similar values are 0.0867 and
  −2.704, so they agree to 0.5% and 2.3%, from a different initial profile.

This is not a reproduction of BSI's Figure 2. It does not include their short-mean-free-path core
physics, and no black hole forms in this model: it stops at 100 times the central density.

---

# Part I — before evolution (written 2026-10-10)

## What was built (`gravothermal_collapse.py`)

- **Halo.** A cored profile, ρ = 1/(1 + r^2.19), in units with G = 1, divided into 200
  Lagrangian mass shells from r = 0 to r = 100. Each shell's mass is the exact integral of the
  profile and never changes.
- **Hydrostatic equilibrium.** Built inward from the outer edge, using the same discrete
  equations the solver later enforces, so the starting state is an exact fixed point.
  - At the outer edge the pressure is the weight of the r^-2.19 envelope beyond r = 100, not
    zero. A zero-pressure edge had produced a 3×10⁷ temperature cliff in an earlier attempt.
  - Against the exact continuum solution this is second-order accurate: v² is within 1.1×10⁻³
    at 200 shells.
  - The project's original Stage 2 turned out to be first-order, with v² 7.6–10% too high. It is
    kept in the file as a labelled, superseded record.
- **Conduction step.** Heat flows down the temperature gradient, solved implicitly with a
  tridiagonal solver. It uses the long-mean-free-path conductivity κ ∝ σρv³, derived from BSI's
  definitions. Total thermal energy is conserved to round-off while up to 8.8% of it is
  redistributed. *(At this stage it had not been used in any time evolution; see Part II.)*
- **Relaxation step: the physics earlier attempts were missing.** After heat moves, each shell
  returns to hydrostatic equilibrium keeping its own entropy. For a monatomic gas that means
  v² ∝ ρ^(2/3), so compressing a shell heats it. This is the PdV work, the shell-level form of
  E = −K.
  - The equations are solved by Newton iteration with an analytical Jacobian, which agrees with
    finite differences to 2×10⁻⁹. Convergence is quadratic.
  - The convergence test is per row: each equation must reach max(10⁻¹⁰, its own computed
    round-off floor), and the final step must be within the step that round-off alone can
    produce.
  - If a solve does not converge, the code raises an error; it never accepts the state.

Earlier attempts at this stage, still in the git history, failed for three diagnosed reasons:
the zero-pressure boundary cliff, a density reconstruction that was not a fixed point of its own
equations, and the missing PdV term. Each was shown numerically in `diagnose_stage3.py` before
being fixed.

## Tests of the relaxation step (`test_relaxation.py`)

All test choices were fixed before the tests were run.

- **Fixed point.** Relaxing the unperturbed halo moves nothing: the forced Newton step is
  1.2×10⁻¹⁶.
- **Single shell (r ≈ 0.1), with 1% of its heat removed.** The shell is compressed (density
  ×1.006) and heats during relaxation (+0.0107 in v²). It still ends cooler than it started
  (−0.016). One shell alone does not show negative heat capacity; the whole structure sinks
  slightly around it. A pre-run prediction that the shell's inner edge would move outward was
  wrong; everything moves inward.
- **The gate: the core r < 1, with 1% of its heat removed.** The criterion, fixed before running,
  was that the core's mass-weighted v² must end higher than before the loss. **It failed.** The
  v² went 2.80996 → 2.78186 (after the loss) → 2.80088 (after relaxation), an effective heat
  capacity of **+0.32**. Compression gave back about 68% of the lost heat, but not all of it.

This region holds 0.4% of the halo's mass and is held in mainly by the pressure of the envelope
around it: its surface-pressure term is about 3.7 times its self-gravity. For a region like
that, positive heat capacity is the expected result. The gate's verdict stands. It was not
redefined to produce a pass, and no time-dependent gate was introduced.

## Where negative heat capacity lives (`scan_heat_capacity.py`)

An exploratory scan, with its choices fixed before running and labelled as unable to overturn
the gate. It repeats the 1% heat-removal test for regions r < R, and checks the virial theorem
with a surface-pressure term, 2K + W = 3p_sV, for each region. That identity holds to ≤ 2×10⁻¹⁶
at every R. The ratio |W| / 3p_sV measures whether a region is held up by its own gravity
(above 1) or by the pressure around it (below 1).

| Region r < R | 0.29 | 0.98 | 2.97 | 9.9 | 28.7 | 100 (whole halo) |
|---|---|---|---|---|---|---|
| Mass fraction | 0.02% | 0.4% | 3.2% | 12.8% | 34.4% | 100% |
| Heat capacity C | +0.558 | +0.323 | −0.074 | −0.353 | −0.460 | −0.268 |
| \|W\| / 3p_sV | 0.030 | 0.274 | 1.049 | 2.105 | 2.845 | 3.180 |

- **Negative heat capacity exists in this halo at t = 0,** in every region r < R with R ≥ 3.
  There, losing 1% of the thermal energy leaves the region hotter after it relaxes.
- **The sign of C changes between R = 1 and R = 3,** the same interval in which self-gravity
  overtakes the surrounding pressure. Small regions held in by pressure have positive heat
  capacity; self-gravitating regions have negative heat capacity. That is consistent with the
  Lynden-Bell & Wood (1968) condition that BSI cite.
- **C is not a simple function of |W| / 3p_sV.** The ratio rises all the way out to R = 100, but
  C is most negative near R ≈ 29 and weaker for the whole halo. The whole halo is not the clean
  E = −K system of simulation 3, because the outer pressure adds a boundary term. Its equilibrium
  identity, ΔH = −ΔK + 4P_ext·ΔV, holds to 2×10⁻¹⁰.

At that stage the conclusion was that such regions exist in this halo, but that the collapse
itself had not been shown. Part II addresses the collapse.

### Limitations as recorded at the end of Part I

The first and fourth are superseded by Part II.

- **No time evolution was run** (superseded).
- **The crossover is bracketed coarsely,** by a single grid interval (R = 1 to 3). Every heat
  capacity uses a single 1% perturbation, and none has been tested at other resolutions.
- **The test is quasi-static.** Relaxation assumes each shell keeps its entropy. Energy released
  during relaxation is second-order: 0.3% of the heat removed in the gate test.
- **The starting profile is a cored halo, not BSI's self-similar profile** (the evolution used
  NFW instead; see Part II). BSI's self-similar equations could not be read reliably from the
  available copy of the paper, so they were not reconstructed from memory.

---

# Part II — time evolution (written 2026-10-10)

## The evolution engine (`evolve()` in `gravothermal_collapse.py`)

Each time step runs four stages:

1. implicit conduction;
2. an update of each shell's entropy from its new v² (the only place entropy changes);
3. relaxation to hydrostatic equilibrium with the unchanged, verified solver. Newton starts from
   a guess extrapolated linearly from the previous two steps;
4. an adaptive time step, dt × min(1.2, 0.02 / max|Δρ/ρ|), so no shell's density changes by more
   than about 2% per step.

A failed step is rejected and retried with dt halved. Below dt = 10⁻¹⁶ the run stops as a
breakdown. Total mass is conserved by construction, since each Lagrangian shell's mass is fixed.

## The cored halo: expansion in a short test (`run_evolution.py`)

A 50-step loop check on the cored halo reached t = 0.48. The central density rose slightly,
then fell to 0.943 of its starting value. The number of shell edges carrying heat inward grew
from 76 to 139, which is the cored profile's temperature inversion warming and expanding the
centre. There were 7 rejections, all at residuals near the round-off floor.

The cored halo was **not** run long. It is not claimed that it cannot collapse.

## The NFW halo (`build_nfw_halo()`, `check_nfw_initial.py`)

The evolution then used the standard cuspy halo of the SIDM literature (Navarro, Frenk & White
1996), with ρ_s = r_s = 1 and 200 shells from r = 0.01 to 30 r_s.

- **Construction.** Shell masses come from the closed-form enclosed mass and are exact to 10⁻¹⁶.
  The starting state is a fixed point of the relaxation solver: its residual is 7×10⁻¹⁵.
- **Accuracy.** v² is second-order accurate against the continuum solution: 1.9×10⁻⁴ at 400 shells.
- **Temperature.** v² is **inverted**. The centre is cold (0.138) and v² peaks at 1.185 at
  r = 0.756 r_s, with 108 of 199 edges carrying heat inward at t = 0.
- **Self-gravity.** |W|/3p_sV is 0.29 at R = 0.1 and 0.55 at R = 0.3, crossing 1 near r_s. The
  centre of the cusp is not self-gravitating.

## The pre-registered run (`run_collapse.py`, committed in `9c2b84c` before it ran)

**Criterion.** Fixed and committed before the run. Gravothermal collapse is demonstrated if and
only if all four hold:

1. the central density reaches 10 times its t = 0 value;
2. the gravity-to-pressure ratio of the shells initially inside r = 0.3 exceeds 1;
3. mass is conserved every step;
4. the solver converges on every accepted step.

Here the central density is the density of the innermost shell, ρ_c0 = 148.02. The run stops at
100 times ρ_c0, at 20,000 steps, at t = 5,000, or on breakdown.

**What happened** (record: `stage4_collapse_history.csv`, 639 steps, 7.4 s of wall time):

| Phase | Behaviour |
|---|---|
| Core formation, t = 0 → 18.5 | ρ_c **fell** to 0.0173 ρ_c0, a factor of 58, while the central v² rose from 0.138 to about 1.1. Heat flowed inward and the cusp became a hot core. The binding ratio fell from 0.55 to 0.25. |
| Slow phase, t ≈ 18 → 150 | ρ_c rose gradually and the half-density core began to lose mass. |
| Runaway, t ≈ 150 → 154.88 | ρ_c rose to **100.1 ρ_c0** (5,800 times its minimum) while the central v² rose to **2.52**. The core heated while contracting and losing mass. dt fell from 4.7 to 3.8×10⁻⁴, the signature of a runaway. The binding ratio reached **3.81**. |

All four criteria were met, with a mass error of at most 1.2×10⁻¹⁶ and no rejections. **Verdict:
gravothermal collapse demonstrated.** The verdict was held as provisional until the convergence
test below. This is negative heat capacity, the mechanism of `virial_theorem.py`, running away
inside a halo.

## The convergence test (`run_convergence.py`, committed in `9ffd969` before it ran)

**The concern.** The scheme loses a little energy (−1.43×10⁻³ of the total over the run). Lost
energy acts as an artificial heat sink, and a heat sink speeds up gravothermal collapse. The test
asked whether that leak drove the result.

**The variants.** A halved the per-step density limit (ξ = 0.01). B doubled the resolution to 400
shells. All quantities were computed identically for every run, and the pass conditions were
fixed in advance.

| Run | t10 (first t with ρ_c ≥ 10 ρ_c0) | d ln v²_c / d ln ρ_c | d ln ρ_c / d ln M_core | Energy drift |
|---|---|---|---|---|
| Baseline (ξ 0.02, 200 shells) | 154.70 | 0.09052 | −2.8090 | −1.431×10⁻³ |
| A (ξ 0.01, 200 shells) | 149.58 (−3.3%) | 0.08883 (−1.9%) | −2.7872 (−0.8%) | −7.04×10⁻⁴ |
| B (ξ 0.02, 400 shells) | 155.21 (+0.3%) | 0.09028 (−0.3%) | −2.8099 (+0.0%) | −1.406×10⁻³ |
| BSI self-similar | n/a | 0.0867 | −2.704 | n/a |

The exponents are least-squares slopes over every step with 2 ≤ ρ_c/ρ_c0 ≤ 100.

All four pass conditions were met:

- collapse occurs in both variants;
- the drift falls in A;
- the exponents stay within ±10%;
- t10 stays within 25%.

**Result: robust.**

- **The leak is a first-order step-size error.** Halving ξ halved the drift (a ratio of 2.03),
  while doubling the resolution left it unchanged.
- **The positive evidence: halving the leak made the collapse come earlier** (149.6 against
  154.7). If the artificial heat sink had been driving the collapse, reducing it would have
  delayed it.
- **200 shells resolve these quantities.** At 400 shells they change by 0.3% or less.
- **Extrapolation to zero step size** (linear in ξ, which the drift ratio supports; this is not
  part of the pre-registered test):

  | Quantity | Extrapolated value | BSI self-similar | Difference |
  |---|---|---|---|
  | d ln v²_c / d ln ρ_c | **0.0871** | 0.0867 | 0.5% |
  | d ln ρ_c / d ln M_core | **−2.765** | −2.704 | 2.3% |
  | t10 | about 144.5 | n/a | n/a |

  Neither exponent depends on the uncalibrated time unit.

## How this connects to the rest of the project

Simulation 3 derived negative heat capacity for an idealised sphere. Here the same mechanism was
first measured in a halo (Part I), then shown to drive a runaway: the core heated as it lost
energy, contracted and shed mass. The late-time scaling matches BSI's self-similar solution. That
runaway is the process the dissertation's black-hole-seed argument rests on.

This model follows it to 100 times the central density. It does not follow it to relativistic
instability or black hole formation.

## Limitations of the result

- **Long-mean-free-path conduction only.** BSI show that the late core enters the
  short-mean-free-path regime, which slows the final collapse. That physics is absent here.
- **The time unit is not calibrated.** `C_LMFP_PLACEHOLDER = 1.0`. A collapse time of t10 ≈ 150
  code units cannot be compared with BSI's ~290 relaxation times. A pre-run estimate put collapse
  at 10²–10³ code units.
- **The start is NFW, not BSI's self-similar profile.** What agrees with BSI is the late-time
  exponents, not the full history, and not their Figure 2.
- **The outer boundary was not varied:** r = 30 r_s, fixed outer pressure, insulating edge.
- **"Central density" is the mean density of the innermost shell,** and the core mass is counted
  in whole shells.
- **The cored-halo gate still stands as FAIL** at t = 0. It is a different halo and a different
  question.
- **No black hole forms in this model.** The run stops at 100 times the central density.

## Corrections on the record

Kept visible as part of the method.

- **Advisor:** the first brief gave the conductivity as ∝ v; the paper's text gives v³. Corrected
  by ruling.
- **Advisor:** explained the cored profile's off-centre temperature peak as imperfect
  isothermality; it is a genuine temperature inversion.
- **Advisor:** predicted that the NFW cusp would be hot, with no inversion. The check showed a
  strong inversion (centre 0.138, peak 1.185 at 0.756 r_s), as Claude had predicted.
- **Advisor:** approved a reformulation of the Newton solve in logarithmic variables. Claude
  showed it would be a no-op (column scaling leaves a Thomas solve unchanged, and the round-off
  floor comes from evaluating the residual); it was cancelled.
- **Claude:** predicted that pure conduction could only cool the core. Test E of
  `diagnose_stage3.py` showed it warming transiently; the claim was corrected to "cannot run
  away".
- **Claude:** a fixed step bound of 10⁻¹³ was said to sit "600×" above round-off. That rested on
  one case and was wrong; it was replaced by a computed per-row criterion.
- **Claude:** predicted that a cooled shell's inner edge would move outward. It moved inward.

## Reproducing

```
python gravothermal_collapse.py        # setup checks (no evolution)
python test_relaxation.py              # relaxation tests 1-2
python test_relaxation.py 3            # the t=0 gate (FAIL)
python scan_heat_capacity.py           # the heat-capacity scan
python diagnose_stage3.py              # the three defects of the earlier Stage 3
python diagnose_stage2_resolution.py   # Stage 2 accuracy against the continuum
python run_evolution.py                # 50-step loop check on the cored halo
python check_nfw_initial.py            # the NFW initial halo
python run_collapse.py                 # the pre-registered collapse run
python run_convergence.py A            # convergence variant A (xi = 0.01)
python run_convergence.py B            # convergence variant B (400 shells)
python run_convergence.py analyze      # evaluates the convergence test
```

The git history records each step, including the failures, and shows each pre-registration
committed before the run it governed.
