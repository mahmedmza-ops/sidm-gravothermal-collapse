# Capstone: gravothermal physics of a self-interacting dark matter halo — findings

This is the fourth and last simulation in a project that takes the mathematics of my BSc
dissertation, *Galactic Black Holes*, to working code. The dissertation followed Balberg,
Shapiro & Inagaki (2002, "BSI") in arguing that supermassive black hole seeds can form through
the gravothermal collapse of self-interacting dark matter (SIDM) halos. The engine of that
collapse is negative heat capacity: a self-gravitating system that loses energy contracts and
gets hotter. Simulation 3 (`virial_theorem.py`) demonstrates that for a uniform sphere, where
the virial theorem gives E = −K. This capstone asks whether the same mechanism is present, and
measurable, in a real halo.

What follows is exactly what was built and found. It is not a reproduction of BSI's Figure 2,
and no collapse was run.

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
  redistributed. This step is built and tested, but it was not used in any time evolution.
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

## The finding: where negative heat capacity lives (`scan_heat_capacity.py`)

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

## How this connects to the rest of the project

Simulation 3 showed negative heat capacity for an idealised uniform sphere with no surroundings.
This capstone measures the same mechanism inside an SIDM halo with a realistic density profile.
It shows the mechanism is present on self-gravitating scales and absent in the small,
pressure-confined core. That is the mechanism the dissertation's black-hole-seed argument
depends on: gravothermal collapse needs a region that heats as it loses energy.

This work establishes that such regions exist in this halo. It does **not** show the collapse
itself, nor the formation of a black hole.

## Limitations

- **No time evolution was run.** The full conduction-plus-relaxation evolution to core collapse
  was not attempted and remains future work. BSI themselves could integrate only until the
  central v² had grown by a few tens.
- **The crossover is bracketed coarsely,** by a single grid interval (R = 1 to 3). Every heat
  capacity uses a single 1% perturbation, and none has been tested at other resolutions.
- **The test is quasi-static.** Relaxation assumes each shell keeps its entropy (adiabatic,
  hydrostatic). Energy released during relaxation is second-order: 0.3% of the heat removed in
  the gate test.
- **The starting profile is a cored halo, not BSI's self-similar profile.** BSI's equations for
  that profile could not be read reliably from the available copy of the paper, so they were not
  reconstructed from memory. The cored profile has a genuine temperature inversion: the centre is
  cooler than r ≈ 6. An evolution from it would begin with heat flowing inward.
- **The time unit is not calibrated.** The conductivity's overall coefficient is a named
  placeholder (`C_LMFP_PLACEHOLDER = 1.0`). In this limit it sets only the time unit, and it would
  have to be taken from the paper before any comparison with BSI's collapse time (~290 relaxation
  times).
- **Boundary assumptions.** The outer pressure is held fixed and the outer edge is insulating.
  Both are stated approximations for an isolated halo.

## Reproducing

```
python gravothermal_collapse.py        # setup checks (no evolution)
python test_relaxation.py              # tests 1-2
python test_relaxation.py 3            # the gate (FAIL)
python scan_heat_capacity.py           # the heat-capacity scan
python diagnose_stage3.py              # the three defects of the earlier Stage 3
python diagnose_stage2_resolution.py   # Stage 2 accuracy against the continuum
```

The git history records each step, including the failures.
