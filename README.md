# Dissertation to Simulation

**Mudrik Zubeir Ahmed** · [linkedin.com/in/mdrk](https://linkedin.com/in/mdrk)

Taking the mathematics of my undergraduate physics dissertation and carrying it through to working, verified simulation code.

In 2020 I wrote a BSc dissertation, *Galactic Black Holes*, which argued that supermassive black holes form from the gravothermal collapse of self-interacting dark matter halos, following Balberg, Shapiro & Inagaki (2002). The dissertation was analytical: derivations on paper, no code. This repository takes those same derivations and makes them run.

**What this is, stated plainly.** Every result here reproduces established physics. Nothing in this repository is a new discovery. What it offers is an independent implementation, real verification, and claims held to exactly what the numbers support.

## The four foundations

Each builds on the last, and each is verified against a known result.

**1. The Eddington limit** — `eddington_limit.py`
The maximum luminosity at which radiation pressure balances gravity, derived from first principles. For a 10⁸ solar-mass black hole the code gives 3.29 × 10¹² solar luminosities, matching the textbook value. Radiation against gravity.

**2. The Jeans instability** — `jeans_instability.py`, `jeans_animation.py`
When a gas cloud collapses under its own weight. The Jeans mass at 10 K and 10⁻¹⁸ kg/m³ comes out at 34.25 solar masses, typical of a cold molecular cloud at that density. The animation shows a cloud above the threshold collapsing and one below it holding, from the same model with only the mass changed. Pressure against gravity. *The time evolution is a simplified uniform-sphere model, not a hydrodynamic simulation.*

**3. The virial theorem and negative heat capacity** — `virial_theorem.py`
The strangest result in the set: a self-gravitating system gets **hotter** when it loses energy. The code applies the virial relation (2K + U = 0, so E = −K) and then shows the consequence, a sphere radiating energy while contracting and heating. Gravity against its own thermodynamics. This is the engine of the capstone.

**4. Capstone: gravothermal collapse of a dark matter halo** — `gravothermal_collapse.py` and associated scripts
The dissertation's own conclusion, made to run. Full account in `CAPSTONE_FINDINGS.md`.

## The capstone, in short

A Lagrangian gravothermal fluid solver built from scratch: implicit conduction, entropy-conserving relaxation, an analytical Jacobian, and a convergence criterion derived from the machine's own precision rather than chosen by hand.

Starting from an NFW halo, the model produces the sequence the SIDM literature describes: the cusp heats and expands into a core, the core slowly contracts, and then it collapses, the central density rising past 100× its initial value while the core heats and sheds mass.

**The independent check.** The late-collapse exponents, extrapolated to zero step size, are 0.0871 and −2.765, against Balberg, Shapiro & Inagaki's self-similar values of 0.0867 and −2.704. Agreement to within a few per cent, from a different initial profile, with no tuning.

**Method note.** The capstone's collapse criterion and its convergence test were each committed to version control before the run that tested them; earlier tests had their criteria fixed in the script before running. The git history shows the pre-registration commits preceding their runs. In the convergence test, halving the numerical energy leak made the collapse occur *earlier*, which rules out the leak having driven it.

**What it does not show.** Long-mean-free-path conduction only. The time unit is uncalibrated. The starting profile is NFW, not the paper's self-similar one, so the exponents match but the history does not. No black hole forms: the run stops at 100× central density. A separate cored halo failed an earlier mechanism test at t = 0, and that failure is recorded, not hidden.

## Running it

```
pip install numpy scipy matplotlib
python eddington_limit.py
python jeans_instability.py
python jeans_animation.py
python virial_theorem.py
python run_collapse.py          # the pre-registered capstone run
python run_convergence.py analyze
```

## On the record

`CAPSTONE_FINDINGS.md` records the full sequence, including the parts that failed: a mechanism gate that failed at t = 0 and was left standing, a halo that expanded instead of collapsing, and seven errors made during the work, each corrected by measurement or by checking the source. The failures are kept deliberately. They are the evidence that the successes were tested.

## Sources

- Balberg, Shapiro & Inagaki (2002), *Self-Interacting Dark Matter Halos and the Gravothermal Catastrophe*, arXiv:astro-ph/0110561
- Lynden-Bell & Wood (1968), on the gravothermal instability of self-gravitating systems
- Navarro, Frenk & White (1996), the NFW halo profile
- Meng et al. (2026), arXiv:2607.02151, for the implicit (backward-Euler) treatment of the conduction term. The mass-coordinate formulation used here is my own.
