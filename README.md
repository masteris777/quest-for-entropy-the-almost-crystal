# The Almost-Crystal — companion code

Everything the article claims, runnable from scratch.

## Run it

```
pip install numpy scipy matplotlib
python run_all.py
```

`run_all.py` runs every experiment and checks each claim the article makes —
eleven checks. It reports PASS or FAIL per claim, and names the sentence each
one is testing. The two surrogate attacks are the slow part;
`python run_all.py --quick` skips them.

## What is in here

The article puts the almost-crystal through two auditions. The code follows the
same split.

**The first audition — could it be the clockwork?**

| file | what it does |
|---|---|
| `bell_readout.py` | one listener inside a mass-spring chain: hear a jumble, recover the tune, predict the rest |
| `detector_transfer.py` | tune the detector on one chain, then score the others — including a chaotic one |
| `tone_matched_fake.py` | builds a fake with the same tune and no deep order, and watches the first probe collapse |
| `return_time_fake.py` | builds a fake with the same short runs, and watches the second probe collapse |
| `chain_builders.py` | the substitution rule and the decoy layouts |
| `resonator_bank.py` | the listener itself |
| `dynamical_matrix.py` | turns a spring layout into the thing that rings |

**The second audition — could it be the floor?**

| file | what it does |
|---|---|
| `make_three_floors.py` | lays the same grain down three ways and computes each one's diffraction fingerprint |
| `floor_ruler_test.py` | the traveler's grain-count: does a moving observer count the same grain as a standing one? |

`article.md` is the piece itself, with its figures in `assets/`. Figures the
scripts generate land in `figures/`.

## Scope

The article's "What this does NOT claim" section is the scope fence, and it
ships with the code in `article.md`. Short version: these are small home-built
engines and one-dimensional simulations, not a verdict on real quasicrystals or
on anyone's research programme.

## Licence

MIT for the code. The article text is © Marijus Masteika.
