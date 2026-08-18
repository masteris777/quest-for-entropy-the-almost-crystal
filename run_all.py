"""Reproduce every claim the article makes, from scratch.

    python run_all.py            everything (a few minutes)
    python run_all.py --quick    skip the two slow surrogate attacks

Each check below names the sentence in the article it is testing. If a check
fails, the article is wrong and I want to know.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = []


def run(script):
    print(f"\n$ python {script}")
    proc = subprocess.run([sys.executable, str(HERE / script)],
                          cwd=HERE, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise SystemExit(f"{script} failed")
    print(proc.stdout.strip()[-700:])
    return proc.stdout


def metrics(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def check(claim, ok, detail):
    RESULTS.append((claim, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {claim}: {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the two slow surrogate attacks")
    args = ap.parse_args()

    # ---- the first audition: can a listener inside hear the chain? -------
    run("bell_readout.py")
    m = metrics("metrics_bell_readout.json")["readout"]
    check("one listener inside the chain gets its tune (article: a couple of percent)",
          m["mean_error"] < 0.03 and m["worst_error"] < 0.035,
          f"mean {100 * m['mean_error']:.1f}%, worst {100 * m['worst_error']:.1f}%")
    check("and it never looks at the whole chain",
          m["local_time_series_used"] and not m["global_projection_used_as_detector"],
          "local time-series only, no global projection")

    # ---- the detector cannot tell the classes apart ----------------------
    run("detector_transfer.py")
    d = metrics("metrics_detector_transfer.json")
    cross = d["cross_class_confusion"]["mean_confusion"]
    check("tuned on the ordered chain, it still matches a chaotic one "
          "(article: my detectors treat the two as identical)",
          cross > 0.90, f"cross-class match {cross:.3f} where 1 is perfect")
    every = all(f["pct_gt_0_8"] == 100.0 for f in d["families"].values())
    check("every sample of every family clears the bar (article: it passed everything)",
          every, ", ".join(f"{k} {v['pct_gt_0_8']:.0f}%" for k, v in d["families"].items()))

    # ---- the two fakes, each built to break one probe --------------------
    if args.quick:
        print("\n(skipping the two surrogate attacks: --quick)")
    else:
        run("tone_matched_fake.py")
        t = metrics("metrics_tone_matched_fake.json")["conditions"]["G1_T5"]
        honest = t["separation_sturmian_iid_random"]
        fake = t["separation_sturmian_phase_surrogate_mean"]
        check("a fake with the same tune collapses the first probe "
              "(article: the same tune, no deep order)",
              honest > 0.05 and fake < 0.001,
              f"gap against noise {honest:.4f}, against the fake {fake:.6f} "
              f"({honest / fake:.0f}x smaller)")
        check("the fake really does carry the same tune",
              t["phase_surrogate_spectrum_error_mean"] == 0.0,
              f"tone error against the real one: {t['phase_surrogate_spectrum_error_mean']:.4f}")

        run("return_time_fake.py")
        r = metrics("metrics_return_time_fake.json")
        worst = min(c["markov_order_4_separation_min"]
                    for c in r["conditions"].values())
        check("a fake with the same short runs collapses the second probe "
              "(article: twice, with two different detectors)",
              worst < 1e-6, f"worst gap left for it to find: {worst:.7f}")

    # ---- the second audition: can a floor keep a secret? -----------------
    run("floor_ruler_test.py")
    f = metrics("floor_ruler_metrics.json")
    broken = f["frame_carrying_ratio"]
    ratio = f["ordered_ratio"]
    check("a repeating lattice hands a moving traveler a frame "
          "(article: the lattice cheats by a fixed amount)",
          abs(ratio["lattice"] - broken) < 0.02,
          f"{ratio['lattice']:.3f} against the frame-carrying value {broken:.3f}")
    check("the almost-crystal hands over the same frame "
          "(article: the same amount, digit for digit)",
          round(ratio["almost_crystal"], 3) == round(ratio["lattice"], 3),
          f"almost-crystal {ratio['almost_crystal']:.3f} vs lattice {ratio['lattice']:.3f}")
    check("the random sprinkle deals fair (article: it hides)",
          ratio["sprinkle"] > 0.90, f"{ratio['sprinkle']:.3f}, boost-invariant")
    fine = f["lattice_refinement"]
    check("making the grain finer does not rescue the lattice",
          all(abs(v - broken) < 0.05 for v in fine.values()),
          f"coarse {fine['coarse']:.3f}, twice as fine {fine['twice_as_fine']:.3f} "
          f"- both stuck at {broken:.3f}")

    # ---- the fingerprint gallery -----------------------------------------
    run("make_three_floors.py")
    png = HERE / "figures" / "three_floors.png"
    check("the three fingerprints are computed, not drawn",
          png.exists() and png.stat().st_size > 50_000,
          f"figures/three_floors.png regenerated ({png.stat().st_size // 1024} KB)")

    print("\n" + "=" * 66)
    for claim, ok, _ in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {claim}")
    print("=" * 66)
    bad = [c for c, ok, _ in RESULTS if not ok]
    if bad:
        raise SystemExit(f"{len(bad)} check(s) FAILED")
    print(f"all {len(RESULTS)} checks reproduced")


if __name__ == "__main__":
    main()
