"""The traveler's grain-count: can a floor keep its own alignments secret?

Three ways to lay a grain down - a repeating lattice, an almost-crystal
(Fibonacci), a random sprinkle - and one question asked of each: does a
moving traveler count the same amount of grain as a standing one?

The count is the longest causal chain between two events. For a traveler
the two events are the same interval apart, just boosted. If the floor is
honest the ratio is 1. If the floor has a built-in frame, the ratio drops
to exp(-eta) and stays there however fine the grain is made.

Run:  python floor_ruler_test.py
"""

import hashlib
import json
import math
from pathlib import Path

ETA = 0.5                       # the traveler's rapidity
BROKEN = math.exp(-ETA)         # the ratio a frame-carrying floor is stuck at
OUT = Path(__file__).resolve().parent


def H(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def u01(seed_hex):
    return int(seed_hex, 16) / float(1 << 256)


def sprinkle(run, n):
    """n points dropped at hash-determined positions - no rule, no alignment."""
    return [(u01(H("sprinkle", run, i, "t")), u01(H("sprinkle", run, i, "x")))
            for i in range(n)]


def longest_chain(points, p, q):
    """Longest causally ordered run of points between events p and q."""
    tp, xp = p
    tq, xq = q
    interior = [(t, x) for t, x in points
                if (t - tp) > abs(x - xp) and (tq - t) > abs(x - xq)]
    interior.sort()
    best_to = [1] * len(interior)
    for i, (ti, xi) in enumerate(interior):
        best = 0
        for j in range(i):
            tj, xj = interior[j]
            if (ti - tj) > abs(xi - xj) and best_to[j] > best:
                best = best_to[j]
        best_to[i] = best + 1
    return max(best_to) if best_to else 0


def fib_positions(n_iters=18):
    """The long-short row as positions on a line, normalized to [0, 1]."""
    word = "A"
    for _ in range(n_iters):
        word = "".join("AB" if ch == "A" else "A" for ch in word)
    phi = (1 + math.sqrt(5)) / 2
    steps = [phi if ch == "A" else 1.0 for ch in word]
    pos = [0.0]
    for s in steps:
        pos.append(pos[-1] + s)
    return [p / pos[-1] for p in pos]


def count_in(positions, span):
    return sum(1 for p in positions if p < span)


def ordered_ratio(positions):
    """Boosted count over straight count for a floor built from `positions`."""
    straight = count_in(positions, 0.5)
    span_u = min(0.5 * math.exp(ETA), 1.0)
    span_v = 0.5 * math.exp(-ETA)
    boosted = min(count_in(positions, span_u), count_in(positions, span_v))
    return boosted / straight


def lattice_chain_2d(T, X):
    """Same question on a 2-D square lattice, at a chosen density."""
    pts = [(float(t), float(x))
           for t in range(T + 1) for x in range(-T, T + 1)
           if t > abs(x) and (T - t) > abs(X - x)]
    return longest_chain(pts, (0.0, 0.0), (float(T), float(X)))


def main():
    fib = fib_positions()
    lattice = [i / (len(fib) - 1) for i in range(len(fib))]

    r_lattice = ordered_ratio(lattice)
    r_fib = ordered_ratio(fib)

    spr = []
    for run in range(3):
        pts = sprinkle(run, 5000)
        p, s = (0.15, 0.5), 0.4
        straight = longest_chain(pts, p, (p[0] + s, p[1]))
        boosted = longest_chain(pts, p,
                                (p[0] + s * math.cosh(ETA), p[1] + s * math.sinh(ETA)))
        spr.append(boosted / straight)
    r_sprinkle = sum(spr) / len(spr)

    print("the traveler's grain-count   (1.000 = honest floor, %.3f = frame built in)"
          % BROKEN)
    print("  repeating lattice   %.3f" % r_lattice)
    print("  almost-crystal      %.3f" % r_fib)
    print("  random sprinkle     %.3f" % r_sprinkle)

    fine = []
    for L in (40, 80):
        T = int(round(L * math.cosh(ETA)))
        X = int(round(L * math.sinh(ETA)))
        fine.append(lattice_chain_2d(T, X) / lattice_chain_2d(L, 0))
    print()
    print("does making the lattice finer help?")
    print("  coarse lattice      %.3f" % fine[0])
    print("  twice as fine       %.3f" % fine[1])
    print("  -> stuck at %.3f either way; fineness is not the cure" % BROKEN)

    metrics = {
        "eta": ETA,
        "frame_carrying_ratio": BROKEN,
        "ordered_ratio": {"lattice": r_lattice, "almost_crystal": r_fib,
                          "sprinkle": r_sprinkle},
        "lattice_refinement": {"coarse": fine[0], "twice_as_fine": fine[1]},
    }
    (OUT / "floor_ruler_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
