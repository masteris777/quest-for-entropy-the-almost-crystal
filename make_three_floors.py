"""Three floors + their fingerprints.

Top: the same grain laid three ways - lattice, almost-crystal (Ammann-Beenker
cut-and-project), random sprinkle (hash-seeded). Bottom: each one's diffraction
fingerprint S(q) - every sharp spot is a built-in direction/spacing.
Computed, not drawn."""

import hashlib
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_PTS = 900


def hash_points(seed, n):
    pts, h = [], hashlib.sha256(seed.encode()).hexdigest()
    while len(pts) < n:
        h = hashlib.sha256(h.encode()).hexdigest()
        pts.append((int(h[:8], 16) / 16 ** 8, int(h[8:16], 16) / 16 ** 8))
    return np.array(pts)


def lattice_points(n):
    m = int(round(math.sqrt(n)))
    g = (np.arange(m) + 0.5) / m
    X, Y = np.meshgrid(g, g)
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def ammann_beenker():
    N = 12
    rng = np.arange(-N, N + 1)
    grid = np.stack(np.meshgrid(rng, rng, rng, rng, indexing="ij"), axis=-1).reshape(-1, 4)
    ks = np.arange(4)
    par = np.stack([np.cos(np.pi * ks / 4), np.sin(np.pi * ks / 4)], axis=1)
    per = np.stack([np.cos(3 * np.pi * ks / 4), np.sin(3 * np.pi * ks / 4)], axis=1)
    n = grid.astype(float) - np.array([0.1237, 0.2141, 0.0533, 0.1719])
    p = (n @ par)[np.linalg.norm(n @ per, axis=1) < 1.15]
    L = 0.55 * np.linalg.norm(p, axis=1).max()
    p = p[(np.abs(p[:, 0]) < L) & (np.abs(p[:, 1]) < L)]
    return (p / (2 * L)) + 0.5


def structure_factor(pts, qmax=320.0, nq=241):
    qs = np.linspace(-qmax, qmax, nq)
    S = np.zeros((nq, nq))
    x, y = pts[:, 0], pts[:, 1]
    for i, qy in enumerate(qs):
        ph = np.exp(1j * (qs[:, None] * x[None, :] + qy * y[None, :]))
        S[i] = np.abs(ph.sum(axis=1)) ** 2
    return S / len(pts)


def main():
    sets = [
        ("the crystal (lattice)", lattice_points(N_PTS)),
        ("the almost-crystal", ammann_beenker()),
        ("the sprinkle (random)", hash_points("genesis-42", N_PTS)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 9.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    for col, (name, pts) in enumerate(sets):
        ax = axes[0, col]
        ax.scatter(pts[:, 0], pts[:, 1], s=6, c="#1a3a6b", linewidths=0)
        ax.set_title(name, fontsize=13)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        S = structure_factor(pts)
        c = len(S) // 2
        S[c - 2:c + 3, c - 2:c + 3] = np.median(S)
        ax2 = axes[1, col]
        ax2.imshow(np.log1p(S), cmap="magma", origin="lower")
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel("its fingerprint", fontsize=11)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "figures", "three_floors.png")
    fig.savefig(os.path.normpath(out), dpi=150, facecolor="white")
    print("saved:", os.path.normpath(out))


if __name__ == "__main__":
    main()
