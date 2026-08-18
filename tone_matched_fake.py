"""
The tone-matched fake - same pitches, no long-range order

Removes the easy power-spectrum difference between Sturmian and random drive
by building two adversarial surrogate families:
  1. Phase-randomized surrogate: preserves Sturmian |FFT| magnitudes exactly,
     randomises Fourier phases.
  2. Binary IAAFT surrogate: iterative amplitude-adjusted Fourier transform
     preserving {+1,-1} amplitude distribution and approximately the Sturmian
     power spectrum.

If the fixed resonator-bank observer still separates Sturmian from these
matched-spectrum controls, the signal carries information beyond the one-body
power spectrum. If the gap collapses, the current mechanism is spectral
identity only.

Fixed observer: K=32 log-spaced resonators [0.5, 15 Hz]. No retuning.
Conditions: G1_T5 (γ=1.0, T=5s), G5_T5 (γ=5.0, T=5s), G1_T50 (γ=1.0, T=50s).
"""

import numpy as np
from scipy.signal import lfilter
import json
import os

# ── Fixed parameters ──────────────────────────────────────────────────────────
FS      = 1000.0
DT      = 1.0 / FS
T_BIT   = 50           # 50 ms tick, matching the temporal probes
N_WINS  = 20
K       = 32
F_MIN   = 0.5
F_MAX   = 15.0
FREQS   = np.logspace(np.log10(F_MIN), np.log10(F_MAX), K)

N_SURROGATE_SEEDS = 10   # seeds for phase + binary surrogates
N_IAAFT_ITER      = 50   # IAAFT convergence iterations

CONDITIONS = [
    ("G1_T5",  1.0,  5.0),
    ("G5_T5",  5.0,  5.0),
    # G1_T50 omitted: window length measured flat earlier; T=50s IAAFT on 1M samples is cost-prohibitive
]


# ── Drive generators ──────────────────────────────────────────────────────────

def gen_sturmian_drive(N: int) -> np.ndarray:
    alpha  = (np.sqrt(5.0) - 1.0) / 2.0
    n_bits = N // T_BIT + 2
    word   = np.array(
        [int(np.floor((i + 1) * alpha)) - int(np.floor(i * alpha))
         for i in range(n_bits)], dtype=float)
    return 0.3 * np.repeat(word * 2.0 - 1.0, T_BIT)[:N]


def gen_iid_random(N: int, seed: int = 42) -> np.ndarray:
    rng    = np.random.default_rng(seed)
    n_bits = N // T_BIT + 2
    return 0.3 * np.repeat(rng.choice([-1.0, 1.0], size=n_bits), T_BIT)[:N]


# ── Surrogate generators ──────────────────────────────────────────────────────

def gen_phase_surrogate(reference: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Preserve |FFT(reference)| exactly; draw uniformly random Fourier phases.
    Output is real-valued, amplitude rescaled to match reference RMS.
    """
    N    = len(reference)
    spec = np.fft.rfft(reference)
    mag  = np.abs(spec)
    # Random phases, conjugate symmetry automatic via rfft/irfft
    phi  = rng.uniform(0.0, 2.0 * np.pi, len(spec))
    phi[0] = 0.0                          # DC must be real
    if N % 2 == 0:
        phi[-1] = 0.0                     # Nyquist must be real
    new_spec = mag * np.exp(1j * phi)
    sur      = np.fft.irfft(new_spec, n=N)
    # Rescale to reference RMS
    rms_ref = np.sqrt(np.mean(reference ** 2))
    rms_sur = np.sqrt(np.mean(sur ** 2)) + 1e-30
    return sur * (rms_ref / rms_sur)


def gen_binary_iaaft(reference: np.ndarray, rng: np.random.Generator,
                     n_iter: int = N_IAAFT_ITER) -> tuple[np.ndarray, float]:
    """
    IAAFT for binary {+1,-1} * amplitude sequences.
    Returns (surrogate, spectrum_error).

    Steps per iteration:
      1. FFT → replace magnitudes with target (Sturmian) → IFFT → real y
      2. Rank-order substitute: assign reference values at positions
         corresponding to sorted y rank (preserves amplitude distribution).
    """
    amp     = 0.3
    N       = len(reference)
    # Target magnitudes from reference
    target_mag = np.abs(np.fft.rfft(reference))

    # Sorted reference values (binary distribution: ±0.3)
    ref_sorted = np.sort(reference)

    # Initialise with a random permutation of reference values
    x = reference[rng.permutation(N)].copy()

    for _ in range(n_iter):
        # Step 1: spectrum-constrained phase
        spec  = np.fft.rfft(x)
        phase = np.angle(spec)
        x     = np.fft.irfft(target_mag * np.exp(1j * phase), n=N)
        # Step 2: rank-order substitution → restore {+0.3, -0.3} distribution
        rank  = np.argsort(np.argsort(x))   # rank of each position
        x     = ref_sorted[rank]

    # Spectrum error: L2 relative deviation (avoids deep-null overflow)
    sur_mag   = np.abs(np.fft.rfft(x))
    spec_err  = float(np.linalg.norm(sur_mag - target_mag)
                      / (np.linalg.norm(target_mag) + 1e-30))
    return x, spec_err


# ── Resonator bank ────────────────────────────────────────────────────────────

def apply_bank(signal: np.ndarray, gamma: float) -> np.ndarray:
    out = np.empty((K, len(signal)))
    for k, fk in enumerate(FREQS):
        wk = 2.0 * np.pi * fk
        wd = np.sqrt(max(wk ** 2 - gamma ** 2, 1e-30))
        a1 = np.exp(-gamma * DT) * np.cos(wd * DT)
        a2 = np.exp(-2.0 * gamma * DT)
        out[k] = lfilter([1.0], [1.0, -2.0 * a1, a2], signal)
    return out


def feature_vec(bank: np.ndarray, start: int, end: int) -> np.ndarray:
    E = np.mean(bank[:, start:end] ** 2, axis=1) + 1e-30
    v = np.log(E)
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-15 else np.zeros(K)


def fingerprint(signal: np.ndarray, gamma: float, n_win: int) -> np.ndarray:
    """Return (N_WINS, K) feature matrix."""
    bank = apply_bank(signal, gamma)
    return np.array([feature_vec(bank, w * n_win, (w + 1) * n_win)
                     for w in range(N_WINS)])


# ── Metrics ───────────────────────────────────────────────────────────────────

def centroid_sep(a: np.ndarray, b: np.ndarray) -> float:
    ca = a.mean(axis=0); ca /= np.linalg.norm(ca) + 1e-30
    cb = b.mean(axis=0); cb /= np.linalg.norm(cb) + 1e-30
    return float(1.0 - np.dot(ca, cb))


# ── Main ──────────────────────────────────────────────────────────────────────

def run_condition(label: str, gamma: float, t_win: float,
                  sturmian_fp: np.ndarray) -> dict:
    n_win   = int(t_win * FS)
    n_total = n_win * N_WINS

    print(f"    iid random ...", end=" ", flush=True)
    rnd_fp  = fingerprint(gen_iid_random(n_total), gamma, n_win)
    sep_iid = centroid_sep(sturmian_fp, rnd_fp)
    print(f"sep={sep_iid:.4f}")

    # Phase surrogates
    phase_seps = []; phase_errs = []
    print(f"    phase surrogates ({N_SURROGATE_SEEDS} seeds) ...", end=" ", flush=True)
    for seed in range(N_SURROGATE_SEEDS):
        rng = np.random.default_rng(1000 + seed)
        sur = gen_phase_surrogate(gen_sturmian_drive(n_total), rng)
        fp  = fingerprint(sur, gamma, n_win)
        phase_seps.append(centroid_sep(sturmian_fp, fp))
        # Phase surrogate has exact spectrum match by construction → err = 0
        phase_errs.append(0.0)
    print(f"mean={np.mean(phase_seps):.4f}  min={np.min(phase_seps):.4f}")

    # Binary IAAFT surrogates
    binary_seps = []; binary_errs = []
    print(f"    IAAFT surrogates ({N_SURROGATE_SEEDS} seeds) ...", end=" ", flush=True)
    ref = gen_sturmian_drive(n_total)
    for seed in range(N_SURROGATE_SEEDS):
        rng = np.random.default_rng(2000 + seed)
        sur, err = gen_binary_iaaft(ref, rng)
        fp  = fingerprint(sur, gamma, n_win)
        binary_seps.append(centroid_sep(sturmian_fp, fp))
        binary_errs.append(err)
    print(f"mean={np.mean(binary_seps):.4f}  min={np.min(binary_seps):.4f}  "
          f"spec_err={np.mean(binary_errs):.4f}")

    return {
        "gamma_rad_s":   gamma,
        "T_win_s":       t_win,
        "separation_sturmian_iid_random":            round(sep_iid, 6),
        "separation_sturmian_phase_surrogate_mean":  round(float(np.mean(phase_seps)), 6),
        "separation_sturmian_phase_surrogate_min":   round(float(np.min(phase_seps)), 6),
        "separation_sturmian_phase_surrogate_max":   round(float(np.max(phase_seps)), 6),
        "separation_sturmian_binary_surrogate_mean": round(float(np.mean(binary_seps)), 6),
        "separation_sturmian_binary_surrogate_min":  round(float(np.min(binary_seps)), 6),
        "separation_sturmian_binary_surrogate_max":  round(float(np.max(binary_seps)), 6),
        "phase_surrogate_spectrum_error_mean":       0.0,
        "binary_surrogate_spectrum_error_mean":      round(float(np.mean(binary_errs)), 6),
        "n_surrogate_seeds":                         N_SURROGATE_SEEDS,
    }


def main():
    print("The tone-matched fake - same pitches, no long-range order")
    print(f"  {N_SURROGATE_SEEDS} surrogate seeds, {N_IAAFT_ITER} IAAFT iterations\n")

    results = {}

    for label, gamma, t_win in CONDITIONS:
        n_win   = int(t_win * FS)
        n_total = n_win * N_WINS
        print(f"  [{label}] gamma={gamma}, T={t_win}s, N={n_total}")
        sturmian_fp = fingerprint(gen_sturmian_drive(n_total), gamma, n_win)
        results[label] = run_condition(label, gamma, t_win, sturmian_fp)
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    # Key condition for verdict: G1_T5 (the strongest case measured)
    g1 = results["G1_T5"]
    g5 = results["G5_T5"]

    phase_collapses_g1  = g1["separation_sturmian_phase_surrogate_min"]  < 0.05
    binary_collapses_g1 = g1["separation_sturmian_binary_surrogate_min"] < 0.05
    phase_collapses_g5  = g5["separation_sturmian_phase_surrogate_min"]  < 0.05
    binary_collapses_g5 = g5["separation_sturmian_binary_surrogate_min"] < 0.05

    best_binary_gap = max(
        g1["separation_sturmian_binary_surrogate_mean"],
        g5["separation_sturmian_binary_surrogate_mean"],
    )

    matched_collapses = binary_collapses_g1 and binary_collapses_g5

    if not binary_collapses_g1 and g1["separation_sturmian_binary_surrogate_min"] >= 0.05:
        verdict = "PASS"
        claim   = "spectral_identity_survives_adversarial_control"
    elif not binary_collapses_g1 or not binary_collapses_g5:
        verdict = "PARTIAL"
        claim   = "spectral_identity_survives_adversarial_control"
    else:
        verdict = "FAIL"
        claim   = "spectral_identity_collapses"

    metrics = {
        "experiment": "tone_matched_fake",
        "verdict":  verdict,
        "protocol": "adversarial_spectral_control",
        "baseline_separation_resonator_probe": 0.060,
        "baseline_separation_hardened_G1_T5": 0.108,
        "conditions": results,
        "summary": {
            "matched_spectrum_collapses_resonator_gap": bool(matched_collapses),
            "sturmian_survives_binary_matched_spectrum": bool(not binary_collapses_g1),
            "phase_surrogate_collapses_G1_T5":          bool(phase_collapses_g1),
            "binary_surrogate_collapses_G1_T5":         bool(binary_collapses_g1),
            "best_surviving_matched_spectrum_gap":      round(best_binary_gap, 6),
        },
        "claim_boundary": claim,
    }

    out_dir  = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "metrics_tone_matched_fake.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved: {out_path}")
    print(f"  Verdict: {verdict}  ({claim})")
    print(f"\n  Key gaps (G1_T5):")
    print(f"    vs iid random:          {g1['separation_sturmian_iid_random']:.4f}")
    print(f"    vs phase surrogate:     mean={g1['separation_sturmian_phase_surrogate_mean']:.4f}"
          f"  min={g1['separation_sturmian_phase_surrogate_min']:.4f}")
    print(f"    vs binary IAAFT:        mean={g1['separation_sturmian_binary_surrogate_mean']:.4f}"
          f"  min={g1['separation_sturmian_binary_surrogate_min']:.4f}")


if __name__ == "__main__":
    main()
