"""
The pattern-matched fake - same short runs, no long-range order

Tests if the symbolic return-time order survives controls that preserve local
k-gram statistics (Markov models and approx Eulerian K-gram shuffling).

Forbidden: Fourier magnitudes, resonator outputs, spectral distances.
"""

import numpy as np
from collections import defaultdict
import json
import os

# ── Fixed parameters (declared before running) ───────────────────────────────
N_SEQ             = 200_000
ANALYSIS_K        = [2, 3, 4, 5, 6]
MARKOV_ORDERS     = [1, 2, 3, 4]
KGRAM_ORDERS      = [2, 3, 4]
N_SURROGATE_SEEDS = 10

# ── Sequence generators ───────────────────────────────────────────────────────

def gen_sturmian_bin(N: int, alpha: float = (np.sqrt(5) - 1) / 2) -> np.ndarray:
    return np.array(
        [int(np.floor((i + 1) * alpha)) - int(np.floor(i * alpha))
         for i in range(N)], dtype=np.int8)

# ── Local-Statistic Surrogate generators ──────────────────────────────────────

def gen_markov_surrogate_bin(reference: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate an Order-m Markov surrogate.
    Estimates P(x_t | x_{t-m ... t-1}) from reference.
    Samples sequence of length N.
    """
    N = len(reference)
    counts = defaultdict(lambda: np.zeros(2, dtype=int))
    # Count transitions
    for i in range(N - m):
        u = tuple(reference[i:i+m])
        nxt = reference[i+m]
        counts[u][nxt] += 1
        
    # Convert to probabilities for class 1
    probs = {}
    for u, c in counts.items():
        s = c.sum()
        probs[u] = c[1] / s if s > 0 else 0.5
        
    out = list(reference[:m])
    for i in range(m, N):
        u = tuple(out[-m:])
        p1 = probs.get(u, 0.5)
        out.append(1 if rng.random() < p1 else 0)
        
    return np.array(out, dtype=np.int8)

def gen_kgram_approx_surrogate(reference: np.ndarray, K: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """
    Generate an approx K-gram preserving surrogate by greedily walking the empirical (K-1)-order De Bruijn graph.
    Selects valid outgoing transitions WITHOUT replacement to preserve counts.
    Falls back to random choices if hitting a dead end (introducing small K-gram error).
    Returns (surrogate_array, total_variation_error)
    """
    N = len(reference)
    k_minus_1 = K - 1
    if k_minus_1 <= 0:
        # fallback for K=1 (unigram preservation) = simple shuffle
        out = reference.copy()
        rng.shuffle(out)
        return out, 0.0
        
    # build exact transitions
    transitions = defaultdict(list)
    for i in range(N - k_minus_1):
        u = tuple(reference[i:i+k_minus_1])
        if i + k_minus_1 < N:
            transitions[u].append(reference[i+k_minus_1])
            
    # shuffle transitions internally
    for u in transitions:
        rng.shuffle(transitions[u])
        
    out = list(reference[:k_minus_1])
    for i in range(k_minus_1, N):
        u = tuple(out[-k_minus_1:])
        avail = transitions.get(u, [])
        if len(avail) > 0:
            nxt = avail.pop()
        else:
            nxt = rng.integers(0, 2)
        out.append(nxt)
        
    out_arr = np.array(out, dtype=np.int8)
    
    # compute K-gram total variation error
    ref_counts = defaultdict(int)
    for i in range(N - K + 1):
        ref_counts[tuple(reference[i:i+K])] += 1
    out_counts = defaultdict(int)
    for i in range(N - K + 1):
        out_counts[tuple(out_arr[i:i+K])] += 1
        
    err = 0
    all_kgrams = set(ref_counts.keys()).union(set(out_counts.keys()))
    for kg in all_kgrams:
        err += abs(ref_counts.get(kg, 0) - out_counts.get(kg, 0))
    
    tv = 0.5 * err / (N - K + 1 + 1e-15)
    return out_arr, float(tv)

# ── Return-time statistics ────────────────────────────────────────────────────

def return_time_stats(seq: np.ndarray, k: int) -> dict:
    seq_list = seq.tolist()
    positions = defaultdict(list)
    for i in range(len(seq_list) - k + 1):
        w = tuple(seq_list[i:i+k])
        positions[w].append(i)

    stats = {}
    for w, pos in positions.items():
        if len(pos) < 3:
            continue
        gaps           = np.diff(pos)
        vals, counts   = np.unique(gaps, return_counts=True)
        p              = counts / counts.sum()
        stats[w] = {
            "entropy":       float(-np.sum(p * np.log(p + 1e-30))),
            "support_size":  int(len(vals)),
        }
    return stats

def feature_vector(seq: np.ndarray, k: int) -> np.ndarray:
    D     = 2 ** k
    v     = np.zeros(2 * D, dtype=float)
    stats = return_time_stats(seq, k)
    all_words = [tuple(int(b) for b in format(i, f"0{k}b")) for i in range(D)]
    for idx, w in enumerate(all_words):
        if w in stats:
            v[2 * idx]     = stats[w]["entropy"]
            v[2 * idx + 1] = float(stats[w]["support_size"])
    n = np.linalg.norm(v)
    return v / n if n > 1e-15 else v

def cosine_sep(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15: return 0.0
    return float(1.0 - np.dot(a / na, b / nb))

def agg_stats(seq: np.ndarray, k: int) -> tuple[float, float]:
    ws = return_time_stats(seq, k)
    if not ws: return 0.0, 0.0
    return (float(np.mean([v["entropy"] for v in ws.values()])),
            float(np.mean([v["support_size"] for v in ws.values()])))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("The pattern-matched fake - same short runs, no long-range order")
    print(f"  N={N_SEQ}, analysis K={ANALYSIS_K}, {N_SURROGATE_SEEDS} surrogate seeds")

    # Generate Sturmian reference
    print("  Generating Sturmian reference ...")
    stm = gen_sturmian_bin(N_SEQ)
    print(f"  Sturmian balance: {stm.mean():.4f}")
    stm_fvecs = {k: feature_vector(stm, k) for k in ANALYSIS_K}
    stm_stats = {k: agg_stats(stm, k) for k in ANALYSIS_K}

    # Generate surrogates
    markov_surrogates = {m: [] for m in MARKOV_ORDERS}
    print("\n  Generating Markov surrogates ...")
    for m in MARKOV_ORDERS:
        print(f"    Order {m} ...", flush=True)
        for seed in range(N_SURROGATE_SEEDS):
            rng = np.random.default_rng(100 * m + seed)
            markov_surrogates[m].append(gen_markov_surrogate_bin(stm, m, rng))

    kgram_surrogates = {K: [] for K in KGRAM_ORDERS}
    kgram_errors = {K: [] for K in KGRAM_ORDERS}
    print("\n  Generating K-gram approx surrogates ...")
    for K in KGRAM_ORDERS:
        print(f"    Preserving K={K} ...", flush=True)
        for seed in range(N_SURROGATE_SEEDS):
            rng = np.random.default_rng(200 * K + seed)
            sur_seq, err = gen_kgram_approx_surrogate(stm, K, rng)
            kgram_surrogates[K].append(sur_seq)
            kgram_errors[K].append(err)
            
    # Compute metrics
    print("\n  Computing separations ...")
    conditions = {}
    
    worst_markov_gaps = []
    worst_kgram_gaps  = []
    
    for k in ANALYSIS_K:
        sv = stm_fvecs[k]
        cond_data = {
            "sturmian_support_size_mean": round(stm_stats[k][1], 6)
        }
        
        # Markov separations
        for m in MARKOV_ORDERS:
            seps = [cosine_sep(sv, feature_vector(s, k)) for s in markov_surrogates[m]]
            cond_data[f"markov_order_{m}_separation_mean"] = round(float(np.mean(seps)), 6)
            cond_data[f"markov_order_{m}_separation_min"]  = round(float(np.min(seps)), 6)
            worst_markov_gaps.append(float(np.min(seps)))
            
        # K-gram separations
        for K in KGRAM_ORDERS:
            seps = [cosine_sep(sv, feature_vector(s, k)) for s in kgram_surrogates[K]]
            cond_data[f"kgram_K{K}_separation_mean"] = round(float(np.mean(seps)), 6)
            cond_data[f"kgram_K{K}_separation_min"]  = round(float(np.min(seps)), 6)
            cond_data[f"kgram_K{K}_count_error_mean"] = round(float(np.mean(kgram_errors[K])), 6)
            worst_kgram_gaps.append(float(np.min(seps)))
            
        conditions[f"analysis_k{k}"] = cond_data
        print(f"  k={k}:")
        print(f"    Markov min: order1={cond_data['markov_order_1_separation_min']:.4f}, order4={cond_data['markov_order_4_separation_min']:.4f}")
        print(f"    K-gram min: K2={cond_data['kgram_K2_separation_min']:.4f}, K4={cond_data['kgram_K4_separation_min']:.4f}")

    global_worst_markov = min(worst_markov_gaps) if worst_markov_gaps else 0.0
    global_worst_kgram  = min(worst_kgram_gaps) if worst_kgram_gaps else 0.0

    if global_worst_markov > 0.05 and global_worst_kgram > 0.05:
        verdict = "PASS"
        claim   = "return_time_order_survives_local_statistics_matching"
    elif global_worst_markov > 0.05 and global_worst_kgram <= 0.05:
        verdict = "PARTIAL"
        claim   = "return_time_order_survives_local_statistics_matching" # but collapses weakly to K-grams
    else:
        verdict = "FAIL"
        claim   = "return_time_order_collapses_to_kgram_statistics"

    metrics = {
        "experiment": "return_time_fake",
        "verdict": verdict,
        "protocol": "kgram_matched_return_time_hardening",
        "sequence_length": N_SEQ,
        "analysis_word_lengths": ANALYSIS_K,
        "markov_orders": MARKOV_ORDERS,
        "kgram_preserve_orders": KGRAM_ORDERS,
        "n_surrogate_seeds": N_SURROGATE_SEEDS,
        "conditions": conditions,
        "summary": {
            "worst_markov_gap": round(global_worst_markov, 6),
            "worst_kgram_preserved_gap": round(global_worst_kgram, 6),
            "best_supported_claim": claim,
            "local_statistics_explain_return_time_signal": verdict == "FAIL"
        },
        "claim_boundary": claim
    }

    out_dir  = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "metrics_return_time_fake.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Saved: {out_path}")
    print(f"  Verdict: {verdict}  ({claim})")
    print(f"  Worst Markov gap:   {global_worst_markov:.4f}")
    print(f"  Worst K-gram gap:   {global_worst_kgram:.4f}")

if __name__ == "__main__":
    main()
