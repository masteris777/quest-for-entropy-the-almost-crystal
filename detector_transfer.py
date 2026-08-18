import sys, os, json
import numpy as np
from scipy.linalg import eigh

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from dynamical_matrix import build_dynamical_matrix
from resonator_bank import ResonatorBank

def eigenmodes(D):
    evals, evecs = eigh(D)
    valid = evals > 1e-8
    evals, evecs = evals[valid], evecs[:, valid]
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx]

def learn_source_bank(springs, N, excite_site, top_k, min_gap):
    masses = np.ones(N)
    D = build_dynamical_matrix(masses, springs)
    evals, evecs = eigenmodes(D)
    omegas = np.sqrt(evals)

    order = np.argsort(omegas)[::-1]
    selected = []
    for i in order:
        if len(selected) == 0:
            selected.append(i)
        elif np.min(np.abs(omegas[i] - omegas[selected])) > min_gap:
            selected.append(i)
        if len(selected) == top_k:
            break
            
    sel = np.array(selected)
    omegas_ref = omegas[sel]
    evecs_ref  = evecs[:, sel]
    
    return omegas_ref, evecs_ref

def get_footprint(springs, N, excite_site, obs_site, omegas_ref, T_steps, dt, gamma):
    masses = np.ones(N)
    D = build_dynamical_matrix(masses, springs)
    evals, evecs = eigenmodes(D)
    omegas = np.sqrt(evals)
    
    v0 = np.zeros(N); v0[excite_site] = 1.0
    proj = evecs.T @ v0
    amp = proj / omegas
    t_array = np.arange(T_steps) * dt
    
    signal = np.sum(amp[:, None] * evecs[obs_site, :, None] * np.sin(np.outer(omegas, t_array)), axis=0)
    bank = ResonatorBank(omegas_ref, gamma=gamma)
    bank.run_signal(t_array, signal)
    
    avg_E = bank.energy_accumulator / bank.t_total
    total = np.sum(avg_E)
    if total < 1e-15:
        return np.ones(len(omegas_ref)) / len(omegas_ref)
    return avg_E / total

def score(w_ref, w_ctrl):
    return float(np.dot(w_ref, w_ctrl) / (np.linalg.norm(w_ref) * np.linalg.norm(w_ctrl) + 1e-15))

# GENERATORS
def gen_fibonacci_springs(N, k_A=1.0, k_B=2.0, offset=0):
    S0 = [0]; S1 = [0, 1]
    while len(S1) < N + offset + 10:
        S0, S1 = S1, S1 + S0
    return np.array([k_A if w == 0 else k_B for w in S1[offset:offset+N]])

def gen_sturmian_springs(N, k_A=1.0, k_B=2.0, alpha=np.sqrt(2)-1, phase=0.0):
    word = [int(np.floor((n + 1) * alpha + phase)) - int(np.floor(n * alpha + phase)) for n in range(N)]
    return np.array([k_B if w else k_A for w in word])

def gen_logistic_springs(N, k_A=1.0, k_B=2.0, seed=0.12345):
    word = []
    x = seed
    for _ in range(100): x = 4.0 * x * (1.0 - x)
    for _ in range(N):
        x = 4.0 * x * (1.0 - x)
        word.append(1 if x >= 0.5 else 0)
    return np.array([k_B if w else k_A for w in word])


def main():
    print("Detector transfer: tune on one chain, then score the others")
    
    N = 250
    top_k = 5
    dt = 0.05
    T_steps = 5000
    gamma = 0.0002
    
    excite_site = N // 2
    obs_site = min(10, N - 1)
    if obs_site == excite_site: obs_site = 11

    n_samples = 150
    np.random.seed(42)

    # 1. Define source bank (Sturmian Phase=0)
    source_alpha = np.sqrt(2) - 1
    springs_src = gen_sturmian_springs(N, alpha=source_alpha, phase=0.0)
    omegas_src, _ = learn_source_bank(springs_src, N, excite_site, top_k, min_gap=0.30)
    w_src = get_footprint(springs_src, N, excite_site, obs_site, omegas_src, T_steps, dt, gamma)
    
    # 2. Define cross-class bank (Logistic Seed=0.12345)
    springs_chaos_src = gen_logistic_springs(N, seed=0.12345)
    omegas_chaos, _ = learn_source_bank(springs_chaos_src, N, excite_site, top_k, min_gap=0.35)
    w_chaos_src = get_footprint(springs_chaos_src, N, excite_site, obs_site, omegas_chaos, T_steps, dt, gamma)

    results = {}

    # === FIBONACCI (Shift transfer) ===
    print("\nEvaluating Fibonacci (Shift-only transfer)...")
    scores_fibo = []
    for i in range(n_samples):
        offset = np.random.randint(10, 5000)
        s_test = gen_fibonacci_springs(N, offset=offset)
        w_test = get_footprint(s_test, N, excite_site, obs_site, omegas_src, T_steps, dt, gamma)
        scores_fibo.append(score(w_src, w_test))
    
    sf = np.array(scores_fibo)
    results["fibonacci"] = {
        "evaluation_type": "shift",
        "n_samples": n_samples,
        "max": float(np.max(sf)),
        "mean": float(np.mean(sf)),
        "median": float(np.median(sf)),
        "p10": float(np.percentile(sf, 10)),
        "pct_gt_0_8": float(np.mean(sf > 0.8) * 100)
    }

    # === STURMIAN (Independent Instance / Phase) ===
    print("Evaluating Sturmian (Independent phase)...")
    scores_sturm = []
    for i in range(n_samples):
        phase = np.random.rand()  # Random phase in [0, 1)
        s_test = gen_sturmian_springs(N, alpha=source_alpha, phase=phase)
        w_test = get_footprint(s_test, N, excite_site, obs_site, omegas_src, T_steps, dt, gamma)
        scores_sturm.append(score(w_src, w_test))

    ss = np.array(scores_sturm)
    results["sturmian"] = {
        "evaluation_type": "phase (independent_instance)",
        "n_samples": n_samples,
        "max": float(np.max(ss)),
        "mean": float(np.mean(ss)),
        "median": float(np.median(ss)),
        "p10": float(np.percentile(ss, 10)),
        "pct_gt_0_8": float(np.mean(ss > 0.8) * 100)
    }

    # === LOGISTIC CHAOS (Independent Seed) ===
    print("Evaluating Logistic Chaos (Independent seed vs its own Chaos Bank)...")
    scores_chaos = []
    for i in range(n_samples):
        seed = float(np.random.rand() * 0.9 + 0.05) # Random seed in [0.05, 0.95)]
        s_test = gen_logistic_springs(N, seed=seed)
        w_test = get_footprint(s_test, N, excite_site, obs_site, omegas_chaos, T_steps, dt, gamma)
        scores_chaos.append(score(w_chaos_src, w_test))
        
    sc = np.array(scores_chaos)
    results["logistic_chaos"] = {
        "evaluation_type": "seed (independent_instance)",
        "n_samples": n_samples,
        "max": float(np.max(sc)),
        "mean": float(np.mean(sc)),
        "median": float(np.median(sc)),
        "p10": float(np.percentile(sc, 10)),
        "pct_gt_0_8": float(np.mean(sc > 0.8) * 100)
    }

    # === CROSS-CLASS CONFUSION ===
    print("Evaluating Cross-Class (Logistic independent instances on Sturmian Bank)...")
    scores_x = []
    for i in range(n_samples):
        seed = float(np.random.rand() * 0.9 + 0.05)
        s_test = gen_logistic_springs(N, seed=seed)
        w_test = get_footprint(s_test, N, excite_site, obs_site, omegas_src, T_steps, dt, gamma)
        scores_x.append(score(w_src, w_test))
    
    sx = np.array(scores_x)
    cross_class = {
        "tested": True,
        "max_confusion": float(np.max(sx)),
        "mean_confusion": float(np.mean(sx))
    }

    sturm_survives = results["sturmian"]["mean"] > 0.60
    chaos_fails = results["logistic_chaos"]["mean"] < 0.40 and results["logistic_chaos"]["p10"] < 0.20

    if sturm_survives and chaos_fails:
        verdict = "PASS"
        align1 = True
        align2 = True
        claim = "true_class_transfer"
    elif sturm_survives and not chaos_fails:
        verdict = "FAIL"
        align1 = True
        align2 = False
        if results["logistic_chaos"]["mean"] > 0.60:
            claim = "realization_specific_localization"
        else:
            claim = "shift_only_transfer"
    else:
        verdict = "FAIL"
        align1 = False
        align2 = chaos_fails
        claim = "shift_only_transfer"

    metrics = {
        "experiment": "detector_transfer",
        "verdict": verdict,
        "protocol": "independent_instance_transfer_boundary_audit",
        "source_bank": {
            "family": "sturmian_and_logistic",
            "instance_type": "fixed parameter (phase=0, seed=0.12345)",
            "retuned_per_sample": False
        },
        "families": results,
        "cross_class_confusion": cross_class,
        "boundary_alignment": {
            "deterministic_transfer_survives": align1,
            "chaotic_independent_seed_transfer_fails": align2,
            "supports_true_class_transfer_boundary": (verdict == "PASS")
        },
        "claim_boundary": claim
    }

    out_path = os.path.join(base_dir, "metrics_detector_transfer.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"Verdict: {verdict} ({claim})")

if __name__ == "__main__":
    main()