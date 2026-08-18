import numpy as np

def generate_fibonacci(iterations=16):
    # A -> AB, B -> A
    # Length of iteration 16 is F_17 = 1597
    # Wait, fib sequence lengths grow fast:
    # 0: 1, 1: 2, 2: 3, 3: 5, 4: 8, 5: 13, 6: 21, 7: 34, 8: 55, 9: 89, 10: 144, 11: 233, 12: 377, 13: 610, 14: 987, 15: 1597, 16: 2584, 17: 4181, 18: 6765, 19: 10946, 20: 17711
    # Iteration 20 -> ~17k elements.
    seq = [1]
    for _ in range(iterations):
        next_seq = []
        for v in seq:
            if v == 1:
                next_seq.extend([1, -1])
            else:
                next_seq.extend([1])
        seq = next_seq
    return np.array(seq, dtype=float)

def generate_thue_morse(iterations=14):
    # 2^14 = 16384
    seq = [1]
    for _ in range(iterations):
        next_seq = []
        for v in seq:
            if v == 1:
                next_seq.extend([1, -1])
            else:
                next_seq.extend([-1, 1])
        seq = next_seq
    return np.array(seq, dtype=float)

def generate_sturmian(N=16000, alpha=np.sqrt(2)-1, rho=0.0):
    n = np.arange(1, N+1)
    seq = np.floor((n+1)*alpha + rho) - np.floor(n*alpha + rho)
    return 2.0 * seq - 1.0

def generate_cut_and_project_gaps(N=16000, slope=1.618033988749895):
    # difference between projected points. Basically another Sturmian variation
    n = np.arange(1, N+1)
    seq = np.floor((n+1)/slope) - np.floor(n/slope)
    return 2.0 * seq - 1.0

if __name__ == "__main__":
    fib = generate_fibonacci(20)
    print("Fibonacci length:", len(fib))
    tm = generate_thue_morse(14)
    print("Thue-Morse length:", len(tm))
    sturm = generate_sturmian()
    print("Sturmian length:", len(sturm))
