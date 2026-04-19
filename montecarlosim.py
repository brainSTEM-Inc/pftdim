import math
import random
import time

def F_binom(k, N, q):
    #Binomial CDF: Probability of AT MOST k failures out of N sensors.
    if k < 0:
        return 0.0
    prob = 0.0
    for f in range(k + 1):
        combinations = math.comb(N, f)
        prob += combinations * (q**f) * ((1 - q)**(N - f))
    return prob

def dim_p_CF(t_list, q, p):
    #Closed Form calculation based on Theorem 9 (Greedy Upgrades)
    # 1. Check absolute feasibility (max upgrades)
    max_prob = 1.0
    for t in t_list:
        max_prob *= F_binom(1, t, q)
   
    if max_prob < p - 1e-10:
        return float('inf')
       
    # 2. Base case (n_i = t_i - 1)
    cost = sum(t - 1 for t in t_list)
    current_prob = 1.0
    for t in t_list:
        current_prob *= F_binom(0, t - 1, q)
       
    if current_prob >= p - 1e-10:
        return cost
       
    # 3. Greedy upgrades
    t_sorted = sorted(t_list, reverse=True)
    for t in t_sorted:
        alpha = 1 + (t - 1) * q
        current_prob *= alpha
        cost += 1
        if current_prob >= p - 1e-10:
            return cost
           
    return float('inf')

def dim_p_DP(t_list, q, p):
    #Dynamic Programming implementation based on Algorithm 2.
    M = len(t_list)
    C = [0] + t_list
   
    S = [0] * (M + 1)
    for i in range(1, M + 1):
        S[i] = S[i-1] + C[i]
       
    DP = [[-float('inf')] * (S[M] + 1) for _ in range(M + 1)]
    DP[0][0] = 0.0
   
    for i in range(1, M + 1):
        t = C[i]
        for c in range(S[i] + 1):
            for s in range(min(c, t) + 1):
                if s < t - 1:
                    p_star = 0.0
                elif s == t - 1:
                    p_star = F_binom(0, t - 1, q)
                else:
                    p_star = F_binom(1, t, q)
                   
                if p_star > 0 and c - s >= 0 and DP[i-1][c-s] != -float('inf'):
                    val = DP[i-1][c-s] + math.log(p_star)
                    if val > DP[i][c]:
                        DP[i][c] = val
                       
    target_log_p = math.log(p) if p > 0 else -float('inf')
   
    for c in range(S[M] + 1):
        if DP[M][c] >= target_log_p - 1e-10:
            return c
           
    return float('inf')

def run_random_simulator(num_trials=1000):
    print(f"Initializing simulator for {num_trials} randomized caterpillar trials...")
    print("-" * 67)
   
    mismatches = 0
    start_time = time.time()
   
    for trial in range(1, num_trials + 1):
        # 1. Randomize the Caterpillar structure
        num_major_vertices = random.randint(2, 10)
        t_list = [random.randint(2, 8) for _ in range(num_major_vertices)]
       
        # 2. Randomize the probability parameters
        q = round(random.uniform(0.01, 0.99), 3)
        p = round(random.uniform(0.01, 0.99), 3)
       
        # 3. Compute both methods
        res_dp = dim_p_DP(t_list, q, p)
        res_cf = dim_p_CF(t_list, q, p)
       
        # 4. Cross-verify
        if res_dp != res_cf:
            mismatches += 1
            print(f"\n[!] MISMATCH DETECTED ON TRIAL {trial}")
            print(f"    Terminal Degrees : {t_list}")
            print(f"    q = {q:.3f}, p = {p:.3f}")
            print(f"    DP result: {res_dp} | CF result: {res_cf}")
           
        #Progress bar   
        if trial % 200 == 0:
            print(f"[{trial}/{num_trials}] trials completed securely...")
           
    elapsed = time.time() - start_time
    print("-" * 65)
   
    if mismatches == 0:
        print(f"SUCCESS")
        print(f"Tested {num_trials} random configurations in {elapsed:.2f} seconds.")
    else:
        print(f"FAILED. Found {mismatches} mismatches out of {num_trials} trials.")

# Run the simulation
if __name__ == "__main__":
    run_random_simulator(10000)
