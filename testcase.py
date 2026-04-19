import math

def F_binom(k, N, q):
    #Binomial CDF: Probability of AT MOST k failures out of N sensors.
    if k < 0:
        return 0.0
    prob = 0.0
    for f in range(k + 1):
        combinations = math.comb(N, f)
        prob += combinations * (q**f) * ((1 - q)**(N - f))
    return prob

def dim_p_DP_with_vector(t_list, q, p):
    """
    Dynamic Programming implementation based on Algorithm 2.
    Returns: (Total Cost, Branch Distributions, Exact Probability Achieved)
    """
    M = len(t_list)
    C = [0] + t_list
   
    S = [0] * (M + 1)
    for i in range(1, M + 1):
        S[i] = S[i-1] + C[i]
       
    DP = [[-float('inf')] * (S[M] + 1) for _ in range(M + 1)]
    DP[0][0] = 0.0
   
    Choice = [[0] * (S[M] + 1) for _ in range(M + 1)]
   
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
                        Choice[i][c] = s
                       
    target_log_p = math.log(p) if p > 0 else -float('inf')
   
    C_star = float('inf')
    exact_log_prob = -float('inf')
   
    for c in range(S[M] + 1):
        if DP[M][c] >= target_log_p - 1e-10:
            C_star = c
            exact_log_prob = DP[M][c]
            break
           
    if C_star == float('inf'):
        return float('inf'), [], 0.0
       
    # Convert log-probability back to standard probability
    exact_prob = math.exp(exact_log_prob)
       
    #Phase 4: Backtracking for Exact Distribution
    c_curr = C_star
    vector_distribution = []
   
    for i in range(M, 0, -1):
        s_i = Choice[i][c_curr]
        c_curr -= s_i
       
        branch_vector = [1] * s_i + [0] * (C[i] - s_i)
        vector_distribution.insert(0, branch_vector)
       
    return C_star, vector_distribution, exact_prob

# --- Test ---
t_list = [2,3,5,7]  # A caterpillar with 4 major vertices
q = 0.01
p = 0.95

cost, layout, exact_prob = dim_p_DP_with_vector(t_list, q, p)

print(f"Target Probability (p)   : {p:.6f}")
print(f"Exact Prob. Achieved     : {exact_prob:.6f}")
print(f"Total Sensors Required   : {cost}\n")
print("Exact Sensor Distribution:")
for i, branch in enumerate(layout):
    print(f"  Major Vertex {i+1} (Degree {t_list[i]}): {branch}")
