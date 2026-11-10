import numpy as np
import scipy.optimize as opt
from scipy.stats import truncnorm
import multiprocessing as mp
import time
import pickle
import os
import sys

# --- 1. PROBLEM SPECIFICATION & INSTANCE GENERATION ---

INSTANCE_SIZES = {
    # id: (I, J, K, S, R, X)
    1: (6, 2, 3, 2, 1, 1),
    2: (8, 3, 5, 2, 2, 1),
    3: (10, 4, 7, 3, 2, 1),
    4: (12, 5, 8, 3, 2, 2),
    5: (15, 6, 9, 3, 3, 2),
    6: (18, 6, 9, 4, 3, 2),
    7: (21, 8, 10, 4, 4, 3),
    8: (24, 9, 12, 5, 4, 3),
    9: (27, 9, 14, 5, 5, 4),
    10: (30, 10, 16, 6, 6, 4),
    11: (33, 10, 18, 6, 6, 5),
    12: (36, 12, 18, 8, 8, 5),
    13: (50, 16, 24, 10, 10, 6),
    14: (75, 24, 34, 14, 14, 8),
    15: (100, 32, 45, 18, 18, 10)
}

def solve_lognormal_params(mean, p95):
    # Solve 0.5 * sigma^2 - 1.64485 * sigma + ln(p95/mean) = 0
    # => sigma = (1.64485 - sqrt(1.64485^2 - 2 * ln(p95/mean))) / 1
    ln_ratio = np.log(p95 / mean)
    disc = 1.64485**2 - 2 * ln_ratio
    if disc < 0:
        disc = 0
    sigma = 1.64485 - np.sqrt(disc)
    mu = np.log(mean) - 0.5 * sigma**2
    return mu, sigma

def generate_scenarios(n_scenarios, I, J, S, seed, dist_type='uniform'):
    rng = np.random.default_rng(seed)
    
    # Baseline Uniform parameters
    # returns: U(3500, 3800) -> mean = 3650
    # capacity: U(1600, 2400) -> mean = 2000
    # inspection cost: U(15, 18) -> mean = 16.5
    # storage cost: U(15, 25) -> mean = 20
    # turnaround time: U(2, 10) -> mean = 6
    
    if dist_type == 'uniform':
        R = rng.uniform(3500, 3800, (n_scenarios, I))
        Ca = rng.uniform(1600, 2400, (n_scenarios, J))
        C = rng.uniform(15, 18, (n_scenarios, J))
        W = rng.uniform(15, 25, (n_scenarios, J))
        t = rng.uniform(2, 10, (n_scenarios, S, I))
    elif dist_type == 'normal':
        # Symmetric truncated Normal with same support and mean
        # returns: mean=3650, sd = 300 / 4 = 75
        R = truncnorm.rvs(-2, 2, loc=3650, scale=75, size=(n_scenarios, I), random_state=rng)
        # capacity: mean=2000, sd = 800 / 4 = 200
        Ca = truncnorm.rvs(-2, 2, loc=2000, scale=200, size=(n_scenarios, J), random_state=rng)
        # inspection cost: mean=16.5, sd = 3 / 4 = 0.75
        C = truncnorm.rvs(-2, 2, loc=16.5, scale=0.75, size=(n_scenarios, J), random_state=rng)
        # storage cost: mean=20, sd = 10 / 4 = 2.5
        W = truncnorm.rvs(-2, 2, loc=20, scale=2.5, size=(n_scenarios, J), random_state=rng)
        # turnaround time: mean=6, sd = 8 / 4 = 2.0
        t = truncnorm.rvs(-2, 2, loc=6, scale=2.0, size=(n_scenarios, S, I), random_state=rng)
    elif dist_type == 'lognormal':
        # Lognormal calibrated to same mean and 95th percentile
        # For Uniform U(a, b), mean is (a+b)/2, p95 is a + 0.95*(b-a)
        
        # returns: U(3500, 3800) -> mean=3650, p95=3785
        mu_R, sig_R = solve_lognormal_params(3650, 3785)
        R = rng.lognormal(mu_R, sig_R, (n_scenarios, I))
        
        # capacity: U(1600, 2400) -> mean=2000, p95=2360
        mu_Ca, sig_Ca = solve_lognormal_params(2000, 2360)
        Ca = rng.lognormal(mu_Ca, sig_Ca, (n_scenarios, J))
        
        # C: U(15, 18) -> mean=16.5, p95=17.85
        mu_C, sig_C = solve_lognormal_params(16.5, 17.85)
        C = rng.lognormal(mu_C, sig_C, (n_scenarios, J))
        
        # W: U(15, 25) -> mean=20, p95=24.5
        mu_W, sig_W = solve_lognormal_params(20, 24.5)
        W = rng.lognormal(mu_W, sig_W, (n_scenarios, J))
        
        # t: U(2, 10) -> mean=6, p95=9.6
        mu_t, sig_t = solve_lognormal_params(6, 9.6)
        t = rng.lognormal(mu_t, sig_t, (n_scenarios, S, I))
    else:
        raise ValueError(f"Unknown distribution {dist_type}")
        
    return R, Ca, C, W, t

def generate_instance(instance_id, dist_type='uniform', param_mod=None):
    I, J, K, S, R_cnt, X = INSTANCE_SIZES[instance_id]
    
    # Deterministic parameters (fixed per instance)
    rng = np.random.default_rng(1000 + instance_id)
    
    T_ij = rng.uniform(2, 10, (I, J))
    T_jk = rng.uniform(2, 10, (J, K))
    T_js = rng.uniform(2, 10, (J, S))
    T_si = rng.uniform(2, 10, (S, I))
    T_jr = rng.uniform(2, 10, (J, R_cnt))
    T_rx = rng.uniform(2, 10, (R_cnt, X))
    d_x = rng.uniform(5, 10, (X,))
    tau_i = rng.uniform(2, 8, (I,))
    
    # Deterministic fractions
    beta_K = 0.5
    beta_S = 0.3
    beta_R = 0.2
    alpha = 0.4
    phi_1 = 160
    phi_2 = 200
    
    # Apply parameter modifications for sensitivity analysis
    if param_mod:
        param_name, pct_change = param_mod
        factor = 1.0 + pct_change / 100.0
        if param_name == 'Ca':
            pass # Handled inside scenario generation (capacity)
        elif param_name == 'beta':
            # vary beta_S by factor and renormalize
            beta_S *= factor
            rem = 1.0 - beta_S
            # keep ratio of beta_K and beta_R
            sum_other = beta_K + beta_R
            beta_K = rem * (beta_K / sum_other)
            beta_R = rem * (beta_R / sum_other)
        elif param_name == 'alpha':
            alpha *= factor
        elif param_name == 'phi_1':
            phi_1 *= factor
        elif param_name == 'phi_2':
            phi_2 *= factor
            
    # In-sample scenarios (50)
    scen_seed = 2000 + instance_id
    R, Ca, C, W, t = generate_scenarios(50, I, J, S, scen_seed, dist_type)
    
    if param_mod and param_mod[0] == 'Ca':
        Ca *= (1.0 + param_mod[1]/100.0)
        
    # Out-of-sample scenarios (500)
    oos_seed = 3000 + instance_id
    R_oos, Ca_oos, C_oos, W_oos, t_oos = generate_scenarios(500, I, J, S, oos_seed, dist_type)
    if param_mod and param_mod[0] == 'Ca':
        Ca_oos *= (1.0 + param_mod[1]/100.0)
        
    return {
        'I': I, 'J': J, 'K': K, 'S': S, 'R_cnt': R_cnt, 'X': X,
        'T_ij': T_ij, 'T_jk': T_jk, 'T_js': T_js, 'T_si': T_si, 'T_jr': T_jr, 'T_rx': T_rx,
        'd_x': d_x, 'tau_i': tau_i,
        'beta_K': beta_K, 'beta_S': beta_S, 'beta_R': beta_R, 'alpha': alpha,
        'phi_1': phi_1, 'phi_2': phi_2,
        'R': R, 'Ca': Ca, 'C': C, 'W': W, 't': t,
        'R_oos': R_oos, 'Ca_oos': Ca_oos, 'C_oos': C_oos, 'W_oos': W_oos, 't_oos': t_oos
    }

# --- 2. VECTORISED DECODER ---

class Decoder:
    def __init__(self, inst):
        self.I = inst['I']
        self.J = inst['J']
        self.K = inst['K']
        self.S = inst['S']
        self.R_cnt = inst['R_cnt']
        self.X = inst['X']
        
        self.T_ij = inst['T_ij']
        self.T_jk = inst['T_jk']
        self.T_js = inst['T_js']
        self.T_si = inst['T_si']
        self.T_jr = inst['T_jr']
        self.T_rx = inst['T_rx']
        self.d_x = inst['d_x']
        self.tau_i = inst['tau_i']
        
        self.beta_K = inst['beta_K']
        self.beta_S = inst['beta_S']
        self.beta_R = inst['beta_R']
        self.alpha = inst['alpha']
        self.phi_1 = inst['phi_1']
        self.phi_2 = inst['phi_2']
        
        # Precompute greedy downstream routing choices
        # retailing center: k*(j) = argmin_k T_jk
        self.k_star = np.argmin(self.T_jk, axis=1) # shape (J,)
        self.T_j_kstar = self.T_jk[np.arange(self.J), self.k_star] # shape (J,)
        
        # recycling center: r*(j) = argmin_r T_jr
        self.r_star = np.argmin(self.T_jr, axis=1) # shape (J,)
        self.T_j_rstar = self.T_jr[np.arange(self.J), self.r_star] # shape (J,)
        
        # disposal location: x*(r) = argmin_x (T_rx + d_x)
        self.x_star = np.argmin(self.T_rx + self.d_x, axis=1) # shape (R_cnt,)
        self.T_rx_d_star = (self.T_rx + self.d_x)[np.arange(self.R_cnt), self.x_star] # shape (R_cnt,)
        
        # service center: s*(i, j) = argmin_s (T_js + T_si)
        # T_js has shape (J, S), T_si has shape (S, I) => round_trip cost shape (I, J, S)
        round_trip_cost = self.T_js[np.newaxis, :, :] + self.T_si.T[:, np.newaxis, :] # shape (I, J, S)
        self.s_star = np.argmin(round_trip_cost, axis=2) # shape (I, J)
        self.rt_cost_star = round_trip_cost[np.arange(self.I)[:, np.newaxis], np.arange(self.J)[np.newaxis, :], self.s_star] # shape (I, J)
        
    def precompute_scenario_costs(self, R, Ca, C, W, t):
        # returns precomputed variables for fast decoding
        n_scenarios = R.shape[0]
        
        # unit cost for shipping 1 unit from zone i to collection center j in scenario omega
        C_static = (self.T_ij + 
                    self.beta_K * self.T_j_kstar + 
                    self.beta_S * self.rt_cost_star + 
                    self.beta_R * self.T_j_rstar + 
                    self.beta_R * self.alpha * self.T_rx_d_star[self.r_star]) # shape (I, J)
        
        # Add scenario-dependent C and W
        C_unit = C_static[np.newaxis, :, :] + C[:, np.newaxis, :] + W[:, np.newaxis, :] # shape (n_scenarios, I, J)
        
        # lateness delta: max(t - tau, 0)
        t_indexed = np.zeros((n_scenarios, self.I, self.J))
        for j in range(self.J):
            s_idx = self.s_star[:, j] # shape (I,)
            t_indexed[:, :, j] = t[:, s_idx, np.arange(self.I)]
            
        delta = np.maximum(t_indexed - self.tau_i[np.newaxis, :, np.newaxis], 0.0) # shape (n_scenarios, I, J)
        
        return C_unit, delta
        
    def decode(self, a, rho, R, Ca, C_unit, delta):
        # a: shape (N, I)
        # rho: shape (N, I)
        N = a.shape[0]
        n_scenarios = R.shape[0]
        
        # 1. Tendered volume at each center: V of shape (N, J, n_scenarios)
        mask = (a[:, :, np.newaxis] == np.arange(self.J)[np.newaxis, np.newaxis, :]) # shape (N, I, J)
        V = np.sum(mask[:, :, :, np.newaxis] * 
                   rho[:, :, np.newaxis, np.newaxis] * 
                   R.T[np.newaxis, :, np.newaxis, :], axis=1) # shape (N, J, n_scenarios)
        
        # 2. Scale factor: scale_all of shape (N, J, n_scenarios)
        Ca_w = Ca.T[np.newaxis, :, :] # shape (1, J, n_scenarios)
        scale_all = np.where(V > 0, np.minimum(1.0, Ca_w / np.maximum(V, 1e-9)), 1.0)
        
        # 3. Scale factor for each zone: scale_zone of shape (N, I, n_scenarios)
        idx_p = np.arange(N)[:, np.newaxis, np.newaxis]
        idx_a = a[:, :, np.newaxis]
        idx_w = np.arange(n_scenarios)[np.newaxis, np.newaxis, :]
        scale_zone = scale_all[idx_p, idx_a, idx_w]
        
        # 4. Collected and uncollected quantities: shape (N, I, n_scenarios)
        R_w = R.T[np.newaxis, :, :] # shape (1, I, n_scenarios)
        Q = rho[:, :, np.newaxis] * R_w * scale_zone
        U = R_w - Q
        
        # 5. Costs: C_u of shape (N, I, n_scenarios)
        C_unit_t = C_unit.transpose(1, 2, 0) # shape (I, J, n_scenarios)
        C_u = C_unit_t[np.arange(self.I)[np.newaxis, :, np.newaxis], a[:, :, np.newaxis], np.arange(n_scenarios)[np.newaxis, np.newaxis, :]]
        
        # 6. Lateness: del_u of shape (N, I, n_scenarios)
        delta_t = delta.transpose(1, 2, 0) # shape (I, J, n_scenarios)
        del_u = delta_t[np.arange(self.I)[np.newaxis, :, np.newaxis], a[:, :, np.newaxis], np.arange(n_scenarios)[np.newaxis, np.newaxis, :]]
        
        # 7. Objectives: mean over scenarios (axis 2)
        Z1 = np.mean(np.sum(C_u * Q, axis=1), axis=1)
        Z2 = np.mean(self.phi_1 * np.sum(U, axis=1) + self.phi_2 * self.beta_S * np.sum(del_u * Q, axis=1), axis=1)
        
        return Z1, Z2

# --- 3. EXACT MILP SOLVER ---

def solve_exact(inst, grid_size=21, time_limit=10.0):
    I = inst['I']
    J = inst['J']
    R = inst['R']
    Ca = inst['Ca']
    n_scen = R.shape[0]
    
    dec = Decoder(inst)
    C_unit, delta = dec.precompute_scenario_costs(R, Ca, inst['C'], inst['W'], inst['t'])
    
    n_y = I * J
    n_Q = I * J * n_scen
    n_vars = n_y + n_Q
    
    integrality = np.zeros(n_vars)
    integrality[:n_y] = 1 # y_ij are binary
    
    lb = np.zeros(n_vars)
    ub = np.inf * np.ones(n_vars)
    ub[:n_y] = 1.0
    bounds = opt.Bounds(lb, ub)
    
    c1 = np.zeros(n_vars)
    c2 = np.zeros(n_vars)
    
    p_w = 1.0 / n_scen
    for w in range(n_scen):
        start_q_w = n_y + w * (I * J)
        c1[start_q_w : start_q_w + I*J] = p_w * C_unit[w].flatten()
        
    const_Z2 = dec.phi_1 * np.mean(np.sum(R, axis=1))
    for w in range(n_scen):
        start_q_w = n_y + w * (I * J)
        val = dec.phi_2 * dec.beta_S * delta[w].flatten() - dec.phi_1
        c2[start_q_w : start_q_w + I*J] = p_w * val
        
    A_eq = []
    b_eq = []
    A_ub = []
    b_ub = []
    
    # sum_j y_ij = 1
    for i in range(I):
        row = np.zeros(n_vars)
        row[i*J : (i+1)*J] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)
        
    # Q_ij^omega <= R_i^omega y_ij => Q_ij^omega - R_i^omega y_ij <= 0
    for w in range(n_scen):
        R_w = R[w]
        for i in range(I):
            for j in range(J):
                row = np.zeros(n_vars)
                row[i*J + j] = -R_w[i]
                row[n_y + w*(I*J) + i*J + j] = 1.0
                A_ub.append(row)
                b_ub.append(0.0)
                
    # sum_i Q_ij^omega <= Ca_j^omega
    for w in range(n_scen):
        Ca_w = Ca[w]
        for j in range(J):
            row = np.zeros(n_vars)
            for i in range(I):
                row[n_y + w*(I*J) + i*J + j] = 1.0
            A_ub.append(row)
            b_ub.append(Ca_w[j])
            
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # 1. Minimize Z2
    res_min_z2 = opt.milp(c=c2, integrality=integrality, bounds=bounds,
                          constraints=opt.LinearConstraint(
                              np.vstack([A_eq, A_ub]) if A_ub.size else A_eq,
                              np.concatenate([b_eq, -np.inf * np.ones(len(b_ub))]),
                              np.concatenate([b_eq, b_ub])
                          ))
    if res_min_z2.status != 0:
        return None
    min_z2 = res_min_z2.fun + const_Z2
    
    # 2. Minimize Z1
    res_min_z1 = opt.milp(c=c1, integrality=integrality, bounds=bounds,
                          constraints=opt.LinearConstraint(
                              np.vstack([A_eq, A_ub]) if A_ub.size else A_eq,
                              np.concatenate([b_eq, -np.inf * np.ones(len(b_ub))]),
                              np.concatenate([b_eq, b_ub])
                          ))
    if res_min_z1.status != 0:
        return None
    
    # Find Z2 value corresponding to minimum Z1
    max_z2 = np.dot(c2, res_min_z1.x) + const_Z2
    
    # Epsilon grid on Z2
    eps_vals = np.linspace(min_z2, max_z2, grid_size)
    pareto_front = []
    
    for eps in eps_vals:
        A_all = np.vstack([A_eq, A_ub, c2])
        lb_all = np.concatenate([b_eq, -np.inf * np.ones(len(b_ub) + 1)])
        ub_all = np.concatenate([b_eq, b_ub, [eps - const_Z2]])
        
        res = opt.milp(c=c1, integrality=integrality, bounds=bounds,
                       constraints=opt.LinearConstraint(A_all, lb_all, ub_all))
        if res.status == 0:
            z1_val = res.fun
            z2_val = np.dot(c2, res.x) + const_Z2
            pareto_front.append((z1_val, z2_val))
            
    pareto_front = filter_non_dominated(np.array(pareto_front))
    return pareto_front

def filter_non_dominated(points):
    if len(points) == 0:
        return points
    idx = np.lexsort((-points[:, 1], points[:, 0]))
    sorted_pts = points[idx]
    
    non_dominated = [sorted_pts[0]]
    for pt in sorted_pts[1:]:
        if pt[1] < non_dominated[-1][1]:
            non_dominated.append(pt)
    return np.array(non_dominated)

# --- 4. MULTI-OBJECTIVE INDICATORS ---

def compute_hypervolume(front, ref_point):
    if len(front) == 0:
        return 0.0
    valid_front = front[(front[:, 0] <= ref_point[0]) & (front[:, 1] <= ref_point[1])]
    if len(valid_front) == 0:
        return 0.0
    valid_front = valid_front[np.argsort(valid_front[:, 0])]
    
    hv = (ref_point[0] - valid_front[0, 0]) * (ref_point[1] - valid_front[0, 1])
    for i in range(1, len(valid_front)):
        hv += (ref_point[0] - valid_front[i, 0]) * (valid_front[i-1, 1] - valid_front[i, 1])
    return hv

def compute_igd_plus(front, ref_front):
    if len(front) == 0 or len(ref_front) == 0:
        return 1e6
    sum_dist = 0.0
    for v in ref_front:
        dists = np.sqrt(np.sum(np.maximum(front - v, 0.0)**2, axis=1))
        sum_dist += np.min(dists)
    return sum_dist / len(ref_front)

def compute_spread(front, ref_front):
    if len(front) == 0:
        return 1.0
    if len(front) == 1:
        return 1.0
    
    front = front[np.argsort(front[:, 0])]
    d = np.sqrt(np.sum(np.diff(front, axis=0)**2, axis=1))
    d_bar = np.mean(d)
    
    ref_front = ref_front[np.argsort(ref_front[:, 0])]
    e1 = ref_front[0]
    e2 = ref_front[-1]
    
    df = np.sqrt(np.sum((front[0] - e1)**2))
    dl = np.sqrt(np.sum((front[-1] - e2)**2))
    
    numerator = df + dl + np.sum(np.abs(d - d_bar))
    denominator = df + dl + len(d) * d_bar
    if denominator == 0:
        return 0.0
    return numerator / denominator

def compute_epsilon_indicator(front, ref_front):
    if len(front) == 0 or len(ref_front) == 0:
        return 1e6
    eps_vals = []
    for v in ref_front:
        shift = np.max(front - v, axis=1)
        eps_vals.append(np.min(shift))
    return np.max(eps_vals)

# --- 5. INITIALIZATION HEURISTICS ---

def heuristic_seeding(inst, N_seed, rng):
    I, J = inst['I'], inst['J']
    R_mean = np.mean(inst['R'], axis=0)
    Ca_mean = np.mean(inst['Ca'], axis=0)
    C_mean = np.mean(inst['C'], axis=0)
    W_mean = np.mean(inst['W'], axis=0)
    
    dec = Decoder(inst)
    
    pop_a = []
    pop_rho = []
    
    for s_idx in range(N_seed):
        noise = rng.normal(0, 0.01, (I, J))
        C_coll = inst['T_ij'] + C_mean + W_mean + noise
        
        sorted_zones = np.argsort(-R_mean)
        a = -1 * np.ones(I, dtype=int)
        cap_used = np.zeros(J)
        
        for i in sorted_zones:
            assigned = False
            sorted_centers = np.argsort(C_coll[i])
            for j in sorted_centers:
                if cap_used[j] + R_mean[i] <= Ca_mean[j]:
                    a[i] = j
                    cap_used[j] += R_mean[i]
                    assigned = True
                    break
            if not assigned:
                a[i] = sorted_centers[0]
                cap_used[sorted_centers[0]] += R_mean[i]
                
        C_mean_i = np.zeros(I)
        for i in range(I):
            j = a[i]
            C_mean_i[i] = (inst['T_ij'][i, j] + C_mean[j] + W_mean[j] + 
                           inst['beta_K'] * dec.T_j_kstar[j] + 
                           inst['beta_S'] * dec.rt_cost_star[i, j] + 
                           inst['beta_R'] * dec.T_j_rstar[j] + 
                           inst['beta_R'] * inst['alpha'] * dec.T_rx_d_star[dec.r_star[j]])
            
        ratio = inst['phi_1'] / (C_mean_i + 1e-6)
        r_min, r_max = np.min(ratio), np.max(ratio)
        if r_max > r_min:
            ratio_norm = 0.1 + 0.8 * (ratio - r_min) / (r_max - r_min)
        else:
            ratio_norm = 0.5 * np.ones(I)
            
        alpha_seed = 1.0 - s_idx / max(1, N_seed - 1)
        rho = alpha_seed * ratio_norm + (1.0 - alpha_seed) * 1.0
        rho = np.clip(rho, 0.0, 1.0)
        
        pop_a.append(a)
        pop_rho.append(rho)
        
    return np.array(pop_a), np.array(pop_rho)

# --- 6. METAHEURISTIC OPERATORS ---

def crossover_two_point(p1_a, p1_rho, p2_a, p2_rho, pc, rng):
    I = len(p1_a)
    if rng.random() > pc:
        return p1_a.copy(), p1_rho.copy(), p2_a.copy(), p2_rho.copy()
    
    pt1, pt2 = sorted(rng.choice(I, 2, replace=False))
    
    c1_a = p1_a.copy()
    c1_rho = p1_rho.copy()
    c2_a = p2_a.copy()
    c2_rho = p2_rho.copy()
    
    c1_a[pt1:pt2] = p2_a[pt1:pt2]
    c1_rho[pt1:pt2] = p2_rho[pt1:pt2]
    c2_a[pt1:pt2] = p1_a[pt1:pt2]
    c2_rho[pt1:pt2] = p1_rho[pt1:pt2]
    
    return c1_a, c1_rho, c2_a, c2_rho

def mutate_creep(a, rho, pm, sigma, J, rng):
    a_mut = a.copy()
    mask_a = rng.random(len(a)) < pm
    if np.any(mask_a):
        a_mut[mask_a] = rng.integers(0, J, size=np.sum(mask_a))
        
    rho_mut = rho.copy()
    mask_rho = rng.random(len(rho)) < pm
    if np.any(mask_rho):
        perturb = rng.uniform(-sigma, sigma, size=np.sum(mask_rho))
        rho_mut[mask_rho] = np.clip(rho_mut[mask_rho] + perturb, 0.0, 1.0)
        
    return a_mut, rho_mut

def tournament_selection(pop_a, pop_rho, Z1, Z2, ranks, cd, rng):
    N = len(pop_a)
    i1, i2 = rng.choice(N, 2, replace=False)
    
    if ranks[i1] < ranks[i2]:
        return pop_a[i1], pop_rho[i1]
    elif ranks[i2] < ranks[i1]:
        return pop_a[i2], pop_rho[i2]
    else:
        if cd[i1] > cd[i2]:
            return pop_a[i1], pop_rho[i1]
        else:
            return pop_a[i2], pop_rho[i2]

def fast_non_dominated_sort(Z1, Z2):
    N = len(Z1)
    z1_i = Z1[:, np.newaxis]
    z1_j = Z1[np.newaxis, :]
    z2_i = Z2[:, np.newaxis]
    z2_j = Z2[np.newaxis, :]
    
    cond1 = (z1_i <= z1_j) & (z2_i <= z2_j)
    cond2 = (z1_i < z1_j) | (z2_i < z2_j)
    dom = cond1 & cond2 # True if i dominates j
    
    dom_count = np.sum(dom, axis=0) # number of individuals that dominate j
    
    ranks = -1 * np.ones(N, dtype=int)
    fronts = []
    curr_dom_count = dom_count.copy()
    remaining = np.arange(N)
    
    while len(remaining) > 0:
        f = remaining[curr_dom_count[remaining] == 0]
        if len(f) == 0:
            break
        ranks[f] = len(fronts)
        fronts.append(f)
        curr_dom_count -= np.sum(dom[f], axis=0)
        remaining = np.setdiff1d(remaining, f)
        
    fronts = [np.array(f, dtype=int) for f in fronts]
    return ranks, fronts

def calculate_crowding_distance(Z1, Z2, fronts):
    N = len(Z1)
    cd = np.zeros(N)
    for front in fronts:
        if len(front) <= 2:
            cd[front] = np.inf
            continue
            
        cd[front] = 0.0
        idx_1 = np.argsort(Z1[front])
        cd[front[idx_1[0]]] = np.inf
        cd[front[idx_1[-1]]] = np.inf
        z1_range = Z1[front[idx_1[-1]]] - Z1[front[idx_1[0]]]
        if z1_range > 0:
            for i in range(1, len(front) - 1):
                cd[front[idx_1[i]]] += (Z1[front[idx_1[i+1]]] - Z1[front[idx_1[i-1]]]) / z1_range
                
        idx_2 = np.argsort(Z2[front])
        cd[front[idx_2[0]]] = np.inf
        cd[front[idx_2[-1]]] = np.inf
        z2_range = Z2[front[idx_2[-1]]] - Z2[front[idx_2[0]]]
        if z2_range > 0:
            for i in range(1, len(front) - 1):
                cd[front[idx_2[i]]] += (Z2[front[idx_2[i+1]]] - Z2[front[idx_2[i-1]]]) / z2_range
                
    return cd

# --- 8. RUN ALGORITHMS ---

def run_nsga2_base(inst, is_enhanced=True, ablation=None, seed=42):
    N = 100
    T_eval = 20000
    pc = 0.9
    
    I, J = inst['I'], inst['J']
    dec = Decoder(inst)
    R, Ca = inst['R'], inst['Ca']
    C_unit, delta = dec.precompute_scenario_costs(R, Ca, inst['C'], inst['W'], inst['t'])
    
    rng = np.random.default_rng(seed)
    
    eta = 0.3
    if not is_enhanced or ablation == 'no-init':
        eta = 0.0
    elif ablation == 'eta-0.1':
        eta = 0.1
    elif ablation == 'eta-0.5':
        eta = 0.5
        
    N_seed = int(np.ceil(eta * N))
    N_rand = N - N_seed
    
    if N_seed > 0:
        pop_a, pop_rho = heuristic_seeding(inst, N_seed, rng)
        if N_rand > 0:
            rand_a = rng.integers(0, J, (N_rand, I))
            rand_rho = rng.uniform(0.0, 1.0, (N_rand, I))
            pop_a = np.vstack([pop_a, rand_a])
            pop_rho = np.vstack([pop_rho, rand_rho])
    else:
        pop_a = rng.integers(0, J, (N, I))
        pop_rho = rng.uniform(0.0, 1.0, (N, I))
        
    Z1, Z2 = dec.decode(pop_a, pop_rho, R, Ca, C_unit, delta)
    eval_count = N
    
    ranks, fronts = fast_non_dominated_sort(Z1, Z2)
    cd = calculate_crowding_distance(Z1, Z2, fronts)
    
    s_stall = 0
    h_best = -1.0
    
    worst_z1 = 1.1 * np.max(Z1)
    worst_z2 = 1.1 * np.max(Z2)
    ref_pt = np.array([worst_z1, worst_z2])
    
    p_min, p_max = 0.05, 0.30
    sigma_0 = 0.2
    g_s = 10
    
    t = 0
    T_max_gen = T_eval // N - 1
    
    spread_max = 0.0
    
    while eval_count < T_eval:
        f1_idx = fronts[0]
        f1_cd = cd[f1_idx]
        finite_cd = f1_cd[f1_cd < np.inf]
        if len(finite_cd) > 0:
            cd_avg = np.mean(finite_cd)
        else:
            cd_avg = 0.0
            
        spread_max = max(spread_max, cd_avg)
        delta_t = cd_avg / spread_max if spread_max > 0 else 0.5
        
        f1_objs = np.column_stack([Z1[f1_idx], Z2[f1_idx]])
        h_val = compute_hypervolume(f1_objs, ref_pt)
        if h_val > h_best:
            h_best = h_val
            s_stall = 0
        else:
            s_stall += 1
            
        if is_enhanced and ablation != 'no-adapt':
            pm = p_min + (p_max - p_min) * (1.0 - delta_t)
            boost = 2.0 if s_stall >= g_s else 1.0
            sigma = sigma_0 * (1.0 - t / T_max_gen) * boost
        else:
            pm = 0.1 if ablation == 'no-adapt' else 1.0 / (2 * I)
            sigma = sigma_0 * (1.0 - t / T_max_gen)
            
        off_a = []
        off_rho = []
        for _ in range(N // 2):
            p1_a, p1_rho = tournament_selection(pop_a, pop_rho, Z1, Z2, ranks, cd, rng)
            p2_a, p2_rho = tournament_selection(pop_a, pop_rho, Z1, Z2, ranks, cd, rng)
            
            c1_a, c1_rho, c2_a, c2_rho = crossover_two_point(p1_a, p1_rho, p2_a, p2_rho, pc, rng)
            
            c1_a, c1_rho = mutate_creep(c1_a, c1_rho, pm, sigma, J, rng)
            c2_a, c2_rho = mutate_creep(c2_a, c2_rho, pm, sigma, J, rng)
            
            off_a.append(c1_a)
            off_rho.append(c1_rho)
            off_a.append(c2_a)
            off_rho.append(c2_rho)
            
        off_a = np.array(off_a)
        off_rho = np.array(off_rho)
        
        o_Z1, o_Z2 = dec.decode(off_a, off_rho, R, Ca, C_unit, delta)
        eval_count += N
        
        comb_a = np.vstack([pop_a, off_a])
        comb_rho = np.vstack([pop_rho, off_rho])
        comb_Z1 = np.concatenate([Z1, o_Z1])
        comb_Z2 = np.concatenate([Z2, o_Z2])
        
        ranks, fronts = fast_non_dominated_sort(comb_Z1, comb_Z2)
        cd = calculate_crowding_distance(comb_Z1, comb_Z2, fronts)
        
        next_pop_idx = []
        for front in fronts:
            if len(next_pop_idx) + len(front) <= N:
                next_pop_idx.extend(front)
            else:
                rem = N - len(next_pop_idx)
                front_sorted = front[np.argsort(-cd[front])]
                next_pop_idx.extend(front_sorted[:rem])
                break
                
        pop_a = comb_a[next_pop_idx]
        pop_rho = comb_rho[next_pop_idx]
        Z1 = comb_Z1[next_pop_idx]
        Z2 = comb_Z2[next_pop_idx]
        
        ranks, fronts = fast_non_dominated_sort(Z1, Z2)
        cd = calculate_crowding_distance(Z1, Z2, fronts)
        
        t += 1
        
    f1_idx = fronts[0]
    final_front = np.column_stack([Z1[f1_idx], Z2[f1_idx]])
    final_front = np.unique(final_front, axis=0)
    final_front = filter_non_dominated(final_front)
    
    final_a = pop_a[f1_idx]
    final_rho = pop_rho[f1_idx]
    
    return final_front, final_a, final_rho

def run_moead_de(inst, seed=42):
    N = 100
    T_eval = 20000
    T_neigh = 20
    delta = 0.9
    n_r = 2
    F = 0.5
    
    I, J = inst['I'], inst['J']
    dec = Decoder(inst)
    R, Ca = inst['R'], inst['Ca']
    C_unit, delta_lateness = dec.precompute_scenario_costs(R, Ca, inst['C'], inst['W'], inst['t'])
    
    rng = np.random.default_rng(seed)
    
    lambdas = []
    for i in range(N):
        val = i / (N - 1)
        lambdas.append(np.array([1.0 - val, val]))
    lambdas = np.array(lambdas)
    
    dists = np.sqrt(np.sum((lambdas[:, np.newaxis, :] - lambdas[np.newaxis, :, :])**2, axis=2))
    neighbors = np.argsort(dists, axis=1)[:, :T_neigh]
    
    pop_a = rng.integers(0, J, (N, I))
    pop_rho = rng.uniform(0.0, 1.0, (N, I))
    
    Z1, Z2 = dec.decode(pop_a, pop_rho, R, Ca, C_unit, delta_lateness)
    eval_count = N
    
    z = np.array([np.min(Z1), np.min(Z2)])
    pm = 1.0 / I
    
    while eval_count < T_eval:
        off_a = []
        off_rho = []
        for i in range(N):
            if rng.random() < delta:
                pool = neighbors[i]
            else:
                pool = np.arange(N)
            r1, r2, r3 = rng.choice(pool, 3, replace=False)
            v_rho = pop_rho[r1] + F * (pop_rho[r2] - pop_rho[r3])
            v_rho = np.clip(v_rho, 0.0, 1.0)
            v_a = pop_a[r1].copy()
            mask = rng.random(I) < pm
            if np.any(mask):
                v_a[mask] = rng.integers(0, J, size=np.sum(mask))
            off_a.append(v_a)
            off_rho.append(v_rho)
            
        off_a = np.array(off_a)
        off_rho = np.array(off_rho)
        o_Z1, o_Z2 = dec.decode(off_a, off_rho, R, Ca, C_unit, delta_lateness)
        eval_count += N
        
        z[0] = min(z[0], np.min(o_Z1))
        z[1] = min(z[1], np.min(o_Z2))
        
        for i in range(N):
            o_z1_val = o_Z1[i]
            o_z2_val = o_Z2[i]
            v_a = off_a[i]
            v_rho = off_rho[i]
            
            if rng.random() < delta:
                pool = neighbors[i]
            else:
                pool = np.arange(N)
                
            indices = pool.copy()
            rng.shuffle(indices)
            
            replaced_count = 0
            for idx in indices:
                if replaced_count >= n_r:
                    break
                g_curr = max(lambdas[idx, 0] * abs(Z1[idx] - z[0]), lambdas[idx, 1] * abs(Z2[idx] - z[1]))
                g_off = max(lambdas[idx, 0] * abs(o_z1_val - z[0]), lambdas[idx, 1] * abs(o_z2_val - z[1]))
                
                if g_off < g_curr:
                    pop_a[idx] = v_a
                    pop_rho[idx] = v_rho
                    Z1[idx] = o_z1_val
                    Z2[idx] = o_z2_val
                    replaced_count += 1
                    
    objs = np.column_stack([Z1, Z2])
    front = filter_non_dominated(objs)
    return front, pop_a, pop_rho

def run_sms_emoa(inst, seed=42):
    N = 100
    T_eval = 20000
    pc = 0.9
    
    I, J = inst['I'], inst['J']
    dec = Decoder(inst)
    R, Ca = inst['R'], inst['Ca']
    C_unit, delta_lateness = dec.precompute_scenario_costs(R, Ca, inst['C'], inst['W'], inst['t'])
    
    rng = np.random.default_rng(seed)
    
    pop_a = rng.integers(0, J, (N, I))
    pop_rho = rng.uniform(0.0, 1.0, (N, I))
    
    Z1, Z2 = dec.decode(pop_a, pop_rho, R, Ca, C_unit, delta_lateness)
    eval_count = N
    
    worst_z1 = 1.1 * np.max(Z1)
    worst_z2 = 1.1 * np.max(Z2)
    ref_pt = np.array([worst_z1, worst_z2])
    
    ranks, fronts = fast_non_dominated_sort(Z1, Z2)
    cd = calculate_crowding_distance(Z1, Z2, fronts)
    
    sigma_0 = 0.2
    T_max_gen = T_eval - N
    pm = 1.0 / (2 * I)
    
    while eval_count < T_eval:
        t = eval_count - N
        sigma = sigma_0 * (1.0 - t / T_max_gen)
        
        p1_a, p1_rho = tournament_selection(pop_a, pop_rho, Z1, Z2, ranks, cd, rng)
        p2_a, p2_rho = tournament_selection(pop_a, pop_rho, Z1, Z2, ranks, cd, rng)
        
        c1_a, c1_rho, _, _ = crossover_two_point(p1_a, p1_rho, p2_a, p2_rho, pc, rng)
        c1_a, c1_rho = mutate_creep(c1_a, c1_rho, pm, sigma, J, rng)
        
        o_Z1, o_Z2 = dec.decode(c1_a[np.newaxis, :], c1_rho[np.newaxis, :], R, Ca, C_unit, delta_lateness)
        eval_count += 1
        o_Z1, o_Z2 = o_Z1[0], o_Z2[0]
        
        comb_a = np.vstack([pop_a, c1_a])
        comb_rho = np.vstack([pop_rho, c1_rho])
        comb_Z1 = np.concatenate([Z1, [o_Z1]])
        comb_Z2 = np.concatenate([Z2, [o_Z2]])
        
        ranks, fronts = fast_non_dominated_sort(comb_Z1, comb_Z2)
        worst_front_idx = fronts[-1]
        
        if len(worst_front_idx) == 1:
            discard_idx = worst_front_idx[0]
        else:
            front_objs = np.column_stack([comb_Z1[worst_front_idx], comb_Z2[worst_front_idx]])
            sorted_f_idx = np.argsort(front_objs[:, 0])
            
            m = len(worst_front_idx)
            contr = np.zeros(m)
            
            contr[sorted_f_idx[0]] = (front_objs[sorted_f_idx[1], 0] - front_objs[sorted_f_idx[0], 0]) * (ref_pt[1] - front_objs[sorted_f_idx[0], 1])
            for i in range(1, m - 1):
                contr[sorted_f_idx[i]] = (front_objs[sorted_f_idx[i+1], 0] - front_objs[sorted_f_idx[i], 0]) * (front_objs[sorted_f_idx[i-1], 1] - front_objs[sorted_f_idx[i], 1])
            contr[sorted_f_idx[-1]] = (ref_pt[0] - front_objs[sorted_f_idx[-1], 0]) * (front_objs[sorted_f_idx[-2], 1] - front_objs[sorted_f_idx[-1], 1])
            
            min_contr_f_idx = np.argmin(contr)
            discard_idx = worst_front_idx[min_contr_f_idx]
            
        keep_mask = np.ones(N + 1, dtype=bool)
        keep_mask[discard_idx] = False
        
        pop_a = comb_a[keep_mask]
        pop_rho = comb_rho[keep_mask]
        Z1 = comb_Z1[keep_mask]
        Z2 = comb_Z2[keep_mask]
        
        ranks, fronts = fast_non_dominated_sort(Z1, Z2)
        cd = calculate_crowding_distance(Z1, Z2, fronts)
        
    f1_idx = fronts[0]
    front = np.column_stack([Z1[f1_idx], Z2[f1_idx]])
    front = np.unique(front, axis=0)
    front = filter_non_dominated(front)
    return front, pop_a, pop_rho

def run_spea2(inst, seed=42):
    N = 100
    N_arc = 100
    T_eval = 20000
    pc = 0.9
    
    I, J = inst['I'], inst['J']
    dec = Decoder(inst)
    R, Ca = inst['R'], inst['Ca']
    C_unit, delta_lateness = dec.precompute_scenario_costs(R, Ca, inst['C'], inst['W'], inst['t'])
    
    rng = np.random.default_rng(seed)
    
    pop_a = rng.integers(0, J, (N, I))
    pop_rho = rng.uniform(0.0, 1.0, (N, I))
    
    Z1, Z2 = dec.decode(pop_a, pop_rho, R, Ca, C_unit, delta_lateness)
    eval_count = N
    
    arc_a = np.zeros((0, I), dtype=int)
    arc_rho = np.zeros((0, I))
    arc_Z1 = np.zeros(0)
    arc_Z2 = np.zeros(0)
    
    sigma_0 = 0.2
    T_max_gen = T_eval // N - 1
    t_gen = 0
    pm = 1.0 / (2 * I)
    
    while eval_count < T_eval:
        sigma = sigma_0 * (1.0 - t_gen / T_max_gen)
        
        comb_a = np.vstack([pop_a, arc_a])
        comb_rho = np.vstack([pop_rho, arc_rho])
        comb_Z1 = np.concatenate([Z1, arc_Z1])
        comb_Z2 = np.concatenate([Z2, arc_Z2])
        
        M_total = len(comb_Z1)
        
        z1_i = comb_Z1[:, np.newaxis]
        z1_j = comb_Z1[np.newaxis, :]
        z2_i = comb_Z2[:, np.newaxis]
        z2_j = comb_Z2[np.newaxis, :]
        
        cond1 = (z1_i <= z1_j) & (z2_i <= z2_j)
        cond2 = (z1_i < z1_j) | (z2_i < z2_j)
        dom = cond1 & cond2
        
        S = np.sum(dom, axis=1)
        raw_F = np.sum(dom * S[:, np.newaxis], axis=0)
                    
        objs = np.column_stack([comb_Z1, comb_Z2])
        dists = np.sqrt(np.sum((objs[:, np.newaxis, :] - objs[np.newaxis, :, :])**2, axis=2))
        
        k = int(np.floor(np.sqrt(M_total)))
        sorted_dists = np.sort(dists, axis=1)
        density = 1.0 / (sorted_dists[:, k] + 2.0)
        
        fitness = raw_F + density
        
        non_dom_idx = np.where(raw_F == 0)[0]
        
        if len(non_dom_idx) == N_arc:
            arc_idx = non_dom_idx
        elif len(non_dom_idx) < N_arc:
            dom_idx = np.where(raw_F > 0)[0]
            sorted_dom = dom_idx[np.argsort(fitness[dom_idx])]
            rem = N_arc - len(non_dom_idx)
            arc_idx = np.concatenate([non_dom_idx, sorted_dom[:rem]])
        else:
            curr_arc = list(non_dom_idx)
            while len(curr_arc) > N_arc:
                curr_dists = dists[curr_arc][:, curr_arc]
                sorted_curr_dists = np.sort(curr_dists, axis=1)
                
                min_val_idx = 0
                for rank in range(1, len(curr_arc)):
                    col_vals = sorted_curr_dists[:, rank]
                    min_val = np.min(col_vals)
                    ties = np.where(col_vals == min_val)[0]
                    if len(ties) == 1:
                        min_val_idx = ties[0]
                        break
                curr_arc.pop(min_val_idx)
            arc_idx = np.array(curr_arc)
            
        arc_a = comb_a[arc_idx]
        arc_rho = comb_rho[arc_idx]
        arc_Z1 = comb_Z1[arc_idx]
        arc_Z2 = comb_Z2[arc_idx]
        arc_fit = fitness[arc_idx]
        
        pop_a_new = []
        pop_rho_new = []
        for _ in range(N // 2):
            i1, i2 = rng.choice(N_arc, 2, replace=False)
            p1_idx = i1 if arc_fit[i1] < arc_fit[i2] else i2
            
            i1, i2 = rng.choice(N_arc, 2, replace=False)
            p2_idx = i1 if arc_fit[i1] < arc_fit[i2] else i2
            
            c1_a, c1_rho, c2_a, c2_rho = crossover_two_point(arc_a[p1_idx], arc_rho[p1_idx],
                                                             arc_a[p2_idx], arc_rho[p2_idx], pc, rng)
            c1_a, c1_rho = mutate_creep(c1_a, c1_rho, pm, sigma, J, rng)
            c2_a, c2_rho = mutate_creep(c2_a, c2_rho, pm, sigma, J, rng)
            
            pop_a_new.append(c1_a)
            pop_rho_new.append(c1_rho)
            pop_a_new.append(c2_a)
            pop_rho_new.append(c2_rho)
            
        pop_a = np.array(pop_a_new)
        pop_rho = np.array(pop_rho_new)
        
        Z1, Z2 = dec.decode(pop_a, pop_rho, R, Ca, C_unit, delta_lateness)
        eval_count += N
        t_gen += 1
        
    final_objs = np.column_stack([arc_Z1, arc_Z2])
    final_front = filter_non_dominated(final_objs)
    return final_front, arc_a, arc_rho

# --- 9. PIPELINE COORDINATORS ---

def run_replication(args):
    instance_id, algo_name, rep_id, seed = args
    inst = generate_instance(instance_id)
    
    t0 = time.time()
    if algo_name == 'E-NSGA-II':
        front, pop_a, pop_rho = run_nsga2_base(inst, is_enhanced=True, seed=seed)
    elif algo_name == 'NSGA-II':
        front, pop_a, pop_rho = run_nsga2_base(inst, is_enhanced=False, seed=seed)
    elif algo_name == 'MOEA/D-DE':
        front, pop_a, pop_rho = run_moead_de(inst, seed=seed)
    elif algo_name == 'SMS-EMOA':
        front, pop_a, pop_rho = run_sms_emoa(inst, seed=seed)
    elif algo_name == 'SPEA2':
        front, pop_a, pop_rho = run_spea2(inst, seed=seed)
    else:
        raise ValueError(f"Unknown algorithm {algo_name}")
    runtime = time.time() - t0
    
    return {
        'instance_id': instance_id,
        'algo_name': algo_name,
        'rep_id': rep_id,
        'front': front,
        'runtime': runtime
    }

def run_exact_solver(instance_id):
    inst = generate_instance(instance_id)
    t0 = time.time()
    front = solve_exact(inst, grid_size=21)
    runtime = time.time() - t0
    return {
        'instance_id': instance_id,
        'front': front,
        'runtime': runtime
    }

def run_ablation_replication(args):
    instance_id, variant_name, rep_id, seed = args
    inst = generate_instance(instance_id)
    
    t0 = time.time()
    if variant_name == 'E-NSGA-II':
        front, _, _ = run_nsga2_base(inst, is_enhanced=True, seed=seed)
    elif variant_name == '--Init':
        front, _, _ = run_nsga2_base(inst, is_enhanced=True, ablation='no-init', seed=seed)
    elif variant_name == '--Adapt':
        front, _, _ = run_nsga2_base(inst, is_enhanced=True, ablation='no-adapt', seed=seed)
    elif variant_name == 'Plain NSGA-II':
        front, _, _ = run_nsga2_base(inst, is_enhanced=False, seed=seed)
    else:
        raise ValueError(f"Unknown variant {variant_name}")
    runtime = time.time() - t0
    
    return {
        'instance_id': instance_id,
        'variant_name': variant_name,
        'rep_id': rep_id,
        'front': front,
        'runtime': runtime
    }

def run_robustness_replication(args):
    instance_id, dist_type, algo_name, rep_id, seed = args
    inst = generate_instance(instance_id, dist_type=dist_type)
    
    t0 = time.time()
    if algo_name == 'E-NSGA-II':
        front, _, _ = run_nsga2_base(inst, is_enhanced=True, seed=seed)
    elif algo_name == 'MOEA/D-DE':
        front, _, _ = run_moead_de(inst, seed=seed)
    elif algo_name == 'SMS-EMOA':
        front, _, _ = run_sms_emoa(inst, seed=seed)
    elif algo_name == 'SPEA2':
        front, _, _ = run_spea2(inst, seed=seed)
    runtime = time.time() - t0
    
    return {
        'instance_id': instance_id,
        'dist_type': dist_type,
        'algo_name': algo_name,
        'rep_id': rep_id,
        'front': front,
        'runtime': runtime
    }

def select_knee_point(front):
    if len(front) <= 2:
        return front[0], 0
    
    p1 = front[0]
    p2 = front[-1]
    
    A = p2[1] - p1[1]
    B = -(p2[0] - p1[0])
    C = (p2[0] - p1[0]) * p1[1] - (p2[1] - p1[1]) * p1[0]
    
    dist = np.abs(A * front[:, 0] + B * front[:, 1] + C) / np.sqrt(A**2 + B**2)
    knee_idx = np.argmax(dist)
    return front[knee_idx], knee_idx

def select_topsis(front, weights):
    norm = np.sqrt(np.sum(front**2, axis=0))
    norm_front = front / norm
    
    w_norm_front = norm_front * weights
    
    ideal_best = np.min(w_norm_front, axis=0)
    ideal_worst = np.max(w_norm_front, axis=0)
    
    d_best = np.sqrt(np.sum((w_norm_front - ideal_best)**2, axis=1))
    d_worst = np.sqrt(np.sum((w_norm_front - ideal_worst)**2, axis=1))
    
    closeness = d_worst / (d_best + d_worst + 1e-6)
    topsis_idx = np.argmax(closeness)
    return front[topsis_idx], topsis_idx

def main():
    print("Starting Reverse Logistics Computational Study...")
    
    # folder for the result files: first command-line argument, default ./results_rerun
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "results_rerun"
    os.makedirs(out_dir, exist_ok=True)
    
    n_cores = max(1, mp.cpu_count() - 1)
    print(f"Parallelizing runs on {n_cores} cores.")
    
    # STEP 1: VALIDATION AGAINST EXACT
    print("Running exact solver on Instances 1-3...")
    exact_results = {}
    for inst_id in [1, 2, 3]:
        res = run_exact_solver(inst_id)
        exact_results[inst_id] = res
        print(f"  Instance {inst_id} exact solver done. Front size: {len(res['front']) if res['front'] is not None else 0}, CPU: {res['runtime']:.2f}s")
        
    with open(f"{out_dir}/exact_results.pkl", "wb") as f:
        pickle.dump(exact_results, f)
        
    # STEP 2: MAIN COMPARISON (Instances 1-12)
    print("Running metaheuristics comparison on Instances 1-12...")
    algos = ['E-NSGA-II', 'MOEA/D-DE', 'SMS-EMOA', 'SPEA2']
    instances = list(range(1, 13))
    replications = 30
    
    tasks = []
    rep_seeds = np.random.default_rng(9999).integers(10000, 99999, (12, len(algos), replications))
    
    for i_idx, inst_id in enumerate(instances):
        for a_idx, algo in enumerate(algos):
            for rep in range(replications):
                seed = int(rep_seeds[i_idx, a_idx, rep])
                tasks.append((inst_id, algo, rep, seed))
                
    print(f"Total metaheuristic runs: {len(tasks)}. Launching pool...")
    t_start = time.time()
    with mp.Pool(n_cores) as pool:
        raw_results = pool.map(run_replication, tasks)
    print(f"All comparison runs completed in {(time.time() - t_start)/60.0:.2f} minutes.")
    
    comparison_results = {inst_id: {algo: {} for algo in algos} for inst_id in instances}
    for res in raw_results:
        inst_id = res['instance_id']
        algo = res['algo_name']
        rep = res['rep_id']
        comparison_results[inst_id][algo][rep] = res
        
    with open(f"{out_dir}/comparison_results.pkl", "wb") as f:
        pickle.dump(comparison_results, f)
        
    # STEP 3: ABLATION STUDY (Instance 8)
    print("Running ablation study on Instance 8...")
    variants = ['E-NSGA-II', '--Init', '--Adapt', 'Plain NSGA-II']
    ablation_tasks = []
    ablation_seeds = np.random.default_rng(8888).integers(10000, 99999, (len(variants), replications))
    for v_idx, variant in enumerate(variants):
        for rep in range(replications):
            seed = int(ablation_seeds[v_idx, rep])
            ablation_tasks.append((8, variant, rep, seed))
            
    with mp.Pool(n_cores) as pool:
        raw_ablation = pool.map(run_ablation_replication, ablation_tasks)
        
    ablation_results = {variant: {} for variant in variants}
    for res in raw_ablation:
        variant = res['variant_name']
        rep = res['rep_id']
        ablation_results[variant][rep] = res
        
    with open(f"{out_dir}/ablation_results.pkl", "wb") as f:
        pickle.dump(ablation_results, f)
        
    # STEP 4: OUT-OF-SAMPLE EVALUATION
    print("Running out-of-sample evaluation on Instances 4, 8, 12...")
    oos_results = {}
    for inst_id in [4, 8, 12]:
        inst_data = generate_instance(inst_id)
        dec = Decoder(inst_data)
        
        R, Ca = inst_data['R'], inst_data['Ca']
        C_unit_in, delta_in = dec.precompute_scenario_costs(R, Ca, inst_data['C'], inst_data['W'], inst_data['t'])
        
        R_oos, Ca_oos = inst_data['R_oos'], inst_data['Ca_oos']
        C_unit_oos, delta_oos = dec.precompute_scenario_costs(R_oos, Ca_oos, inst_data['C_oos'], inst_data['W_oos'], inst_data['t_oos'])
        
        reps = comparison_results[inst_id]['E-NSGA-II']
        hvs = []
        all_objs = []
        for algo in algos:
            for rep in range(replications):
                front = comparison_results[inst_id][algo][rep]['front']
                if len(front) > 0:
                    all_objs.extend(front)
        all_objs = np.array(all_objs)
        ref_pt = 1.1 * np.max(all_objs, axis=0)
        
        for rep in range(replications):
            hv = compute_hypervolume(reps[rep]['front'], ref_pt)
            hvs.append((hv, rep))
        hvs.sort()
        median_rep = hvs[replications // 2][1]
        
        seed = int(rep_seeds[inst_id - 1, 0, median_rep])
        _, pop_a, pop_rho = run_nsga2_base(inst_data, is_enhanced=True, seed=seed)
        
        Z1_f1, Z2_f1 = dec.decode(pop_a, pop_rho, R, Ca, C_unit_in, delta_in)
        ranks, fronts = fast_non_dominated_sort(Z1_f1, Z2_f1)
        f1_idx = fronts[0]
        f1_objs = np.column_stack([Z1_f1[f1_idx], Z2_f1[f1_idx]])
        f1_objs_f = filter_non_dominated(f1_objs)
        
        knee_val, knee_idx_in_f1 = select_knee_point(f1_objs_f)
        pop_idx = -1
        for idx in f1_idx:
            if abs(Z1_f1[idx] - knee_val[0]) < 1e-5 and abs(Z2_f1[idx] - knee_val[1]) < 1e-5:
                pop_idx = idx
                break
                
        if pop_idx == -1:
            pop_idx = f1_idx[0]
            
        knee_a = pop_a[pop_idx : pop_idx+1]
        knee_rho = pop_rho[pop_idx : pop_idx+1]
        
        Z1_in, Z2_in = dec.decode(knee_a, knee_rho, R, Ca, C_unit_in, delta_in)
        Z1_in, Z2_in = Z1_in[0], Z2_in[0]
        
        Z1_scens = []
        Z2_scens = []
        for w in range(500):
            R_w = R_oos[w : w+1]
            Ca_w = Ca_oos[w : w+1]
            C_unit_w = C_unit_oos[w : w+1]
            delta_w = delta_oos[w : w+1]
            
            z1, z2 = dec.decode(knee_a, knee_rho, R_w, Ca_w, C_unit_w, delta_w)
            Z1_scens.append(z1[0])
            Z2_scens.append(z2[0])
            
        Z1_scens = np.array(Z1_scens)
        Z2_scens = np.array(Z2_scens)
        
        oos_results[inst_id] = {
            'Z1_in': Z1_in,
            'Z2_in': Z2_in,
            'Z1_oos_mean': np.mean(Z1_scens),
            'Z1_oos_sd': np.std(Z1_scens),
            'Z2_oos_mean': np.mean(Z2_scens),
            'Z2_oos_sd': np.std(Z2_scens),
            'Z1_gap': (np.mean(Z1_scens) - Z1_in) / Z1_in * 100.0,
            'Z2_gap': (np.mean(Z2_scens) - Z2_in) / Z2_in * 100.0
        }
        print(f"  Instance {inst_id} OOS done. Z1 in: {Z1_in:.2f}, Z1 oos: {np.mean(Z1_scens):.2f}")
        
    with open(f"{out_dir}/oos_results.pkl", "wb") as f:
        pickle.dump(oos_results, f)
        
    # STEP 5: DISTRIBUTIONAL ROBUSTNESS
    print("Running distributional robustness on Instances 4, 8, 12...")
    dists_list = ['normal', 'lognormal']
    robustness_tasks = []
    rob_seeds = np.random.default_rng(7777).integers(10000, 99999, (3, len(dists_list), len(algos), replications))
    for i_idx, inst_id in enumerate([4, 8, 12]):
        for d_idx, dist in enumerate(dists_list):
            for a_idx, algo in enumerate(algos):
                for rep in range(replications):
                    seed = int(rob_seeds[i_idx, d_idx, a_idx, rep])
                    robustness_tasks.append((inst_id, dist, algo, rep, seed))
                    
    with mp.Pool(n_cores) as pool:
        raw_robustness = pool.map(run_robustness_replication, robustness_tasks)
        
    robustness_results = {inst_id: {dist: {algo: {} for algo in algos} for dist in dists_list} for inst_id in [4, 8, 12]}
    for res in raw_robustness:
        inst_id = res['instance_id']
        dist = res['dist_type']
        algo = res['algo_name']
        rep = res['rep_id']
        robustness_results[inst_id][dist][algo][rep] = res
        
    with open(f"{out_dir}/robustness_results.pkl", "wb") as f:
        pickle.dump(robustness_results, f)
        
    # STEP 6: SCALABILITY ANALYSIS (Instances 13-15)
    print("Running scalability analysis on Instances 13-15...")
    scalability_tasks = []
    scalability_seeds = np.random.default_rng(6666).integers(10000, 99999, (3, len(algos), replications))
    for i_idx, inst_id in enumerate([13, 14, 15]):
        for a_idx, algo in enumerate(algos):
            for rep in range(replications):
                seed = int(scalability_seeds[i_idx, a_idx, rep])
                scalability_tasks.append((inst_id, algo, rep, seed))
                
    with mp.Pool(n_cores) as pool:
        raw_scalability = pool.map(run_replication, scalability_tasks)
        
    scalability_results = {inst_id: {algo: {} for algo in algos} for inst_id in [13, 14, 15]}
    for res in raw_scalability:
        inst_id = res['instance_id']
        algo = res['algo_name']
        rep = res['rep_id']
        scalability_results[inst_id][algo][rep] = res
        
    with open(f"{out_dir}/scalability_results.pkl", "wb") as f:
        pickle.dump(scalability_results, f)
        
    # STEP 7: SENSITIVITY ANALYSIS & DECISION CLOSURE (Instance 8)
    print("Running sensitivity analysis and decision closure on Instance 8...")
    inst_8 = generate_instance(8)
    dec_8 = Decoder(inst_8)
    
    reps_8 = comparison_results[8]['E-NSGA-II']
    all_objs_8 = []
    for algo in algos:
        for rep in range(replications):
            front = comparison_results[8][algo][rep]['front']
            if len(front) > 0:
                all_objs_8.extend(front)
    all_objs_8 = np.array(all_objs_8)
    ref_pt_8 = 1.1 * np.max(all_objs_8, axis=0)
    
    hvs_8 = []
    for rep in range(replications):
        hv = compute_hypervolume(reps_8[rep]['front'], ref_pt_8)
        hvs_8.append((hv, rep))
    hvs_8.sort()
    median_rep_8 = hvs_8[replications // 2][1]
    
    seed_8 = int(rep_seeds[8 - 1, 0, median_rep_8])
    _, pop_a_8, pop_rho_8 = run_nsga2_base(inst_8, is_enhanced=True, seed=seed_8)
    
    C_unit_8, delta_8 = dec_8.precompute_scenario_costs(inst_8['R'], inst_8['Ca'], inst_8['C'], inst_8['W'], inst_8['t'])
    Z1_pop, Z2_pop = dec_8.decode(pop_a_8, pop_rho_8, inst_8['R'], inst_8['Ca'], C_unit_8, delta_8)
    ranks_pop, fronts_pop = fast_non_dominated_sort(Z1_pop, Z2_pop)
    f1_idx_8 = fronts_pop[0]
    f1_objs_8 = np.column_stack([Z1_pop[f1_idx_8], Z2_pop[f1_idx_8]])
    f1_objs_f_8 = filter_non_dominated(f1_objs_8)
    
    knee_val_8, knee_idx_f1 = select_knee_point(f1_objs_f_8)
    topsis_73, topsis_73_f1 = select_topsis(f1_objs_f_8, np.array([0.7, 0.3]))
    topsis_55, topsis_55_f1 = select_topsis(f1_objs_f_8, np.array([0.5, 0.5]))
    topsis_37, topsis_37_f1 = select_topsis(f1_objs_f_8, np.array([0.3, 0.7]))
    
    def get_closure_stats(val):
        p_idx = -1
        for idx in f1_idx_8:
            if abs(Z1_pop[idx] - val[0]) < 1e-5 and abs(Z2_pop[idx] - val[1]) < 1e-5:
                p_idx = idx
                break
        if p_idx == -1:
            p_idx = f1_idx_8[0]
            
        a_dec = pop_a_8[p_idx : p_idx+1]
        rho_dec = pop_rho_8[p_idx : p_idx+1]
        
        total_R = 0.0
        total_Q = 0.0
        total_U = 0.0
        total_late = 0.0
        
        n_scenarios = inst_8['R'].shape[0]
        for w in range(n_scenarios):
            R_w = inst_8['R'][w]
            Ca_w = inst_8['Ca'][w]
            
            scale_all = np.ones((1, dec_8.J))
            for j in range(dec_8.J):
                mask = (a_dec == j)
                V_j = np.sum(mask * rho_dec * R_w, axis=1)
                nonzero = V_j > 0
                scale_all[nonzero, j] = np.minimum(1.0, Ca_w[j] / V_j[nonzero])
                
            scale_zone = scale_all[0, a_dec[0]]
            Q_w = rho_dec[0] * R_w * scale_zone
            U_w = R_w - Q_w
            
            total_R += np.sum(R_w)
            total_Q += np.sum(Q_w)
            total_U += np.sum(U_w)
            
            del_u = delta_8[w, np.arange(dec_8.I), a_dec[0]]
            total_late += np.sum(dec_8.phi_2 * dec_8.beta_S * del_u * Q_w)
            
        return {
            'obj': val,
            'a': a_dec[0],
            'rho': rho_dec[0],
            'coll_rate': (total_Q / total_R) * 100.0,
            'uncoll': total_U / n_scenarios,
            'late_pen': total_late / n_scenarios
        }
        
    closure_results = {
        'knee': get_closure_stats(knee_val_8),
        'topsis_73': get_closure_stats(topsis_73),
        'topsis_55': get_closure_stats(topsis_55),
        'topsis_37': get_closure_stats(topsis_37),
    }
    
    with open(f"{out_dir}/closure_results.pkl", "wb") as f:
        pickle.dump(closure_results, f)
        
    knee_a = pop_a_8[f1_idx_8[knee_idx_f1]][np.newaxis, :]
    knee_rho = pop_rho_8[f1_idx_8[knee_idx_f1]][np.newaxis, :]
    
    sensitivity_results = {}
    params_to_vary = ['Ca', 'beta', 'alpha', 'phi_1', 'phi_2']
    grid = [-35, -20, 0, 20, 35]
    
    for param in params_to_vary:
        sensitivity_results[param] = {}
        for pct in grid:
            inst_sens = generate_instance(8, param_mod=(param, pct))
            dec_sens = Decoder(inst_sens)
            
            R_sens, Ca_sens = inst_sens['R'], inst_sens['Ca']
            C_unit_sens, delta_sens = dec_sens.precompute_scenario_costs(R_sens, Ca_sens, inst_sens['C'], inst_sens['W'], inst_sens['t'])
            
            n_scenarios = R_sens.shape[0]
            
            trans_tot = 0.0
            insp_tot = 0.0
            stor_tot = 0.0
            disp_tot = 0.0
            pen_tot = 0.0
            z1_tot = 0.0
            z2_tot = 0.0
            
            for w in range(n_scenarios):
                R_w = R_sens[w]
                Ca_w = Ca_sens[w]
                C_unit_w = C_unit_sens[w]
                delta_w = delta_sens[w]
                
                scale_all = np.ones((1, dec_sens.J))
                for j in range(dec_sens.J):
                    mask = (knee_a == j)
                    V_j = np.sum(mask * knee_rho * R_w, axis=1)
                    nonzero = V_j > 0
                    scale_all[nonzero, j] = np.minimum(1.0, Ca_w[j] / V_j[nonzero])
                    
                scale_zone = scale_all[0, knee_a[0]]
                Q_w = knee_rho[0] * R_w * scale_zone
                U_w = R_w - Q_w
                
                F_j = np.zeros(dec_sens.J)
                for j in range(dec_sens.J):
                    F_j[j] = np.sum(Q_w[knee_a[0] == j])
                    
                trans_ij = np.sum(dec_sens.T_ij[np.arange(dec_sens.I), knee_a[0]] * Q_w)
                trans_jk = np.sum(dec_sens.T_j_kstar * dec_sens.beta_K * F_j)
                trans_jr = np.sum(dec_sens.T_j_rstar * dec_sens.beta_R * F_j)
                
                flow_r = np.zeros(dec_sens.R_cnt)
                for j in range(dec_sens.J):
                    flow_r[dec_sens.r_star[j]] += dec_sens.beta_R * F_j[j]
                trans_rx = np.sum(dec_sens.T_rx[np.arange(dec_sens.R_cnt), dec_sens.x_star] * dec_sens.alpha * flow_r)
                
                trans_rep = np.sum(dec_sens.beta_S * dec_sens.rt_cost_star[np.arange(dec_sens.I), knee_a[0]] * Q_w)
                
                trans_w = trans_ij + trans_jk + trans_jr + trans_rx + trans_rep
                insp_w = np.sum(inst_sens['C'][w] * F_j)
                stor_w = np.sum(inst_sens['W'][w] * F_j)
                disp_w = np.sum(dec_sens.d_x[dec_sens.x_star] * dec_sens.alpha * flow_r)
                
                del_u = delta_w[np.arange(dec_sens.I), knee_a[0]]
                pen_w = dec_sens.phi_1 * np.sum(U_w) + dec_sens.phi_2 * dec_sens.beta_S * np.sum(del_u * Q_w)
                
                trans_tot += trans_w
                insp_tot += insp_w
                stor_tot += stor_w
                disp_tot += disp_w
                pen_tot += pen_w
                
                z1_w, z2_w = dec_sens.decode(knee_a, knee_rho, R_w[np.newaxis, :], Ca_w[np.newaxis, :], C_unit_sens[w:w+1], delta_sens[w:w+1])
                z1_tot += z1_w[0]
                z2_tot += z2_w[0]
                
            sensitivity_results[param][pct] = {
                'trans': trans_tot / n_scenarios,
                'insp': insp_tot / n_scenarios,
                'stor': stor_tot / n_scenarios,
                'disp': disp_tot / n_scenarios,
                'pen': pen_tot / n_scenarios,
                'Z1': z1_tot / n_scenarios,
                'Z2': z2_tot / n_scenarios
            }
            
    with open(f"{out_dir}/sensitivity_results.pkl", "wb") as f:
        pickle.dump(sensitivity_results, f)
        
    print("Computational Study finished successfully and results are pickled.")

if __name__ == '__main__':
    main()
