"""Sparse epsilon-constraint solver for the scenario-expanded model (same model as study.solve_exact).

Variables: y_ij (binary), Q_ij^w >= 0.  Only the model of study.py: cost-based downstream routing.
Each subproblem: min Z1  s.t.  Z2 <= eps, with a time limit, and the incumbent, the best bound
and the relative MIP gap are recorded whether or not the limit is hit.
"""
import sys, time, pickle
import numpy as np
import scipy.sparse as sp
import scipy.optimize as opt


def build(st, inst):
    I, J, R, Ca = inst['I'], inst['J'], inst['R'], inst['Ca']
    W = R.shape[0]
    dec = st.Decoder(inst)
    C_unit, delta = dec.precompute_scenario_costs(R, Ca, inst['C'], inst['W'], inst['t'])
    ny, nq = I * J, I * J * W
    n = ny + nq
    c1 = np.zeros(n)
    c2 = np.zeros(n)
    c1[ny:] = (C_unit / W).reshape(-1)                       # order (w,i,j)
    c2[ny:] = ((dec.phi_2 * dec.beta_S * delta - dec.phi_1) / W).reshape(-1)
    const2 = dec.phi_1 * R.sum(axis=1).mean()
    # sum_j y_ij = 1
    rows = np.repeat(np.arange(I), J)
    A1 = sp.csr_matrix((np.ones(ny), (rows, np.arange(ny))), shape=(I, n))
    # Q_ij^w - R_i^w y_ij <= 0
    r = np.arange(nq)
    qcol = ny + r
    ycol = np.tile(np.arange(ny), W)
    Rv = np.repeat(R, J, axis=1).reshape(-1)                 # (w,i,j) -> R[w,i]
    A2 = sp.csr_matrix((np.concatenate([np.ones(nq), -Rv]),
                        (np.concatenate([r, r]), np.concatenate([qcol, ycol]))), shape=(nq, n))
    # sum_i Q_ij^w <= Ca_j^w
    w_idx = np.repeat(np.arange(W), I * J)
    j_idx = np.tile(np.tile(np.arange(J), I), W)
    A3 = sp.csr_matrix((np.ones(nq), (w_idx * J + j_idx, qcol)), shape=(W * J, n))
    A = sp.vstack([A1, A2, A3]).tocsr()
    lo = np.concatenate([np.ones(I), -np.inf * np.ones(nq + W * J)])
    hi = np.concatenate([np.ones(I), np.zeros(nq), Ca.reshape(-1)])
    integ = np.zeros(n)
    integ[:ny] = 1
    ub = np.full(n, np.inf)
    ub[:ny] = 1.0
    return dict(A=A, lo=lo, hi=hi, c1=c1, c2=c2, const2=const2, integ=integ,
                bounds=opt.Bounds(np.zeros(n), ub), n=n)


def solve(m, obj, eps=None, time_limit=None, gap=1e-4):
    A, lo, hi = m['A'], m['lo'], m['hi']
    if eps is not None:
        A = sp.vstack([A, sp.csr_matrix(m['c2'])]).tocsr()
        lo = np.append(lo, -np.inf)
        hi = np.append(hi, eps - m['const2'])
    o = {'mip_rel_gap': gap, 'disp': False}
    if time_limit:
        o['time_limit'] = time_limit
    t0 = time.time()
    r = opt.milp(c=obj, integrality=m['integ'], bounds=m['bounds'],
                 constraints=opt.LinearConstraint(A, lo, hi), options=o)
    out = dict(status=int(r.status), time=time.time() - t0, x=r.x,
               bound=getattr(r, 'mip_dual_bound', None), gap=getattr(r, 'mip_gap', None),
               nodes=getattr(r, 'mip_node_count', None))
    if r.x is not None:
        out['z1'] = float(m['c1'] @ r.x)
        out['z2'] = float(m['c2'] @ r.x + m['const2'])
    return out


if __name__ == "__main__":
    code_dir = sys.argv[1]
    sys.path.insert(0, code_dir)
    import study as st
    ids = [int(x) for x in sys.argv[2].split(",")]
    tl = float(sys.argv[3])
    for i in ids:
        inst = st.generate_instance(i)
        t0 = time.time()
        m = build(st, inst)
        a = solve(m, m['c2'], time_limit=tl)                       # min Z2
        zmin2 = a['z2']
        zmax2 = m['const2']                                        # Q = 0
        mid = solve(m, m['c1'], eps=0.5 * (zmin2 + zmax2), time_limit=tl)
        print(f"inst {i} I={inst['I']} J={inst['J']} vars={m['n']}  build+minZ2 {a['time']:.1f}s st={a['status']} gap={a['gap']} | "
              f"mid-eps: {mid['time']:.1f}s st={mid['status']} gap={mid['gap']} nodes={mid['nodes']} "
              f"z1={mid.get('z1')} bound={mid['bound']}", flush=True)
