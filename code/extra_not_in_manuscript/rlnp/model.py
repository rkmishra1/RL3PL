"""
Bi-objective two-stage stochastic reverse logistics network planning model
for a third-party logistics (3PL) provider.

Implements Section 3 of the manuscript exactly:

  first stage : y_ij   assignment of customer zone i to collection centre j
                       (single sourcing, constraint c1)
  second stage: Q_ij, Q_jk, Q_js, Q_si, Q_jr, Q_rx, U_i  per scenario

  Z1 = E[ transport(i->j, j->k, j->s, s->i, j->r, r->x)
          + inspection/sorting + storage + disposal ]
  Z2 = E[ phi1 * uncollected + phi2 * lateness-weighted redelivered units ]

Constraints c1-c11 as numbered in the manuscript.
"""

from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr as _ndtr, ndtri as _ndtri


# ----------------------------------------------------------------------
# Instance definition
# ----------------------------------------------------------------------

@dataclass
class Instance:
    """A problem instance: sizes, deterministic parameters, scenario data."""
    nI: int          # customer zones
    nJ: int          # collection centres
    nK: int          # retailing centres
    nS: int          # service centres
    nR: int          # recycling centres
    nX: int          # disposal locations

    # deterministic parameters
    T_ij: np.ndarray  # (I,J) unit transport cost zone -> collection centre
    T_jk: np.ndarray  # (J,K)
    T_js: np.ndarray  # (J,S)
    T_si: np.ndarray  # (S,I) return leg, service centre -> zone
    T_jr: np.ndarray  # (J,R)
    T_rx: np.ndarray  # (R,X)
    d_x: np.ndarray   # (X,) unit disposal cost
    beta_K: float
    beta_S: float
    beta_R: float
    alpha: float
    phi1: float
    phi2: float
    tau_i: np.ndarray  # (I,) contractual delivery time

    # scenario-dependent parameters, first axis = scenario
    R: np.ndarray      # (W,I) realised returns
    Ca: np.ndarray     # (W,J) collection centre capacity
    C: np.ndarray      # (W,J) inspection/sorting cost per unit
    W_: np.ndarray     # (W,J) storage cost per unit
    t_si: np.ndarray   # (W,S,I) repair channel turnaround time

    @property
    def nW(self) -> int:
        return self.R.shape[0]

    def check(self):
        assert abs(self.beta_K + self.beta_S + self.beta_R - 1.0) < 1e-12, \
            "Assumption 7 violated: sorting fractions must sum to 1"
        assert self.R.shape == (self.nW, self.nI)
        assert self.Ca.shape == (self.nW, self.nJ)
        assert self.t_si.shape == (self.nW, self.nS, self.nI)


def make_instance(nI, nJ, nK, nS, nR, nX, nW, rng, dist="uniform"):
    """Instance generator using the parameter ranges of Table 'Values'."""

    def draw(lo, hi, size):
        """
        Baseline: Uniform(lo, hi).

        'normal'    : Normal truncated to the same support [lo, hi] with the same
                      mean (lo+hi)/2 and sd (hi-lo)/6, so the support spans
                      +/- 3 sd. Sampled by inverse-cdf on the truncated law, not
                      by clipping, so no probability mass piles up at the bounds.
        'lognormal' : right-skewed, calibrated to the SAME mean and the SAME 95th
                      percentile as the Uniform. Solving
                          mu + sigma^2/2      = log(mean)
                          mu + 1.6449*sigma   = log(q95)
                      gives sigma as the smaller root of
                          sigma^2 - 2*1.6449*sigma + 2*log(q95/mean) = 0.
                      The upper tail is not truncated, so draws may exceed hi.
        """
        if dist == "uniform":
            return rng.uniform(lo, hi, size)

        if dist == "normal":
            mu, sd = (lo + hi) / 2.0, (hi - lo) / 6.0
            a, b = (lo - mu) / sd, (hi - mu) / sd
            u = rng.random(size)
            p = _ndtr(a) + u * (_ndtr(b) - _ndtr(a))
            return mu + sd * _ndtri(np.clip(p, 1e-12, 1 - 1e-12))

        if dist == "lognormal":
            mean = (lo + hi) / 2.0
            q95 = lo + 0.95 * (hi - lo)
            z = 1.6448536269514722                       # Phi^{-1}(0.95)
            c = np.log(q95 / mean)
            disc = z * z - 2.0 * c
            sigma = z - np.sqrt(max(disc, 0.0))          # smaller root
            mu = np.log(mean) - sigma ** 2 / 2.0
            return rng.lognormal(mu, sigma, size)

        raise ValueError(dist)

    return Instance(
        nI=nI, nJ=nJ, nK=nK, nS=nS, nR=nR, nX=nX,
        T_ij=rng.uniform(2, 10, (nI, nJ)),
        T_jk=rng.uniform(2, 10, (nJ, nK)),
        T_js=rng.uniform(2, 10, (nJ, nS)),
        T_si=rng.uniform(2, 10, (nS, nI)),
        T_jr=rng.uniform(2, 10, (nJ, nR)),
        T_rx=rng.uniform(2, 10, (nR, nX)),
        d_x=rng.uniform(5, 10, nX),
        beta_K=0.5, beta_S=0.3, beta_R=0.2, alpha=0.4,
        phi1=160.0, phi2=200.0,
        tau_i=rng.uniform(2, 8, nI),
        R=draw(3500, 3800, (nW, nI)),
        Ca=draw(1600, 2400, (nW, nJ)),
        C=draw(15, 18, (nW, nJ)),
        W_=draw(15, 25, (nW, nJ)),
        t_si=draw(2, 10, (nW, nS, nI)),
    )


# ----------------------------------------------------------------------
# Decoder: chromosome -> feasible flows -> (Z1, Z2)
# ----------------------------------------------------------------------

def decode(inst: Instance, a: np.ndarray, rho: np.ndarray, return_flows=False):
    """
    a   : (I,) int   collection centre index per zone (encodes y_ij, c1)
    rho : (I,) float collection intensity in [0,1]

    Returns (Z1, Z2), the sample-average objectives over all scenarios.
    Flows satisfy c1-c11 by construction.
    """
    W, I, J = inst.nW, inst.nI, inst.nJ

    # --- step 1: attempted collection ---------------------------------
    tendered = rho[None, :] * inst.R                      # (W,I)

    # --- step 2: proportional capacity repair (c4) --------------------
    # volume tendered at each centre, per scenario
    onehot = np.zeros((I, J))
    onehot[np.arange(I), a] = 1.0                          # y_ij
    V = tendered @ onehot                                  # (W,J)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(V > inst.Ca, inst.Ca / np.maximum(V, 1e-12), 1.0)
    scale_per_zone = scale[:, a]                           # (W,I)
    Q_ij_zone = tendered * scale_per_zone                  # collected per zone
    U = inst.R - Q_ij_zone                                 # c2: uncollected

    inbound = Q_ij_zone @ onehot                           # (W,J) collected per centre

    # --- step 3: sorting at collection centres (c5-c7) ----------------
    to_K = inst.beta_K * inbound
    to_S = inst.beta_S * inbound
    to_R = inst.beta_R * inbound

    # --- step 4: downstream allocation (greedy, optimal: Assumption 3)-
    # commercial returns: each centre -> cheapest retailer
    k_star = np.argmin(inst.T_jk, axis=1)                  # (J,)
    cost_jk = inst.T_jk[np.arange(J), k_star]              # (J,)
    # EOL: each centre -> cheapest recycler
    r_star = np.argmin(inst.T_jr, axis=1)
    cost_jr = inst.T_jr[np.arange(J), r_star]
    # e-waste: each recycler -> cheapest disposal (transport + disposal fee)
    rx_total = inst.T_rx + inst.d_x[None, :]
    x_star = np.argmin(rx_total, axis=1)                   # (R,)
    cost_rx = rx_total[np.arange(inst.nR), x_star]         # (R,)

    # repair: zone i routed via the service centre minimising T_js + T_si
    # where j = a_i  (round-trip cost of that zone)
    roundtrip = inst.T_js[a, :] + inst.T_si.T              # (I,S)
    s_star = np.argmin(roundtrip, axis=1)                  # (I,)
    cost_repair_zone = roundtrip[np.arange(I), s_star]     # (I,)

    # --- step 5: objective accumulation -------------------------------
    # Z1 transport + handling
    unit_ij = inst.T_ij[np.arange(I), a]                   # (I,)
    z1 = (Q_ij_zone * unit_ij[None, :]).sum(axis=1)        # i->j transport
    z1 = z1 + (Q_ij_zone * (inst.C[:, a] + inst.W_[:, a])).sum(axis=1)  # C + W
    z1 = z1 + (to_K * cost_jk[None, :]).sum(axis=1)        # j->k
    z1 = z1 + (to_R * cost_jr[None, :]).sum(axis=1)        # j->r

    # repair channel: j->s and s->i for the zone's own repair volume (c9)
    repair_zone = inst.beta_S * Q_ij_zone                  # (W,I)
    z1 = z1 + (repair_zone * cost_repair_zone[None, :]).sum(axis=1)

    # e-waste r->x with disposal fee (c10)
    ewaste_centre = inst.alpha * to_R                      # (W,J) leaving each centre
    # volume arriving at recycler r, then to disposal
    ewaste_r = np.zeros((W, inst.nR))
    np.add.at(ewaste_r.T, r_star, ewaste_centre.T)
    z1 = z1 + (ewaste_r * cost_rx[None, :]).sum(axis=1)

    # Z2 penalties
    lateness = np.maximum(inst.t_si[:, s_star, np.arange(I)] - inst.tau_i[None, :], 0.0)
    z2 = inst.phi1 * U.sum(axis=1) + inst.phi2 * (lateness * repair_zone).sum(axis=1)

    Z1, Z2 = float(z1.mean()), float(z2.mean())
    if not return_flows:
        return Z1, Z2

    flows = dict(Q_ij_zone=Q_ij_zone, U=U, inbound=inbound, to_K=to_K,
                 to_S=to_S, to_R=to_R, repair_zone=repair_zone,
                 ewaste_r=ewaste_r, onehot=onehot, s_star=s_star)
    return (Z1, Z2), flows


def decode_batch(inst: Instance, A: np.ndarray, Rho: np.ndarray):
    """
    Vectorised decoder for a whole population.
    A   : (P,I) int, Rho : (P,I) float  ->  (P,2) objectives.
    Mathematically identical to decode(), just batched over individuals.
    """
    P, I = A.shape
    W, J = inst.nW, inst.nJ

    tendered = Rho[:, None, :] * inst.R[None, :, :]            # (P,W,I)

    # volume per collection centre: scatter-add over zones
    onehot = np.zeros((P, I, J))
    onehot[np.arange(P)[:, None], np.arange(I)[None, :], A] = 1.0
    idx = np.broadcast_to(A[:, None, :], (P, W, I))
    V = np.einsum('pwi,pij->pwj', tendered, onehot, optimize=True)
    Ca = inst.Ca[None, :, :]                                   # (1,W,J)
    scale = np.where(V > Ca, Ca / np.maximum(V, 1e-12), 1.0)   # (P,W,J)
    scale_zone = np.take_along_axis(scale, idx, axis=2)        # (P,W,I)

    Qz = tendered * scale_zone                                 # collected per zone
    U = inst.R[None, :, :] - Qz

    # handling costs at the assigned centre
    CW = (inst.C + inst.W_)                                    # (W,J)
    cw_zone = CW[None, :, :].repeat(P, 0)
    cw_zone = np.take_along_axis(cw_zone, idx, axis=2)         # (P,W,I)
    unit_ij = inst.T_ij[np.arange(I)[None, :], A]              # (P,I)

    z1 = (Qz * (unit_ij[:, None, :] + cw_zone)).sum(axis=2)    # (P,W)

    # downstream greedy targets (instance-level, precomputed)
    k_cost = inst.T_jk.min(axis=1)                             # (J,)
    r_cost = inst.T_jr.min(axis=1)
    rx_total = inst.T_rx + inst.d_x[None, :]
    r_star = np.argmin(inst.T_jr, axis=1)                      # (J,)
    rx_cost = rx_total.min(axis=1)                             # (R,)

    inbound = np.einsum('pwi,pij->pwj', Qz, onehot, optimize=True)
    z1 = z1 + inst.beta_K * (inbound * k_cost[None, None, :]).sum(axis=2)
    z1 = z1 + inst.beta_R * (inbound * r_cost[None, None, :]).sum(axis=2)
    # e-waste: centre -> its cheapest recycler -> that recycler's cheapest disposal
    z1 = z1 + inst.alpha * inst.beta_R * (
        inbound * rx_cost[r_star][None, None, :]).sum(axis=2)

    # repair channel: zone-specific round trip via best service centre
    roundtrip = inst.T_js[A] + inst.T_si.T[None, :, :]         # (P,I,S)
    s_star = np.argmin(roundtrip, axis=2)                      # (P,I)
    rt_cost = np.take_along_axis(roundtrip, s_star[:, :, None], axis=2)[:, :, 0]
    repair = inst.beta_S * Qz                                  # (P,W,I)
    z1 = z1 + (repair * rt_cost[:, None, :]).sum(axis=2)

    # lateness of the chosen service centre, per scenario
    lat = np.maximum(
        inst.t_si[None, :, :, :][0][:, s_star, np.arange(I)[None, :]]
        - inst.tau_i[None, None, :], 0.0)                      # (W,P,I)
    lat = np.transpose(lat, (1, 0, 2))                         # (P,W,I)

    z2 = inst.phi1 * U.sum(axis=2) + inst.phi2 * (lat * repair).sum(axis=2)
    return np.stack([z1.mean(axis=1), z2.mean(axis=1)], axis=1)


# ----------------------------------------------------------------------
# Constraint verification (used by the test suite)
# ----------------------------------------------------------------------

def verify(inst: Instance, a, rho, tol=1e-6):
    """Check every constraint c1-c11 on the decoded flows. Raises on failure."""
    (_, _), f = decode(inst, a, rho, return_flows=True)
    I = inst.nI

    # c1 single sourcing
    assert np.allclose(f["onehot"].sum(axis=1), 1.0), "c1"
    # c2 collection balance
    assert np.allclose(f["Q_ij_zone"] + f["U"], inst.R, atol=tol), "c2"
    # c3 flow only on assigned arc, and not above realised returns
    assert (f["Q_ij_zone"] <= inst.R + tol).all(), "c3"
    # c4 capacity
    assert (f["inbound"] <= inst.Ca + tol).all(), "c4"
    # c5-c7 sorting conserves flow
    tot = f["to_K"] + f["to_S"] + f["to_R"]
    assert np.allclose(tot, f["inbound"], atol=tol), "c5-c7 conservation"
    # c8 service centre balance: all repairs shipped back
    assert np.allclose(f["repair_zone"].sum(axis=1), f["to_S"].sum(axis=1), atol=tol), "c8"
    # c9 zone-level redelivery
    assert np.allclose(f["repair_zone"], inst.beta_S * (inst.R - f["U"]), atol=tol), "c9"
    # c10 e-waste
    assert np.allclose(f["ewaste_r"].sum(axis=1), inst.alpha * f["to_R"].sum(axis=1),
                       atol=tol), "c10"
    # c11 domains
    assert (f["Q_ij_zone"] >= -tol).all() and (f["U"] >= -tol).all(), "c11"
    return True
