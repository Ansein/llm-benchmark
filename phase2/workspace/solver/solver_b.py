import itertools
import math
from typing import List, Tuple, Dict, Any, Optional

import numpy as np


def _build_cov(N: int, rho: float) -> np.ndarray:
    Sigma = np.full((N, N), rho, dtype=float)
    np.fill_diagonal(Sigma, 1.0)
    return Sigma


def _posterior_cov(Sigma: np.ndarray, S: Tuple[int, ...]) -> np.ndarray:
    N = Sigma.shape[0]
    if len(S) == 0:
        return Sigma.copy()
    S_idx = np.array(S, dtype=int)
    mask = np.ones(N, dtype=bool)
    mask[S_idx] = False

    Sigma_SS = Sigma[np.ix_(S_idx, S_idx)]
    Sigma_allS = Sigma[:, S_idx]
    Sigma_Sall = Sigma[S_idx, :]

    # Use pseudo-inverse for numerical stability in degenerate cases.
    inv_SS = np.linalg.pinv(Sigma_SS)
    Sigma_post = Sigma - Sigma_allS @ inv_SS @ Sigma_Sall
    # Symmetrize and clip tiny negatives on diagonal
    Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)
    diag = np.clip(np.diag(Sigma_post), 0.0, None)
    np.fill_diagonal(Sigma_post, diag)
    return Sigma_post


def _safe_logdet(A: np.ndarray) -> float:
    sign, ld = np.linalg.slogdet(A)
    if sign <= 0:
        # For near-singular matrices, fall back to a tiny ridge.
        eps = 1e-12
        sign, ld = np.linalg.slogdet(A + eps * np.eye(A.shape[0]))
    return float(ld)


def _information_leakage(Sigma_prior: np.ndarray, Sigma_post: np.ndarray) -> float:
    # Sum_i 0.5 * log(Var_prior_i / Var_post_i) using scalar marginal variances.
    prior_var = np.diag(Sigma_prior)
    post_var = np.diag(Sigma_post)
    post_var = np.maximum(post_var, 1e-15)
    prior_var = np.maximum(prior_var, 1e-15)
    return float(0.5 * np.sum(np.log(prior_var) - np.log(post_var)))


def _user_leakage_vector(Sigma_prior: np.ndarray, Sigma_post: np.ndarray) -> np.ndarray:
    prior_var = np.maximum(np.diag(Sigma_prior), 1e-15)
    post_var = np.maximum(np.diag(Sigma_post), 1e-15)
    return 0.5 * (np.log(prior_var) - np.log(post_var))


def solve_scenario_b(N: int, rho: float, v_lo: float, v_hi: float, alpha: float, seed: int = 0) -> Dict[str, Any]:
    """Enumerate all sharing sets and return the platform-optimal outcome.

    Parameters
    ----------
    N : int
        Number of users.
    rho : float
        Common correlation parameter; covariance matrix has 1 on diagonal and rho off-diagonal.
    v_lo, v_hi : float
        Low/high privacy values used to assign heterogeneous user privacy costs.
    alpha : float
        Platform weight on information leakage.
    seed : int
        Random seed for deterministic assignment of v_i values.

    Returns
    -------
    dict with keys:
        sharing_set, share_rate, platform_profit, social_welfare, total_leakage, payments, values
    """
    rng = np.random.default_rng(seed)
    # Deterministic but heterogeneous privacy values.
    if N == 1:
        v = np.array([v_lo], dtype=float)
    else:
        # Mix of low/high values with deterministic shuffle.
        vals = np.array([v_lo] * (N // 2) + [v_hi] * (N - N // 2), dtype=float)
        rng.shuffle(vals)
        v = vals

    Sigma_prior = _build_cov(N, rho)

    best = None
    all_results = []
    for bits in range(1 << N):
        S = tuple(i for i in range(N) if (bits >> i) & 1)
        Sigma_post = _posterior_cov(Sigma_prior, S)
        leakage_vec = _user_leakage_vector(Sigma_prior, Sigma_post)
        total_leakage = float(np.sum(leakage_vec))

        # Platform-side payments: take the minimum incentive-compatible payment given realized leakage.
        # In a best-response / Nash interpretation, user i shares iff p_i >= v_i * I_i.
        payments = np.zeros(N, dtype=float)
        for i in S:
            payments[i] = v[i] * leakage_vec[i]

        platform_profit = float(alpha * total_leakage - np.sum(payments))
        social_welfare = float(total_leakage - np.sum(payments))
        share_rate = len(S) / N if N > 0 else 0.0

        res = {
            "sharing_set": list(S),
            "share_rate": share_rate,
            "platform_profit": platform_profit,
            "social_welfare": social_welfare,
            "total_leakage": total_leakage,
            "payments": payments.tolist(),
            "privacy_values": v.tolist(),
            "cov_prior": Sigma_prior.tolist(),
            "cov_post": Sigma_post.tolist(),
        }
        all_results.append(res)

        if best is None or platform_profit > best["platform_profit"] + 1e-12:
            best = res

    best["all_results"] = all_results
    return best


# Backward-compatible alias expected by the validation harness.
def solve(N: int, rho: float, v_lo: float, v_hi: float, alpha: float, seed: int = 0) -> Dict[str, Any]:
    return solve_scenario_b(N, rho, v_lo, v_hi, alpha, seed)


if __name__ == "__main__":
    out = solve_scenario_b(4, 0.3, 0.5, 1.5, 1.0, 0)
    print(out)
