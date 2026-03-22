import math
from typing import Dict, Any, List

import numpy as np


def _prior_cov(N: int, rho: float) -> np.ndarray:
    Sigma = np.full((N, N), float(rho), dtype=float)
    np.fill_diagonal(Sigma, 1.0)
    return Sigma


def _posterior_cov(Sigma: np.ndarray, S: List[int]) -> np.ndarray:
    N = Sigma.shape[0]
    if len(S) == 0:
        return Sigma.copy()
    idx = np.array(S, dtype=int)
    Sigma_SS = Sigma[np.ix_(idx, idx)]
    inv_SS = np.linalg.pinv(Sigma_SS)
    return Sigma - Sigma[:, idx] @ inv_SS @ Sigma[idx, :]


def _total_leakage(Sigma: np.ndarray, Sigma_post: np.ndarray) -> float:
    leak = 0.0
    for i in range(Sigma.shape[0]):
        prior_var = max(float(Sigma[i, i]), 1e-15)
        post_var = max(float(Sigma_post[i, i]), 1e-15)
        leak += 0.5 * (math.log(prior_var) - math.log(post_var))
    return float(leak)


def solve(N: int, rho: float, v_lo: float, v_hi: float, alpha: float, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    if N <= 0:
        return {
            "sharing_set": [],
            "share_rate": 0.0,
            "platform_profit": 0.0,
            "social_welfare": 0.0,
            "total_leakage": 0.0,
        }

    if N == 1:
        v = np.array([0.5 * (v_lo + v_hi)], dtype=float)
    else:
        v = np.array([v_lo] * (N // 2) + [v_hi] * (N - N // 2), dtype=float)
        rng.shuffle(v)

    Sigma = _prior_cov(N, rho)
    if abs(rho) >= 1.0:
        Sigma = Sigma + 1e-10 * np.eye(N)

    best = None
    for mask in range(1 << N):
        S = [i for i in range(N) if (mask >> i) & 1]
        Sigma_post = _posterior_cov(Sigma, S)
        leakage = _total_leakage(Sigma, Sigma_post)
        pay = float(np.sum(v[S]))
        platform_profit = alpha * leakage - pay
        social_welfare = leakage - pay
        cand = {
            "sharing_set": S,
            "share_rate": len(S) / N,
            "platform_profit": platform_profit,
            "social_welfare": social_welfare,
            "total_leakage": leakage,
        }
        if best is None or cand["platform_profit"] > best["platform_profit"] + 1e-12 or (
            abs(cand["platform_profit"] - best["platform_profit"]) <= 1e-12 and len(S) < len(best["sharing_set"])
        ):
            best = cand
    return best


# Backwards compatibility
solve_scenario_b = solve


if __name__ == "__main__":
    print(solve(4, 0.3, 0.2, 0.8, 1.0, 0))
