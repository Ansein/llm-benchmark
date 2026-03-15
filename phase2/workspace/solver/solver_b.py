"""
Scenario B Solver — Inference Externality (Too Much Data)
Auto-generated fallback by SolverBuilderAgent.
"""
import itertools
import numpy as np


def solve(N: int, rho: float, v_lo: float, v_hi: float,
          alpha: float = 1.0, seed: int = 42) -> dict:
    """
    Compute the platform-optimal sharing set and equilibrium metrics.

    Parameters
    ----------
    N       : number of users
    rho     : inter-user type correlation
    v_lo    : lower bound of privacy preference uniform distribution
    v_hi    : upper bound of privacy preference uniform distribution
    alpha   : platform's marginal value of information
    seed    : random seed for reproducibility

    Returns
    -------
    dict with keys: sharing_set, share_rate, platform_profit,
                   social_welfare, total_leakage
    """
    rng = np.random.default_rng(seed)
    v_values = rng.uniform(v_lo, v_hi, N)   # privacy costs

    # Prior covariance matrix: rho off-diagonal, 1 on diagonal
    Sigma = np.full((N, N), rho)
    np.fill_diagonal(Sigma, 1.0)

    best_profit = -np.inf
    best_set = []

    for size in range(N + 1):
        for S in itertools.combinations(range(N), size):
            S = list(S)
            profit, leakage = _platform_profit(S, Sigma, v_values, alpha, N)
            if profit > best_profit:
                best_profit = profit
                best_set = S
                best_leakage = leakage

    # Compute welfare metrics for the optimal set
    share_rate = len(best_set) / N
    # Social welfare = platform profit + sum of (v_i - payment) for sharers
    payments = v_values[best_set] if best_set else np.array([])
    consumer_benefit = float(np.sum(payments))  # payment equals their cost
    social_welfare = best_profit + consumer_benefit

    return {
        "sharing_set": best_set,
        "share_rate": share_rate,
        "platform_profit": float(best_profit),
        "social_welfare": float(social_welfare),
        "total_leakage": float(best_leakage),
    }


def _platform_profit(S: list, Sigma: np.ndarray,
                      v_values: np.ndarray, alpha: float, N: int):
    """Compute platform profit for sharing set S."""
    if not S:
        return 0.0, 0.0

    S_arr = np.array(S)
    Sigma_S = Sigma[np.ix_(S_arr, S_arr)]
    try:
        Sigma_S_inv = np.linalg.inv(Sigma_S)
    except np.linalg.LinAlgError:
        return -np.inf, 0.0

    log_det_prior = np.log(np.linalg.det(Sigma))

    total_leakage = 0.0
    for i in range(N):
        # Posterior variance for user i given S
        sigma_iS = Sigma[i, S_arr]
        sigma_post_i = Sigma[i, i] - sigma_iS @ Sigma_S_inv @ sigma_iS
        sigma_post_i = max(sigma_post_i, 1e-12)
        leakage_i = 0.5 * (np.log(Sigma[i, i]) - np.log(sigma_post_i))
        total_leakage += leakage_i

    # Platform pays each sharer their privacy cost v_i
    total_payment = float(np.sum(v_values[S_arr]))
    profit = alpha * total_leakage - total_payment
    return profit, total_leakage
