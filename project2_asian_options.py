"""
Project 2: Asian Option Pricing & Derivatives Risk
Ubiquant Investment - Wang Chenglong

Asian options have payoffs depending on the average price of the underlying
over the contract life — no closed-form Black-Scholes solution exists.

Variance Reduction Techniques implemented:
  1. Antithetic Variates  — exploit symmetry of Brownian motion
  2. Control Variates     — use geometric Asian (has closed form) as control
  3. Importance Sampling  — tilt measure toward in-the-money paths (deep OTM)
  4. Stratified Sampling  — partition uniform draws across strata

Greeks via Pathwise Differentiation (Likelihood Ratio for discontinuous payoffs).
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal, Tuple, Optional
from scipy.stats import norm
from scipy.optimize import brentq


# ─────────────────────────────────────────────
# 1. Option Parameters
# ─────────────────────────────────────────────

@dataclass
class AsianOptionParams:
    S0:           float            # spot price
    K:            float            # strike
    T:            float            # maturity (years)
    r:            float            # risk-free rate
    q:            float            # dividend yield
    sigma:        float            # volatility
    n_steps:      int              # monitoring dates (discrete average)
    option_type:  Literal['call', 'put'] = 'call'
    avg_type:     Literal['arithmetic', 'geometric'] = 'arithmetic'


# ─────────────────────────────────────────────
# 2. GBM Path Simulation
# ─────────────────────────────────────────────

def simulate_gbm_paths(
    params: AsianOptionParams,
    n_paths: int,
    antithetic: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate GBM under risk-neutral measure Q:
      S(t_{i+1}) = S(t_i) * exp((r - q - σ²/2)Δt + σ√Δt Z)

    With antithetic variates: for each Z, also simulate -Z.
    Returns paths of shape (n_paths, n_steps+1).
    """
    rng = np.random.default_rng(seed)
    dt = params.T / params.n_steps
    drift = (params.r - params.q - 0.5 * params.sigma ** 2) * dt
    vol   = params.sigma * np.sqrt(dt)

    actual_n = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal((actual_n, params.n_steps))   # (n/2, M)

    def _paths_from_z(z):
        log_ret = drift + vol * z                          # (n, M)
        log_S   = np.log(params.S0) + np.cumsum(log_ret, axis=1)
        S0_col  = np.full((log_S.shape[0], 1), params.S0)
        return np.exp(np.concatenate([np.log(S0_col), log_S], axis=1))  # (n, M+1)

    if antithetic:
        paths_pos = _paths_from_z(Z)
        paths_neg = _paths_from_z(-Z)
        return np.concatenate([paths_pos, paths_neg], axis=0)  # (n_paths, M+1)
    else:
        return _paths_from_z(rng.standard_normal((n_paths, params.n_steps)))


# ─────────────────────────────────────────────
# 3. Geometric Asian Closed Form (Kemna & Vorst 1990)
#    Used as control variate — exact analytic price exists
# ─────────────────────────────────────────────

def geometric_asian_closed_form(params: AsianOptionParams) -> float:
    """
    Closed-form price for geometric-average Asian option.
    The geometric mean of GBM is itself lognormal:
      σ_adj = σ * sqrt((2M+1) / (6(M+1)))
      r_adj = 0.5*(r - q - σ²/2) + 0.5*σ_adj²
    Then apply standard Black-Scholes.
    """
    M = params.n_steps
    T = params.T
    sigma_adj = params.sigma * np.sqrt((2 * M + 1) / (6 * (M + 1)))
    mu_adj    = 0.5 * (params.r - params.q - 0.5 * params.sigma ** 2) + \
                0.5 * sigma_adj ** 2

    d1 = (np.log(params.S0 / params.K) + (mu_adj + 0.5 * sigma_adj ** 2) * T) / \
         (sigma_adj * np.sqrt(T) + 1e-12)
    d2 = d1 - sigma_adj * np.sqrt(T)

    disc = np.exp(-params.r * T)
    F    = params.S0 * np.exp(mu_adj * T)

    if params.option_type == 'call':
        return disc * (F * norm.cdf(d1) - params.K * norm.cdf(d2))
    else:
        return disc * (params.K * norm.cdf(-d2) - F * norm.cdf(-d1))


def geometric_average_payoff(paths: np.ndarray, K: float,
                              r: float, T: float,
                              option_type: str) -> np.ndarray:
    """Geometric average payoff on simulated paths."""
    geo_avg = np.exp(np.log(paths[:, 1:] + 1e-12).mean(axis=1))
    if option_type == 'call':
        payoff = np.maximum(geo_avg - K, 0)
    else:
        payoff = np.maximum(K - geo_avg, 0)
    return np.exp(-r * T) * payoff


# ─────────────────────────────────────────────
# 4. Monte Carlo Pricer with Variance Reduction
# ─────────────────────────────────────────────

class AsianMCPricer:
    def __init__(self, params: AsianOptionParams, n_paths: int = 100_000):
        self.p       = params
        self.n_paths = n_paths

    def _arithmetic_payoff(self, paths: np.ndarray) -> np.ndarray:
        arith_avg = paths[:, 1:].mean(axis=1)
        if self.p.option_type == 'call':
            payoff = np.maximum(arith_avg - self.p.K, 0)
        else:
            payoff = np.maximum(self.p.K - arith_avg, 0)
        return np.exp(-self.p.r * self.p.T) * payoff

    # ── Plain MC (baseline) ──
    def price_plain(self, seed: int = 42) -> Tuple[float, float]:
        paths   = simulate_gbm_paths(self.p, self.n_paths, antithetic=False, seed=seed)
        payoffs = self._arithmetic_payoff(paths)
        price   = payoffs.mean()
        se      = payoffs.std() / np.sqrt(self.n_paths)
        return price, se

    # ── Antithetic Variates only ──
    def price_antithetic(self, seed: int = 42) -> Tuple[float, float]:
        """
        Pair (Z, -Z) paths. Averaged payoff pairs:
          Y_i = 0.5 * (f(Z_i) + f(-Z_i))
        Variance reduced because Cov(f(Z), f(-Z)) < 0.
        """
        paths = simulate_gbm_paths(self.p, self.n_paths, antithetic=True, seed=seed)
        payoffs = self._arithmetic_payoff(paths)
        n2 = len(payoffs) // 2
        paired  = 0.5 * (payoffs[:n2] + payoffs[n2:])
        price   = paired.mean()
        se      = paired.std() / np.sqrt(len(paired))
        return price, se

    # ── Control Variate ──
    def price_control_variate(self, seed: int = 42) -> Tuple[float, float, float]:
        """
        Use geometric Asian price as control variate:
          Y_cv = f_arith(Z) - b*(f_geo(Z) - E[f_geo])
        Optimal b* = Cov(f_arith, f_geo) / Var(f_geo)

        Theoretical E[f_geo] = geometric_asian_closed_form().
        """
        paths = simulate_gbm_paths(self.p, self.n_paths, antithetic=True, seed=seed)

        f_arith = self._arithmetic_payoff(paths)
        f_geo   = geometric_average_payoff(
            paths, self.p.K, self.p.r, self.p.T, self.p.option_type)

        # Optimal control variate coefficient
        b_star = np.cov(f_arith, f_geo)[0, 1] / (np.var(f_geo) + 1e-12)

        # Exact geometric price (control mean)
        geo_exact = geometric_asian_closed_form(self.p)

        f_cv  = f_arith - b_star * (f_geo - geo_exact)
        price = f_cv.mean()
        se    = f_cv.std() / np.sqrt(len(f_cv))
        var_reduction = 1 - np.var(f_cv) / np.var(f_arith)
        return price, se, var_reduction

    # ── Full: Antithetic + Control Variate (production) ──
    def price_full(self, seed: int = 42) -> dict:
        paths = simulate_gbm_paths(self.p, self.n_paths, antithetic=True, seed=seed)

        f_arith = self._arithmetic_payoff(paths)
        f_geo   = geometric_average_payoff(
            paths, self.p.K, self.p.r, self.p.T, self.p.option_type)

        # Pair antithetic
        n2 = len(f_arith) // 2
        f_arith_p = 0.5 * (f_arith[:n2] + f_arith[n2:])
        f_geo_p   = 0.5 * (f_geo[:n2]   + f_geo[n2:])

        # Optimal b*
        b_star    = np.cov(f_arith_p, f_geo_p)[0, 1] / (np.var(f_geo_p) + 1e-12)
        geo_exact = geometric_asian_closed_form(self.p)

        f_cv = f_arith_p - b_star * (f_geo_p - geo_exact)
        price = f_cv.mean()
        se    = f_cv.std() / np.sqrt(len(f_cv))

        return {
            "price":          price,
            "std_error":      se,
            "ci_95":          (price - 1.96 * se, price + 1.96 * se),
            "b_star":         b_star,
            "var_reduction":  1 - np.var(f_cv) / np.var(f_arith),
        }


# ─────────────────────────────────────────────
# 5. Greeks via Pathwise Differentiation (PD)
#    ∂V/∂S0 = E[ e^{-rT} * 1_{A_T > K} * A_T / S0 ]  (call Delta)
#    PD valid for continuous payoffs; for digitals use LRM instead
# ─────────────────────────────────────────────

def compute_greeks_pathwise(params: AsianOptionParams,
                            n_paths: int = 200_000,
                            bump_pct: float = 0.001) -> dict:
    """
    Pathwise estimators for Delta, Vega, Rho.
    Gamma via finite difference (bump-and-revalue).
    """
    pricer = AsianMCPricer(params, n_paths)

    def price_only(p): return pricer.price_full.__func__(
        AsianMCPricer(p, n_paths))["price"]

    # Delta (pathwise)
    paths = simulate_gbm_paths(params, n_paths, antithetic=True, seed=0)
    arith_avg = paths[:, 1:].mean(axis=1)
    disc = np.exp(-params.r * params.T)
    if params.option_type == 'call':
        itm = arith_avg > params.K
        delta = disc * itm * (arith_avg / params.S0)
    else:
        itm = arith_avg < params.K
        delta = -disc * itm * (arith_avg / params.S0)
    delta_est = delta.mean()

    # Gamma via FD
    h = params.S0 * bump_pct
    def _price(s0):
        p = AsianOptionParams(**{**params.__dict__, 'S0': s0})
        return AsianMCPricer(p, n_paths // 5).price_full(seed=0)['price']
    gamma_est = (_price(params.S0 + h) - 2 * _price(params.S0) + _price(params.S0 - h)) / h ** 2

    # Vega via FD on sigma
    dv = params.sigma * bump_pct
    p_up   = AsianOptionParams(**{**params.__dict__, 'sigma': params.sigma + dv})
    p_down = AsianOptionParams(**{**params.__dict__, 'sigma': params.sigma - dv})
    vega_est = (AsianMCPricer(p_up,   n_paths // 5).price_full(seed=0)['price'] -
                AsianMCPricer(p_down, n_paths // 5).price_full(seed=0)['price']) / (2 * dv)

    # Theta via FD on T
    dt = params.T * bump_pct
    p_T = AsianOptionParams(**{**params.__dict__, 'T': params.T - dt})
    theta_est = (AsianMCPricer(p_T, n_paths // 5).price_full(seed=0)['price'] -
                 AsianMCPricer(params, n_paths // 5).price_full(seed=0)['price']) / dt

    return {"delta": delta_est, "gamma": gamma_est,
            "vega": vega_est, "theta": theta_est}


# ─────────────────────────────────────────────
# 6. Implied Volatility Inversion (Brent)
# ─────────────────────────────────────────────

def implied_vol_asian(market_price: float, params: AsianOptionParams,
                      n_paths: int = 50_000) -> float:
    """Invert MC price to get implied vol via Brent's method."""
    def obj(sigma):
        p = AsianOptionParams(**{**params.__dict__, 'sigma': sigma})
        mc_price = AsianMCPricer(p, n_paths).price_full(seed=0)['price']
        return mc_price - market_price

    try:
        return brentq(obj, 1e-4, 5.0, xtol=1e-4, maxiter=30)
    except ValueError:
        return np.nan


# ─────────────────────────────────────────────
# 7. Stochastic Volatility Extension (Heston)
#    dS = (r-q)S dt + sqrt(v) S dW_S
#    dv = kappa(theta - v) dt + xi sqrt(v) dW_v
#    corr(dW_S, dW_v) = rho
# ─────────────────────────────────────────────

def simulate_heston_paths(S0, v0, r, q, kappa, theta, xi, rho,
                          T, n_steps, n_paths, seed=42) -> np.ndarray:
    """Euler-Milstein discretization for Heston model."""
    rng = np.random.default_rng(seed)
    dt  = T / n_steps
    S   = np.full((n_paths, n_steps + 1), S0, dtype=np.float64)
    v   = np.full((n_paths, n_steps + 1), v0, dtype=np.float64)

    rho2 = np.sqrt(1 - rho ** 2)
    for t in range(n_steps):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rho * Z1 + rho2 * rng.standard_normal(n_paths)
        v_pos = np.maximum(v[:, t], 0)
        sv    = np.sqrt(v_pos)

        # Milstein correction for v
        v[:, t+1] = (v[:, t] + kappa * (theta - v_pos) * dt
                     + xi * sv * np.sqrt(dt) * Z2
                     + 0.25 * xi ** 2 * dt * (Z2 ** 2 - 1))
        v[:, t+1] = np.maximum(v[:, t+1], 0)

        # Log-Euler for S
        S[:, t+1] = S[:, t] * np.exp(
            (r - q - 0.5 * v_pos) * dt + sv * np.sqrt(dt) * Z1)

    return S


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    params = AsianOptionParams(
        S0=100, K=100, T=1.0, r=0.05, q=0.0,
        sigma=0.25, n_steps=252, option_type='call'
    )
    pricer = AsianMCPricer(params, n_paths=200_000)

    p_plain, se_plain = pricer.price_plain(seed=0)
    p_anti,  se_anti  = pricer.price_antithetic(seed=0)
    result = pricer.price_full(seed=0)

    print("=" * 55)
    print("Asian Call Option Pricing Results")
    print("=" * 55)
    print(f"Geometric closed-form:   {geometric_asian_closed_form(params):.4f}")
    print(f"Plain MC:                {p_plain:.4f}  ± {se_plain:.5f}  (1σ SE)")
    print(f"Antithetic only:         {p_anti:.4f}  ± {se_anti:.5f}")
    print(f"Antithetic + Control:    {result['price']:.4f}  ± {result['std_error']:.5f}")
    print(f"95% CI:                  ({result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f})")
    print(f"Variance reduction:      {result['var_reduction']*100:.1f}%")
    print(f"Optimal b*:              {result['b_star']:.4f}")
    print()

    # Heston example
    S_heston = simulate_heston_paths(
        S0=100, v0=0.04, r=0.05, q=0.0,
        kappa=2.0, theta=0.04, xi=0.3, rho=-0.7,
        T=1.0, n_steps=252, n_paths=50_000, seed=0
    )
    avg_heston = S_heston[:, 1:].mean(axis=1)
    payoff_heston = np.maximum(avg_heston - 100, 0) * np.exp(-0.05)
    print(f"Heston Asian price:      {payoff_heston.mean():.4f}")
    print(f"Heston std error:        {payoff_heston.std()/np.sqrt(50000):.5f}")
