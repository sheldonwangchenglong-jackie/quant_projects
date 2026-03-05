"""
Project 1: Multi-Factor Alpha & GAT-Based Stock Representation
Ubiquant Investment - Wang Chenglong

Pipeline:
  1. Build stock relation graph (sector / supply-chain / style-factor edges)
  2. Learn node embeddings via multi-head GAT
  3. Concat GAT embedding with 200+ alpha factors
  4. Cross-sectional return forecast via LGBM / XGBoost
  5. Construct long-short portfolio with convex risk budgeting
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


# ─────────────────────────────────────────────
# 1. Graph Construction
# ─────────────────────────────────────────────

@dataclass
class StockGraph:
    """Adjacency representation for A-share universe."""
    n_stocks: int           # ~4000 A-share securities
    edge_index: np.ndarray  # (2, E) source / target
    edge_weight: np.ndarray # (E,)  relation strength
    edge_type: np.ndarray   # (E,)  0=sector 1=supplychain 2=style


def build_stock_graph(
    sector_matrix: np.ndarray,       # (N, N) 1 if same sector
    supplychain_matrix: np.ndarray,  # (N, N) supply-chain strength
    factor_corr_matrix: np.ndarray,  # (N, N) rolling factor return corr
    sc_threshold: float = 0.3,
    style_threshold: float = 0.6,
) -> StockGraph:
    """
    Fuse three relation types into a single heterogeneous graph.
    Traditional stat-arb treats stocks as isolated one-hot vectors;
    this graph explicitly encodes inter-asset topology.
    """
    N = sector_matrix.shape[0]
    edges_src, edges_dst, weights, types = [], [], [], []

    def _add_edges(matrix, threshold, etype, symmetric=True):
        rows, cols = np.where(matrix > threshold)
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]
        for r, c in zip(rows, cols):
            edges_src.append(r); edges_dst.append(c)
            weights.append(float(matrix[r, c]))
            types.append(etype)
            if symmetric and r != c:
                edges_src.append(c); edges_dst.append(r)
                weights.append(float(matrix[r, c]))
                types.append(etype)

    _add_edges(sector_matrix, 0.5, 0, symmetric=True)
    _add_edges(supplychain_matrix, sc_threshold, 1, symmetric=False)
    _add_edges(factor_corr_matrix, style_threshold, 2, symmetric=True)

    edge_index = np.array([edges_src, edges_dst], dtype=np.int64)  # (2, E)
    return StockGraph(
        n_stocks=N,
        edge_index=edge_index,
        edge_weight=np.array(weights, dtype=np.float32),
        edge_type=np.array(types, dtype=np.int32),
    )


# ─────────────────────────────────────────────
# 2. Multi-Head Graph Attention Network (NumPy impl)
#    Production version uses PyTorch Geometric:
#    from torch_geometric.nn import GATConv
# ─────────────────────────────────────────────

def softmax_over_neighbors(scores: np.ndarray, dst: np.ndarray, N: int) -> np.ndarray:
    """Numerically stable scatter softmax for attention coefficients."""
    alpha = np.full_like(scores, -np.inf)
    for i in range(N):
        mask = dst == i
        if mask.any():
            s = scores[mask]
            s -= s.max()
            exp_s = np.exp(s)
            alpha[mask] = exp_s / (exp_s.sum() + 1e-9)
    return alpha


class GATLayer:
    """
    Single GAT layer:
      e_ij = LeakyReLU( a^T [W h_i || W h_j] )
      alpha_ij = softmax_j(e_ij)
      h_i' = sigma( sum_j alpha_ij W h_j )

    Multi-head: run K independent heads, concat (or mean) outputs.
    """
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 concat: bool = True, alpha: float = 0.2):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.concat = concat
        self.leaky_alpha = alpha

        # Weight matrices: one per head
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.W = [np.random.randn(in_dim, out_dim) * scale for _ in range(n_heads)]
        # Attention vector a = [a_src || a_dst] ∈ R^{2*out_dim}
        self.a_src = [np.random.randn(out_dim) * 0.01 for _ in range(n_heads)]
        self.a_dst = [np.random.randn(out_dim) * 0.01 for _ in range(n_heads)]

    def forward(self, X: np.ndarray, graph: StockGraph) -> np.ndarray:
        """
        X: (N, in_dim) node features
        Returns: (N, n_heads*out_dim) if concat else (N, out_dim)
        """
        src, dst = graph.edge_index[0], graph.edge_index[1]
        N = graph.n_stocks
        head_outputs = []

        for k in range(self.n_heads):
            # Linear projection
            H = X @ self.W[k]                          # (N, out_dim)

            # Compute attention logits on each edge
            e_src = H[src] @ self.a_src[k]             # (E,)
            e_dst = H[dst] @ self.a_dst[k]             # (E,)
            e = e_src + e_dst                           # (E,)

            # LeakyReLU
            e = np.where(e >= 0, e, self.leaky_alpha * e)

            # Weight by edge confidence
            e = e * graph.edge_weight

            # Scatter softmax per destination node
            alpha = softmax_over_neighbors(e, dst, N)  # (E,)

            # Aggregate: h'_i = sum_{j in N(i)} alpha_ij * H_j
            H_new = np.zeros((N, self.out_dim), dtype=np.float32)
            np.add.at(H_new, dst, (alpha[:, None] * H[src]).astype(np.float32))

            # ELU activation
            H_new = np.where(H_new >= 0, H_new, np.exp(H_new) - 1)
            head_outputs.append(H_new)

        if self.concat:
            return np.concatenate(head_outputs, axis=-1)   # (N, K*out_dim)
        else:
            return np.mean(head_outputs, axis=0)           # (N, out_dim)


class GAT:
    """
    2-layer GAT:
      Layer 1: in_dim → hidden_dim (K heads, concat) → K*hidden_dim
      Layer 2: K*hidden_dim → embed_dim (1 head or mean, final embedding)
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64,
                 embed_dim: int = 32, n_heads: int = 4):
        self.layer1 = GATLayer(in_dim, hidden_dim, n_heads=n_heads, concat=True)
        self.layer2 = GATLayer(hidden_dim * n_heads, embed_dim, n_heads=1, concat=False)

    def forward(self, X: np.ndarray, graph: StockGraph) -> np.ndarray:
        """Returns (N, embed_dim) stock embeddings."""
        H = self.layer1.forward(X, graph)   # (N, K*hidden_dim)
        H = self.layer2.forward(H, graph)   # (N, embed_dim)
        # L2 normalize
        norms = np.linalg.norm(H, axis=1, keepdims=True) + 1e-9
        return H / norms


# ─────────────────────────────────────────────
# 3. Multi-Factor Feature Engineering
# ─────────────────────────────────────────────

def compute_alpha_factors(prices: np.ndarray, volumes: np.ndarray,
                          fundamentals: np.ndarray) -> np.ndarray:
    """
    Compute 200+ cross-sectional alpha factors.
    prices:       (T, N) daily close
    volumes:      (T, N) daily turnover
    fundamentals: (N, F) PE / PB / ROE / etc.

    Returns (N, num_factors) factor matrix for current date.
    """
    T, N = prices.shape
    factors = []

    # ── Momentum factors ──
    ret_1d  = prices[-1] / prices[-2] - 1
    ret_5d  = prices[-1] / prices[-6] - 1
    ret_20d = prices[-1] / prices[-21] - 1
    ret_60d = prices[-1] / prices[-61] - 1
    factors += [ret_1d, ret_5d, ret_20d, ret_60d]

    # Risk-adjusted momentum (CAPM residual momentum)
    market_ret = prices.mean(axis=1)  # equal-weight market
    betas = np.array([
        np.cov(prices[:, i], market_ret)[0, 1] / (np.var(market_ret) + 1e-9)
        for i in range(N)
    ])
    alpha_ret = ret_20d - betas * (market_ret[-1] / market_ret[-21] - 1)
    factors.append(alpha_ret)

    # ── Volatility / risk factors ──
    daily_rets = np.diff(prices, axis=0) / prices[:-1]
    vol_20d = daily_rets[-20:].std(axis=0)
    vol_60d = daily_rets[-60:].std(axis=0)
    skew_20d = _skewness(daily_rets[-20:])
    kurt_20d = _kurtosis(daily_rets[-20:])
    factors += [vol_20d, vol_60d, skew_20d, kurt_20d]

    # ── Liquidity / microstructure factors ──
    amihud = (np.abs(daily_rets[-20:]) / (volumes[-20:] + 1e-9)).mean(axis=0)
    turnover_ratio = volumes[-20:].mean(axis=0) / volumes[-60:].mean(axis=0)
    factors += [amihud, turnover_ratio]

    # ── Value factors (from fundamentals) ──
    # fundamentals columns: [pe, pb, ps, pcf, roe, roa, gpm, debt_ratio]
    factors += [fundamentals[:, i] for i in range(fundamentals.shape[1])]

    # ── Reversal ──
    reversal_1d = -ret_1d
    reversal_5d = -ret_5d
    factors += [reversal_1d, reversal_5d]

    # ── Size ──
    log_mktcap = np.log(prices[-1] * volumes[-1] + 1)
    factors.append(log_mktcap)

    factor_matrix = np.column_stack(factors)  # (N, num_factors)

    # Cross-sectional winsorize and z-score each factor
    factor_matrix = _winsorize_zscore(factor_matrix)
    return factor_matrix


def _skewness(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=0)
    s = x.std(axis=0) + 1e-9
    return ((x - m) ** 3).mean(axis=0) / s ** 3


def _kurtosis(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=0)
    s = x.std(axis=0) + 1e-9
    return ((x - m) ** 4).mean(axis=0) / s ** 4 - 3


def _winsorize_zscore(X: np.ndarray, q: float = 0.01) -> np.ndarray:
    """Winsorize at q/1-q then cross-sectional z-score."""
    lo = np.nanquantile(X, q, axis=0)
    hi = np.nanquantile(X, 1 - q, axis=0)
    X = np.clip(X, lo, hi)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + 1e-9
    return (X - mu) / sd


# ─────────────────────────────────────────────
# 4. Cross-Sectional Return Forecast (LGBM wrapper)
#    In production: lgb.LGBMRegressor / xgb.XGBRegressor
# ─────────────────────────────────────────────

class CrossSectionalForecaster:
    """
    Combines GAT embedding with alpha factors for return prediction.
    Feature vector per stock: [gat_embed (32) || alpha_factors (200+)] ≈ 232-dim
    Target: forward 20-day cross-sectional rank return
    """
    def __init__(self, embed_dim: int = 32):
        self.embed_dim = embed_dim
        self.feature_importance: Optional[np.ndarray] = None
        # In production: self.model = lgb.LGBMRegressor(...)

    def build_features(self, gat_embeddings: np.ndarray,
                       alpha_factors: np.ndarray) -> np.ndarray:
        """Concat GAT embedding with engineered factors."""
        return np.concatenate([gat_embeddings, alpha_factors], axis=1)

    def predict_rank_return(self, features: np.ndarray) -> np.ndarray:
        """
        Dummy linear predictor (replace with trained LGBM).
        Returns cross-sectional rank score ∈ [0, 1].
        """
        W = np.random.randn(features.shape[1]) * 0.01
        raw = features @ W
        # Cross-sectional rank
        ranks = raw.argsort().argsort().astype(float)
        return ranks / (len(ranks) - 1)


# ─────────────────────────────────────────────
# 5. Long-Short Portfolio via Convex Risk Budgeting
#    Lagrangian formulation:
#      min  w^T Σ w  -  λ * α^T w
#      s.t. sum(w_long) = 1, sum(w_short) = -1
#           ||w||_1 <= C  (position limit)
#           w_i >= 0  for longs, w_i <= 0 for shorts
# ─────────────────────────────────────────────

def construct_longshort_portfolio(
    alpha_scores: np.ndarray,     # (N,) predicted rank returns
    cov_matrix: np.ndarray,       # (N, N) return covariance (rolling 60d)
    top_k: int = 50,
    lam: float = 2.0,             # risk-aversion λ
    tc_penalty: float = 0.001,    # transaction cost penalty per unit turnover
    prev_weights: Optional[np.ndarray] = None,
    max_iter: int = 200,
    lr: float = 0.01,
) -> np.ndarray:
    """
    Gradient-based mean-variance optimization with transaction cost penalty.

    Lagrangian:
      L(w) = w^T Σ w - λ * α^T w + ρ * ||w - w_prev||_1
    Solved via projected gradient descent.
    """
    N = len(alpha_scores)
    if prev_weights is None:
        prev_weights = np.zeros(N)

    # Select long / short universe by rank
    order = alpha_scores.argsort()
    short_idx = order[:top_k]
    long_idx = order[-top_k:]

    # Initialize: uniform within each leg
    w = np.zeros(N)
    w[long_idx]  = 1.0 / top_k
    w[short_idx] = -1.0 / top_k

    def portfolio_risk(w): return w @ cov_matrix @ w
    def alpha_term(w):     return alpha_scores @ w
    def tc_term(w):        return tc_penalty * np.abs(w - prev_weights).sum()
    def objective(w):      return portfolio_risk(w) - lam * alpha_term(w) + tc_term(w)

    def grad(w):
        g_risk  = 2 * cov_matrix @ w
        g_alpha = -lam * alpha_scores
        g_tc    = tc_penalty * np.sign(w - prev_weights)
        return g_risk + g_alpha + g_tc

    for _ in range(max_iter):
        g = grad(w)
        w_new = w - lr * g

        # Project: enforce long/short sign constraints
        w_new[long_idx]  = np.maximum(w_new[long_idx],  0)
        w_new[short_idx] = np.minimum(w_new[short_idx], 0)

        # Renormalize each leg to unit notional
        long_sum = w_new[long_idx].sum()
        if long_sum > 1e-9:
            w_new[long_idx] /= long_sum

        short_sum = np.abs(w_new[short_idx]).sum()
        if short_sum > 1e-9:
            w_new[short_idx] /= short_sum
            w_new[short_idx] *= -1

        w = w_new

    return w


# ─────────────────────────────────────────────
# 6. End-to-End Pipeline
# ─────────────────────────────────────────────

def run_gat_alpha_pipeline(
    prices: np.ndarray,        # (T, N)
    volumes: np.ndarray,       # (T, N)
    fundamentals: np.ndarray,  # (N, F)
    node_features: np.ndarray, # (N, D) initial node features (e.g. price stats)
    sector_matrix: np.ndarray, # (N, N)
    supplychain_matrix: np.ndarray,
    factor_corr_matrix: np.ndarray,
    prev_weights: Optional[np.ndarray] = None,
) -> Dict:
    T, N = prices.shape

    # Step 1: build heterogeneous stock graph
    graph = build_stock_graph(sector_matrix, supplychain_matrix, factor_corr_matrix)
    print(f"[Graph] nodes={N}, edges={graph.edge_index.shape[1]}")

    # Step 2: GAT forward pass → dense embeddings
    gat = GAT(in_dim=node_features.shape[1], hidden_dim=64, embed_dim=32, n_heads=4)
    embeddings = gat.forward(node_features, graph)        # (N, 32)
    print(f"[GAT] embedding shape: {embeddings.shape}")

    # Step 3: compute alpha factors
    alpha_factors = compute_alpha_factors(prices, volumes, fundamentals)  # (N, F)
    print(f"[Factors] factor matrix shape: {alpha_factors.shape}")

    # Step 4: cross-sectional return forecast
    forecaster = CrossSectionalForecaster(embed_dim=32)
    features = forecaster.build_features(embeddings, alpha_factors)       # (N, 32+F)
    alpha_scores = forecaster.predict_rank_return(features)               # (N,)

    # Step 5: rolling covariance estimate (shrinkage towards identity)
    daily_rets = np.diff(prices[-62:], axis=0) / prices[-62:-1]
    Sigma_sample = np.cov(daily_rets.T)
    # Ledoit-Wolf-style linear shrinkage
    shrink = 0.2
    Sigma = (1 - shrink) * Sigma_sample + shrink * np.eye(N) * np.diag(Sigma_sample).mean()

    # Step 6: portfolio optimization
    weights = construct_longshort_portfolio(
        alpha_scores, Sigma, top_k=50, lam=2.0, tc_penalty=0.001,
        prev_weights=prev_weights
    )

    long_stocks  = np.where(weights > 1e-4)[0]
    short_stocks = np.where(weights < -1e-4)[0]
    print(f"[Portfolio] long={len(long_stocks)}, short={len(short_stocks)}")
    print(f"[Portfolio] expected alpha={alpha_scores @ weights:.4f}, "
          f"risk={np.sqrt(weights @ Sigma @ weights) * np.sqrt(252):.4f}")

    return {"weights": weights, "alpha_scores": alpha_scores, "embeddings": embeddings}


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    N, T, F = 200, 120, 8      # scaled-down demo (real: N=4000)

    prices       = 10 * np.exp(np.cumsum(np.random.randn(T, N) * 0.02, axis=0))
    volumes      = np.abs(np.random.randn(T, N)) * 1e6
    fundamentals = np.abs(np.random.randn(N, F))
    node_features = np.random.randn(N, 16)   # initial node features

    sector_matrix = (np.random.randint(0, 10, (N, N)) == 0).astype(float)
    np.fill_diagonal(sector_matrix, 0)
    supplychain_matrix = np.random.rand(N, N) * (np.random.rand(N, N) > 0.97)
    factor_corr_matrix = np.corrcoef(fundamentals)

    result = run_gat_alpha_pipeline(
        prices, volumes, fundamentals, node_features,
        sector_matrix, supplychain_matrix, factor_corr_matrix
    )
    print("\nTop 5 long positions:", np.argsort(result['weights'])[-5:][::-1])
    print("Top 5 short positions:", np.argsort(result['weights'])[:5])
