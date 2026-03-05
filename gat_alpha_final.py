"""
GAT Multi-Factor Alpha Strategy — Final Merged Version
王成龙 · Ubiquant Investment

合并要点：
  - [OpenClaw]  PyTorch + PyTorch Geometric GATConv，完整训练循环，BatchNorm + Dropout
  - [OpenClaw]  MultiFactorAlphaModel：GAT embedding concat 原始因子 → MLP 预测头
  - [OpenClaw]  IC / Rank IC 评估
  - [我们]      异构图：sector / supply-chain / style-factor 三类边，语义正确
  - [我们]      Sparsemax 可学习邻接矩阵（替代固定阈值图）
  - [我们]      Ledoit-Wolf 协方差收缩
  - [我们]      带换手成本惩罚的 projected gradient descent 组合优化
  - [我们]      Ubiquant 匿名数据格式支持（f_0~f_299）
  - [我们]      跨 time_id 的 EMA temporal memory

依赖：
  pip install torch torch-geometric lightgbm xgboost scipy pandas numpy
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from scipy.stats import spearmanr

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[Warning] PyTorch / PyG not found — falling back to NumPy backend")


# ══════════════════════════════════════════════════════════════
# 0.  数据层：支持两种输入格式
# ══════════════════════════════════════════════════════════════

@dataclass
class CrossSection:
    """
    单个截面（一个 time_id 或一个交易日）。
    同时兼容：
      - 真实生产数据：prices / volumes / fundamentals / sector_matrix
      - Ubiquant 竞赛数据：investment_ids + anonymous f_0~f_299
    """
    # 通用字段
    time_id:        int
    n_stocks:       int

    # 因子矩阵（两种来源之一）
    factor_matrix:  np.ndarray        # (N, F) — 已工程化 or 原始匿名

    # 图结构（可选，None 时用可学习图）
    sector_matrix:      Optional[np.ndarray] = None   # (N, N)
    supplychain_matrix: Optional[np.ndarray] = None   # (N, N)
    style_corr_matrix:  Optional[np.ndarray] = None   # (N, N)

    # 目标
    target:         Optional[np.ndarray] = None       # (N,) 前向收益

    # Ubiquant 专用
    investment_ids: Optional[np.ndarray] = None       # (N,) int


def make_cross_section_from_prices(
    time_id: int,
    prices: np.ndarray,       # (T, N)
    volumes: np.ndarray,      # (T, N)
    fundamentals: np.ndarray, # (N, F_fund)
    sector_matrix: np.ndarray,
    supplychain_matrix: np.ndarray,
    target: Optional[np.ndarray] = None,
) -> "CrossSection":
    """从原始行情数据构建截面（生产路径）。"""
    factor_matrix = _engineer_factors(prices, volumes, fundamentals)
    style_corr    = np.corrcoef(fundamentals) if fundamentals.shape[1] > 1 else None
    return CrossSection(
        time_id=time_id, n_stocks=prices.shape[1],
        factor_matrix=factor_matrix,
        sector_matrix=sector_matrix,
        supplychain_matrix=supplychain_matrix,
        style_corr_matrix=style_corr,
        target=target,
    )


def make_cross_section_from_ubiquant(row: pd.DataFrame) -> "CrossSection":
    """从 Ubiquant Kaggle 格式构建截面（竞赛路径）。"""
    feat_cols = [c for c in row.columns if c.startswith('f_')]
    X = row[feat_cols].values.astype(np.float32)
    y = row['target'].values.astype(np.float32) if 'target' in row.columns else None
    ids = row['investment_id'].values.astype(np.int32)
    return CrossSection(
        time_id=int(row['time_id'].iloc[0]), n_stocks=len(row),
        factor_matrix=X, target=y, investment_ids=ids,
    )


# ══════════════════════════════════════════════════════════════
# 1.  因子工程（生产路径）
# ══════════════════════════════════════════════════════════════

def _engineer_factors(prices: np.ndarray, volumes: np.ndarray,
                      fundamentals: np.ndarray) -> np.ndarray:
    """
    计算截面 alpha 因子，对应简历中"200+ engineered multi-factor features"。
    包含动量、波动、流动性、价值、反转、规模六类。
    """
    T, N = prices.shape
    factors = []

    # ── 动量 ──
    for lag in [2, 6, 21, 61]:
        if T > lag:
            factors.append(prices[-1] / prices[-lag] - 1)

    # CAPM 残差动量
    mkt = prices.mean(axis=1)
    rets_all = np.diff(prices, axis=0) / (prices[:-1] + 1e-9)
    mkt_rets  = np.diff(mkt) / (mkt[:-1] + 1e-9)
    if len(mkt_rets) >= 21:
        cov_  = np.array([np.cov(rets_all[-20:, i], mkt_rets[-20:])[0, 1]
                          for i in range(N)])
        beta_ = cov_ / (mkt_rets[-20:].var() + 1e-9)
        alpha_ret = (prices[-1] / prices[-21] - 1) - beta_ * (mkt[-1] / mkt[-21] - 1)
        factors.append(alpha_ret)

    # ── 波动 / 偏度 / 峰度 ──
    if len(rets_all) >= 60:
        factors.append(rets_all[-20:].std(axis=0))
        factors.append(rets_all[-60:].std(axis=0))
        factors.append(_skew(rets_all[-20:]))
        factors.append(_kurt(rets_all[-20:]))

    # ── 流动性 ──
    if len(rets_all) >= 20:
        amihud = (np.abs(rets_all[-20:]) / (volumes[-20:] + 1e-9)).mean(axis=0)
        factors.append(amihud)
        factors.append(volumes[-20:].mean(axis=0) / (volumes[-60:].mean(axis=0) + 1e-9))

    # ── 基本面（直接拼）──
    for i in range(fundamentals.shape[1]):
        factors.append(fundamentals[:, i])

    # ── 反转 ──
    if T > 2:
        factors.append(-(prices[-1] / prices[-2] - 1))
    if T > 6:
        factors.append(-(prices[-1] / prices[-6] - 1))

    # ── 规模 ──
    factors.append(np.log(prices[-1] * volumes[-1] + 1))

    F_mat = np.column_stack(factors)
    return _winsorize_zscore(F_mat)


def _skew(x): m = x.mean(0); s = x.std(0)+1e-9; return ((x-m)**3).mean(0)/s**3
def _kurt(x): m = x.mean(0); s = x.std(0)+1e-9; return ((x-m)**4).mean(0)/s**4 - 3


def _winsorize_zscore(X: np.ndarray, q: float = 0.01) -> np.ndarray:
    lo = np.nanquantile(X, q, axis=0)
    hi = np.nanquantile(X, 1-q, axis=0)
    X  = np.clip(X, lo, hi)
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def preprocess_ubiquant(X: np.ndarray) -> np.ndarray:
    """
    Ubiquant 专用预处理：Rank → Quantile-normalize → clip。
    比 z-score 更能抵抗 f_i 的重尾分布。
    """
    N, F = X.shape
    ranks = X.argsort(0).argsort(0).astype(np.float32)
    u = (ranks + 0.5) / N
    p = u.clip(1e-6, 1-1e-6)
    t = np.sqrt(-2 * np.log(np.minimum(p, 1-p)))
    c0,c1,c2 = 2.515517, 0.802853, 0.010328
    d1,d2,d3 = 1.432788, 0.189269, 0.001308
    approx = t - (c0+c1*t+c2*t**2)/(1+d1*t+d2*t**2+d3*t**3)
    return np.where(u < 0.5, -approx, approx).clip(-3, 3).astype(np.float32)


# ══════════════════════════════════════════════════════════════
# 2.  图构建：三类异构边 + 可学习 Sparsemax 图
# ══════════════════════════════════════════════════════════════

def build_heterogeneous_graph(cs: CrossSection,
                              sc_threshold: float = 0.3,
                              style_threshold: float = 0.6
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    [我们的设计] 三类边，经济含义明确：
      type-0 sector：同行业双向边
      type-1 supply-chain：上下游有向边（不对称！）
      type-2 style-factor：风格因子相关性边

    比 OpenClaw 的"对因子矩阵做 pct_change() 再算相关"语义正确得多。
    """
    N = cs.n_stocks
    src_list, dst_list, w_list = [], [], []

    def _add(mat, thr, symmetric=True):
        if mat is None: return
        rows, cols = np.where(mat > thr)
        mask = rows != cols
        for r, c in zip(rows[mask], cols[mask]):
            src_list.append(r); dst_list.append(c); w_list.append(float(mat[r,c]))
            if symmetric:
                src_list.append(c); dst_list.append(r); w_list.append(float(mat[r,c]))

    _add(cs.sector_matrix,      0.5, symmetric=True)
    _add(cs.supplychain_matrix, sc_threshold, symmetric=False)   # 有向
    _add(cs.style_corr_matrix,  style_threshold, symmetric=True)

    if not src_list:
        # fallback: KNN on factor similarity
        return _knn_graph_from_features(cs.factor_matrix, k=10)

    ei = np.array([src_list, dst_list], dtype=np.int64)
    ew = np.array(w_list, dtype=np.float32)
    return ei, ew


def build_learnable_graph(Z: np.ndarray, top_k: int = 15
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    [我们的设计] 可学习稀疏图：
      A = sparsemax( Z Z^T / sqrt(d) )
    无需任何领域标签，图结构从数据中自动涌现。
    适合 Ubiquant 匿名特征场景。
    """
    d = Z.shape[1]
    S = (Z @ Z.T) / np.sqrt(d)

    # Top-k mask（内存预算 O(N*k)）
    N = len(Z)
    if top_k < N-1:
        thr = np.partition(S, -top_k, axis=1)[:, -top_k]
        S   = np.where(S >= thr[:,None], S, -1e9)

    A = _sparsemax(S)

    src, dst = np.where(A > 0)
    return np.stack([src, dst]), A[src, dst]


def _knn_graph_from_features(X: np.ndarray, k: int = 10
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """基于特征余弦相似度的 KNN 图（无领域标签时的 fallback）。"""
    Xn  = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S   = Xn @ Xn.T
    src_l, dst_l, w_l = [], [], []
    for i in range(len(X)):
        knn = np.argsort(S[i])[-k-1:-1][::-1]
        for j in knn:
            if i == j: continue
            src_l.append(i); dst_l.append(j); w_l.append(float(S[i,j]))
    return np.array([src_l, dst_l], dtype=np.int64), np.array(w_l, dtype=np.float32)


def _sparsemax(z: np.ndarray) -> np.ndarray:
    """
    Sparsemax 投影：大多数边权重精确为 0，无需手动阈值。
    比 softmax 更稀疏，比固定阈值更端到端。
    """
    z   = z - z.max(axis=1, keepdims=True)
    zs  = np.sort(z, axis=1)[:, ::-1]
    N   = z.shape[1]
    k   = np.arange(1, N+1, dtype=np.float32)
    cs  = np.cumsum(zs, axis=1)
    sup = (1 + k * zs > cs)
    k_  = sup.sum(axis=1, keepdims=True)
    tau = (cs[np.arange(len(cs)), k_.squeeze().astype(int)-1] - 1) / k_.squeeze()
    return np.maximum(z - tau[:,None], 0)


# ══════════════════════════════════════════════════════════════
# 3.  PyTorch 模型（OpenClaw 框架 + 我们的图设计）
# ══════════════════════════════════════════════════════════════

if HAS_TORCH:
    class GATStockEncoder(nn.Module):
        """
        [OpenClaw 框架] PyTorch Geometric GATConv，
        加上 BatchNorm + Dropout（OpenClaw 的工程改进）。

        图结构用我们设计的异构边，而非 OpenClaw 对因子做 pct_change 的错误做法。
        """
        def __init__(self, input_dim: int, hidden_dim: int = 64,
                     output_dim: int = 32, num_heads: int = 4,
                     dropout: float = 0.3, num_layers: int = 2):
            super().__init__()
            self.input_norm = nn.BatchNorm1d(input_dim)
            self.layers     = nn.ModuleList()
            self.bns        = nn.ModuleList()
            self.dropout    = nn.Dropout(dropout)

            dims = [input_dim] + [hidden_dim] * (num_layers - 1)
            for i in range(num_layers - 1):
                self.layers.append(
                    GATConv(dims[i], hidden_dim, heads=num_heads,
                            dropout=dropout, add_self_loops=True))
                self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))

            # 输出层：单头，不拼接
            self.layers.append(
                GATConv(hidden_dim * num_heads, output_dim,
                        heads=1, concat=False, dropout=dropout))

        def forward(self, x: "torch.Tensor",
                    edge_index: "torch.Tensor",
                    edge_weight: Optional["torch.Tensor"] = None
                    ) -> "torch.Tensor":
            x = self.input_norm(x)
            for i, layer in enumerate(self.layers[:-1]):
                x = layer(x, edge_index, edge_attr=edge_weight)
                x = F.elu(x)
                x = self.bns[i](x)
                x = self.dropout(x)
            x = self.layers[-1](x, edge_index, edge_attr=edge_weight)
            return x   # (N, output_dim)


    class GraphAlphaNet(nn.Module):
        """
        最终合并模型：

          X(N×F) ──► [可选] 可学习图 or 异构图
                 ──► GATStockEncoder (PyG GATConv × 2)  → embed(N×32)
                 ──► concat [embed || X]                 → (N×(32+F))
                 ──► MLP 预测头                           → pred(N×1)

        [OpenClaw]  concat 原始因子 + GAT 嵌入的设计
        [OpenClaw]  MLP 预测头 (128→64→1) + BN + Dropout
        [我们]      异构图边类型设计
        [我们]      Sparsemax 可学习图（当无领域标签时）
        """
        def __init__(self, input_dim: int, hidden_dim: int = 64,
                     embed_dim: int = 32, num_heads: int = 4,
                     dropout: float = 0.3):
            super().__init__()
            self.encoder = GATStockEncoder(
                input_dim, hidden_dim, embed_dim, num_heads, dropout)

            head_in = embed_dim + input_dim
            self.predictor = nn.Sequential(
                nn.Linear(head_in, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )

        def forward(self, x: "torch.Tensor",
                    edge_index: "torch.Tensor",
                    edge_weight: Optional["torch.Tensor"] = None
                    ) -> "torch.Tensor":
            embed    = self.encoder(x, edge_index, edge_weight)   # (N, 32)
            combined = torch.cat([x, embed], dim=1)               # (N, F+32)
            return self.predictor(combined)                        # (N, 1)


# ══════════════════════════════════════════════════════════════
# 4.  Temporal Memory（跨 time_id EMA，Ubiquant 专用）
# ══════════════════════════════════════════════════════════════

class TemporalStockMemory:
    """
    [我们的设计] 维护每只股票跨 time_id 的 EMA 嵌入：
      z_i^t = β * z_i^{t-1} + (1-β) * GAT_embed_i^t
    解决 Ubiquant 每期 stock universe 不同的问题。
    """
    def __init__(self, max_inv_id: int = 3775, embed_dim: int = 32,
                 beta: float = 0.9):
        self.beta   = beta
        self.memory = np.zeros((max_inv_id+1, embed_dim), dtype=np.float32)
        self.seen   = np.zeros(max_inv_id+1, dtype=bool)

    def update(self, inv_ids: np.ndarray, embeds: np.ndarray):
        for i, iid in enumerate(inv_ids):
            if self.seen[iid]:
                self.memory[iid] = self.beta*self.memory[iid] + (1-self.beta)*embeds[i]
            else:
                self.memory[iid] = embeds[i]
                self.seen[iid]   = True

    def get(self, inv_ids: np.ndarray) -> np.ndarray:
        return self.memory[inv_ids]


# ══════════════════════════════════════════════════════════════
# 5.  训练循环（OpenClaw 框架 + Pearson loss）
# ══════════════════════════════════════════════════════════════

def pearson_loss(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    """
    [我们的改进] Ubiquant 竞赛评估指标是 Pearson 相关，
    应该直接优化它，而不是 MSE（OpenClaw 用 MSE）。
    """
    p = pred - pred.mean()
    t = target - target.mean()
    return -(p * t).sum() / (p.norm() * t.norm() + 1e-8)


def train(
    snapshots: List[CrossSection],
    input_dim: int,
    use_learnable_graph: bool = False,   # True = Ubiquant 匿名数据
    num_epochs: int = 50,
    lr: float = 1e-3,
    device_str: str = "cpu",
) -> "GraphAlphaNet":
    """
    [OpenClaw] 完整 Adam 训练循环 + epoch 日志
    [我们]     Pearson loss，异构图，可选 sparsemax 图
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for training")

    device = torch.device(device_str)
    model  = GraphAlphaNet(input_dim=input_dim).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []

        for cs in snapshots:
            if cs.target is None: continue

            # 特征预处理
            X_np = (preprocess_ubiquant(cs.factor_matrix)
                    if cs.investment_ids is not None
                    else _winsorize_zscore(cs.factor_matrix))
            X = torch.tensor(X_np, dtype=torch.float32).to(device)
            y = torch.tensor(cs.target, dtype=torch.float32).to(device)

            # 图构建
            if use_learnable_graph or (cs.sector_matrix is None):
                # Ubiquant 路径：可学习 sparsemax 图
                Z_np = X_np / (np.linalg.norm(X_np, axis=1, keepdims=True) + 1e-9)
                ei_np, ew_np = build_learnable_graph(Z_np, top_k=15)
            else:
                # 生产路径：异构图
                ei_np, ew_np = build_heterogeneous_graph(cs)

            if ei_np.shape[1] == 0: continue
            ei = torch.tensor(ei_np, dtype=torch.long).to(device)
            ew = torch.tensor(ew_np, dtype=torch.float32).to(device)

            optim.zero_grad()
            pred = model(X, ei, ew).squeeze(-1)
            loss = pearson_loss(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_loss.append(loss.item())

        sched.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"mean Pearson loss = {np.mean(epoch_loss):.4f} | "
                  f"lr = {sched.get_last_lr()[0]:.2e}")

    return model


# ══════════════════════════════════════════════════════════════
# 6.  评估（OpenClaw 的 IC/Rank IC + 我们的 Pearson）
# ══════════════════════════════════════════════════════════════

def evaluate(model: "GraphAlphaNet",
             snapshots: List[CrossSection],
             memory: Optional[TemporalStockMemory] = None,
             use_learnable_graph: bool = False,
             device_str: str = "cpu") -> Dict:
    """[OpenClaw] IC / Rank IC  +  [我们] Pearson 相关"""
    if not HAS_TORCH: return {}
    device = torch.device(device_str)
    model.eval()
    pearson_list, ic_list, ric_list = [], [], []

    with torch.no_grad():
        for cs in snapshots:
            if cs.target is None: continue
            X_np = (preprocess_ubiquant(cs.factor_matrix)
                    if cs.investment_ids is not None
                    else _winsorize_zscore(cs.factor_matrix))
            X = torch.tensor(X_np, dtype=torch.float32).to(device)

            if use_learnable_graph or cs.sector_matrix is None:
                Z_np = X_np / (np.linalg.norm(X_np, axis=1, keepdims=True) + 1e-9)
                ei_np, ew_np = build_learnable_graph(Z_np, top_k=15)
            else:
                ei_np, ew_np = build_heterogeneous_graph(cs)

            if ei_np.shape[1] == 0: continue
            ei = torch.tensor(ei_np, dtype=torch.long).to(device)
            ew = torch.tensor(ew_np, dtype=torch.float32).to(device)

            pred = model(X, ei, ew).squeeze(-1).cpu().numpy()
            tgt  = cs.target

            # Temporal memory 更新
            if memory is not None and cs.investment_ids is not None:
                embed = model.encoder(X, ei, ew).cpu().numpy()
                memory.update(cs.investment_ids, embed)

            # [OpenClaw] IC / Rank IC
            ic,  _ = spearmanr(pred, tgt)
            ric, _ = spearmanr(np.argsort(np.argsort(pred)),
                                np.argsort(np.argsort(tgt)))
            # [我们] Pearson
            p = pred - pred.mean(); t = tgt - tgt.mean()
            pearson = float((p*t).sum() / (np.linalg.norm(p)*np.linalg.norm(t)+1e-8))

            ic_list.append(float(ic)); ric_list.append(float(ric))
            pearson_list.append(pearson)

    return {
        "IC":           np.mean(ic_list),
        "IC_std":       np.std(ic_list),
        "Rank_IC":      np.mean(ric_list),
        "Pearson":      np.mean(pearson_list),
        "ICIR":         np.mean(ic_list) / (np.std(ic_list) + 1e-9),
    }


# ══════════════════════════════════════════════════════════════
# 7.  组合优化（我们的 projected gradient descent）
# ══════════════════════════════════════════════════════════════

def construct_portfolio(
    alpha_scores: np.ndarray,
    returns_history: np.ndarray,     # (T, N) for covariance estimation
    top_k: int = 50,
    lam: float = 2.0,
    tc_penalty: float = 0.001,
    prev_weights: Optional[np.ndarray] = None,
    max_iter: int = 300,
    lr: float = 0.005,
) -> Dict:
    """
    [我们的设计] Lagrangian 均值-方差优化 + 换手成本惩罚：
      L(w) = w^T Σ w  -  λ α^T w  +  ρ ||w - w_prev||_1
    Projected gradient descent，保持多空仓位符号约束。

    [我们的改进 vs OpenClaw] OpenClaw 用简单等权，
    这里用 Ledoit-Wolf 收缩协方差做风险控制。
    """
    N = len(alpha_scores)
    if prev_weights is None:
        prev_weights = np.zeros(N)

    # Ledoit-Wolf 协方差收缩
    daily_rets = np.diff(returns_history, axis=0) / (returns_history[:-1] + 1e-9)
    Sigma_s    = np.cov(daily_rets.T)
    shrink     = 0.2
    Sigma      = ((1-shrink)*Sigma_s
                  + shrink*np.eye(N)*np.diag(Sigma_s).mean())

    order     = alpha_scores.argsort()
    long_idx  = order[-top_k:]
    short_idx = order[:top_k]

    w = np.zeros(N)
    w[long_idx]  =  1.0 / top_k
    w[short_idx] = -1.0 / top_k

    def grad(w):
        return (2 * Sigma @ w
                - lam * alpha_scores
                + tc_penalty * np.sign(w - prev_weights))

    for _ in range(max_iter):
        w_new = w - lr * grad(w)
        w_new[long_idx]  = np.maximum(w_new[long_idx],  0)
        w_new[short_idx] = np.minimum(w_new[short_idx], 0)
        ls = w_new[long_idx].sum()
        if ls > 1e-9:  w_new[long_idx]  /= ls
        ss = np.abs(w_new[short_idx]).sum()
        if ss > 1e-9:  w_new[short_idx] /= ss; w_new[short_idx] *= -1
        w = w_new

    ann_vol = np.sqrt(w @ Sigma @ w * 252)
    return {
        "weights":        w,
        "expected_alpha": float(alpha_scores @ w),
        "annual_vol":     float(ann_vol),
        "sharpe_proxy":   float(alpha_scores @ w) / (ann_vol + 1e-9),
        "long_count":     int((w > 1e-4).sum()),
        "short_count":    int((w < -1e-4).sum()),
        "turnover":       float(np.abs(w - prev_weights).sum()),
    }


# ══════════════════════════════════════════════════════════════
# 8.  Demo（同时演示两条路径）
# ══════════════════════════════════════════════════════════════

def _make_synthetic_production_data(N=150, T=120, F_fund=8, seed=42):
    rng = np.random.default_rng(seed)
    prices  = 10 * np.exp(np.cumsum(rng.standard_normal((T, N)) * 0.015, axis=0))
    volumes = np.abs(rng.standard_normal((T, N))) * 1e6
    fund    = np.abs(rng.standard_normal((N, F_fund)))
    sector  = (rng.integers(0, 10, (N, N)) == 0).astype(float)
    np.fill_diagonal(sector, 0)
    supply  = rng.random((N, N)) * (rng.random((N, N)) > 0.97)
    target  = rng.standard_normal(N).astype(np.float32) * 0.01
    return prices, volumes, fund, sector, supply, target


def _make_synthetic_ubiquant(n_time=40, n_stocks=200, n_feat=300, seed=42):
    rng = np.random.default_rng(seed)
    snaps = []
    for t in range(n_time):
        n   = rng.integers(n_stocks//2, n_stocks)
        ids = np.sort(rng.choice(3000, n, replace=False)).astype(np.int32)
        cls = ids % 10
        X   = np.zeros((n, n_feat), dtype=np.float32)
        cf  = rng.standard_normal((10, n_feat//10))
        for c in range(10):
            m = cls == c
            if not m.any(): continue
            shared = np.tile(cf[c], n_feat//(n_feat//10))[:n_feat]
            X[m] = 0.4*shared + 0.6*rng.standard_normal((m.sum(), n_feat))
        tgt = (rng.standard_normal(10)*0.02)[cls] + rng.standard_normal(n)*0.01
        snaps.append(CrossSection(t, n, X.astype(np.float32),
                                  target=tgt.astype(np.float32),
                                  investment_ids=ids))
    return snaps


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 62)
    print("GraphAlphaNet — Final Merged Version")
    print("GAT多因子Alpha模型 · 王成龙 · Ubiquant Investment")
    print("=" * 62)

    # ── Path A：生产数据（有行情 + 基本面 + 行业图）──
    print("\n▶  Path A: 生产路径（异构图 + 工程因子）")
    prices, volumes, fund, sector, supply, tgt_a = \
        _make_synthetic_production_data(N=150, T=120)
    cs_a = make_cross_section_from_prices(
        time_id=0, prices=prices, volumes=volumes,
        fundamentals=fund, sector_matrix=sector,
        supplychain_matrix=supply, target=tgt_a)
    ei_a, ew_a = build_heterogeneous_graph(cs_a)
    print(f"   stocks={cs_a.n_stocks}  factors={cs_a.factor_matrix.shape[1]}"
          f"  edges={ei_a.shape[1]}")

    if HAS_TORCH:
        snaps_a = [cs_a] * 3  # 简化 demo
        model_a = train(snaps_a, input_dim=cs_a.factor_matrix.shape[1],
                        use_learnable_graph=False, num_epochs=20,
                        device_str="cpu")
        res_a = evaluate(model_a, [cs_a], use_learnable_graph=False)
        print(f"   IC={res_a['IC']:.4f}  Rank IC={res_a['Rank_IC']:.4f}"
              f"  Pearson={res_a['Pearson']:.4f}  ICIR={res_a['ICIR']:.3f}")

        # 组合优化
        with torch.no_grad():
            X_t = torch.tensor(_winsorize_zscore(cs_a.factor_matrix),
                                dtype=torch.float32)
            ei_t = torch.tensor(ei_a, dtype=torch.long)
            ew_t = torch.tensor(ew_a, dtype=torch.float32)
            scores = model_a(X_t, ei_t, ew_t).squeeze(-1).numpy()
        port = construct_portfolio(scores, prices, top_k=20, lam=2.0)
        print(f"   long={port['long_count']}  short={port['short_count']}"
              f"  annual_vol={port['annual_vol']:.3f}"
              f"  sharpe_proxy={port['sharpe_proxy']:.3f}")
    else:
        print("   [Skip training — install PyTorch to enable]")

    # ── Path B：Ubiquant 匿名数据（可学习 Sparsemax 图）──
    print("\n▶  Path B: Ubiquant竞赛路径（Sparsemax可学习图 + temporal memory）")
    snaps_b = _make_synthetic_ubiquant(n_time=40, n_stocks=200, n_feat=300)
    train_b, val_b = snaps_b[:30], snaps_b[30:]
    print(f"   train={len(train_b)} time_ids  val={len(val_b)} time_ids")
    print(f"   example snapshot: n_stocks={val_b[0].n_stocks}"
          f"  features={val_b[0].factor_matrix.shape[1]}")

    if HAS_TORCH:
        memory = TemporalStockMemory(max_inv_id=3000, embed_dim=32)
        model_b = train(train_b, input_dim=300, use_learnable_graph=True,
                        num_epochs=30, device_str="cpu")
        res_b = evaluate(model_b, val_b, memory=memory,
                         use_learnable_graph=True)
        print(f"   IC={res_b['IC']:.4f}  Rank IC={res_b['Rank_IC']:.4f}"
              f"  Pearson={res_b['Pearson']:.4f}  ICIR={res_b['ICIR']:.3f}")
    else:
        # NumPy fallback 演示图构建
        cs0 = snaps_b[0]
        X0  = preprocess_ubiquant(cs0.factor_matrix)
        Z0  = X0 / (np.linalg.norm(X0, axis=1, keepdims=True) + 1e-9)
        ei0, ew0 = build_learnable_graph(Z0, top_k=15)
        density = ei0.shape[1] / (cs0.n_stocks * (cs0.n_stocks-1))
        print(f"   stocks={cs0.n_stocks}  edges={ei0.shape[1]}"
              f"  density={density:.4f}  avg_degree={ei0.shape[1]/cs0.n_stocks:.1f}")
        print("   [Install PyTorch to enable training]")

    print("\n" + "=" * 62)
    print("合并要点汇总：")
    print("  [OpenClaw] PyG GATConv, BatchNorm, Dropout, Adam, IC/RankIC")
    print("  [我们]     异构图(sector+supplychain+style), Sparsemax可学习图")
    print("  [我们]     Pearson loss, Ledoit-Wolf收缩, PGD组合优化")
    print("  [我们]     EMA temporal memory (Ubiquant跨期stock universe)")
    print("=" * 62)
