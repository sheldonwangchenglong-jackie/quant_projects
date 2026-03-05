"""
量化项目演示 - 纯 NumPy 版本
王成龙 - 量化研究项目

无需 PyTorch，直接运行展示核心逻辑
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

print("=" * 70)
print("量化项目演示 - 纯 NumPy 版本")
print("王成龙 - 三个核心项目")
print("=" * 70)

np.random.seed(42)

# ============================================================
# 项目 1: GAT 多因子 Alpha 策略 (简化版)
# ============================================================
print("\n" + "=" * 70)
print("项目 1: GAT 多因子 Alpha 策略")
print("=" * 70)

print("\n[1] 生成模拟股票数据...")
num_stocks = 500
num_factors = 50

# 生成因子数据
factor_data = pd.DataFrame(
    np.random.randn(num_stocks, num_factors),
    columns=[f'factor_{i}' for i in range(num_factors)]
)

# 生成目标收益率 (与部分因子相关)
true_weights = np.zeros(num_factors)
true_weights[:10] = np.random.randn(10) * 0.1
target_returns = factor_data.values @ true_weights + np.random.randn(num_stocks) * 0.05

print(f"   - 股票数量：{num_stocks}")
print(f"   - 因子数量：{num_factors}")
print(f"   - 收益率均值：{target_returns.mean():.4f}")
print(f"   - 收益率标准差：{target_returns.std():.4f}")

print("\n[2] 构建股票相关性图...")
returns_corr = factor_data.pct_change().dropna().corr()

# 简化：用因子相关性代替
factor_corr = factor_data.corr()
edge_threshold = 0.8

num_edges = 0
for i in range(min(50, num_stocks)):  # 只计算前 50 只股票
    for j in range(i + 1, min(50, num_stocks)):
        if abs(factor_corr.iloc[i, j]) > edge_threshold:
            num_edges += 1

print(f"   - 检测到强相关边数 (前 50 只股票): {num_edges}")

print("\n[3] 训练简化版多因子模型 (线性回归)...")
from sklearn.linear_model import Ridge

# 训练集/测试集分割
train_size = int(0.8 * num_stocks)
X_train = factor_data.values[:train_size]
y_train = target_returns[:train_size]
X_test = factor_data.values[train_size:]
y_test = target_returns[train_size:]

# 训练模型
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 计算 IC
ic, ic_pvalue = spearmanr(predictions, y_test)
ric = spearmanr(np.argsort(predictions), np.argsort(y_test))[0]

print(f"   - 测试集 IC: {ic:.4f} (p-value: {ic_pvalue:.4f})")
print(f"   - 测试集 Rank IC: {ric:.4f}")

print("\n[4] 构建多空组合...")
num_long = 50
num_short = 50

long_idx = np.argsort(predictions)[-num_long:]
short_idx = np.argsort(predictions)[:num_short]

# 模拟下一期收益
long_return = y_test[long_idx].mean()
short_return = y_test[short_idx].mean()
portfolio_return = long_return - short_return

print(f"   - 多头组合收益：{long_return:.4f}")
print(f"   - 空头组合收益：{short_return:.4f}")
print(f"   - 多空组合收益：{portfolio_return:.4f}")

print("\n✅ 项目 1 演示完成！")
print("   完整代码：01_gat_multi_factor_alpha/train.py")

# ============================================================
# 项目 2: 亚式期权蒙特卡洛定价
# ============================================================
print("\n" + "=" * 70)
print("项目 2: 亚式期权蒙特卡洛定价")
print("=" * 70)

print("\n[1] 期权参数设置...")
S0 = 100          # 标的价格
K = 100           # 行权价
r = 0.05          # 无风险利率
sigma = 0.25      # 波动率
T = 1.0           # 到期时间
n_avg = 52        # 平均次数 (每周)
num_paths = 50000 # 模拟路径数

print(f"   - 标的价格：{S0}")
print(f"   - 行权价：{K}")
print(f"   - 无风险利率：{r:.2%}")
print(f"   - 波动率：{sigma:.2%}")
print(f"   - 到期时间：{T} 年")
print(f"   - 平均次数：{n_avg}")

print("\n[2] 蒙特卡洛模拟 ( Plain MC )...")
dt = T / n_avg

# 生成路径
prices = np.zeros((num_paths, n_avg + 1))
prices[:, 0] = S0

for t in range(1, n_avg + 1):
    Z = np.random.standard_normal(num_paths)
    prices[:, t] = prices[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# 算术平均
avg_prices = prices.mean(axis=1)

# 看涨期权收益
payoffs = np.maximum(avg_prices - K, 0)
discount_factor = np.exp(-r * T)
price_plain = discount_factor * payoffs.mean()
se_plain = discount_factor * payoffs.std() / np.sqrt(num_paths)

print(f"   - 亚式期权价格：{price_plain:.4f}")
print(f"   - 标准误差：{se_plain:.4f}")
print(f"   - 95% 置信区间：[{price_plain - 1.96*se_plain:.4f}, {price_plain + 1.96*se_plain:.4f}]")

print("\n[3] 对偶变量法 (Antithetic Variates)...")
num_paths_half = num_paths // 2

prices_antithetic = np.zeros((num_paths_half, n_avg + 1))
prices_antithetic[:, 0] = S0

for t in range(1, n_avg + 1):
    Z = np.random.standard_normal(num_paths_half)
    Z_combined = np.concatenate([Z, -Z])  # 对偶变量
    prices_antithetic_full = prices_antithetic[:, t-1].repeat(2) * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_combined)
    prices_antithetic = np.column_stack([prices_antithetic, prices_antithetic_full])

avg_prices_anti = prices_antithetic.mean(axis=1)
payoffs_anti = np.maximum(avg_prices_anti - K, 0)
price_anti = discount_factor * payoffs_anti.mean()
se_anti = discount_factor * payoffs_anti.std() / np.sqrt(len(payoffs_anti))

variance_reduction = 1 - (se_anti ** 2) / (se_plain ** 2)

print(f"   - 亚式期权价格：{price_anti:.4f}")
print(f"   - 标准误差：{se_anti:.4f}")
print(f"   - 方差缩减：{variance_reduction:.2%}")

print("\n[4] Black-Scholes 近似比较...")
# 调整波动率
sigma_adj = sigma * np.sqrt((2 * n_avg + 1) / (6 * (n_avg + 1)))

d1 = (np.log(S0 / K) + (r + 0.5 * sigma_adj**2) * T) / (sigma_adj * np.sqrt(T))
d2 = d1 - sigma_adj * np.sqrt(T)

bs_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

print(f"   - BS 近似价格：{bs_price:.4f}")
print(f"   - 与 MC 差异：{abs(price_plain - bs_price):.4f} ({abs(price_plain - bs_price)/bs_price*100:.2f}%)")

print("\n✅ 项目 2 演示完成！")
print("   完整代码：02_asian_option_monte_carlo/pricing.py")

# ============================================================
# 项目 3: 深度强化学习期货执行 (简化版)
# ============================================================
print("\n" + "=" * 70)
print("项目 3: 深度强化学习期货执行")
print("=" * 70)

print("\n[1] 执行环境设置...")
num_contracts = 5
time_horizon = 50
initial_position = 1000
market_impact = 0.001
transaction_cost = 0.0005

print(f"   - 合约数量：{num_contracts}")
print(f"   - 执行时间：{time_horizon} 步")
print(f"   - 初始仓位：{initial_position}")
print(f"   - 市场冲击系数：{market_impact}")
print(f"   - 交易成本：{transaction_cost}")

print("\n[2] 生成模拟价格路径...")
price_data = np.random.randn(time_horizon + 100, num_contracts).cumsum(axis=0) + 100
print(f"   - 价格数据形状：{price_data.shape}")

print("\n[3] 执行策略比较...")

# 策略 1: TWAP (时间加权平均价格)
twap_quantity = initial_position / time_horizon
twap_costs = []
position = initial_position

for t in range(time_horizon):
    current_price = price_data[t]
    impact = market_impact * np.abs(twap_quantity)
    cost = (current_price.mean() - price_data[0].mean()) * twap_quantity + transaction_cost * twap_quantity
    twap_costs.append(cost)
    position -= twap_quantity

twap_total_cost = sum(twap_costs)

# 策略 2: 简化版 RL 策略 (根据价格动量调整)
rl_costs = []
position = initial_position

for t in range(time_horizon):
    # 简单动量策略：价格下跌时多执行
    if t > 0:
        momentum = (price_data[t] - price_data[t-1]).mean()
        base_quantity = initial_position / time_horizon
        adjustment = 1 - np.sign(momentum) * 0.2  # 价格跌时多卖
        quantity = base_quantity * adjustment * (position / initial_position)
    else:
        quantity = initial_position / time_horizon
    
    current_price = price_data[t]
    impact = market_impact * np.abs(quantity)
    cost = (current_price.mean() - price_data[0].mean()) * quantity + transaction_cost * quantity + market_impact * quantity**2
    rl_costs.append(cost)
    position -= quantity

rl_total_cost = sum(rl_costs)

improvement = (twap_total_cost - rl_total_cost) / twap_total_cost * 100

print(f"   - TWAP 总成本：{twap_total_cost:.4f}")
print(f"   - 简化 RL 总成本：{rl_total_cost:.4f}")
print(f"   - 成本改善：{improvement:.2f}%")

print("\n[4] 执行轨迹分析...")
print(f"   - TWAP 平均每步成本：{np.mean(twap_costs):.4f}")
print(f"   - RL 平均每步成本：{np.mean(rl_costs):.4f}")
print(f"   - TWAP 成本波动率：{np.std(twap_costs):.4f}")
print(f"   - RL 成本波动率：{np.std(rl_costs):.4f}")

print("\n✅ 项目 3 演示完成！")
print("   完整代码：03_deep_rl_futures_execution/train_ppo.py")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("🎉 所有项目演示完成！")
print("=" * 70)
print("\n📊 项目汇总:")
print("  1. GAT 多因子 Alpha - IC: {:.4f}, Rank IC: {:.4f}".format(ic, ric))
print("  2. 亚式期权定价 - 价格：{:.4f}, 方差缩减：{:.2f}%".format(price_plain, variance_reduction * 100))
print("  3. 期货执行 RL - 成本改善：{:.2f}%".format(improvement))
print("\n💡 完整代码已保存在:")
print("   /Users/jackiewang/.openclaw/workspace/quant_projects/")
print("=" * 70)
