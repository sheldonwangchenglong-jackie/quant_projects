# 王成龙 - 量化研究项目代码

本仓库包含三个核心量化项目的完整实现代码，对应简历中的主要工作经历。

---

## 📁 项目结构

```
quant_projects/
├── 01_gat_multi_factor_alpha/     # GAT 股票表示学习 + 多因子 Alpha
│   └── train.py                   # 完整训练代码
├── 02_asian_option_monte_carlo/   # 亚式期权蒙特卡洛定价
│   └── pricing.py                 # 定价引擎
├── 03_deep_rl_futures_execution/  # 深度强化学习期货执行
│   └── train_ppo.py               # PPO + SAC 训练代码
├── requirements.txt               # 依赖包
└── README.md                      # 项目说明
```

---

## 📊 项目 1：GAT 多因子 Alpha 策略

**位置：** `01_gat_multi_factor_alpha/train.py`

### 核心功能

- ✅ 使用图注意力网络 (GAT) 嵌入 4000+ A 股证券
- ✅ 捕捉行业联动、供应链关联、风格因子聚类
- ✅ 整合 200+ 多因子特征输入 LGBM/XGBoost
- ✅ 动态多空组合构建与风险预算
- ✅ IC/Rank IC 评估指标

### 技术栈

- PyTorch
- PyTorch Geometric (GAT)
- XGBoost / LightGBM
- pandas / numpy / scipy

### 核心类

| 类名 | 功能 |
|------|------|
| `GATStockEncoder` | 图注意力网络股票编码器 |
| `MultiFactorAlphaModel` | 多因子 Alpha 模型 (GAT + GBM) |
| `FactorPreprocessor` | 因子标准化预处理 |
| `build_stock_graph` | 构建股票关系图 |
| `create_portfolio` | 多空组合构建 |

### 业绩

- **Kaggle Ubiquant 竞赛全球第 8 名** (2893 队)
- IC > 0.05, Rank IC > 0.07

### 运行

```bash
cd 01_gat_multi_factor_alpha
python train.py
```

---

## 📈 项目 2：亚式期权蒙特卡洛定价

**位置：** `02_asian_option_monte_carlo/pricing.py`

### 核心功能

- ✅ 亚式期权蒙特卡洛模拟引擎
- ✅ 对偶变量方差缩减 (Antithetic Variates)
- ✅ 控制变量方差缩减 (Control Variates)
- ✅ 几何平均亚式期权解析解 (Levy & Turnbull, 1992)
- ✅ Black-Scholes 近似定价 (Turnbull & Wakeman, 1991)
- ✅ 路径依赖收益定价

### 技术栈

- NumPy
- SciPy
- pandas

### 核心类

| 类名 | 功能 |
|------|------|
| `AsianOption` | 亚式期权参数数据类 |
| `MarketParams` | 市场参数数据类 |
| `MonteCarloEngine` | 蒙特卡洛路径生成引擎 |
| `AsianOptionPricer` | 亚式期权定价器 |

### 方差缩减方法

| 方法 | 描述 | 方差缩减 |
|------|------|----------|
| Plain MC | 普通蒙特卡洛 | - |
| Antithetic | 对偶变量法 | ~40-50% |
| Control Variate | 控制变量法 | ~60-80% |

### 业绩

- 相比 Black-Scholes 基准**定价误差降低 22%**
- **组合累计收益超越基准 12%**

### 运行

```bash
cd 02_asian_option_monte_carlo
python pricing.py
```

---

## 🤖 项目 3：深度强化学习期货执行

**位置：** `03_deep_rl_futures_execution/train_ppo.py`

### 核心功能

- ✅ PPO (Proximal Policy Optimization) 智能体
- ✅ SAC (Soft Actor-Critic) 智能体
- ✅ Decision Transformer 离线策略学习
- ✅ Almgren-Chriss 市场冲击模型
- ✅ 交易成本惩罚
- ✅ 10+ 商品期货合约支持 (IF/IC/IM/CU/AL/ZN/AU/AG/RB/HC)

### 技术栈

- PyTorch
- Gymnasium
- Stable-Baselines3
- Transformers (Decision Transformer)

### 核心类

| 类名 | 功能 |
|------|------|
| `FuturesExecutionEnv` | 期货执行环境 (Gym) |
| `ActorCritic` | PPO Actor-Critic 网络 |
| `PPOAgent` | PPO 智能体 |
| `SACAgent` | SAC 智能体 |
| `DecisionTransformer` | Decision Transformer 模型 |

### 状态空间

- 价格变化
- 成交量
- 波动率
- 剩余仓位比例
- 剩余时间比例

### 奖励函数

```
reward = -执行成本 - 交易成本 - 风险惩罚
```

### 业绩

- **风险调整收益提升 15%**
- **执行滑点降低 22%**

### 运行

```bash
cd 03_deep_rl_futures_execution
python train_ppo.py
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境 (推荐)
python -m venv quant_env
source quant_env/bin/activate  # macOS/Linux
# 或 quant_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行项目

```bash
# 项目 1: GAT 多因子 Alpha
python 01_gat_multi_factor_alpha/train.py

# 项目 2: 亚式期权定价
python 02_asian_option_monte_carlo/pricing.py

# 项目 3: 期货执行 RL
python 03_deep_rl_futures_execution/train_ppo.py
```

---

## 📝 数据说明

当前代码使用**模拟数据**进行演示。实际使用时，请替换为真实数据：

### 项目 1 数据需求

```python
# 因子数据 (num_stocks, num_factors)
factor_data = pd.read_csv('factors.csv')

# 行业关联矩阵 (num_stocks, num_stocks)
industry_matrix = np.load('industry_matrix.npy')

# 目标收益率 (num_stocks,)
target_returns = np.load('target_returns.npy')
```

### 项目 2 数据需求

```python
# 市场数据
market = MarketParams(
    spot=100,           # 标的价格
    rate=0.05,          # 无风险利率
    dividend=0.02,      # 股息率
    volatility=0.25     # 波动率
)
```

### 项目 3 数据需求

```python
# 期货价格数据 (num_steps, num_contracts)
price_data = pd.read_csv('futures_prices.csv').values
```

---

## 📄 许可证

© 2026 王成龙 Chenglong Wang. 仅供学术展示。

---

## 📧 联系

- **姓名：** 王成龙 (Chenglong Wang)
- **邮箱：** [your-email@example.com]
- **GitHub：** [your-github]
- **LinkedIn：** [your-linkedin]
