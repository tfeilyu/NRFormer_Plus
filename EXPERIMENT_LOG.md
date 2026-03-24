# NRFormer+ Experiment Log

> TKDE Extension: NRFormer (KDD'25) → NRFormer+ (TKDE'26)
> Full data analysis report: `../data_analysis/DATA_ANALYSIS_REPORT.md`

---

## Baseline Results (11 baselines + NRFormer on extended dataset)

### Japan-4H (4-hour resolution, 3627 stations, 9222 timesteps)

| Model | 6th MAE | 6th RMSE | 12th MAE | 12th RMSE | 24th MAE | 24th RMSE | 24th MAPE |
|-------|---------|----------|----------|-----------|----------|-----------|-----------|
| HA | 2.35 | 9.04 | 2.48 | 9.83 | 2.66 | 11.10 | 3.12% |
| LR | 2.08 | 6.74 | 2.29 | 8.04 | 2.43 | 9.89 | 3.00% |
| XGBoost | 2.17 | 8.49 | 2.35 | 9.27 | 2.52 | 10.47 | 2.95% |
| DCRNN | 1.96 | 6.75 | 2.26 | 7.95 | 2.36 | 8.73 | 2.95% |
| STID | 1.81 | 5.94 | 1.96 | 6.76 | 2.15 | 7.95 | 2.72% |
| DLinear | 1.96 | 6.36 | 2.12 | 7.26 | 2.29 | 8.49 | 2.97% |
| PatchTST | 1.85 | 6.44 | 2.01 | 7.35 | 2.18 | 8.53 | 2.76% |
| Koopa | 1.85 | 6.34 | 1.98 | 7.23 | 2.15 | 8.48 | 2.71% |
| StemGNN | 1.95 | 6.72 | 2.19 | 7.91 | 2.34 | 8.44 | 2.94% |
| GWN | 1.93 | 6.65 | 2.07 | 7.52 | 2.24 | 8.60 | 2.84% |
| LightCTS | 1.82 | 6.03 | 1.97 | 6.88 | 2.15 | 8.08 | 2.69% |
| **NRFormer** | **1.74** | **5.81** | **1.89** | **6.67** | **2.07** | **7.92** | **2.63%** |
| NRFormer+ | - | - | - | - | - | - | - |

### Japan-1D (daily resolution, 3627 stations, 1537 timesteps)

| Model | 6th MAE | 6th RMSE | 12th MAE | 12th RMSE | 24th MAE | 24th RMSE | 24th MAPE |
|-------|---------|----------|----------|-----------|----------|-----------|-----------|
| HA | 2.37 | 11.70 | 2.57 | 13.67 | 2.94 | 17.08 | 3.37% |
| LR | 2.03 | 6.77 | 2.18 | 8.91 | 2.41 | 11.48 | 3.17% |
| XGBoost | 2.25 | 10.25 | 2.46 | 12.59 | 2.69 | 15.91 | 3.02% |
| DCRNN | 2.19 | 9.56 | 2.42 | 9.12 | 2.61 | 12.76 | 3.29% |
| STID | 2.02 | 6.93 | 2.16 | 8.99 | 2.42 | 11.57 | 3.15% |
| DLinear | 1.93 | 6.57 | 2.11 | 8.96 | 2.37 | 11.79 | 3.10% |
| PatchTST | 1.83 | 6.69 | 2.02 | 8.97 | 2.29 | 11.80 | 2.97% |
| Koopa | 1.89 | 6.78 | 2.07 | 8.99 | 2.32 | 11.69 | 2.98% |
| StemGNN | 2.19 | 9.51 | 2.46 | 10.45 | 2.61 | 12.65 | 3.22% |
| GWN | 2.15 | 9.24 | 2.37 | 11.24 | 2.59 | 12.60 | 3.21% |
| LightCTS | 1.95 | 6.60 | 2.13 | 8.18 | 2.37 | 11.50 | 3.09% |
| **NRFormer** | **1.84** | **6.58** | **2.01** | **8.92** | **2.28** | **11.65** | **2.93%** |
| NRFormer+ | - | - | - | - | - | - | - |

### Targets

**NRFormer+ must beat NRFormer on all metrics.**

Primary target (Japan-1D, 24-step): **MAE < 2.28, RMSE < 11.65, MAPE < 2.93%**

---

## Key Data Analysis Findings (驱动模型设计)

> Full report with plots: `../data_analysis/DATA_ANALYSIS_REPORT.md`

### Finding → Design Decision Mapping

| # | Data Finding | Evidence | Design Decision |
|---|-------------|----------|-----------------|
| F1 | Extreme autocorrelation | ACF(1)=0.946, lag-60 still 0.611 | **Residual learning**: predict ΔX, not X |
| F2 | 2-order-of-magnitude range | Skewness=18, 40-7170 nSv/h | **Log-space modeling**: log(1+x) transform |
| F3 | Radon washout on rain days | Humid +1.25 vs dry -0.02 nSv/h | **Rain-aware gating**: dryness index feature |
| F4 | Graph fragmentation | 235 isolated nodes, 288 components | **Virtual global nodes** in spatial attention |
| F5 | Anomalies span 636 km | Far beyond 10km graph range | **Multi-scale spatial**: local + regional + global |
| F6 | Prefecture clustering 3.45x | Within r=0.473, between r=0.137 | **Region-aware attention** with cluster bias |
| F7 | No station-to-station propagation | 9.5% neighbor sync, lag-0 dominant | **Revise diffusion module** → synchronous response |
| F8 | Wind speed useless, temp/dew useful | r_wind=-0.015, r_temp=0.214 | Keep temperature path, de-emphasize wind speed |
| F9 | Non-stationary (3%/yr decay) | ADF p=0.106, effective half-life 22.6yr | RevIN necessary; consider explicit detrending |
| F10 | 36% stations hard to predict | AR(1) R²<0.5 for 36% stations | Focus model capacity on hard-to-predict stations |

---

## Optimization Roadmap (Data-Driven)

```
Iteration 1 (P0): Residual Learning + Log-space          ← START HERE
    Basis: F1 (ACF=0.946) + F2 (skewness=18)
    Change: ~25 lines in NRFormer_Plus.py
    Goal: Establish improved NRFormer+ baseline

Iteration 2 (P0): Rain-aware Gating
    Basis: F3 (radon washout confirmed)
    Change: ~30 lines
    Goal: Improve sudden-change prediction

Iteration 3 (P1): Multi-scale Spatial Graph
    Basis: F4 (graph broken) + F5 (636km anomaly range)
    Change: ~60 lines
    Goal: Fix isolated-node information deficit

Iteration 4 (P1): Region-aware Attention
    Basis: F6 (prefecture 3.45x) + F10 (36% hard stations)
    Change: ~40 lines
    Goal: Better spatial grouping

Iteration 5 (P2): Revised Physics Module
    Basis: F7 (no propagation) + F8 (wind useless)
    Change: ~50 lines
    Goal: Physically correct constraints

Iteration 6 (P2): Hyperparameter Tuning + Final
    Basis: All previous iteration results
    Goal: Optimal architecture for paper
```

---

## NRFormer+ Iteration History

### Iteration 0: Architecture Foundation (Before Data Analysis)

**Date:** 2026-03-24

**Changes from KDD NRFormer:**
1. Physics module: AtmosphericDiffusionModule with real graph Laplacian
2. Temporal gradient encoding dC/dt added to physics triplet [C, D, ∇²C, dC/dt]
3. MeteorologicalEncoder: separate wind/temperature pathways with temporal conv
4. LocationEncoder: deeper MLP (2→32→64→D)
5. TemporalEncoder: day-of-year embedding
6. 3-way gated fusion: radiation + temporal + spatial (was 2-way concat)
7. FFN expansion 1x → 4x in temporal self-attention
8. Learned temporal positional encoding
9. Reduced temporal dropout: 0.3 → 0.1
10. Spatial attention heads: 8 → 4 (head_dim doubled)
11. Fixed LR schedule: milestones [50,60,70,80] γ=0.01 → [100,200,250] γ=0.5
12. Fixed NOAA normalization data leakage
13. Fixed missing data threshold
14. Train-time output clamping

**Status:** Completed — Phase 1 capacity search results below.

#### Phase 1 Results: Capacity Search (Japan-1D)

| Exp ID | hidden | TL | SL | BS | Params | Best Epoch | V-MAE | T-MAE | T-RMSE | T-MAPE |
|--------|--------|----|----|----|--------|------------|-------|-------|--------|--------|
| **p1_baseline** | **32** | 3 | 2 | 8 | **0.39M** | **10** | **2.2548** | **2.3231** | **10.9806** | **0.03%** |
| p1_h64 | 64 | 3 | 2 | 8 | 0.97M | 15 | 2.2528 | 2.3341 | 11.0096 | 0.03% |

Per-horizon MAE (best run p1_baseline):
| Step | MAE |
|------|-----|
| 6 | 2.1070 |
| 9 | 2.2825 |
| 12 | 2.3786 |
| 24 | 2.7343 |

#### Phase 1 Analysis

**1. NRFormer+ vs NRFormer (current gap):**

| Metric | NRFormer (target) | NRFormer+ (p1_baseline) | Gap | Status |
|--------|------------------|------------------------|-----|--------|
| 6th MAE | 1.84 | 2.107 | +0.267 (+14.5%) | :x: Behind |
| 12th MAE | 2.01 | 2.379 | +0.369 (+18.4%) | :x: Behind |
| 24th MAE | 2.28 | 2.734 | +0.454 (+19.9%) | :x: Behind |
| 24th RMSE | 11.65 | 10.98 | **-0.67 (-5.7%)** | :white_check_mark: Better! |
| 24th MAPE | 2.93% | 0.03% | - | MAPE异常低，可能是计算问题 |

**2. Key observations:**
- **MAE明显落后NRFormer** (14-20%)，但**RMSE反而更好** (-5.7%)。这说明NRFormer+在极端值（大误差）上表现更好，但在普通样本上不如NRFormer。
- **MAPE异常** (0.03% vs NRFormer 2.93%)：几乎可以确定是MAPE计算有bug或单位问题，需要检查。
- **hidden=32优于hidden=64**：更大模型 (+0.01 MAE, +2.5x参数) 反而更差，说明当前数据量下模型已经过参数化，或者训练不充分。
- **Early stopping过早** (epoch 10/15)：200 epochs但只训练了10-15轮就停了，说明validation loss快速上升，存在**过拟合或训练不稳定**问题。
- **V-MAE ≈ T-MAE**：验证和测试性能接近，没有过拟合到验证集。

**3. Root cause analysis:**
- **Early stopping 10轮就触发** → 模型初始学习很快但随后validation loss开始恶化。可能原因：
  - (a) 学习率0.001过大，初始下降快但随后震荡
  - (b) 缺乏warmup导致初期训练不稳定
  - (c) batch_size=8太小，梯度噪声大
- **MAE落后但RMSE更好** → NRFormer+可能在高辐射站点（>1000 nSv/h）表现好（降低极端误差），但在大量低辐射站点（78%在40-100 nSv/h）上不如NRFormer。这恰好支持我们的**对数空间建模**方案。

**4. Decision:**
- **hidden=32** 是更好的选择（0.39M参数足够）
- 不继续Phase 2 (depth/batch search)，**直接进入数据驱动改进 Iteration 1**
- 优先解决：(a) 对数空间建模解决MAE问题 (b) 残差学习利用高自相关 (c) 检查MAPE计算bug (d) 考虑训练策略（warmup, 更小的初始LR）

---

### Iteration 1: Residual Learning + Log-space Modeling

**Date:** TBD

**Data motivation:**
- F1: ACF(1)=0.946 → 94.6% signal is "repeat yesterday", model should learn the 5.4% residual
- F2: Skewness=18, range 40-7170 → log transform compresses 2 orders of magnitude

**Technical changes:**

**A. Log-space transform:**
```python
# In PGRT2.forward(), before RevIN:
x = torch.log1p(x)  # log(1+x)
# At output, after denormalization:
output = torch.expm1(output)  # exp(x)-1
```

**B. Residual prediction:**
```python
# In PGRT2.forward(), at output stage:
last_value = inputs[:, 0:1, :, -1:]  # last known radiation
predicted_delta = end_conv2(F.relu(end_conv1(x_fused)))
output = last_value + predicted_delta  # residual connection
```

**Experiments:**

| Exp ID | Log-space | Residual | hidden | epochs | Test MAE | Test RMSE | Test MAPE | vs NRFormer |
|--------|-----------|----------|--------|--------|----------|-----------|-----------|-------------|
| i1_baseline | No | No | 32 | 200 | - | - | - | - |
| i1_log | Yes | No | 32 | 200 | - | - | - | - |
| i1_res | No | Yes | 32 | 200 | - | - | - | - |
| i1_log_res | Yes | Yes | 32 | 200 | - | - | - | - |

**Analysis:** (fill after experiments)

**Decision:** (which combination to keep)

---

### Iteration 2: Rain-aware Gating

**Date:** TBD

**Data motivation:**
- F3: Radon washout confirmed — humid days see +1.25 nSv/h radiation spike
- Sudden changes affect 3500+ stations simultaneously, driven by precipitation

**Technical changes:**
```python
# Compute dryness index
dryness = air_temperature - dew_point  # [B, N, T]
rain_gate = torch.sigmoid(-self.rain_fc(dryness.mean(dim=-1)))  # [B, N, 1]

# Gate meteorological/physics features
meteo_feat = meteo_feat * (1 + rain_gate * self.rain_scale)
physics_feat = physics_feat * (1 + rain_gate * self.physics_boost)
```

**Experiments:**

| Exp ID | Rain gate | rain_scale init | Test MAE | Test RMSE | Test MAPE | Δ MAE |
|--------|-----------|----------------|----------|-----------|-----------|-------|
| i2_no_gate | No | - | - | - | - | baseline |
| i2_gate_01 | Yes | 0.1 | - | - | - | - |
| i2_gate_05 | Yes | 0.5 | - | - | - | - |

**Analysis:** (fill after)

---

### Iteration 3: Multi-scale Spatial Graph

**Date:** TBD

**Data motivation:**
- F4: 235 isolated nodes (6.5%) receive NO spatial info, 288 disconnected components
- F5: Anomaly decorrelation length = 636 km, far beyond 10km graph
- Current graph captures only local (<10km) relationships

**Technical changes:**
```python
# Add K virtual global nodes to spatial attention
K_global = 10  # number of virtual nodes
self.global_tokens = nn.Parameter(torch.randn(1, K_global, hidden_dim))
self.global_mask = ...  # allow all-to-global connections

# In forward:
x_aug = torch.cat([x, self.global_tokens.expand(B, -1, -1)], dim=1)
x_spatial = self.spatial_attn(x_aug, mask_augmented)
x = x_spatial[:, :N, :]  # remove virtual nodes
```

**Experiments:**

| Exp ID | K_global | Test MAE | Test RMSE | Δ MAE |
|--------|----------|----------|-----------|-------|
| i3_k5 | 5 | - | - | - |
| i3_k10 | 10 | - | - | - |
| i3_k20 | 20 | - | - | - |

---

### Iteration 4: Region-aware Attention

**Date:** TBD

**Data motivation:**
- F6: Within-prefecture correlation (0.473) is 3.45x between-prefecture (0.137)
- F10: Temporal clustering with k=15-20 shows geographic coherence 0.4

**Technical changes:**
```python
# Pre-compute station clusters from temporal behavior (k-means)
self.cluster_embed = nn.Embedding(K_clusters, hidden_dim)
# In spatial attention Q/K:
region_bias = self.cluster_embed(cluster_ids)  # [N, H]
Q = Q + region_bias
K = K + region_bias
```

**Experiments:**

| Exp ID | K_clusters | Cluster source | Test MAE | Test RMSE | Δ MAE |
|--------|-----------|----------------|----------|-----------|-------|
| i4_k10_pref | 10 | Prefecture | - | - | - |
| i4_k15_kmeans | 15 | K-means temporal | - | - | - |
| i4_k20_kmeans | 20 | K-means temporal | - | - | - |

---

### Iteration 5: Revised Physics Module

**Date:** TBD

**Data motivation:**
- F7: Only 9.5% of neighbors synchronize within ±2 days → no station-to-station diffusion
- Lag-0 dominance → changes are simultaneous (atmospheric-scale forcing)
- Estimated D=5814 km²/day is unrealistically large → classical diffusion model is wrong

**Technical changes:** Replace Laplacian diffusion with regional synchronous response:
```python
# Old: laplacian = A_norm @ C - C  (assumes gradual spreading)
# New: Regional synchronous response model
regional_mean = scatter_mean(C, cluster_ids, dim=1)  # [B, K]
deviation = C - regional_mean[:, cluster_ids]  # [B, N]
weather_forcing = self.weather_net(meteo_features)  # [B, K]

# Physics features: [C, deviation, weather_forcing, dC/dt]
# Physics loss: ||dC/dt - weather_forcing - α·deviation||²
```

**Experiments:**

| Exp ID | Physics type | λ_physics | Test MAE | Test RMSE | Δ MAE |
|--------|-------------|-----------|----------|-----------|-------|
| i5_diffusion | Original Laplacian | 0 | - | - | baseline |
| i5_sync_001 | Synchronous | 0.01 | - | - | - |
| i5_sync_01 | Synchronous | 0.1 | - | - | - |

---

### Iteration 6: Hyperparameter Tuning + Final

**Date:** TBD

**Search space** (based on Iter 1-5 best config):

| Parameter | Search values |
|-----------|--------------|
| hidden_channels | 32, 64, 96 |
| num_temporal_att_layer | 2, 3, 4 |
| num_spatial_att_layer | 2, 3 |
| batch_size | 4, 8, 16 |
| learning_rate | 0.0005, 0.001, 0.002 |
| weight_decay | 0.0001, 0.001 |

**Best config:** TBD

**Final results (3-run mean ± std):**

| Dataset | MAE | RMSE | MAPE |
|---------|-----|------|------|
| Japan-1D | - ± - | - ± - | - ± - |
| Japan-4H | - ± - | - ± - | - ± - |

---

## How to Run Experiments

```bash
# Single experiment
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name NRFormer_Plus --dataset 1D-data \
    --model_des <exp_id> --epochs 200 \
    --IsDayOfYearEmbedding True --temporal_dropout 0.1 \
    --ffn_ratio 4 --spatial_heads 4

# Phased parallel experiments (3 GPUs)
bash go.sh --phase 1

# Compare all results
python compare_results.py
```

## How to Update This Log

After each experiment:
1. Run `python compare_results.py` to get results
2. Fill in the MAE/RMSE/MAPE in the corresponding iteration table
3. Write analysis notes and decision
4. Commit: `git add EXPERIMENT_LOG.md && git commit -m "Iter X: <brief description>"`
