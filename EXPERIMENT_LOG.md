# NRFormer+ Experiment Log

## Baseline Results (from TKDE paper, 11 baselines + NRFormer)

### Japan-4H (4-hour resolution, 3627 stations)

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

### Japan-1D (daily resolution, 3627 stations)

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

### Target: NRFormer+ needs to beat NRFormer on all metrics

**Key targets (Japan-1D, 24th step):** MAE < 2.28, RMSE < 11.65, MAPE < 2.93%

---

## NRFormer+ Iteration History

### Iteration 0: Architecture Fixes (Before First Run)

**Date:** 2026-03-24

**Changes from KDD NRFormer:**
1. Physics module: AtmosphericDiffusionModule with real graph Laplacian (was fake global mean)
2. Temporal gradient encoding dC/dt added to physics triplet [C, D, nabla^2(C), dC/dt]
3. MeteorologicalEncoder: separate wind/temperature pathways with temporal conv
4. LocationEncoder: deeper MLP (2->32->64->D)
5. TemporalEncoder: day-of-year embedding
6. 3-way gated fusion: radiation + temporal + spatial (was 2-way simple concat)
7. FFN expansion 1x -> 4x in temporal self-attention
8. Learned temporal positional encoding
9. Reduced temporal dropout: 0.3 -> 0.1
10. Spatial attention heads: 8 -> 4 (head_dim doubled)
11. Fixed LR schedule: milestones [50,60,70,80] gamma=0.01 -> [100,200,250] gamma=0.5
12. Fixed NOAA normalization data leakage
13. Fixed missing data threshold (was destroying low readings)
14. Train-time output clamping for train/eval consistency

**Status:** Waiting for first experiment results

---

### Iteration 1: Phase 1 - Capacity Search

**Date:** 2026-03-24

**Experiments:**
| Exp ID | hidden | temp_layers | spat_layers | batch | dropout | ffn | heads | Params | Best Epoch | Test MAE | Test RMSE | Test MAPE | Notes |
|--------|--------|-------------|-------------|-------|---------|-----|-------|--------|------------|----------|-----------|-----------|-------|
| p1_baseline | 32 | 3 | 2 | 8 | 0.1 | 4x | 4 | - | - | - | - | - | |
| p1_h64 | 64 | 3 | 2 | 8 | 0.1 | 4x | 4 | - | - | - | - | - | |
| p1_h96 | 96 | 3 | 2 | 8 | 0.1 | 4x | 4 | - | - | - | - | - | |

**Analysis:** (fill after experiments)


**Decision:** (which hidden_channels to use going forward)

---

### Iteration 2: Phase 2 - Depth & Batch Size

**Date:** TBD

**Experiments:**
| Exp ID | hidden | temp_layers | spat_layers | batch | Params | Best Epoch | Test MAE | Test RMSE | Notes |
|--------|--------|-------------|-------------|-------|--------|------------|----------|-----------|-------|
| p2_t4 | ? | 4 | 2 | 8 | - | - | - | - | |
| p2_s3 | ? | 3 | 3 | 8 | - | - | - | - | |
| p2_t4s3 | ? | 4 | 3 | 8 | - | - | - | - | |
| p2_bs16 | ? | 3 | 2 | 16 | - | - | - | - | |
| p2_bs32 | ? | 3 | 2 | 32 | - | - | - | - | |

---

### Iteration 3: Phase 3 - Advanced Features

**Planned experiments:**
- Physics loss regularization (lambda=0.01, 0.1)
- Adaptive adjacency learning
- Multi-step diffusion (use all T timesteps, not just last)
- Cross-variable meteo attention (wind x temperature interaction)

---

## Optimization Roadmap

```
Phase 1: Capacity Search          ← CURRENT
    └─ Find best hidden_channels (32/64/96)
Phase 2: Depth & Batch Size
    └─ Find best layer counts and batch size
Phase 3: Advanced Features
    └─ Physics loss, adaptive adj, cross-attention
Phase 4: Final Tuning
    └─ LR schedule, weight decay, ensemble
Phase 5: 4H-data Validation
    └─ Run best config on Japan-4H dataset
Phase 6: Paper Results
    └─ 3-run mean±std, sudden change eval, ablation
```

## How to Update This Log

After each experiment run:
1. `python compare_results.py` - get results table
2. Copy MAE/RMSE/MAPE into the iteration table above
3. Add analysis notes and decision
4. Commit: `git add EXPERIMENT_LOG.md && git commit -m "Update experiment log"`
