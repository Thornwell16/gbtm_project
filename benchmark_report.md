# AutoTraj Benchmark Report

*Generated: 2026-03-16 22:44*

This report provides parameter recovery evidence for the AutoTraj validation paper.
Simulated benchmarks compare AutoTraj estimates against known ground-truth parameters.
The Cambridge benchmark validates against the canonical Nagin (1999) real-world dataset.

---

## Executive Summary

| Benchmark | N | T | Dist | LL | BIC(Nagin) | True k | BIC k | Assign Acc | β Recovery |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cambridge LOGIT [2,2] | 195 | 23 | LOGIT | -788.8 | -807.3 | N/A | 2 | N/A | N/A (0/0) |
| LOGIT Simulation [1,1] | 500 | 10 | LOGIT | -2885.3 | -2900.8 | 2 | 2 | 90.2% | 100% (4/4) |
| CNORM Simulation [1,1] | 500 | 10 | CNORM | -5805.4 | -5824.0 | 2 | 2 | 100.0% | 100% (5/5) |
| Poisson Simulation [1,1] | 500 | 10 | POISSON | -10813.1 | -10828.7 | 2 | 2 | 100.0% | 100% (4/4) |

---

## Cambridge LOGIT [2,2]

### Dataset Summary

- **N subjects:** 195
- **Time points:** 23
- **Distribution:** LOGIT
- **Log-likelihood:** -788.8127
- **BIC (Nagin):** -807.2682

### Estimated Group Proportions

| Group | Est. π (%) |
| --- | --- |
| Group 1 | 80.1% |
| Group 2 | 19.9% |

### Parameter Recovery Table

> 95% CI = Estimate ± 1.96 × SE(model-based Hessian). Recovery = YES if true value falls within 95% CI.

| Parameter | True Value | AutoTraj Estimate | Std Error | 95% CI | Recovery |
| --- | --- | --- | --- | --- | --- |
| β_0 Group 1 (80%) | N/A | -3.4351 | 0.2386 | [-3.9026, -2.9675] | N/A |
| β_1 Group 1 (80%) | N/A | -0.0848 | 0.3141 | [-0.7004, 0.5308] | N/A |
| β_2 Group 1 (80%) | N/A | -1.9702 | 0.5372 | [-3.0232, -0.9172] | N/A |
| β_0 Group 2 (20%) | N/A | -0.8359 | 0.1641 | [-1.1575, -0.5143] | N/A |
| β_1 Group 2 (20%) | N/A | -0.6139 | 0.1726 | [-0.9522, -0.2755] | N/A |
| β_2 Group 2 (20%) | N/A | -1.6526 | 0.2877 | [-2.2166, -1.0886] | N/A |

### Adequacy Metrics (Nagin 2005 thresholds: AvePP > 0.70, OCC > 5.0, H_rel > 0.50)

| Group | Assigned N | Estimated Pi (%) | AvePP | OCC |
| --- | --- | --- | --- | --- |
| Group 1 | 162 | 80.1 | 0.9548 | 5.25 |
| Group 2 | 33 | 19.9 | 0.9539 | 83.3 |

**Relative entropy (H_rel): 0.8386**

### Trajectory Figure

![Cambridge LOGIT [2,2]](benchmark_figures/Cambridge_LOGIT_22.png)

---

## LOGIT Simulation [1,1]

### Dataset Summary

- **N subjects:** 500
- **Time points:** 10
- **Distribution:** LOGIT
- **Log-likelihood:** -2885.3131
- **BIC (Nagin):** -2900.8497

### Estimated Group Proportions

| Group | Est. π (%) |
| --- | --- |
| Group 1 | 42.3% |
| Group 2 | 57.7% |

### Parameter Recovery Table

> 95% CI = Estimate ± 1.96 × SE(model-based Hessian). Recovery = YES if true value falls within 95% CI.

| Parameter | True Value | AutoTraj Estimate | Std Error | 95% CI | Recovery |
| --- | --- | --- | --- | --- | --- |
| β_0 Group 1 (42%) | -2.0000 | -2.1750 | 0.1816 | [-2.5310, -1.8190] | YES |
| β_1 Group 1 (42%) | 3.5000 | 3.3583 | 0.2971 | [2.7759, 3.9406] | YES |
| β_0 Group 2 (58%) | -0.5000 | -0.4727 | 0.0455 | [-0.5619, -0.3836] | YES |
| β_1 Group 2 (58%) | 0.0000 | -0.1103 | 0.0803 | [-0.2678, 0.0471] | YES |

**Recovery: 4/4 parameters (100%)**

**Group assignment accuracy: 90.2%**

### Trajectory Figure

![LOGIT Simulation [1,1]](benchmark_figures/LOGIT_Simulation_11.png)

---

## CNORM Simulation [1,1]

### Dataset Summary

- **N subjects:** 500
- **Time points:** 10
- **Distribution:** CNORM
- **Log-likelihood:** -5805.3906
- **BIC (Nagin):** -5824.0344

### Estimated Group Proportions

| Group | Est. π (%) |
| --- | --- |
| Group 1 | 40.0% |
| Group 2 | 60.0% |

### Parameter Recovery Table

> 95% CI = Estimate ± 1.96 × SE(model-based Hessian). Recovery = YES if true value falls within 95% CI.

| Parameter | True Value | AutoTraj Estimate | Std Error | 95% CI | Recovery |
| --- | --- | --- | --- | --- | --- |
| β_0 Group 1 (40%) | 1.0000 | 0.9924 | 0.0211 | [0.9510, 1.0338] | YES |
| β_1 Group 1 (40%) | -2.5000 | -2.4739 | 0.0361 | [-2.5447, -2.4032] | YES |
| β_0 Group 2 (60%) | 4.5000 | 4.4950 | 0.0146 | [4.4665, 4.5236] | YES |
| β_1 Group 2 (60%) | 0.0000 | 0.0069 | 0.0228 | [-0.0378, 0.0516] | YES |
| σ (residual SD) | 0.8000 | 0.7958 | 0.0087 | [0.7788, 0.8128] | YES |

**Recovery: 5/5 parameters (100%)**

**Group assignment accuracy: 100.0%**

### Trajectory Figure

![CNORM Simulation [1,1]](benchmark_figures/CNORM_Simulation_11.png)

---

## Poisson Simulation [1,1]

### Dataset Summary

- **N subjects:** 500
- **Time points:** 10
- **Distribution:** POISSON
- **Log-likelihood:** -10813.1361
- **BIC (Nagin):** -10828.6726

### Estimated Group Proportions

| Group | Est. π (%) |
| --- | --- |
| Group 1 | 40.0% |
| Group 2 | 60.0% |

### Parameter Recovery Table

> 95% CI = Estimate ± 1.96 × SE(model-based Hessian). Recovery = YES if true value falls within 95% CI.

| Parameter | True Value | AutoTraj Estimate | Std Error | 95% CI | Recovery |
| --- | --- | --- | --- | --- | --- |
| β_0 Group 1 (40%) | 0.5000 | 0.4903 | 0.0177 | [0.4557, 0.5250] | YES |
| β_1 Group 1 (40%) | 0.3000 | 0.3374 | 0.0275 | [0.2835, 0.3913] | YES |
| β_0 Group 2 (60%) | 2.0000 | 2.0058 | 0.0068 | [1.9926, 2.0191] | YES |
| β_1 Group 2 (60%) | -0.3000 | -0.2883 | 0.0105 | [-0.3089, -0.2677] | YES |

**Recovery: 4/4 parameters (100%)**

**Group assignment accuracy: 100.0%**

### Trajectory Figure

![Poisson Simulation [1,1]](benchmark_figures/Poisson_Simulation_11.png)

---

