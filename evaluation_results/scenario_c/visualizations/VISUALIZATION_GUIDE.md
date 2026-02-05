# Scenario C Visualization Guide

## Overview

This directory contains comprehensive visualization of **11 LLM models** evaluated on **Scenario C: The Economics of Social Data**.

**Evaluation Date**: 2026-02-05  
**Theoretical Optimum**: Profit R* = 1.311, Compensation m* = 0.6, Participation Rate = 3.42%

---

## 8 Visualization Charts

### Chart 1: Config D Profit Comparison
**Filename**: `1_config_D_profit_comparison.png`

- **Type**: Horizontal bar chart
- **Purpose**: Compare intermediary profits across 11 models in Config D (most challenging)
- **Color Coding**:
  - Green = Profitable
  - Red = Loss
- **Red Dashed Line**: Theoretical optimum (1.311)

**Key Finding**: qwen3-max-2026-01-23 achieves 2.346 profit (+78.9% above theory)

---

### Chart 2: Config C Intermediary Strategy
**Filename**: `2_config_C_intermediary_strategy.png`

- **Type**: Dual subplot (2 horizontal bar charts)
- **Left Plot**: LLM intermediary profit comparison
- **Right Plot**: Compensation strategy choice
  - Blue = Anonymized data
  - Orange = Identified data
- **Red Dashed Lines**: Theoretical optima

**Key Finding**: Most LLMs choose higher compensation (0.65-0.78) vs theory (0.6)

---

### Chart 3: Config B Consumer Accuracy
**Filename**: `3_config_B_consumer_accuracy.png`

- **Type**: Horizontal bar chart with annotations
- **Purpose**: Test LLM consumer decision quality given optimal compensation
- **Annotations**: Show participation rate for each model
- **Red Dashed Line**: Perfect accuracy (100%)

**Key Finding**: qwen3-max-2026-01-23 achieves 100% accuracy

---

### Chart 4: Multi-Config Heatmap
**Filename**: `4_multi_config_heatmap.png`

- **Type**: Heatmap (matrix visualization)
- **Purpose**: Show profit loss percentage across 3 configs (B, C, D) for all models
- **Color Coding**:
  - Red = Loss (worse than theory)
  - Yellow = Neutral
  - Green = Exceeds theory
- **Values**: Profit loss percentage

**Key Finding**: Performance varies significantly across configs

---

### Chart 5: Config D Strategy Analysis
**Filename**: `5_config_D_strategy_analysis.png`

- **Type**: Dual scatter plots
- **Left Plot**: Compensation (m) vs Profit
  - Blue dots = Anonymized
  - Orange dots = Identified
  - Green vertical line = Theoretical m* (0.6)
  - Red horizontal line = Theoretical profit (1.311)
  
- **Right Plot**: Participation Rate (%) vs Profit
  - Green vertical line = Theoretical rate (3.4%)
  - Red horizontal line = Theoretical profit

**Key Finding**: Wide strategy diversity, no clear correlation pattern

---

### Chart 6: Consumer Welfare Analysis
**Filename**: `6_consumer_welfare_analysis.png`

- **Type**: Dual subplot (bar chart + boxplot)
- **Left Plot**: Consumer surplus gap (Config D)
  - Red bars = Consumer harmed (negative gap)
  - Green bars = Consumer benefited
  
- **Right Plot**: Gini coefficient distribution across configs
  - Lower Gini = More equal consumer surplus distribution
  - Red line = Theoretical Gini coefficient

**Key Finding**: Most models (9 out of 11) harm consumers in Config D, but 2 models benefit consumers:
- deepseek-v3-0324: +1.19 (most beneficial)
- gpt-5.2: +0.61

---

### Chart 7: Top 5 Model Performance Radar
**Filename**: `7_top5_models_radar.png`

- **Type**: Radar chart (5 dimensions)
- **Dimensions**:
  1. **Profit**: Normalized intermediary profit (Config D)
  2. **Consumer Accuracy**: LLM consumer decision quality (Config D)
  3. **Participation Match**: Closeness to theoretical participation rate (Config D)
  4. **Welfare Protection**: Consumer surplus preservation (Config D)
  5. **Cross-Config Consistency**: Performance stability across Configs B, C, D
     - Calculated as: Lower variance in profits across configs = Higher consistency
     - If all 3 configs are profitable: 1/(1+CV) where CV=coefficient of variation
     - Otherwise: (# of profitable configs) / 3

**Key Finding**: qwen3-max-2026-01-23 leads in most dimensions

---

### Chart 8: Comprehensive Ranking Table
**Filename**: `8_comprehensive_ranking.png`

- **Type**: Professional table visualization
- **Columns**:
  - Rank
  - Model name
  - Config B consumer accuracy
  - Config C profit & loss%
  - Config D profit & loss%
  - Config D consumer accuracy
  - Overall score
- **Yellow Highlight**: Top 3 models

**Key Finding**: Complete ranking of all 11 models

---

## Model Ranking Summary

| Rank | Model | B-Cons.Acc | C-Profit | D-Profit | Overall Score |
|------|-------|------------|----------|----------|---------------|
| 1 | **qwen3-max-2026-01-23** | 100.00% | 1.577 | **2.346** ⭐ | **1.000** |
| 2 | **gpt-5.1** | 95.00% | 1.733 | 1.858 | **0.980** |
| 3 | **gpt-5.2** | 95.00% | 1.706 | 1.680 | **0.940** |
| 4 | deepseek-v3.2 | 90.00% | 1.607 | 1.177 | 0.769 |
| 5 | gpt-5.1-2025-11-13 | 90.00% | 1.714 | 0.561 | 0.698 |

---

## Configuration Descriptions

### Config A (Theory)
- **Intermediary**: Rational (optimal policy)
- **Consumers**: Rational (game-theoretic response)
- **Purpose**: Theoretical benchmark

### Config B
- **Intermediary**: Rational (fixed m* = 0.6)
- **Consumers**: LLM agents
- **Purpose**: Test LLM consumer decision quality

### Config C
- **Intermediary**: LLM agent (choose m and anonymization)
- **Consumers**: Rational (game-theoretic response)
- **Purpose**: Test LLM intermediary strategy formulation

### Config D
- **Intermediary**: LLM agent
- **Consumers**: LLM agents
- **Purpose**: Test full LLM market equilibrium (most challenging)

---

## Key Metrics Explained

- **Intermediary Profit**: Main optimization objective (R* = revenue - cost)
- **Individual Accuracy**: Percentage of correct consumer participation decisions
- **Participation Rate**: Percentage of consumers sharing data
- **Consumer Surplus Gap**: Change in consumer welfare vs theory
  - **Positive Gap**: Consumers benefit (surplus increases)
  - **Negative Gap**: Consumers harmed (surplus decreases)
- **Gini Coefficient**: Inequality measure (0 = perfect equality, 1 = maximum inequality)
- **Profit Loss %**: (Profit_LLM - Profit_Theory) / Profit_Theory × 100%

---

## How to Use

### Quick Start
1. Open `8_comprehensive_ranking.png` for overall model ranking
2. Check `1_config_D_profit_comparison.png` for profit distribution
3. Review `4_multi_config_heatmap.png` for cross-config performance

### Deep Dive
- **Strategy Analysis**: Charts 2, 5
- **Consumer Analysis**: Charts 3, 6
- **Comprehensive View**: Charts 4, 7, 8

### Data Files
- **model_ranking.csv**: Structured ranking data (Excel-compatible)
- **场景C评估结果分析报告.md**: Full analysis report (Chinese)
- **README.md**: Additional documentation (Chinese)

---

## Regenerate Charts

To regenerate all visualizations:

```bash
python visualize_scenario_c_results.py
```

The script will:
1. Read `evaluation_results/summary_report_20260205_205348.csv`
2. Generate 8 visualization charts
3. Generate ranking CSV
4. Print key findings

---

## Technical Details

- **Font**: Times New Roman (11pt)
- **Resolution**: 300 DPI
- **Format**: PNG
- **Color Palette**: Seaborn 'husl' with custom color coding
- **Libraries**: pandas, matplotlib, seaborn, numpy

---

**Last Updated**: 2026-02-05  
**Version**: 2.0 (English Edition with Times New Roman)
