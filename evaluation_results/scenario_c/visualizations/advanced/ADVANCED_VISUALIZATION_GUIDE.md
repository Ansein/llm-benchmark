# Advanced Visualization Guide for Scenario C

## Overview

This directory contains **6 advanced scientific visualizations** that provide deeper insights into the Scenario C evaluation results. These charts go beyond basic comparisons to reveal hidden patterns, trade-offs, and strategic relationships.

---

## Chart Descriptions

### 1. Pareto Frontier Analysis ⭐
**Filename**: `1_pareto_frontier_analysis.png`

**Purpose**: Analyze the fundamental trade-off between intermediary profit and consumer welfare.

**Key Features**:
- **X-axis**: Intermediary Profit
- **Y-axis**: Consumer Surplus Change (Gap)
- **Color Coding**: By model series (Red=GPT, Blue=DeepSeek, Green=Qwen)
- **Pareto Frontier**: Dashed black line connecting Pareto-optimal points
- **Quadrants**:
  - **Top-Right (Green)**: Win-Win (High Profit + Consumer Benefit)
  - **Top-Left (Yellow)**: Consumer Benefit but Low Profit
  - **Bottom-Right (Red)**: Profit but Consumer Harmed
  - **Bottom-Left (Gray)**: Lose-Lose

**Key Findings**:
- **3 Pareto-optimal models**: deepseek-v3-0324, gpt-5.2, qwen3-max-2026-01-23
- **Only 1 Win-Win model**: gpt-5.2 (Profit=1.68, CS Gap=+0.61)
- Most models fall in the "Profit but Consumer Harmed" quadrant
- **gpt-5** is the only model in "Lose-Lose" (negative profit + consumer harm)

**Scientific Insight**: The Pareto frontier reveals that achieving both high profit and consumer benefit is extremely difficult. gpt-5.2 is the only model that achieves this balance, making it the **Pareto-dominant strategy**.

---

### 2. Strategy Space Map
**Filename**: `2_strategy_space_map.png`

**Purpose**: Visualize the strategic positioning of all models in the compensation-participation space.

**Key Features**:
- **X-axis**: Compensation (m)
- **Y-axis**: Participation Rate (%)
- **Bubble Size**: Proportional to profit (larger = higher profit)
- **Color**: Anonymization strategy (Blue=Anonymized, Orange=Identified)
- **Gold Star**: Theoretical optimum (m=0.6, r=3.42%)

**Key Findings**:
- **Wide strategy diversity**: Compensation ranges from 0.4 to 1.7
- **Low participation dominant**: 9 out of 11 models have 0% participation
- **High-profit models**: Use either very low compensation (qwen3-max-2026-01-23: m=0.4) or moderate compensation with high participation (gpt-5.1: m=0.7, r=10%)
- **Theoretical gap**: Most models deviate significantly from theory

**Scientific Insight**: The strategy space shows **two distinct strategic clusters**:
1. **Low-m, Low-r cluster** (most models): Conservative, risk-averse
2. **High-m, High-r cluster** (GPT series): Aggressive, market-expansion oriented

---

### 3. Model Series Evolution Analysis
**Filename**: `3_model_series_evolution.png`

**Purpose**: Compare how different versions within each model series (GPT, DeepSeek, Qwen) perform across 4 key metrics.

**Key Features**:
- **4 Subplots**: Profit, CS Gap, Participation Rate, Accuracy
- **3 Series**: GPT (red), DeepSeek (blue), Qwen (green)
- **Evolution**: From older to newer versions (left to right)

**Key Findings**:

**GPT Series**:
- Profit: Highly variable (gpt-5 fails, others succeed)
- CS Gap: Improves from gpt-5 (-42.77) → gpt-5.2 (+0.61) ✨
- Accuracy: Stable at 95% (drops to 20% for gpt-5)
- **Best**: gpt-5.2 (balanced evolution)

**DeepSeek Series**:
- Profit: v3.2 succeeds (1.18), others fail
- CS Gap: v3-0324 best (+1.19), v3.2 worst (-20.28)
- Accuracy: Drops from 65% → 10% (concerning trend)
- **Trade-off**: Profit vs Consumer Welfare

**Qwen Series**:
- Profit: Clear winner (qwen3-max-2026-01-23 = 2.35) 🏆
- CS Gap: All negative (consumer harm)
- Participation: qwen3-max-2026-01-23 only non-zero (5%)
- Accuracy: Wide range (30% → 100%)
- **Pattern**: Newer = higher profit, but not necessarily better consumer treatment

**Scientific Insight**: Model evolution is **not monotonic**. Newer versions don't always outperform older ones across all dimensions. **Multi-objective optimization** is needed.

---

### 4. Metric Correlation Heatmap
**Filename**: `4_metric_correlation_heatmap.png`

**Purpose**: Reveal statistical relationships between key metrics using Pearson correlation.

**Key Features**:
- **6 Metrics**: Compensation, Part.Rate, Profit, CS Gap, Accuracy, Gini
- **Color Scale**: Red (negative correlation) → Blue (positive correlation)
- **Lower Triangle Only**: Eliminates redundancy

**Key Findings**:

**Strong Correlations** (|r| > 0.7):
- None found (suggests high independence)

**Moderate Correlations** (|r| > 0.4):
- **Part.Rate ↔ Profit**: r = 0.435 (positive)
  - Higher participation → Higher profit (intuitive)
- **Compensation ↔ Gini**: r = -0.523 (negative)
  - Higher compensation → Lower inequality (more equal distribution)

**Weak/No Correlations**:
- **Compensation ↔ Profit**: r = -0.386 (slightly negative)
  - Suggests **no simple linear relationship** between m and profit
- **Profit ↔ CS Gap**: r = 0.165 (very weak)
  - **Win-win is possible but rare**
- **Accuracy ↔ Profit**: r = 0.005 (essentially zero)
  - Consumer decision quality doesn't directly determine market outcome

**Scientific Insight**: The **weak correlations** suggest that optimal strategy requires **complex, non-linear optimization**. Simple heuristics (e.g., "increase m to increase profit") don't work.

---

### 5. Parallel Coordinates: Cross-Configuration Profit
**Filename**: `5_parallel_coordinates_profit.png`

**Purpose**: Track how each model's profit changes across Configs B, C, D.

**Key Features**:
- **3 Vertical Axes**: Config B, Config C, Config D
- **Lines**: Each model (colored by series)
- **Gold Dashed Line**: Theoretical optimum (1.311)

**Key Findings**:

**Stable High Performers**:
- **gpt-5.1, gpt-5.2**: Consistently above theory in B and C, good in D
- **qwen3-max-2026-01-23**: Excellent in C and D

**High Variance**:
- **gpt-5**: Good in B and C, crashes in D (-0.24)
- **deepseek series**: Variable performance

**Config Difficulty**:
- **Config B**: Easiest (all models ≥ theory)
- **Config C**: Moderate (most above theory)
- **Config D**: Hardest (many models fail)

**Scientific Insight**: Config D (full LLM environment) is the **true test of robustness**. Models that perform well in B or C may fail catastrophically in D. Only **3 models** maintain profitability across all configs.

---

### 6. Efficiency Frontier Analysis
**Filename**: `6_efficiency_frontier_analysis.png`

**Purpose**: Normalize metrics to compare models on a common efficiency scale.

**Key Features**:
- **X-axis**: Profit Efficiency (LLM Profit / Theory Profit)
- **Y-axis**: Welfare Efficiency (LLM CS / Theory CS)
- **Theoretical Optimum**: (1.0, 1.0) marked with gold star
- **45° Line**: Equal efficiency
- **Quadrants**: Define efficiency zones

**Key Findings**:

**Super-Efficient Models** (Both > 1.0):
- **qwen3-max-2026-01-23**: (1.79, 0.984) - High profit, near-theory welfare
- **gpt-5.2**: (1.28, 1.008) - Balanced excellence ✨

**Specialized Strategies**:
- **deepseek-v3-0324**: (0.0, 1.016) - Consumer-focused (profit sacrifice)
- **gpt-5.1**: (1.42, 0.982) - Profit-focused (slight consumer cost)

**Under-Performers**:
- **gpt-5**: (-0.18, 0.441) - Far below theory on both dimensions

**Efficiency vs Absolute Value**:
- High efficiency doesn't mean high absolute profit
- qwen3-max-2026-01-23 has highest profit but only 98.4% welfare efficiency
- gpt-5.2 achieves 100.8% welfare efficiency with good profit

**Scientific Insight**: The efficiency frontier shows that **qwen3-max-2026-01-23** and **gpt-5.2** are the only models that "expand the frontier" beyond theory. Most models are **interior points** (sub-optimal on at least one dimension).

---

## Summary Table: Advanced Insights

| Chart | Primary Insight | Practical Implication |
|-------|----------------|----------------------|
| 1. Pareto Frontier | Win-win is rare; gpt-5.2 is unique | Choose gpt-5.2 for balanced objectives |
| 2. Strategy Space | Two strategic clusters exist | Different strategies for different goals |
| 3. Series Evolution | Newer ≠ Better always | Test each version independently |
| 4. Correlation | Weak linear relationships | Need non-linear optimization |
| 5. Parallel Coords | Config D is true test | Focus evaluation on Config D |
| 6. Efficiency | Only 2 models expand frontier | qwen3-max-2026 (profit) or gpt-5.2 (balance) |

---

## Scientific Methodology Notes

### Pareto Optimality
A point is **Pareto optimal** if no other point is better in all objectives. Formally:
```
Model A dominates Model B iff:
  Profit(A) ≥ Profit(B) AND CS(A) ≥ CS(B)
  with at least one strict inequality
```

### Correlation Strength Guidelines
- |r| > 0.7: Strong correlation
- 0.4 < |r| ≤ 0.7: Moderate correlation
- 0.2 < |r| ≤ 0.4: Weak correlation
- |r| ≤ 0.2: No meaningful correlation

### Efficiency Calculation
```
Profit Efficiency = LLM_Profit / Theory_Profit
Welfare Efficiency = LLM_CS / Theory_CS
```
Values > 1.0 indicate **exceeding theoretical optimum**.

---

## Recommended Viewing Order

For **comprehensive understanding**:
1. Start with **Chart 5** (Parallel Coordinates) - Get overview
2. Then **Chart 1** (Pareto Frontier) - Understand trade-offs
3. Follow with **Chart 6** (Efficiency Frontier) - See normalized performance
4. Review **Chart 2** (Strategy Space) - Understand strategic diversity
5. Check **Chart 3** (Series Evolution) - Compare within series
6. Finally **Chart 4** (Correlation) - Statistical relationships

For **quick decision-making**:
- **Chart 1** + **Chart 6** → Best model for your objective function

---

## Integration with Basic Visualizations

These advanced charts complement the 8 basic charts:
- **Basic Charts 1-3**: What happened? (descriptive)
- **Advanced Charts 1-6**: Why did it happen? (analytical)

Recommended workflow:
1. Use **basic charts** to identify patterns
2. Use **advanced charts** to explain patterns
3. Combine insights for robust decision-making

---

## Regenerate Charts

To regenerate all advanced visualizations:

```bash
python visualize_scenario_c_advanced.py
```

Requirements:
- pandas, matplotlib, seaborn, numpy, scipy
- Times New Roman font installed
- Python 3.8+

---

## Key Takeaways for Practitioners

### For Profit Maximization
- **Top Choice**: qwen3-max-2026-01-23 (Profit=2.35, 79% above theory)
- **Pareto Frontier**: Position (1.79, 0.98) efficiency
- **Trade-off**: Consumer welfare decreases (-12.49 gap)

### For Balanced Performance
- **Top Choice**: gpt-5.2 (Profit=1.68, CS Gap=+0.61)
- **Unique Property**: Only model in Win-Win quadrant
- **Pareto Frontier**: Pareto-optimal position
- **Efficiency**: (1.28, 1.01) - both above theory

### For Consumer Welfare
- **Top Choice**: deepseek-v3-0324 (CS Gap=+1.19)
- **Trade-off**: Zero profit for intermediary
- **Not sustainable** for commercial applications

### For Robustness
- **Top Choices**: gpt-5.1, gpt-5.2
- **Property**: Profitable in all three configs (B, C, D)
- **Parallel Coordinates**: Smooth lines, no crashes

---

**Last Updated**: 2026-02-05  
**Version**: 1.0  
**Contact**: See main README.md for details
