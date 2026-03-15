# 场景B结果图目录说明

本文按“目录”整理当前仓库里与场景B相关的结果图，并说明每类图的含义。

## 1) 根目录

目录：`d:\benchmark`

- `scenario_b_analysis.png`
  - 含义：单次场景B静态评测的6联图诊断面板。
  - 主要内容：隐私偏好与分享决策、关键指标对比、偏差(MAE)、信息泄露、分享规模-利润曲线、社会福利分解。

## 2) 虚拟博弈结果目录（FP）

目录：`d:\benchmark\evaluation_results\fp_*`

- `eval_YYYYMMDD_HHMMSS_strategy_heatmap.png`
  - 含义：用户策略随轮次演化热力图（行=用户，列=轮次，0不分享/1分享），右侧红星为理论均衡分享用户。
- `eval_YYYYMMDD_HHMMSS_share_rate.png`
  - 含义：FP收敛曲线（蓝线=分享率，红线=与理论均衡Jaccard相似度）。
- `eval_YYYYMMDD_HHMMSS_similarity_heatmap.png/.pdf`
  - 含义：相似度折线 + 策略热力图的组合图（用于观察“动态过程 + 终局一致性”）。

补充汇总图（同目录）：

- `fp_similarity_gpt5_vs_deepseek-r1.png/.pdf`
- `fp_similarity_gpt5_vs_deepseek-r1_side_by_side.png/.pdf`
- `fp_similarity_gpt5_vs_deepseek-r1_subplots.png/.pdf`
  - 含义：GPT-5 与 deepseek-r1 在FP过程中的相似度/策略演化对比图（不同排版版本）。

## 3) 3×3敏感度单次实验分析图

目录：`d:\benchmark\sensitivity_results\scenario_b\sensitivity_3x3_b.v4_20260129_143102\analysis_plots`

- `sensitivity_heatmap_matrix.png`
  - 含义：3×3参数网格（ρ × v区间）下多指标热力图矩阵，通常包含Jaccard、Profit MAE、Welfare MAE、LLM分享率。
- `performance_boxplot.png`
  - 含义：不同参数组合/模型下指标分布箱线图（看离散度和异常点）。
- `parameter_impact.png`
  - 含义：参数变化（ρ、v区间）对性能指标的影响趋势图。
- `performance_radar.png`
  - 含义：多指标雷达图（用于横向看“准确性-稳定性-误差”综合轮廓）。

## 4) 跨模型对比图（场景B敏感度）

目录：`d:\benchmark\sensitivity_results\scenario_b\model_comparison_plots`

- `jaccard_comparison_heatmaps.png`
  - 含义：不同模型在3×3网格下Jaccard热力图并排对比。
- `overall_performance_comparison.png`
  - 含义：总体性能条形图（Jaccard、Profit MAE、Welfare MAE、均衡命中率）。
- `parameter_wise_comparison.png`
  - 含义：按参数组合逐格比较模型表现。
- `parameter_impact_comparison.png`
  - 含义：比较ρ与v变化对不同模型性能的影响幅度。
- `model_ranking_by_params.png`
  - 含义：每个参数区域内模型排名。
- `model_win_matrix_12models.png` / `model_win_matrix.png`
  - 含义：模型两两对胜矩阵（在参数区域内谁更优）。
- `model_stability_comparison.png`
  - 含义：稳定性对比（通常基于CV，越低越稳）。
- `cv_stability_heatmap_12models.png`
  - 含义：12模型在参数网格上的CV稳定性热力图。
- `cv_stability_dotplot_12models.png`
  - 含义：12模型稳定性点图分布。
- `model_radar_comparison.png`
  - 含义：多模型多指标雷达图对比。

## 5) GPT家族对比图（场景B敏感度）

目录：`d:\benchmark\sensitivity_results\scenario_b\gpt_family_comparison_plots`

- `gpt_evolution_heatmaps.png`
  - 含义：GPT不同版本在3×3网格下Jaccard热力图演化对比。
- `gpt_overall_performance.png`
  - 含义：GPT家族总体性能对比（Jaccard/MAE/均衡命中率）。
- `gpt_parameter_impact.png`
  - 含义：参数变化对GPT各版本影响的对比。
- `gpt_stability_comparison.png`
  - 含义：GPT各版本稳定性比较（CV视角）。
- `gpt_improvement_matrix.png`
  - 含义：版本间改进幅度矩阵（新版本相对旧版本）。
- `gpt_evolution_trends.png`
  - 含义：版本沿时间/代际的性能趋势图。
- `gpt_best_by_region.png`
  - 含义：参数区域最佳GPT版本分布图。
- `gpt_radar_profiles.png`
  - 含义：GPT家族多指标雷达轮廓对比。

## 6) 论文附录图目录（镜像/拷贝）

主目录：

- `d:\benchmark\docs\appendix_assets\sensitivity_b`
- `d:\benchmark\docs\appendix_assets\fp_b`

镜像目录（内容重复）：

- `d:\benchmark\docs\support material\appendix_assets\sensitivity_b`
- `d:\benchmark\docs\support material\appendix_assets\fp_b`
- `d:\benchmark\docs\support_material\appendix_assets\sensitivity_b`
- `d:\benchmark\docs\support_material\appendix_assets\fp_b`

含义说明：

- 这些目录是论文/附录用出图副本，核心语义与第2~5节一致。
- `sensitivity_b` 主要收录：跨模型与GPT家族对比图 + 3×3诊断图。
- `fp_b` 主要收录：`fp_similarity_gpt5_vs_deepseek-r1_side_by_side.png` 与 `fp_similarity_gpt5_vs_deepseek-r1_subplots.png`。

