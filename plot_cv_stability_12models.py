"""
CV stability comparison for 12 LLMs in Scenario-B sensitivity analysis.

Output:
1) A ranked heatmap (models x parameter regions) for CV (%)
2) A sorted horizontal dot plot for overall mean CV (%)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager


BASE_DIR = Path("sensitivity_results/scenario_b")
OUT_DIR = BASE_DIR / "model_comparison_plots"
OUT_HEATMAP = OUT_DIR / "cv_stability_heatmap_12models.png"
OUT_DOTPLOT = OUT_DIR / "cv_stability_dotplot_12models.png"

RHO_VALUES = [0.3, 0.6, 0.9]
V_RANGES: List[Tuple[float, float]] = [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2)]
REGION_KEYS = [(rho, v0, v1) for rho in RHO_VALUES for (v0, v1) in V_RANGES]


def get_model_sources(base_dir: Path) -> Dict[str, Path]:
    return {
        "deepseek-v3-0324": (base_dir / "summary_all_results_1.json").resolve(),
        "deepseek-v3.1": (base_dir / "summary_all_results_1.json").resolve(),
        "gpt-5.1": (base_dir / "summary_all_results_1.json").resolve(),
        "gpt-5.1-2025-11-13": (base_dir / "summary_all_results_1.json").resolve(),
        "deepseek-r1": (base_dir / "summary_all_results_2.json").resolve(),
        "qwen3-max": (base_dir / "summary_all_results_2.json").resolve(),
        "qwen-plus": (base_dir / "summary_all_results_2.json").resolve(),
        "qwen-plus-2025-12-01": (base_dir / "summary_all_results_2.json").resolve(),
        "gpt-5": (base_dir / "summary_all_results_gpt-5.json").resolve(),
        "gpt-5.2": (base_dir / "summary_all_results_gpt-5.2.json").resolve(),
        "deepseek-v3.2": (base_dir / "summary_all_results_deepseek-v3.2.json").resolve(),
        "qwen3-max-2026-01-23": (base_dir / "summary_all_results_qwen3-max-2026-01-23.json").resolve(),
    }


def extract_model_name(record: Dict) -> str:
    meta = record.get("experiment_meta", {}) if isinstance(record, dict) else {}
    return (
        meta.get("model_name")
        or record.get("model_name")
        or record.get("model")
        or record.get("llm_model")
        or ""
    )


def load_jaccard_by_region() -> Dict[str, Dict[Tuple[float, float, float], List[float]]]:
    sources = get_model_sources(BASE_DIR)
    missing = [f"{m}: {p}" for m, p in sources.items() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(missing))

    out: Dict[str, Dict[Tuple[float, float, float], List[float]]] = {}
    for model, src in sources.items():
        data = json.loads(src.read_text(encoding="utf-8"))
        grouped: Dict[Tuple[float, float, float], List[float]] = defaultdict(list)
        for row in data:
            if extract_model_name(row) != model:
                continue
            sp = row.get("sensitivity_params", {})
            key = (float(sp.get("rho")), float(sp.get("v_min")), float(sp.get("v_max")))
            grouped[key].append(float(row["equilibrium_quality"]["share_set_similarity"]))
        out[model] = grouped
    return out


def cv_percent(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return np.nan
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if mean <= 1e-12:
        return np.nan
    return std / mean * 100.0


def family_of(model: str) -> str:
    m = model.lower()
    if m.startswith("gpt-"):
        return "GPT"
    if m.startswith("deepseek"):
        return "DeepSeek"
    if m.startswith("qwen"):
        return "Qwen"
    return "Other"


def main() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [f for f in candidates if f in available]
    if not chosen:
        chosen = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(
        style="whitegrid",
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": chosen + ["DejaVu Sans"],
            "axes.unicode_minus": False,
        },
    )

    data = load_jaccard_by_region()
    models = list(data.keys())

    # Build CV matrix: rows=models, cols=9 parameter regions
    cv_matrix = np.full((len(models), len(REGION_KEYS)), np.nan)
    for i, model in enumerate(models):
        for j, key in enumerate(REGION_KEYS):
            cv_matrix[i, j] = cv_percent(data[model].get(key, []))

    mean_cv = np.nanmean(cv_matrix, axis=1)
    order = np.argsort(mean_cv)  # lower is better

    sorted_models = [models[i] for i in order]
    sorted_cv = cv_matrix[order, :]
    sorted_mean_cv = mean_cv[order]

    col_labels = [rf"$\rho$={k[0]}, v=[{k[1]},{k[2]}]" for k in REGION_KEYS]

    # 1) Ranked heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        sorted_cv,
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "CV（%）（越低越稳定）"},
        ax=ax,
    )
    ax.set_title("12模型稳定性热力图（按平均CV排序）", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("参数区域", fontsize=12, fontweight="bold")
    ax.set_ylabel("模型（平均CV更低者在上）", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(len(col_labels)) + 0.5)
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(sorted_models)) + 0.5)
    ax.set_yticklabels([f"{m}  ({sorted_mean_cv[i]:.2f}%)" for i, m in enumerate(sorted_models)], fontsize=10)

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_HEATMAP, dpi=300, bbox_inches="tight")
    plt.close()

    # 2) Sorted dot plot (overall mean CV)
    fam_color = {"GPT": "#d81b60", "DeepSeek": "#1e88e5", "Qwen": "#43a047", "Other": "#6d6d6d"}
    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(sorted_models))
    colors = [fam_color.get(family_of(m), fam_color["Other"]) for m in sorted_models]
    ax.hlines(y, 0, sorted_mean_cv, color=colors, alpha=0.45, linewidth=2)
    ax.scatter(sorted_mean_cv, y, s=90, c=colors, edgecolors="black", linewidths=0.7, zorder=3)
    for i, val in enumerate(sorted_mean_cv):
        ax.text(val + 0.05, i, f"{val:.2f}%", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_models, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("9个参数区域平均CV（%）", fontsize=12, fontweight="bold")
    ax.set_title("12模型总体稳定性排名（CV越低越好）", fontsize=15, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="GPT系列", markerfacecolor=fam_color["GPT"], markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="DeepSeek系列", markerfacecolor=fam_color["DeepSeek"], markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Qwen系列", markerfacecolor=fam_color["Qwen"], markeredgecolor="black", markersize=8),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(OUT_DOTPLOT, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved: {OUT_HEATMAP}")
    print(f"[OK] Saved: {OUT_DOTPLOT}")
    print("\nMean CV ranking (lower is better):")
    for rank, (m, v) in enumerate(zip(sorted_models, sorted_mean_cv), start=1):
        print(f"{rank:>2}. {m:<24} {v:>7.3f}%")


if __name__ == "__main__":
    main()
