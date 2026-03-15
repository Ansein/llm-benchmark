"""
Best-model matrix for Scenario-B sensitivity (12 LLMs).

Cell color = winning model (highest mean Jaccard in that parameter region).
Cell intensity = winning score.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import font_manager


BASE_DIR = Path("sensitivity_results/scenario_b")
OUTPUT_DIR = BASE_DIR / "model_comparison_plots"
OUTPUT_FILE = OUTPUT_DIR / "model_win_matrix_12models.png"

RHO_VALUES = [0.3, 0.6, 0.9]
V_RANGES: List[Tuple[float, float]] = [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2)]


def model_sources(base_dir: Path) -> Dict[str, Path]:
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


def load_stats() -> Dict[str, Dict[Tuple[float, float, float], List[float]]]:
    sources = model_sources(BASE_DIR)
    missing = [f"{m}: {p}" for m, p in sources.items() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(missing))

    stats: Dict[str, Dict[Tuple[float, float, float], List[float]]] = {}
    for model_name, src in sources.items():
        data = json.loads(src.read_text(encoding="utf-8"))
        grouped: Dict[Tuple[float, float, float], List[float]] = defaultdict(list)
        for row in data:
            if extract_model_name(row) != model_name:
                continue
            sp = row.get("sensitivity_params", {})
            key = (
                float(sp.get("rho")),
                float(sp.get("v_min")),
                float(sp.get("v_max")),
            )
            jaccard = float(row["equilibrium_quality"]["share_set_similarity"])
            grouped[key].append(jaccard)
        stats[model_name] = grouped
    return stats


def build_color_map(models: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    palette = plt.cm.tab20(np.linspace(0, 1, len(models)))
    return {m: palette[i] for i, m in enumerate(models)}


def main() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [f for f in candidates if f in available]
    if not chosen:
        chosen = ["DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    model_stats = load_stats()
    model_names = list(model_stats.keys())
    model_colors = build_color_map(model_names)

    fig, ax = plt.subplots(figsize=(15, 11))
    best_matrix = {}

    for r, rho in enumerate(sorted(RHO_VALUES, reverse=True)):
        for c, (v_min, v_max) in enumerate(V_RANGES):
            key = (rho, v_min, v_max)
            means = {}
            for model in model_names:
                vals = model_stats[model].get(key, [])
                if vals:
                    means[model] = float(np.mean(vals))

            if not means:
                winner = "N/A"
                score = np.nan
                base_color = (0.8, 0.8, 0.8, 1.0)
                alpha = 0.4
            else:
                winner = max(means, key=means.get)
                score = means[winner]
                base_color = model_colors[winner]
                alpha = 0.25 + 0.7 * score

            best_matrix[key] = (winner, score)

            rect = plt.Rectangle(
                (c, r), 1, 1, facecolor=base_color, edgecolor="black", linewidth=2, alpha=alpha
            )
            ax.add_patch(rect)

            if winner == "N/A":
                ax.text(c + 0.5, r + 0.56, "N/A", ha="center", va="center", fontsize=11, fontweight="bold")
            else:
                ax.text(
                    c + 0.5,
                    r + 0.58,
                    winner,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.text(c + 0.5, r + 0.28, f"{score:.3f}", ha="center", va="center", fontsize=12)
            ax.text(
                c + 0.5,
                r + 0.10,
                rf"$\rho$={rho}, v=[{v_min},{v_max}]",
                ha="center",
                va="center",
                fontsize=9,
                style="italic",
            )

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(
        ["低 v\n[0.3,0.6]", "中 v\n[0.6,0.9]", "高 v\n[0.9,1.2]"],
        fontsize=15,
        fontweight="bold",
    )
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(
        [r"高 $\rho$=0.9", r"中 $\rho$=0.6", r"低 $\rho$=0.3"],
        fontsize=15,
        fontweight="bold",
    )

    ax.set_title(
        "参数区域最优模型（12个LLM）\n"
        "（颜色=模型，深浅=平均Jaccard）",
        fontsize=20,
        fontweight="bold",
        pad=18,
    )

    legend_patches = [
        mpatches.Patch(facecolor=model_colors[m], edgecolor="black", label=m, alpha=0.8) for m in model_names
    ]
    ax.legend(
        handles=legend_patches,
        title="模型图例",
        title_fontsize=14,
        fontsize=11,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        framealpha=0.95,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved: {OUTPUT_FILE}")
    print("\nWinner by region:")
    for rho in sorted(RHO_VALUES, reverse=True):
        for v_min, v_max in V_RANGES:
            winner, score = best_matrix[(rho, v_min, v_max)]
            if np.isnan(score):
                print(f"  rho={rho}, v=[{v_min},{v_max}] -> N/A")
            else:
                print(f"  rho={rho}, v=[{v_min},{v_max}] -> {winner} ({score:.3f})")


if __name__ == "__main__":
    main()
