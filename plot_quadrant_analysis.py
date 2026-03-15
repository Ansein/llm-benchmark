"""
Quadrant analysis for Scenario-B sensitivity results.
X-axis: Average Jaccard Similarity (higher is better)
Y-axis: EAS (higher is better)
"""
import json
import math
import re
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib import font_manager


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _extract_model_name(record: Dict) -> str:
    meta = record.get("experiment_meta", {}) if isinstance(record, dict) else {}
    return (
        meta.get("model_name")
        or record.get("model_name")
        or record.get("model")
        or record.get("llm_model")
        or ""
    )


def _extract_trial_id(record: Dict) -> int:
    meta = record.get("experiment_meta", {}) if isinstance(record, dict) else {}
    return int(meta.get("trial_index", -1))


def _explicit_model_sources(base_dir: str) -> Dict[str, Path]:
    base = Path(base_dir)
    return {
        "deepseek-v3-0324": (base / "summary_all_results_1.json").resolve(),
        "deepseek-v3.1": (base / "summary_all_results_1.json").resolve(),
        "gpt-5.1": (base / "summary_all_results_1.json").resolve(),
        "gpt-5.1-2025-11-13": (base / "summary_all_results_1.json").resolve(),
        "deepseek-r1": (base / "summary_all_results_2.json").resolve(),
        "qwen3-max": (base / "summary_all_results_2.json").resolve(),
        "qwen-plus": (base / "summary_all_results_2.json").resolve(),
        "qwen-plus-2025-12-01": (base / "summary_all_results_2.json").resolve(),
        "gpt-5": (base / "summary_all_results_gpt-5.json").resolve(),
        "gpt-5.2": (base / "summary_all_results_gpt-5.2.json").resolve(),
        "deepseek-v3.2": (base / "summary_all_results_deepseek-v3.2.json").resolve(),
        "qwen3-max-2026-01-23": (base / "summary_all_results_qwen3-max-2026-01-23.json").resolve(),
    }


def _metric_sig(row: Dict) -> tuple:
    def _norm(v):
        if v is None:
            return "nan"
        try:
            x = float(v)
            if math.isnan(x):
                return "nan"
            return round(x, 8)
        except Exception:
            return "nan"
    return (_norm(row.get("jaccard")), _norm(row.get("llm_share_rate")), _norm(row.get("gt_share_rate")))


def collect_records(base_dir: str = "sensitivity_results/scenario_b", verbose: bool = True) -> List[Dict]:
    files = list(Path(base_dir).rglob("summary*.json"))
    source_map = _explicit_model_sources(base_dir)
    missing_sources = [m for m, p in source_map.items() if not p.exists()]
    if missing_sources:
        raise FileNotFoundError(
            "Required model source files missing: "
            + ", ".join(f"{m} -> {source_map[m]}" for m in missing_sources)
        )
    # Keep one canonical record per (model, rho, v_min, v_max, trial)
    canonical: Dict[tuple, Dict] = {}
    canonical_meta: Dict[tuple, Dict] = {}
    conflict_count = 0
    duplicate_count = 0

    for f in files:
        f_resolved = f.resolve()
        file_mtime = f.stat().st_mtime
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue

        for r in data:
            if not isinstance(r, dict):
                continue

            model = _extract_model_name(r)
            if not model:
                continue
            if model in source_map and f_resolved != source_map[model]:
                continue

            sp = r.get("sensitivity_params", {})
            rho = _safe_float(sp.get("rho"))
            v_min = _safe_float(sp.get("v_min"))
            v_max = _safe_float(sp.get("v_max"))
            if np.isnan(rho) or np.isnan(v_min) or np.isnan(v_max):
                continue

            metrics = r.get("metrics", {})
            llm = metrics.get("llm", {})
            gt = metrics.get("ground_truth", {})

            llm_sr = _safe_float(llm.get("share_rate"))
            gt_sr = _safe_float(gt.get("share_rate"))

            eq = r.get("equilibrium_quality", {})
            jaccard = _safe_float(eq.get("share_set_similarity"))

            trial = _extract_trial_id(r)

            rec = {
                "model": model,
                "rho": rho,
                "v_min": v_min,
                "v_max": v_max,
                "v_mean": (v_min + v_max) / 2.0,
                "trial": trial,
                "llm_share_rate": llm_sr,
                "gt_share_rate": gt_sr,
                "jaccard": jaccard,
            }
            core_key = (model, round(rho, 6), round(v_min, 6), round(v_max, 6), trial)

            if core_key not in canonical:
                canonical[core_key] = rec
                canonical_meta[core_key] = {"mtime": file_mtime, "src": str(f)}
                continue

            prev = canonical[core_key]
            if _metric_sig(prev) != _metric_sig(rec):
                conflict_count += 1
            else:
                duplicate_count += 1

            # Deterministic tie-break: prefer latest file mtime
            if file_mtime >= canonical_meta[core_key]["mtime"]:
                canonical[core_key] = rec
                canonical_meta[core_key] = {"mtime": file_mtime, "src": str(f)}

    if verbose:
        print(f"Canonical dedupe: kept {len(canonical)} unique (model, rho, v_min, v_max, trial) records")
        print(f"  exact duplicates removed: {duplicate_count}")
        print(f"  conflicting duplicates resolved (by latest file): {conflict_count}")

    return list(canonical.values())


def _linear_slope(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return np.nan
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if np.nanstd(x) < 1e-12:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def _safe_corr(a: List[float], b: List[float]) -> float:
    x = np.array(a, dtype=float)
    y = np.array(b, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def compute_eas(model_records: List[Dict]) -> float:
    # param=rho, control=v_mean
    by_vmean = defaultdict(list)
    for r in model_records:
        by_vmean[round(r["v_mean"], 6)].append(r)

    llm_slopes_rho, gt_slopes_rho = [], []
    for rows in by_vmean.values():
        rows = sorted(rows, key=lambda z: z["rho"])
        xs = [z["rho"] for z in rows]
        y_llm = [z["llm_share_rate"] for z in rows]
        y_gt = [z["gt_share_rate"] for z in rows]
        llm_slopes_rho.append(_linear_slope(xs, y_llm))
        gt_slopes_rho.append(_linear_slope(xs, y_gt))

    score_rho = _safe_corr(llm_slopes_rho, gt_slopes_rho)

    # param=v_mean, control=rho
    by_rho = defaultdict(list)
    for r in model_records:
        by_rho[round(r["rho"], 6)].append(r)

    llm_slopes_v, gt_slopes_v = [], []
    for rows in by_rho.values():
        rows = sorted(rows, key=lambda z: z["v_mean"])
        xs = [z["v_mean"] for z in rows]
        y_llm = [z["llm_share_rate"] for z in rows]
        y_gt = [z["gt_share_rate"] for z in rows]
        llm_slopes_v.append(_linear_slope(xs, y_llm))
        gt_slopes_v.append(_linear_slope(xs, y_gt))

    score_v = _safe_corr(llm_slopes_v, gt_slopes_v)

    scores = [s for s in (score_rho, score_v) if not np.isnan(s)]
    return float(np.mean(scores)) if scores else np.nan


def _family(name: str) -> str:
    n = name.lower()
    if n.startswith("gpt-"):
        return "GPT 系列"
    if n.startswith("deepseek"):
        return "DeepSeek 系列"
    if n.startswith("qwen"):
        return "Qwen 系列"
    return "其他"


def _publish_key(name: str) -> float:
    # User-specified release order (top -> bottom means older -> newer)
    ordered_models = [
        "deepseek-v3-0324",
        "deepseek-r1",
        "deepseek-v3.1",
        "deepseek-v3.2",
        "gpt-5",
        "gpt-5.1",
        "gpt-5.1-2025-11-13",
        "gpt-5.2",
        "qwen-plus",
        "qwen3-max",
        "qwen-plus-2025-12-01",
        "qwen3-max-2026-01-23",
    ]
    order_map = {m: i for i, m in enumerate(ordered_models)}

    n = name.lower()
    if n in order_map:
        return float(order_map[n])

    # Fallback to explicit date in name
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", n)
    if m:
        y, mo, d = map(int, m.groups())
        return float(y * 10000 + mo * 100 + d)

    return 1e9


def _family_styles(family_models: Dict[str, List[str]]) -> Dict[str, Dict[str, object]]:
    family_cfg = {
        "GPT 系列": {"marker": "o", "cmap": plt.cm.RdPu},
        "DeepSeek 系列": {"marker": "^", "cmap": plt.cm.Blues},
        "Qwen 系列": {"marker": "s", "cmap": plt.cm.YlGn},
        "其他": {"marker": "D", "cmap": plt.cm.Greys},
    }

    styles: Dict[str, Dict[str, object]] = {}
    for fam, models in family_models.items():
        cfg = family_cfg.get(fam, family_cfg["其他"])
        ordered = sorted(models, key=_publish_key)
        n = len(ordered)
        if n == 1:
            shades = [0.82]
        else:
            # Increase intra-family contrast: much lighter oldest -> much darker newest
            shades = np.linspace(0.35, 0.995, n)
        for m, s in zip(ordered, shades):
            styles[m] = {
                "family": fam,
                "marker": cfg["marker"],
                "color": cfg["cmap"](float(s)),
            }
    return styles


def plot_quadrant(model_points: List[Dict], output_path: str):
    candidates = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS']
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [f for f in candidates if f in available]
    if not chosen:
        chosen = ['DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = chosen + ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    pts = [p for p in model_points if not np.isnan(p["avg_jaccard"]) and not np.isnan(p["eas"])]
    if not pts:
        raise RuntimeError("No valid points to plot")

    x = [p["avg_jaccard"] for p in pts]
    y = [p["eas"] for p in pts]

    x_med = float(np.median(x))
    y_med = float(np.median(y))

    fam_models = defaultdict(list)
    for p in pts:
        fam_models[_family(p["model_name"])].append(p["model_name"])
    styles = _family_styles(fam_models)

    fig, ax = plt.subplots(figsize=(13.5, 10.5))

    # 画点
    for p in pts:
        st = styles.get(p["model_name"], {"marker": "D", "color": "#777777"})
        ax.scatter(
            p["avg_jaccard"], p["eas"],
            s=290,
            marker=st["marker"],
            color=st["color"],
            edgecolors='black',
            linewidth=1.1,
            alpha=0.9,
            zorder=3
        )

    # 轴范围延展（给四象限说明留出更大边距，避免遮挡散点）
    x_min = min(min(x), 0.0)
    x_max = max(max(x), 1.0)
    y_min = min(min(y), 0.0)
    y_max = max(max(y), 1.0)
    x_pad = max(0.12, 0.12 * (x_max - x_min + 1e-9))
    y_pad = max(0.18, 0.14 * (y_max - y_min + 1e-9))
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # 四象限背景色（参考示例风格）
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    quadrant_alpha = 0.30
    ax.add_patch(Rectangle((xl[0], y_med), x_med - xl[0], yl[1] - y_med,
                           facecolor="#DCEED8", edgecolor="none", alpha=quadrant_alpha, zorder=1))  # top-left
    ax.add_patch(Rectangle((x_med, y_med), xl[1] - x_med, yl[1] - y_med,
                           facecolor="#DDE4F6", edgecolor="none", alpha=quadrant_alpha, zorder=1))  # top-right
    ax.add_patch(Rectangle((xl[0], yl[0]), x_med - xl[0], y_med - yl[0],
                           facecolor="#F7EBD0", edgecolor="none", alpha=quadrant_alpha, zorder=1))  # bottom-left
    ax.add_patch(Rectangle((x_med, yl[0]), xl[1] - x_med, y_med - yl[0],
                           facecolor="#F4DEDE", edgecolor="none", alpha=quadrant_alpha, zorder=1))  # bottom-right

    # 中位数分界线
    ax.axvline(x_med, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)
    ax.axhline(y_med, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)

    # 四象限说明（x高好, y高好）
    x_left = xl[0] + 0.06 * (x_med - xl[0])
    x_right = x_med + 0.94 * (xl[1] - x_med)
    y_top = yl[1] - 0.06 * (yl[1] - y_med)
    y_bottom = yl[0] + 0.06 * (y_med - yl[0])

    ax.text(x_right, y_top, '强机制模型', ha='right', va='top', fontsize=14.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#D5F4E6', alpha=0.75, edgecolor='green'))
    ax.text(x_left, y_top, '潜力模型', ha='left', va='top', fontsize=14.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#E8F4F8', alpha=0.75, edgecolor='blue'))
    ax.text(x_right, y_bottom, '表面拟合', ha='right', va='bottom', fontsize=14.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#FFF4E0', alpha=0.75, edgecolor='orange'))
    ax.text(x_left, y_bottom, '弱机制模型', ha='left', va='bottom', fontsize=14.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#F8E8E8', alpha=0.75, edgecolor='brown'))

    ax.set_xlabel('平均 Jaccard 相似度（越高越好）', fontsize=17, fontweight='bold')
    ax.set_ylabel('弹性对齐分数 EAS（越高越好）', fontsize=17, fontweight='bold')
    ax.set_title('象限分析：结果准确性 vs 机制理解能力', fontsize=19, fontweight='bold', pad=14)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(True, alpha=0.2, linestyle='--')

    # 中位数标签
    median_box = dict(boxstyle='round,pad=0.25', facecolor='#F7F7F7', edgecolor='#6E6E6E', alpha=0.92)
    # Place x-median label at a fixed x position
    x_median_label_x = 0.77
    ax.text(
        x_median_label_x, yl[0] + 0.02 * (yl[1] - yl[0]), f'中位数={x_med:.3f}',
        ha='center', va='bottom', fontsize=13.5, bbox=median_box
    )
    ax.text(
        xl[0] + 0.01 * (xl[1] - xl[0]), y_med, f'中位数={y_med:.3f}',
        ha='left', va='center', fontsize=13.5, bbox=median_box
    )

    # 单一图例框：按家族分组，松散分布在上下边界之间
    legend_rows = []
    family_order = ["DeepSeek 系列", "GPT 系列", "Qwen 系列"]
    for fi, fam in enumerate(family_order):
        models = sorted(fam_models.get(fam, []), key=_publish_key)
        if not models:
            continue
        legend_rows.append({"type": "header", "label": fam})
        for m in models:
            st = styles[m]
            legend_rows.append({"type": "model", "label": m, "style": st})
        if fi < len(family_order) - 1:
            legend_rows.append({"type": "spacer", "label": ""})

    # 右侧单一图例面板：上下边界与主图对齐
    ax_leg = fig.add_axes([0.79, 0.10, 0.20, 0.82])
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis('off')

    panel = FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.0, edgecolor="#A0A0A0", facecolor="#F7F7F7", alpha=0.95
    )
    ax_leg.add_patch(panel)

    rows = [{"type": "title", "label": "模型图例"}] + legend_rows
    n_rows = len(rows)
    y_top, y_bottom = 0.96, 0.04
    ys = np.linspace(y_top, y_bottom, n_rows)
    x_marker, x_text = 0.08, 0.13

    for row, yy in zip(rows, ys):
        rtype = row["type"]
        label = row["label"]
        if rtype == "title":
            ax_leg.text(0.5, yy, label, ha='center', va='center', fontsize=20, fontweight='bold')
        elif rtype == "header":
            ax_leg.text(0.08, yy, label, ha='left', va='center', fontsize=18, fontweight='bold')
        elif rtype == "model":
            st = row["style"]
            ax_leg.plot(
                [x_marker], [yy],
                marker=st["marker"], markersize=10.8, linestyle='None',
                markerfacecolor=st["color"], markeredgecolor='black', markeredgewidth=0.8
            )
            ax_leg.text(x_text, yy, label, ha='left', va='center', fontsize=17.2)

    # 固定右侧留白，避免导出时裁掉外侧图例
    fig.subplots_adjust(left=0.09, right=0.78, top=0.92, bottom=0.10)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    print('Collecting summary records from sensitivity_results/scenario_b ...')
    records = collect_records('sensitivity_results/scenario_b', verbose=True)
    if not records:
        raise RuntimeError('No valid records found in sensitivity_results/scenario_b')

    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_model[r['model']].append(r)

    print(f'Total records: {len(records)}')
    print(f'Models found: {len(by_model)}')

    model_points = []
    for m in sorted(by_model.keys()):
        recs = by_model[m]
        avg_jaccard = float(np.nanmean([_safe_float(z.get('jaccard')) for z in recs]))
        eas = compute_eas(recs)
        model_points.append({
            'model_name': m,
            'avg_jaccard': avg_jaccard,
            'eas': eas,
            'n_records': len(recs)
        })
        print(f'  {m}: n={len(recs)}, jaccard={avg_jaccard:.4f}, EAS={eas:.4f}')

    out_dir = Path('evaluation_results/eas_analysis')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'quadrant_analysis.png'
    plot_quadrant(model_points, str(out_path))

    table_path = out_dir / 'quadrant_analysis_points.json'
    table_path.write_text(json.dumps(model_points, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'\nSaved quadrant figure: {out_path}')
    print(f'Saved point table: {table_path}')


if __name__ == '__main__':
    main()
