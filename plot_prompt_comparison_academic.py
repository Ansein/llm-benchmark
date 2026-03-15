import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def zh(s: str) -> str:
    return s.encode('utf-8').decode('utf-8')


def configure_fonts() -> None:
    candidates = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS']
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [f for f in candidates if f in available]
    if not chosen:
        chosen = ['DejaVu Sans']

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = chosen + ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'


def extract_model_name(filepath: Path) -> str:
    filename = Path(filepath).stem
    match = re.match(r'summary_(.+?)_\d{8}_\d{6}', filename)
    if match:
        return match.group(1)
    return filename.replace('summary_', '')


def classify_model(model_name: str) -> str:
    lower_name = model_name.lower()
    if 'gpt' in lower_name:
        return 'GPT'
    if 'deepseek' in lower_name:
        return 'DeepSeek'
    if 'qwen' in lower_name:
        return 'Qwen'
    return 'Other'


def main() -> None:
    configure_fonts()

    results_dir = Path('evaluation_results/prompt_experiments_b')
    summary_files = sorted(results_dir.glob('summary_*.json'))
    print(f"\u52a0\u8f7d {len(summary_files)} \u4e2a\u6a21\u578b\u7ed3\u679c\u6587\u4ef6\n")

    raw_versions = ['b.v0', 'b.v1', 'b.v2', 'b.v3', 'b.v4', 'b.v6']
    display_versions = ['v0', 'v1', 'v2', 'v3', 'v4', 'v5']
    version_labels = [
        'v0\n\u57fa\u7ebf',
        'v1\n+\u5e02\u573a\n\u53c2\u6570',
        'v2\n+\u53c2\u6570\n\u89e3\u91ca',
        'v3\n+\u63a8\u65ad\n\u5916\u90e8\u6027',
        'v4\n+\u6b21\u6a21\u6027\n\u4e0e\u8865\u507f',
        'v5\n+\u5b8c\u6574\n\u516c\u5f0f',
    ]

    models_data = {}
    for filepath in summary_files:
        model_name = extract_model_name(filepath)
        try:
            data = json.loads(Path(filepath).read_text(encoding='utf-8'))
            distances = [data['versions'].get(v, {}).get('decision_distance_mean', 1) for v in raw_versions]
            models_data[model_name] = {
                'decision_distances': distances,
                'series': classify_model(model_name),
            }
            print(f'[OK] {model_name}')
        except Exception as e:
            print(f'[ERR] {model_name}: {e}')

    print(f"\n\u6210\u529f\u52a0\u8f7d {len(models_data)} \u4e2a\u6a21\u578b\n")

    series_models = {}
    for model_name, info in models_data.items():
        series_models.setdefault(info['series'], []).append(model_name)

    colors = ['#6094ce', '#4dbe93', '#f5cc2f', '#4c2e90', '#FF5252']
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.subplots_adjust(hspace=0.35)
    x = np.arange(len(display_versions))

    for idx, (series, model_list) in enumerate(sorted(series_models.items())):
        ax = axes[idx]
        for i, model in enumerate(sorted(model_list)):
            d = models_data[model]['decision_distances']
            ax.plot(
                x,
                d,
                marker='o',
                color=colors[i % len(colors)],
                linewidth=2.5,
                markersize=6,
                label=model,
                markeredgewidth=1.0,
                markeredgecolor='white',
                alpha=0.85,
            )

        ax.set_title(f"{series} \u7cfb\u5217", fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('\u63d0\u793a\u8bcd\u7248\u672c', fontsize=12, fontweight='bold')
        ax.set_ylabel('\u51b3\u7b56\u8ddd\u79bb', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(version_labels, fontsize=9, ha='center')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax.axhline(
            y=0,
            color='#2E7D32',
            linestyle='--',
            linewidth=2,
            alpha=0.6,
            label='\u7406\u8bba\u6700\u4f18\u5bf9\u9f50',
            zorder=0,
        )
        ax.legend(loc='best', fontsize=9, framealpha=0.95, edgecolor='gray', fancybox=False, shadow=False)

        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#333333')

    fig.suptitle('\u4e0d\u540c\u6a21\u578b\u7cfb\u5217\u7684\u63d0\u793a\u8bcd\u5de5\u7a0b\u8868\u73b0', fontsize=16, fontweight='bold', y=0.995)

    output_path = Path('evaluation_results/prompt_experiments_b/academic_comparison.png')
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n\u5b66\u672f\u98ce\u683c\u56fe\u5df2\u4fdd\u5b58: {output_path}\n")


if __name__ == '__main__':
    main()
