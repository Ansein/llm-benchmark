"""
从场景 C 的 detailed JSON 结果中，用 Config B 的混淆矩阵计算 Jaccard 相似度。
参与集合 Jaccard = TP / (TP + FP + FN)，两集皆空时取 1.0。
无需重跑实验，直接读已有 *_detailed.json。
"""
import json
from pathlib import Path

def jaccard_from_confusion(tp: float, fp: float, fn: float) -> float:
    denom = tp + fp + fn
    if denom <= 0:
        return 1.0
    return float(tp) / denom

def main():
    base = Path("evaluation_results/scenario_c")
    if not base.exists():
        base = Path(__file__).resolve().parent.parent / "evaluation_results" / "scenario_c"
    pattern = "scenario_c_common_preferences_*_*_detailed.json"
    files = sorted(base.glob(pattern))
    # 每个模型取最新一份（按时间戳）
    by_model = {}
    for f in files:
        # scenario_c_common_preferences_{model}_{YYYYMMDD_HHMMSS}_detailed.json
        stem = f.stem.replace("_detailed", "")
        idx = stem.find("common_preferences_") + len("common_preferences_")
        rest = stem[idx:]
        # 时间戳形如 20260205_231126，用 _20 定位
        i = rest.rfind("_20")
        if i >= 0:
            model_name = rest[:i]
            ts = rest[i + 1:]
        else:
            model_name = rest
            ts = ""
        if model_name not in by_model or (ts and (not by_model[model_name][0] or ts > by_model[model_name][0])):
            by_model[model_name] = (ts, f)

    print("Model,Jaccard(B)")
    for model in sorted(by_model.keys()):
        _, path = by_model[model]
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cb = data.get("config_B", {}) or data.get("config_B")
            part = cb.get("participation", {})
            cm = part.get("confusion_matrix", {}) or part.get("decision_confusion_matrix", {})
            tp = float(cm.get("TP", 0))
            fp = float(cm.get("FP", 0))
            fn = float(cm.get("FN", 0))
            j = jaccard_from_confusion(tp, fp, fn)
            print(f"{model},{j:.4f}")
        except Exception as e:
            print(f"{model},ERROR:{e}")

if __name__ == "__main__":
    main()
