"""
快速查看最新的LLM调用日志

自动找到最新的日志目录并显示概览
"""

import os
import sys
from pathlib import Path
from view_llm_logs import load_log_files, print_overview, print_statistics


def find_latest_log_dir(base_dir: str = "evaluation_results/prompt_experiments_b/llm_logs") -> str:
    """找到最新的日志目录"""
    if not os.path.exists(base_dir):
        return None
    
    log_dirs = sorted(Path(base_dir).glob("*_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(log_dirs[0]) if log_dirs else None


def main():
    # 查找最新日志目录
    latest_dir = find_latest_log_dir()
    
    if not latest_dir:
        print("❌ 未找到日志目录")
        print("💡 运行实验后会自动生成日志")
        return
    
    print(f"📂 最新日志目录: {latest_dir}")
    
    # 加载并显示
    logs = load_log_files(latest_dir)
    
    if not logs:
        print("⚠️ 日志目录为空")
        return
    
    # 显示概览和统计
    print_overview(logs)
    print_statistics(logs)
    
    print(f"\n{'='*60}")
    print(f"💡 使用以下命令查看更多详情:")
    print(f"   python view_llm_logs.py --dir \"{latest_dir}\" --call-id <id>")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
