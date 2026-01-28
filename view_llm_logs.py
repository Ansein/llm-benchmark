"""
LLM调用日志查看器

用于查看和分析缓存的LLM调用记录

使用方法：
1. 查看所有日志概览：
   python view_llm_logs.py --dir evaluation_results/prompt_experiments_b/llm_logs/b_v0_20260126_235959

2. 查看特定调用的详细信息：
   python view_llm_logs.py --dir evaluation_results/prompt_experiments_b/llm_logs/b_v0_20260126_235959 --call-id 5

3. 导出所有失败的调用：
   python view_llm_logs.py --dir evaluation_results/prompt_experiments_b/llm_logs/b_v0_20260126_235959 --export-failed failed_calls.json

4. 统计分析：
   python view_llm_logs.py --dir evaluation_results/prompt_experiments_b/llm_logs/b_v0_20260126_235959 --stats
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def load_log_files(log_dir: str) -> List[Dict[str, Any]]:
    """加载所有日志文件"""
    log_files = sorted(Path(log_dir).glob("call_*.json"))
    logs = []
    
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8') as f:
            logs.append(json.load(f))
    
    return logs


def print_overview(logs: List[Dict[str, Any]]):
    """打印日志概览"""
    total = len(logs)
    success = sum(1 for log in logs if log["response"]["success"])
    failed = total - success
    
    print(f"\n{'='*60}")
    print(f"📊 LLM调用日志概览")
    print(f"{'='*60}")
    print(f"总调用次数: {total}")
    print(f"✅ 成功: {success} ({success/total*100:.1f}%)")
    print(f"❌ 失败: {failed} ({failed/total*100:.1f}%)")
    
    if failed > 0:
        print(f"\n失败调用详情:")
        for log in logs:
            if not log["response"]["success"]:
                call_id = log["call_id"]
                error = log["response"]["error"]
                print(f"  Call #{call_id}: {error[:100]}...")


def print_statistics(logs: List[Dict[str, Any]]):
    """打印详细统计"""
    print(f"\n{'='*60}")
    print(f"📈 详细统计")
    print(f"{'='*60}")
    
    # 响应长度统计
    lengths = [log["response"]["length"] for log in logs if log["response"]["success"]]
    if lengths:
        print(f"\n响应长度统计:")
        print(f"  平均: {sum(lengths)/len(lengths):.0f} 字符")
        print(f"  最小: {min(lengths)} 字符")
        print(f"  最大: {max(lengths)} 字符")
    
    # 按用户决策统计（如果有share字段）
    share_stats = defaultdict(int)
    truncated_count = 0
    
    for log in logs:
        if log["response"]["success"]:
            response_text = log["response"]["text"]
            if "截断" in response_text:
                truncated_count += 1
            
            # 尝试解析share字段
            try:
                import re
                share_match = re.search(r'"share"\s*:\s*(\d+)', response_text)
                if share_match:
                    share = int(share_match.group(1))
                    share_stats[share] += 1
            except:
                pass
    
    if share_stats:
        print(f"\n决策分布:")
        for share, count in sorted(share_stats.items()):
            decision = "分享" if share == 1 else "不分享"
            print(f"  {decision} (share={share}): {count} 次")
    
    if truncated_count > 0:
        print(f"\n⚠️ 检测到 {truncated_count} 次截断（已修复）")


def print_call_detail(logs: List[Dict[str, Any]], call_id: int):
    """打印特定调用的详细信息"""
    log = next((l for l in logs if l["call_id"] == call_id), None)
    
    if not log:
        print(f"❌ 未找到 Call ID {call_id}")
        return
    
    print(f"\n{'='*60}")
    print(f"🔍 Call #{call_id} 详细信息")
    print(f"{'='*60}")
    print(f"时间: {log['timestamp']}")
    print(f"模型: {log['model_name']}")
    
    print(f"\n📤 请求:")
    for msg in log["messages"]:
        role = msg["role"]
        content = msg["content"]
        print(f"\n[{role}]")
        print(content[:500] + "..." if len(content) > 500 else content)
    
    print(f"\n📥 响应:")
    response = log["response"]
    if response["success"]:
        print(f"状态: ✅ 成功")
        print(f"长度: {response['length']} 字符")
        print(f"\n内容:")
        print(response["text"])
    else:
        print(f"状态: ❌ 失败")
        print(f"错误: {response['error']}")


def export_failed(logs: List[Dict[str, Any]], output_file: str):
    """导出所有失败的调用"""
    failed_logs = [log for log in logs if not log["response"]["success"]]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(failed_logs, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 已导出 {len(failed_logs)} 个失败调用到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="LLM调用日志查看器")
    parser.add_argument("--dir", type=str, required=True, help="日志目录路径")
    parser.add_argument("--call-id", type=int, help="查看特定调用的详细信息")
    parser.add_argument("--stats", action="store_true", help="显示详细统计")
    parser.add_argument("--export-failed", type=str, help="导出失败调用到指定文件")
    
    args = parser.parse_args()
    
    # 加载日志
    if not os.path.exists(args.dir):
        print(f"❌ 日志目录不存在: {args.dir}")
        return
    
    print(f"📂 加载日志目录: {args.dir}")
    logs = load_log_files(args.dir)
    
    if not logs:
        print("⚠️ 未找到日志文件")
        return
    
    # 根据参数执行不同操作
    if args.call_id:
        print_call_detail(logs, args.call_id)
    elif args.export_failed:
        export_failed(logs, args.export_failed)
    elif args.stats:
        print_statistics(logs)
    else:
        print_overview(logs)
        print(f"\n💡 使用 --stats 查看详细统计")
        print(f"💡 使用 --call-id <id> 查看特定调用详情")


if __name__ == "__main__":
    main()
