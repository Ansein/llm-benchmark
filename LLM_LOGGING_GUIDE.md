# LLM调用日志系统使用指南

## 概述

LLM调用日志系统会自动缓存所有模型调用的详细信息，包括：
- 完整的输入提示词（system + user）
- 原始响应文本（即使被截断也会保存）
- 调用时间戳
- 模型配置参数
- 成功/失败状态

## 日志存储位置

日志会自动保存在：
```
evaluation_results/prompt_experiments_b/llm_logs/
├── b_v0_20260126_235959/
│   ├── call_0001_20260126_235959_123456.json
│   ├── call_0002_20260126_235959_234567.json
│   └── ...
├── b_v1_20260126_240100/
│   └── ...
```

每个实验版本会创建独立的日志目录，按时间戳命名。

## 日志文件格式

每个日志文件是一个JSON，包含：

```json
{
  "call_id": 1,
  "timestamp": "2026-01-26T23:59:59.123456",
  "model_name": "gemini-3-flash-preview",
  "messages": [
    {
      "role": "system",
      "content": "你是理性经济主体..."
    },
    {
      "role": "user",
      "content": "你是用户 5，正在参与一个数据市场..."
    }
  ],
  "response_format": {"type": "json_object"},
  "generate_params": {
    "temperature": 0.7,
    "max_tokens": 1500
  },
  "response": {
    "success": true,
    "text": "{\n  \"share\": 1,\n  \"reason\": \"...\"\n}",
    "length": 234,
    "error": null
  }
}
```

## 快速查看最新日志

```bash
# 自动查看最新的日志概览和统计
python view_latest_logs.py
```

输出示例：
```
📂 最新日志目录: evaluation_results/prompt_experiments_b/llm_logs/b_v0_20260126_235959

============================================================
📊 LLM调用日志概览
============================================================
总调用次数: 20
✅ 成功: 18 (90.0%)
❌ 失败: 2 (10.0%)

失败调用详情:
  Call #5: Unterminated string starting at: line 3 column 13...
  Call #12: Expecting value: line 1 column 1...

============================================================
📈 详细统计
============================================================

响应长度统计:
  平均: 187 字符
  最小: 45 字符
  最大: 456 字符

决策分布:
  不分享 (share=0): 12 次
  分享 (share=1): 8 次

⚠️ 检测到 2 次截断（已修复）
```

## 详细日志查看

### 1. 查看所有日志概览

```bash
python view_llm_logs.py --dir evaluation_results/prompt_experiments_b/llm_logs/b_v0_20260126_235959
```

### 2. 查看详细统计

```bash
python view_llm_logs.py --dir evaluation_results/prompt_experiments_b/llm_logs/b_v0_20260126_235959 --stats
```

### 3. 查看特定调用的详细信息

```bash
python view_llm_logs.py --dir evaluation_results/prompt_experiments_b/llm_logs/b_v0_20260126_235959 --call-id 5
```

输出示例：
```
============================================================
🔍 Call #5 详细信息
============================================================
时间: 2026-01-26T23:59:59.123456
模型: gemini-3-flash-preview

📤 请求:

[system]
你是理性经济主体，目标是在不确定他人行为的情况下最大化你的期望效用。
你必须输出严格JSON格式，不要包含任何额外的文本。

[user]
你是用户 5，正在参与一个数据市场。

**你的个人信息**：
- 平台给你的报价：p[5] = 0.3333
- 你的隐私偏好（单位信息的成本）：v[5] = 0.440

**决策框架**：
- 如果你分享数据，你会得到补偿 p = 0.3333
- 分享会产生隐私成本 = v × 边际信息泄露量
- 你需要权衡：补偿收益 vs 隐私成本

请输出严格JSON：
{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明你的决策理由（不超过150字）"
}

📥 响应:
状态: ✅ 成功
长度: 187 字符

内容:
{
  "share": 1,
  "reason": "我的隐私偏好v=0.440处于较低水平，而补偿p=0.3333相对合理。在边际信息泄露较小的情况下，补偿收益超过隐私成本，因此选择分享。...(截断)"
}
```

### 4. 导出所有失败的调用

```bash
python view_llm_logs.py --dir evaluation_results/prompt_experiments_b/llm_logs/b_v0_20260126_235959 --export-failed failed_calls.json
```

## 常见使用场景

### 场景1：发现模型总是输出0，想知道原因

```bash
# 1. 查看最新日志
python view_latest_logs.py

# 2. 如果看到很多失败，查看具体某个失败的调用
python view_llm_logs.py --dir <日志目录> --call-id 5

# 3. 检查原始响应是否被截断
```

### 场景2：对比不同提示词版本的效果

```bash
# 查看 v0 版本
python view_llm_logs.py --dir evaluation_results/prompt_experiments_b/llm_logs/b_v0_20260126_235959 --stats

# 查看 v1 版本
python view_llm_logs.py --dir evaluation_results/prompt_experiments_b/llm_logs/b_v1_20260126_235959 --stats

# 对比决策分布和响应长度
```

### 场景3：调试JSON解析问题

```bash
# 1. 导出所有失败的调用
python view_llm_logs.py --dir <日志目录> --export-failed failed.json

# 2. 查看 failed.json，分析失败模式
# 3. 如果看到"Unterminated string"，说明被截断了
# 4. 如果看到"Expecting value"，说明模型没有输出JSON
```

## 日志分析技巧

### 1. 检查响应长度分布

如果发现：
- **平均长度接近 max_tokens**：说明输出经常被截断，需要增加 max_tokens
- **长度差异很大**：说明模型对不同情况的解释详细程度不同

### 2. 检查截断标记

如果日志中显示"检测到 X 次截断（已修复）"：
- 说明模型输出被截断，但修复逻辑成功提取了 share 字段
- 可以查看具体调用，确认修复是否正确

### 3. 对比成功和失败的调用

```bash
# 查看失败的调用
python view_llm_logs.py --dir <日志目录> --call-id <失败的ID>

# 查看成功的调用
python view_llm_logs.py --dir <日志目录> --call-id <成功的ID>

# 对比提示词是否有差异
```

## 禁用日志（如果需要）

如果不需要日志（例如大规模实验），可以在代码中注释掉：

```python
# run_prompt_experiments.py 第 394 行附近
# llm_client = LLMClient(config=llm_config, log_dir=log_dir)
llm_client = LLMClient(config=llm_config)  # 不启用日志
```

## 注意事项

1. **磁盘空间**：每次调用会生成一个小的JSON文件，大规模实验可能产生大量文件
2. **隐私**：日志包含完整的提示词和响应，请勿上传到公开仓库
3. **性能**：写入日志有轻微的性能开销（通常可忽略）

## 总结

日志系统的主要优势：
- ✅ **完整追溯**：可以回溯任何一次LLM调用的完整上下文
- ✅ **调试利器**：快速定位JSON解析失败、截断等问题
- ✅ **提示词优化**：对比不同版本的实际效果
- ✅ **数据分析**：统计决策分布、响应长度等指标
