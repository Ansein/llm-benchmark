"""
测试 JSON 生成问题的修复

对比使用和不使用 response_format 的效果
"""

import json
from src.evaluators.llm_client import LLMClient

# 加载配置
with open("configs/model_configs.json", 'r', encoding='utf-8') as f:
    configs = json.load(f)

# 找到 gemini 配置
gemini_config = None
for config in configs:
    if config["config_name"] == "gemini-3-flash-preview":
        gemini_config = config
        break

if not gemini_config:
    print("❌ 未找到 Gemini 配置")
    exit(1)

# 覆盖参数
gemini_config["generate_args"]["temperature"] = 0.7
gemini_config["generate_args"]["max_tokens"] = 1500

# 测试提示词
system_prompt = """你是理性经济主体，目标是在不确定他人行为的情况下最大化你的期望效用。

【重要】你必须只输出一个有效的JSON对象，不要包含任何额外的文本、解释或markdown标记。
JSON必须包含 "share" 和 "reason" 两个字段。
确保 "reason" 字段的字符串正确闭合（以引号结束）。"""

user_prompt = """你是用户 0，正在参与一个数据市场决策。

**你的私有信息**：
- 你的隐私偏好：v[0] = 0.637
- 平台给你的个性化报价：p[0] = 0.4821

请输出 JSON：
{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明你的决策理由（不超过150字）"
}"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

print("="*60)
print("🧪 测试 1: 不使用 response_format（新方法）")
print("="*60)

client1 = LLMClient(config=gemini_config)
try:
    result1 = client1.generate_json(messages, force_json_mode=False)
    print(f"✅ 成功解析")
    print(f"响应长度: {len(str(result1))} 字符")
    print(f"Share: {result1.get('share')}")
    print(f"Reason: {result1.get('reason')[:100]}..." if len(result1.get('reason', '')) > 100 else f"Reason: {result1.get('reason')}")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n" + "="*60)
print("🧪 测试 2: 使用 response_format（旧方法）")
print("="*60)

client2 = LLMClient(config=gemini_config)
try:
    result2 = client2.generate_json(messages, force_json_mode=True)
    print(f"✅ 成功解析")
    print(f"响应长度: {len(str(result2))} 字符")
    print(f"Share: {result2.get('share')}")
    print(f"Reason: {result2.get('reason')[:100]}..." if len(result2.get('reason', '')) > 100 else f"Reason: {result2.get('reason')}")
except Exception as e:
    print(f"❌ 失败: {e}")

print("\n" + "="*60)
print("💡 结论")
print("="*60)
print("如果测试1的响应长度远大于测试2，说明 response_format 是问题根源")
