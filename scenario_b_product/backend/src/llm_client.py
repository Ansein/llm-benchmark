"""
LLM客户端封装
支持OpenAI兼容的API接口
"""

import json
import re
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from openai import OpenAI


class LLMClient:
    """LLM客户端封装类"""
    
    def __init__(self, config: Dict[str, Any], log_dir: Optional[str] = None):
        """
        初始化LLM客户端
        
        Args:
            config: 模型配置字典，包含：
                - config_name: 配置名称
                - model_type: 模型类型（目前支持 openai_chat）
                - model_name: 模型名称
                - api_key: API密钥
                - client_args: 客户端参数（如 base_url）
                - generate_args: 生成参数（如 temperature）
            log_dir: 日志目录路径，如果提供则会保存所有LLM调用的详细日志
        """
        self.config_name = config["config_name"]
        self.model_type = config["model_type"]
        self.model_name = config["model_name"]
        self.generate_args = config.get("generate_args", {})
        
        # 日志设置
        self.log_dir = log_dir
        self.call_counter = 0  # 调用计数器
        if self.log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            print(f"📝 LLM调用日志已启用，保存路径: {self.log_dir}")
        
        # 初始化OpenAI客户端
        if self.model_type == "openai_chat":
            client_args = config.get("client_args", {})
            self.client = OpenAI(
                api_key=config["api_key"],
                **client_args
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        生成响应
        
        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            response_format: 响应格式（如 {"type": "json_object"}）
            **kwargs: 额外的生成参数
        
        Returns:
            生成的文本响应
        """
        self.call_counter += 1
        call_id = self.call_counter
        
        # 合并默认参数和自定义参数
        generate_params = {**self.generate_args, **kwargs}
        
        # 记录请求信息（用于日志）
        request_log = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "messages": messages,
            "response_format": response_format,
            "generate_params": generate_params
        }
        
        # 调用API
        try:
            if response_format:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format=response_format,
                    **generate_params
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **generate_params
                )
            
            response_text = response.choices[0].message.content
            
            # 保存日志
            if self.log_dir:
                self._save_call_log(request_log, response_text, success=True)
            
            return response_text
        
        except Exception as e:
            # 保存失败日志
            if self.log_dir:
                self._save_call_log(request_log, str(e), success=False, error=str(e))
            
            print(f"❌ LLM调用失败: {e}")
            raise
    
    def generate_json(
        self, 
        messages: List[Dict[str, str]], 
        force_json_mode: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成JSON格式的响应
        
        Args:
            messages: 消息列表
            force_json_mode: 是否强制使用 API 的 JSON 模式（可能导致截断，默认 False）
            **kwargs: 额外的生成参数
        
        Returns:
            解析后的JSON字典
        """
        # 请求JSON格式输出（默认不使用 response_format，因为 Gemini 有 bug）
        if force_json_mode:
            response_text = self.generate(
                messages=messages,
                response_format={"type": "json_object"},
                **kwargs
            )
        else:
            response_text = self.generate(
                messages=messages,
                **kwargs
            )
        
        # 清理可能的Markdown代码块标记（针对DeepSeek等模型）
        response_text = self._clean_json_response(response_text)
        
        # 解析JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败: {e}")
            print(f"原始响应: {response_text}")
            
            # 尝试修复不完整的JSON（主要针对截断的reason字段）
            repaired_json = self._repair_truncated_json(response_text)
            if repaired_json:
                print(f"✅ 成功修复JSON，提取到的字段: {list(repaired_json.keys())}")
                return repaired_json
            
            raise
    
    def _clean_json_response(self, response_text: str) -> str:
        """
        清理JSON响应中的Markdown标记和其他格式问题
        
        某些模型（如DeepSeek）可能会输出 ```json ... ``` 格式
        这个方法会去掉这些标记，只保留纯JSON
        
        Args:
            response_text: 原始响应文本
        
        Returns:
            清理后的JSON文本
        """
        text = response_text.strip()
        
        # 方法1: 去掉开头的 ```json 或 ```
        if text.startswith("```json"):
            text = text[7:].strip()
        elif text.startswith("```"):
            text = text[3:].strip()
        
        # 去掉结尾的 ```
        if text.endswith("```"):
            text = text[:-3].strip()
        
        # 方法2: 使用正则表达式提取JSON（更鲁棒）
        # 匹配```json 或 ``` 包裹的内容
        json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(json_block_pattern, text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        
        # 方法3: 如果还是解析失败，尝试找到第一个{和最后一个}之间的内容
        if '{' in text and '}' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            text = text[start:end]
        
        return text
    
    def _repair_truncated_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        尝试修复被截断的JSON（通常是reason字段被截断）
        
        策略：使用正则表达式提取关键字段，即使整体JSON不完整
        
        Args:
            text: 不完整的JSON文本
        
        Returns:
            修复后的字典，如果无法修复则返回None
        """
        result = {}
        
        # 提取 share 字段（最关键）
        share_match = re.search(r'"share"\s*:\s*(\d+)', text)
        if share_match:
            result["share"] = int(share_match.group(1))
        
        # 提取 belief_share_rate 字段（如果有）
        belief_match = re.search(r'"belief_share_rate"\s*:\s*([\d.]+)', text)
        if belief_match:
            result["belief_share_rate"] = float(belief_match.group(1))
        
        # 提取 reason 字段（尽力而为，可能不完整）
        reason_match = re.search(r'"reason"\s*:\s*"([^"]*)', text)
        if reason_match:
            result["reason"] = reason_match.group(1) + "...(截断)"
        
        # 如果至少提取到 share 字段，就认为修复成功
        if "share" in result:
            return result
        
        return None
    
    def _save_call_log(self, request_log: Dict[str, Any], response_text: str, 
                       success: bool, error: Optional[str] = None):
        """
        保存LLM调用日志
        
        Args:
            request_log: 请求信息
            response_text: 响应文本
            success: 是否成功
            error: 错误信息（如果失败）
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            call_id = request_log["call_id"]
            filename = f"call_{call_id:04d}_{timestamp}.json"
            filepath = os.path.join(self.log_dir, filename)
            
            log_data = {
                **request_log,
                "response": {
                    "success": success,
                    "text": response_text,
                    "length": len(response_text) if response_text else 0,
                    "error": error
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"⚠️ 保存日志失败: {e}")


def load_model_configs(config_path: str = "configs/model_configs.json") -> Dict[str, Dict]:
    """
    加载模型配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典，key为config_name，value为配置
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        configs = json.load(f)
    
    # 转换为字典格式
    return {cfg["config_name"]: cfg for cfg in configs}


def create_llm_client(config_name: str, config_path: str = "configs/model_configs.json") -> LLMClient:
    """
    创建LLM客户端
    
    Args:
        config_name: 配置名称
        config_path: 配置文件路径
    
    Returns:
        LLMClient实例
    """
    configs = load_model_configs(config_path)
    
    if config_name not in configs:
        available = list(configs.keys())
        raise ValueError(f"配置 '{config_name}' 不存在。可用配置: {available}")
    
    return LLMClient(configs[config_name])
