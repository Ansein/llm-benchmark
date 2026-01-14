"""
LLM客户端封装
支持OpenAI兼容的API接口
"""

import json
from typing import Dict, List, Any, Optional
from openai import OpenAI


class LLMClient:
    """LLM客户端封装类"""
    
    def __init__(self, config: Dict[str, Any]):
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
        """
        self.config_name = config["config_name"]
        self.model_type = config["model_type"]
        self.model_name = config["model_name"]
        self.generate_args = config.get("generate_args", {})
        
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
        # 合并默认参数和自定义参数
        generate_params = {**self.generate_args, **kwargs}
        
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
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"❌ LLM调用失败: {e}")
            raise
    
    def generate_json(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成JSON格式的响应
        
        Args:
            messages: 消息列表
            **kwargs: 额外的生成参数
        
        Returns:
            解析后的JSON字典
        """
        # 请求JSON格式输出
        response_text = self.generate(
            messages=messages,
            response_format={"type": "json_object"},
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
        import re
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
