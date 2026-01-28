"""
场景B提示词版本实验控制器

功能：
1. 从 docs/prompts_b.md 中解析不同版本的提示词（b.v0 到 b.v6）
2. 依次用每个版本运行评估实验
3. 保存每个版本的实验结果到 evaluation_results/prompt_experiments/
"""

import re
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# 导入评估器
from src.evaluators.llm_client import LLMClient
from src.evaluators.evaluate_scenario_b import ScenarioBEvaluator


class PromptVersionParser:
    """提示词版本解析器（硬编码版本）"""
    
    def __init__(self):
        """初始化解析器，硬编码所有提示词版本"""
        self.versions = self._get_hardcoded_prompts()
        print(f"✅ 加载 {len(self.versions)} 个提示词版本: {list(self.versions.keys())}")
    
    def _get_hardcoded_prompts(self) -> Dict[str, Dict[str, str]]:
        """硬编码所有提示词版本"""
        
        # 所有版本共用的系统提示
        system_prompt = """你是理性经济主体，目标是在不确定他人行为的情况下最大化你的期望效用。

【重要】你必须只输出一个有效的JSON对象，不要包含任何额外的文本、解释或markdown标记。
JSON必须包含 "share" 和 "reason" 两个字段。
确保 "reason" 字段的字符串正确闭合（以引号结束）。"""
        
        return {
            "b.v0": {
                "system": system_prompt,
                "user_template": """# 数据市场决策

你是用户 {user_id}，正在参与一个数据市场。

## 你的信息

- **报价**：p[{user_id}] = {price:.4f}
- **隐私偏好**：v[{user_id}] = {v_i:.3f}（单位信息成本）

## 决策

如果分享数据：
- 获得补偿 p = {price:.4f}
- 产生隐私成本 = v × 边际泄露量

**请权衡：补偿 vs 隐私成本**

## 输出

请输出严格JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明决策理由（不超过150字）"
}}"""
            },
            
            "b.v1": {
                "system": system_prompt,
                "user_template": """# 数据市场决策

你是用户 {user_id}，正在参与一个数据市场。

## 你的私有信息

- **报价**：p[{user_id}] = {price:.4f}
- **隐私偏好**：v[{user_id}] = {v_i:.3f}（单位信息成本）

## 市场环境

- 用户总数：n = {n}
- 信息相关系数：ρ = {rho:.2f}
- 观测噪声：σ² = {sigma_noise_sq}
- 隐私偏好分布：所有用户的v ∈ [{v_min}, {v_max}]

你的位置：v = {v_i:.3f}（{v_description}）

## 决策框架

如果分享数据：
- 获得补偿 p = {price:.4f}
- 产生隐私成本 = v × 边际泄露量

**请权衡：补偿 vs 隐私成本**

## 输出

请输出严格JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明决策理由（不超过150字）"
}}"""
            },
            
            "b.v2": {
                "system": system_prompt,
                "user_template": """# 数据市场决策

你是用户 {user_id}，正在参与一个数据市场。

## 你的私有信息

- **报价**：p[{user_id}] = {price:.4f}
- **隐私偏好**：v[{user_id}] = {v_i:.3f}（单位信息成本）

## 市场环境

**用户总数**：n = {n}

**信息相关系数**：ρ = {rho:.2f}
- 你的类型与其他用户的类型相关
- ρ = 0：他人信息无法推断你
- ρ = 1：他人信息完美推断你
- 当前 ρ = {rho:.2f}，推断能力{rho_level}

**观测噪声**：σ² = {sigma_noise_sq}
- σ² 越大：数据噪声越大，泄露程度越低
- σ² 越小：数据越准确，泄露程度越高

**隐私偏好分布**：v ∈ [{v_min}, {v_max}]
- 你的位置：v = {v_i:.3f}（{v_description}）

## 决策框架

如果分享数据：
- 获得补偿 p = {price:.4f}
- 产生隐私成本 = v × 边际泄露量

**请权衡：补偿 vs 隐私成本**

## 输出

请输出严格JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明决策理由（不超过150字）"
}}"""
            },
            
            "b.v3": {
                "system": system_prompt,
                "user_template": """# 数据市场决策（推断外部性）

你是用户 {user_id}，正在参与一个数据市场。

## 你的私有信息

- **报价**：p[{user_id}] = {price:.4f}
- **隐私偏好**：v[{user_id}] = {v_i:.3f}（单位信息成本）

## 市场环境

**用户总数**：n = {n}

**信息相关系数**：ρ = {rho:.2f}
- 你的类型与其他用户的类型相关
- ρ = 0：他人信息无法推断你
- ρ = 1：他人信息完美推断你
- 当前 ρ = {rho:.2f}，推断能力{rho_level}

**观测噪声**：σ² = {sigma_noise_sq}
- σ² 越大：数据噪声越大，泄露程度越低
- σ² 越小：数据越准确，泄露程度越高

**隐私偏好分布**：v ∈ [{v_min}, {v_max}]
- 你的位置：v = {v_i:.3f}（{v_description}）

## 关键机制

**推断泄露**：即使你不分享，平台也能通过其他人的数据推断你的信息

**两种泄露**：
- **基础泄露**：其他人分享导致你的信息间接泄露
- **边际泄露**：你自己分享带来的额外泄露

**分享的真正成本 = 边际泄露**

## 决策框架

**如果分享**：
- 获得补偿：p = {price:.4f}
- 隐私成本：v × 边际泄露量
- 你的信息从部分泄露变为完全泄露

**如果不分享**：
- 无补偿
- 保护未间接泄露的部分信息
- 但仍有基础泄露

**请权衡：补偿 vs 边际隐私成本**

## 输出

请输出严格JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明决策理由（不超过150字）"
}}"""
            },
            
            "b.v4": {
                "system": system_prompt,
                "user_template": """# 数据市场决策（推断外部性）

你是用户 {user_id}，正在参与一个数据市场。

## 你的私有信息

- **报价**：p[{user_id}] = {price:.4f}
- **隐私偏好**：v[{user_id}] = {v_i:.3f}（单位信息成本）

## 市场环境

**用户总数**：n = {n}

**信息相关系数**：ρ = {rho:.2f}
- 你的信息与其他用户相关
- ρ = 0：他人信息无法推断你
- ρ = 1：他人信息完美推断你
- 当前 ρ = {rho:.2f}，推断能力{rho_level}

**观测噪声**：σ² = {sigma_noise_sq}
- σ² 越大：噪声越大，泄露越低
- σ² 越小：数据越准确，泄露越高

**隐私偏好分布**：v ∈ [{v_min}, {v_max}]
- 你的位置：v = {v_i:.3f}（{v_description}）

## 核心机制

### 推断外部性

**关键洞察**：泄露信息量不仅取决于你是否分享，还取决于其他人是否分享。任何人的分享都会增加所有人（包括不分享者）的信息泄露量。

**如果你分享**：
- 获得补偿 p = {price:.4f}
- 你的信息从间接部分泄露 → 完全泄露

**如果你不分享**：
- 无补偿
- 保护未间接泄露的部分
- 但仍有基础泄露（他人分享导致）

### 次模性

**分享的人越多 → 你的边际泄露越小**
- 基础泄露越高 → 边际泄露越低
- 其他人分享得多 → 你再分享的额外成本减少

### 补偿逻辑

**平台报价旨在覆盖边际隐私损失**
- 报价 p = {price:.4f} 反映你的边际价值
- 你需要判断：p 是否足以覆盖 v × 边际泄露

## 决策框架

**隐私成本** = v × 边际泄露量

**权衡**：补偿收益 p vs 隐私成本 v × 边际泄露

## 输出

请输出严格JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明决策理由（不超过150字）"
}}"""
            },
            
            "b.v5": {
                "system": system_prompt,
                "user_template": """# 数据市场静态博弈（推断外部性）

你是用户 {user_id}，正在参与一个**一次性的数据市场决策**。

## 你的私有信息

- **隐私偏好**：v[{user_id}] = {v_i:.3f}（单位信息成本）
- **平台报价**：p[{user_id}] = {price:.4f}

注意：每个用户的报价可能不同

## 公共知识（所有人都知道）

### 市场规模
- 用户总数：n = {n}

### 信息相关性
- 类型相关系数：ρ = {rho:.2f}
- 你的信息与其他用户的信息相关
- ρ = 0：他人信息无法推断你
- ρ = 1：他人信息完美推断你
- 当前 ρ = {rho:.2f}：推断能力{rho_level}

### 数据质量
- 观测噪声：σ² = {sigma_noise_sq}
- σ² 越大：噪声越大，泄露程度越低
- σ² 越小：数据越准确，泄露程度越高

### 隐私偏好分布
- 所有用户的 v ∈ [{v_min}, {v_max}]
- 你的 v = {v_i:.3f}，相对位置：{v_description}
- 你属于{v_description}隐私偏好群体

## 推断外部性机制

### 核心外部性
- **泄露信息量**：不仅取决于你是否分享，还取决于其他人是否分享。任何人的分享都会增加所有人（包括不分享者）的信息泄露量。

### 两种泄露类型

**基础泄露**（Base Leakage）：
- 来源：其他人分享导致你的信息间接泄露
- 影响：即使不分享也会存在
- 原理：通过相关性推断

**边际泄露**（Marginal Leakage）：
- 来源：你自己分享带来的额外泄露
- 计算：完全泄露 - 基础泄露
- 这才是分享的真正成本

### 次模性（Submodularity）

**关键性质**：分享的人越多，你的边际泄露越小

原因：
- 基础泄露越高 → 边际泄露越低
- 其他人分享多 → 你再分享的额外成本减少

### 补偿机制

**平台报价逻辑**：
- 报价 p[{user_id}] = {price:.4f} 旨在覆盖你的边际隐私损失
- 你的单位信息成本为 v[{user_id}] = {v_i:.3f}

## 决策分析

### 如果分享
- **收益**：获得补偿 p = {price:.4f}
- **成本**：隐私成本 = v × 边际泄露量
- **结果**：信息从部分泄露变为完全泄露

### 如果不分享
- **收益**：保护未间接泄露的部分信息
- **成本**：无法得到补偿
- **结果**：仍有基础泄露存在

### 你的任务

在**不知道其他人具体决策**的情况下，权衡：
- 补偿收益 p = {price:.4f}
- 隐私成本 v × 边际泄露量

## 输出

请输出严格JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明你的权衡与信念依据（不超过150字）"
}}"""
            },
            
            "b.v6": {
                "system": system_prompt,
                "user_template": """# 场景：数据市场静态博弈（推断外部性）

你是用户 {user_id}，正在参与一个**一次性的数据市场决策**。

## 基本信息

**你的私有信息**：
- 你的隐私偏好：v[{user_id}] = {v_i:.3f}
- 平台给你的报价：p[{user_id}] = {price:.4f}

**公共知识**（所有人都知道）：
- 用户总数：n = {n}
- 类型相关系数：ρ = {rho:.2f}
  （你的类型与其他用户的类型相关，相关系数为 {rho:.2f}）
- 观测噪声：σ² = {sigma_noise_sq}
- 隐私偏好分布：所有用户的 v 均匀分布在 [{v_min}, {v_max}]
  （你的 v = {v_i:.3f}，相对位置：{v_description}，属于{v_level}隐私偏好群体）

**你不知道的信息**：
- 其他用户的具体 v 值（你只知道分布）
- 其他用户会如何决策（因为是同时决策）

## 推断外部性机制

**关键概念**：即使你不分享数据，平台也能通过其他人的数据推断你的信息。

**泄露信息量 I_i(a)**：
- 给定分享集合 S，平台对你的推断精度提升量
- 通过贝叶斯更新计算：I_i(S) = σ_i² - Var(X_i | S)
- **核心外部性**：I_i 不仅取决于你是否分享，还取决于其他人是否分享

**你的效用函数**：
- 如果你**分享**：u_i = p_i - v_i × I_i(你分享, 其他人的决策)
- 如果你**不分享**：u_i = 0 - v_i × I_i(你不分享, 其他人的决策)

**关键洞察**：
- 不分享也会有**基础泄露**（因为其他人分享会泄露你的信息）
- 分享的真正成本是**边际泄露** = I_i(分享) - I_i(不分享)
- 补偿价格 p_i 旨在覆盖你的边际隐私损失

## 理性预期决策框架

因为你不知道其他人会如何选择，你需要：

**1. 基于分布推测其他人的行为**：
- v 值较低的用户更可能分享（隐私成本低）
- v 值较高的用户更不可能分享（隐私成本高）
- 你的 v = {v_i:.3f}，处于{v_level}水平

**2. 计算期望效用**：
- E[u_i | 分享] = E[p_i - v_i × I_i(1, a_{{-i}})]
- E[u_i | 不分享] = E[- v_i × I_i(0, a_{{-i}})]

**3. 理解次模性**：
- 分享的人越多，你的边际信息价值越低
- 基础泄露越高（别人分享多），你分享的边际泄露越小

**4. 做出最佳反应**：
- 如果 E[u_i | 分享] > E[u_i | 不分享]，则分享
- 否则不分享

## 你的任务

基于上述机制，在**不知道其他人具体决策**的情况下，通过**理性预期**判断是否分享数据。

**思考要点**：
1. 你的 v 值在分布中的位置如何？（v = {v_i:.3f}，属于{v_level}群体）
2. 预期会有多少比例的用户分享？
3. 在那个预期下，你分享的边际价值是多少？
4. 报价 p = {price:.4f} 能否覆盖你的边际隐私损失？
5. 相关性 ρ = {rho:.2f} 如何影响外部性？

## 输出格式

请输出严格JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明你的权衡与信念依据（不超过150字）"
}}"""
            }
        }
    
    def get_version(self, version_id: str) -> Dict[str, str]:
        """
        获取指定版本的提示词
        
        Args:
            version_id: 版本ID，如 "b.v0"
        
        Returns:
            {"system": str, "user_template": str}
        """
        if version_id not in self.versions:
            raise ValueError(f"版本 {version_id} 不存在。可用版本: {list(self.versions.keys())}")
        return self.versions[version_id]
    
    def list_versions(self) -> List[str]:
        """列出所有可用版本"""
        return sorted(self.versions.keys())


class CustomScenarioBEvaluator(ScenarioBEvaluator):
    """自定义场景B评估器，支持替换提示词"""
    
    def __init__(self, llm_client: LLMClient, ground_truth_path: str, 
                 custom_system_prompt: str = None, custom_user_prompt_template: str = None,
                 use_theory_platform: bool = True):
        """
        初始化自定义评估器
        
        Args:
            llm_client: LLM客户端
            ground_truth_path: ground truth文件路径
            custom_system_prompt: 自定义系统提示词（如果为None则使用默认）
            custom_user_prompt_template: 自定义用户决策提示词模板（如果为None则使用默认）
            use_theory_platform: 是否使用理论平台价格
        """
        super().__init__(llm_client, ground_truth_path, use_theory_platform)
        
        self.custom_system_prompt = custom_system_prompt
        self.custom_user_prompt_template = custom_user_prompt_template
    
    def build_system_prompt_user(self) -> str:
        """构建用户的系统提示（可被自定义覆盖）"""
        if self.custom_system_prompt:
            return self.custom_system_prompt
        else:
            return super().build_system_prompt_user()
    
    def build_user_decision_prompt(self, user_id: int, price: float) -> str:
        """
        构建用户决策提示词（可被自定义覆盖）
        
        Args:
            user_id: 用户ID
            price: 平台给出的报价
        
        Returns:
            提示文本
        """
        if self.custom_user_prompt_template:
            # 使用自定义模板，需要填充变量
            v_i = self.params.v[user_id]
            n = self.params.n
            rho = self.params.rho
            sigma_noise_sq = self.params.sigma_noise_sq
            v_min, v_max = 0.3, 1.2
            v_mean = (v_min + v_max) / 2
            
            # 判断用户v在分布中的相对位置
            if v_i < v_mean - 0.2:
                v_description = "偏低"
                v_level = "低"
            elif v_i < v_mean + 0.2:
                v_description = "中等"
                v_level = "中"
            else:
                v_description = "偏高"
                v_level = "高"
            
            # 判断rho的水平
            if rho < 0.3:
                rho_level = "较弱"
            elif rho >= 0.3 and rho < 0.7:
                rho_level = "中等"
            else:
                rho_level = "较强"
            
            # 填充模板变量
            prompt = self.custom_user_prompt_template.format(
                user_id=user_id,
                v_i=v_i,
                price=price,
                n=n,
                rho=rho,
                sigma_noise_sq=sigma_noise_sq,
                v_min=v_min,
                v_max=v_max,
                v_description=v_description,
                v_level=v_level,
                rho_level=rho_level
            )
            return prompt
        else:
            # 使用默认提示词
            return super().build_user_decision_prompt(user_id, price)


class PromptExperimentController:
    """提示词实验控制器"""
    
    def __init__(self, 
                 model_name: str = "gpt-5.2",
                 ground_truth_path: str = "data/ground_truth/scenario_b_result.json",
                 output_dir: str = "evaluation_results/prompt_experiments_b",
                 use_theory_platform: bool = True,
                 config_file: str = "configs/model_configs.json"):
        """
        初始化实验控制器
        
        Args:
            model_name: LLM模型名称（config_name）
            ground_truth_path: ground truth文件路径
            output_dir: 输出目录
            use_theory_platform: 是否使用理论平台价格
            config_file: 模型配置文件路径
        """
        self.model_name = model_name
        self.ground_truth_path = ground_truth_path
        self.output_dir = output_dir
        self.use_theory_platform = use_theory_platform
        
        # 加载模型配置
        with open(config_file, 'r', encoding='utf-8') as f:
            all_configs = json.load(f)
        
        # 查找匹配的配置
        self.model_config = None
        for config in all_configs:
            if config["config_name"] == model_name:
                self.model_config = config
                break
        
        if self.model_config is None:
            raise ValueError(f"未找到模型 {model_name} 的配置。可用配置: {[c['config_name'] for c in all_configs]}")
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化解析器
        self.parser = PromptVersionParser()
        
        print(f"📊 实验控制器初始化完成")
        print(f"   模型: {model_name} ({self.model_config['model_name']})")
        print(f"   Ground Truth: {ground_truth_path}")
        print(f"   输出目录: {output_dir}")
        print(f"   可用提示词版本: {self.parser.list_versions()}")
    
    def run_single_experiment(self, version_id: str, num_rounds: int = 5) -> Dict[str, Any]:
        """
        运行单个版本的实验
        
        Args:
            version_id: 版本ID，如 "b.v0"
            num_rounds: 运行轮数
        
        Returns:
            实验结果字典
        """
        print(f"\n{'='*60}")
        print(f"🚀 开始实验: {version_id}")
        print(f"{'='*60}")
        
        # 获取该版本的提示词
        prompts = self.parser.get_version(version_id)
        system_prompt = prompts["system"]
        user_prompt_template = prompts["user_template"]
        
        print(f"📝 System Prompt 长度: {len(system_prompt)} 字符")
        print(f"📝 User Prompt Template 长度: {len(user_prompt_template)} 字符")
        
        # 创建日志目录（按版本和时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_version_id = version_id.replace(".", "_")
        log_dir = os.path.join(self.output_dir, "llm_logs", f"{safe_version_id}_{timestamp}")
        
        # 初始化LLM客户端（使用配置文件中的配置，并覆盖temperature和max_tokens）
        llm_config = self.model_config.copy()
        llm_config["generate_args"] = llm_config.get("generate_args", {}).copy()
        llm_config["generate_args"]["temperature"] = 0.7
        llm_config["generate_args"]["max_tokens"] = 1500  # 增加到1500以避免截断（中文reason字段需要更多tokens）
        
        llm_client = LLMClient(config=llm_config, log_dir=log_dir)
        
        # 初始化自定义评估器
        evaluator = CustomScenarioBEvaluator(
            llm_client=llm_client,
            ground_truth_path=self.ground_truth_path,
            custom_system_prompt=system_prompt,
            custom_user_prompt_template=user_prompt_template,
            use_theory_platform=self.use_theory_platform
        )
        
        # 运行多轮评估
        print(f"\n⏳ 运行 {num_rounds} 轮评估...")
        all_rounds = []
        for round_idx in range(num_rounds):
            print(f"\n--- 第 {round_idx + 1}/{num_rounds} 轮 ---")
            round_result = evaluator.simulate_static_game(num_trials=1)
            all_rounds.append(round_result)
        
        # 汇总多轮结果
        results = self._aggregate_rounds(all_rounds)
        
        # 添加实验元信息
        results["experiment_meta"] = {
            "version_id": version_id,
            "model_name": self.model_name,
            "num_rounds": num_rounds,
            "timestamp": datetime.now().isoformat(),
            "use_theory_platform": self.use_theory_platform
        }
        
        # 保存所有轮次的原始数据
        results["rounds"] = all_rounds
        
        print(f"✅ 实验 {version_id} 完成")
        
        return results
    
    def _aggregate_rounds(self, all_rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        汇总多轮评估结果
        
        Args:
            all_rounds: 所有轮次的结果列表
        
        Returns:
            汇总后的结果
        """
        num_rounds = len(all_rounds)
        
        # 提取关键指标
        share_rates = [r["metrics"]["llm"]["share_rate"] for r in all_rounds]
        profits = [r["metrics"]["llm"]["profit"] for r in all_rounds]
        welfares = [r["metrics"]["llm"]["welfare"] for r in all_rounds]
        
        # 计算与GT的距离
        jaccard_sims = [r["equilibrium_quality"]["share_set_similarity"] for r in all_rounds]
        profit_maes = [r["metrics"]["deviations"]["profit_mae"] for r in all_rounds]
        welfare_maes = [r["metrics"]["deviations"]["welfare_mae"] for r in all_rounds]
        
        # 构造汇总结果
        return {
            "metrics": {
                "share_rate_mean": float(np.mean(share_rates)),
                "share_rate_std": float(np.std(share_rates)),
                "profit_mean": float(np.mean(profits)),
                "profit_std": float(np.std(profits)),
                "welfare_mean": float(np.mean(welfares)),
                "welfare_std": float(np.std(welfares)),
                "jaccard_similarity_mean": float(np.mean(jaccard_sims)),
                "jaccard_similarity_std": float(np.std(jaccard_sims)),
                "decision_distance_mean": float(1 - np.mean(jaccard_sims)),  # 1 - jaccard 作为距离
                "decision_distance_std": float(np.std([1-j for j in jaccard_sims])),
                "profit_mae_mean": float(np.mean(profit_maes)),
                "welfare_mae_mean": float(np.mean(welfare_maes)),
            },
            "ground_truth": all_rounds[0]["metrics"]["ground_truth"],  # GT在所有轮次中相同
        }
    
    def run_all_experiments(self, versions: List[str] = None, num_rounds: int = 1) -> Dict[str, Any]:
        """
        运行所有版本的实验
        
        Args:
            versions: 要运行的版本列表，如果为None则运行所有版本
            num_rounds: 每个版本的运行轮数
        
        Returns:
            所有实验结果的汇总
        """
        if versions is None:
            versions = self.parser.list_versions()
        
        print(f"\n{'='*60}")
        print(f"🔬 批量实验开始")
        print(f"{'='*60}")
        print(f"📋 计划运行版本: {versions}")
        print(f"🔄 每个版本运行轮数: {num_rounds}")
        print(f"📊 预计总实验数: {len(versions)} 个版本")
        
        all_results = {}
        
        for i, version_id in enumerate(versions, 1):
            print(f"\n[{i}/{len(versions)}] 正在运行: {version_id}")
            
            try:
                results = self.run_single_experiment(version_id, num_rounds)
                all_results[version_id] = results
                
                # 保存单个版本的结果
                self._save_single_result(version_id, results)
                
            except Exception as e:
                print(f"❌ 实验 {version_id} 失败: {str(e)}")
                all_results[version_id] = {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        # 保存汇总结果
        self._save_summary_results(all_results)
        
        print(f"\n{'='*60}")
        print(f"🎉 所有实验完成!")
        print(f"{'='*60}")
        print(f"📁 结果保存在: {self.output_dir}")
        
        return all_results
    
    def _save_single_result(self, version_id: str, results: Dict[str, Any]):
        """保存单个实验结果"""
        # 创建安全的文件名
        safe_version_id = version_id.replace(".", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_version_id}_{self.model_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # 处理numpy类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_converted = convert_numpy(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_converted, f, indent=2, ensure_ascii=False)
        
        print(f"💾 结果已保存: {filepath}")
    
    def _save_summary_results(self, all_results: Dict[str, Any]):
        """保存汇总结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_{self.model_name}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # 提取关键指标汇总
        summary = {
            "experiment_meta": {
                "model_name": self.model_name,
                "timestamp": timestamp,
                "total_versions": len(all_results)
            },
            "versions": {}
        }
        
        for version_id, results in all_results.items():
            if "error" in results:
                summary["versions"][version_id] = {"error": results["error"]}
            else:
                # 提取关键指标
                metrics = results.get("metrics", {})
                summary["versions"][version_id] = {
                    "share_rate_mean": metrics.get("share_rate_mean"),
                    "share_rate_std": metrics.get("share_rate_std"),
                    "decision_distance_mean": metrics.get("decision_distance_mean"),
                    "decision_distance_std": metrics.get("decision_distance_std"),
                    "num_rounds": results.get("experiment_meta", {}).get("num_rounds")
                }
        
        # 处理numpy类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        summary_converted = convert_numpy(summary)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary_converted, f, indent=2, ensure_ascii=False)
        
        print(f"💾 汇总结果已保存: {filepath}")


def main():
    """主函数：命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="场景B提示词版本实验控制器")
    parser.add_argument("--model", type=str, default="gpt-5", 
                        help="LLM模型名称 (默认: gpt-5, 可选: gpt-5.2, gpt-5.1, gpt-5.1-2025-11-13, gpt-5)")
    parser.add_argument("--all-models", action="store_true",
                        help="运行所有配置文件中的模型（忽略--model参数）")
    parser.add_argument("--versions", type=str, nargs="+", 
                        help="要运行的版本列表，如 b.v0 b.v1 (默认: 所有版本)")
    parser.add_argument("--rounds", type=int, default=1, 
                        help="每个版本的运行轮数 (默认: 1)")
    parser.add_argument("--gt-path", type=str, 
                        default="data/ground_truth/scenario_b_result.json",
                        help="Ground truth文件路径")
    parser.add_argument("--output-dir", type=str, 
                        default="evaluation_results/prompt_experiments_b",
                        help="输出目录")
    parser.add_argument("--config-file", type=str,
                        default="configs/model_configs.json",
                        help="模型配置文件路径 (默认: configs/model_configs.json)")
    parser.add_argument("--no-theory-platform", action="store_true",
                        help="不使用理论平台价格（使用LLM定价）")
    
    args = parser.parse_args()
    
    # 如果设置了 --all-models，读取配置文件获取所有模型
    if args.all_models:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            all_configs = json.load(f)
        model_names = [config["config_name"] for config in all_configs]
        
        print(f"\n{'='*80}")
        print(f"🚀 批量运行模式：将依次运行 {len(model_names)} 个模型")
        print(f"{'='*80}")
        print(f"模型列表: {', '.join(model_names)}")
        print(f"提示词版本: {'所有版本' if args.versions is None else ', '.join(args.versions)}")
        print(f"每个版本运行轮数: {args.rounds}")
        print(f"{'='*80}\n")
        
        # 依次运行每个模型
        for i, model_name in enumerate(model_names, 1):
            print(f"\n{'#'*80}")
            print(f"📊 [{i}/{len(model_names)}] 开始运行模型: {model_name}")
            print(f"{'#'*80}\n")
            
            try:
                # 初始化控制器
                controller = PromptExperimentController(
                    model_name=model_name,
                    ground_truth_path=args.gt_path,
                    output_dir=args.output_dir,
                    use_theory_platform=not args.no_theory_platform,
                    config_file=args.config_file
                )
                
                # 运行实验
                controller.run_all_experiments(
                    versions=args.versions,
                    num_rounds=args.rounds
                )
                
                print(f"\n✅ 模型 {model_name} 完成！\n")
                
            except Exception as e:
                print(f"\n❌ 模型 {model_name} 运行失败: {str(e)}\n")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*80}")
        print(f"🎉 所有 {len(model_names)} 个模型运行完成！")
        print(f"{'='*80}\n")
        
    else:
        # 单模型模式
        # 初始化控制器
        controller = PromptExperimentController(
            model_name=args.model,
            ground_truth_path=args.gt_path,
            output_dir=args.output_dir,
            use_theory_platform=not args.no_theory_platform,
            config_file=args.config_file
        )
        
        # 运行实验
        controller.run_all_experiments(
            versions=args.versions,
            num_rounds=args.rounds
        )


if __name__ == "__main__":
    main()
