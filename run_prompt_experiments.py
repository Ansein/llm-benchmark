"""
场景B提示词版本实验控制器

功能：
1. 从 docs/prompts_b.md 中解析不同版本的提示词（b.v0 到 b.v5）
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
你必须输出严格JSON格式，不要包含任何额外的文本。"""
        
        return {
            "b.v0": {
                "system": system_prompt,
                "user_template": """你是用户 {user_id}，正在参与一个数据市场。

**你的个人信息**：
- 平台给你的报价：p[{user_id}] = {price:.4f}
- 你的隐私偏好（单位信息的成本）：v[{user_id}] = {v_i:.3f}

**决策框架**：
- 如果你分享数据，你会得到补偿 p = {price:.4f}
- 分享会产生隐私成本 = v × 边际信息泄露量
- 你需要权衡：补偿收益 vs 隐私成本

请输出严格JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明你的决策理由（不超过150字）"
}}"""
            },
            
            "b.v1": {
                "system": system_prompt,
                "user_template": """你是用户 {user_id}，正在参与一个数据市场。

**你的私有信息**：
- 平台给你的报价：p[{user_id}] = {price:.4f}
- 你的隐私偏好（单位信息的成本）：v[{user_id}] = {v_i:.3f}

**公共知识**：
- 用户总数：n = {n}
- 用户间信息相关系数：ρ = {rho:.2f}
- 观测噪声：σ² = {sigma_noise_sq}
- 隐私偏好分布：所有用户的 v 范围在 [{v_min}, {v_max}]
（你的 v = {v_i:.3f}，相对位置：{v_description}）

**决策框架**：
- 如果你分享数据，你会得到补偿 p = {price:.4f}
- 分享会产生隐私成本 = v × 边际信息泄露量
- 你需要权衡：补偿收益 vs 隐私成本

请输出严格JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明你的决策理由（不超过150字）"
}}"""
            },
            
            "b.v2": {
                "system": system_prompt,
                "user_template": """你是用户 {user_id}，正在参与一个数据市场。

**你的私有信息**：
- 平台给你的报价：p[{user_id}] = {price:.4f}
- 你的隐私偏好（单位信息的成本）：v[{user_id}] = {v_i:.3f}

**公共知识**：
- 用户总数：n = {n}
- 用户间信息相关系数：ρ = {rho:.2f}
  你的类型与其他用户的类型相关，相关系数为 {rho:.2f}，代表其他用户的信息用于推断你的信息的能力。ρ为0时他人的信息完全无法推断你的信息，ρ为1时他人的信息可以完美推断你的信息（这种推断是相互的），ρ越高推断能力越强。
- 观测噪声：σ² = {sigma_noise_sq}
  观测噪声表示数据本身的不确定性。σ²越大，数据的噪声越大，平台从数据中提取有效信息的能力越弱，你的信息泄露程度越低；σ²越小，数据越准确，平台的推断越精确，信息泄露程度越高。
- 隐私偏好分布：所有用户的 v 均匀分布在 [{v_min}, {v_max}]
（你的 v = {v_i:.3f}，相对位置：{v_description}）

**决策框架**：
- 如果你分享数据，你会得到补偿 p = {price:.4f}
- 分享会产生隐私成本 = v × 边际信息泄露量
- 你需要权衡：补偿收益 vs 隐私成本

请输出严格JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明你的决策理由（不超过150字）"
}}"""
            },
            
            "b.v3": {
                "system": system_prompt,
                "user_template": """你是用户 {user_id}，正在参与一个数据市场。

**你的私有信息**：
- 平台给你的报价：p[{user_id}] = {price:.4f}
- 你的隐私偏好（单位信息的成本）：v[{user_id}] = {v_i:.3f}

**公共知识**：
- 用户总数：n = {n}
- 用户间信息相关系数：ρ = {rho:.2f}
  你的类型与其他用户的类型相关，相关系数为 {rho:.2f}，代表其他用户的信息用于推断你的信息的能力。ρ为0时他人的信息完全无法推断你的信息，ρ为1时他人的信息可以完美推断你的信息（这种推断是相互的），ρ越高推断能力越强。
- 观测噪声：σ² = {sigma_noise_sq}
  观测噪声表示数据本身的不确定性。σ²越大，数据的噪声越大，平台从数据中提取有效信息的能力越弱，你的信息泄露程度越低；σ²越小，数据越准确，平台的推断越精确，信息泄露程度越高。
- 隐私偏好分布：所有用户的 v 均匀分布在 [{v_min}, {v_max}]
（你的 v = {v_i:.3f}，相对位置：{v_description}）

**关键机制**：
- 即使你不分享数据，平台也可能通过其他用户的数据推断你的信息（推断外部性）
- 如果你分享，你的信息会从间接部分泄露变成完全泄露
- 如果你不分享，你可以保护未间接泄露的那部分信息
- 不分享也会有基础泄露（因为其他人分享会泄露你的信息），分享的真正成本是边际泄露带来的成本

**决策框架**：
- 如果你分享数据，你会得到补偿 p = {price:.4f}
- 分享会产生隐私成本 = v × 边际信息泄露量
- 你需要权衡：补偿收益 vs 隐私成本

请输出严格JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明你的决策理由（不超过150字）"
}}"""
            },
            
            "b.v4": {
                "system": system_prompt,
                "user_template": """你是用户 {user_id}，正在参与一个数据市场。

**你的私有信息**：
- 平台给你的报价：p[{user_id}] = {price:.4f}
- 你的隐私偏好（单位信息的成本）：v[{user_id}] = {v_i:.3f}

**公共知识**：
- 用户总数：n = {n}
- 用户间信息相关系数：ρ = {rho:.2f}
  你的信息与其他用户的信息相关，相关系数为 {rho:.2f}，代表其他用户的信息用于推断你的信息的能力。ρ为0时他人的信息完全无法推断你的信息，ρ为1时他人的信息可以完美推断你的信息（这种推断是相互的），ρ越高推断能力越强。
- 观测噪声：σ² = {sigma_noise_sq}
  观测噪声表示数据本身的不确定性。σ²越大，数据的噪声越大，平台从数据中提取有效信息的能力越弱，你的信息泄露程度越低；σ²越小，数据越准确，平台的推断越精确，信息泄露程度越高。
- 隐私偏好分布：所有用户的 v 均匀分布在 [{v_min}, {v_max}]
（你的 v = {v_i:.3f}，相对位置：{v_description}）

**核心机制**：
- **推断外部性**：泄露信息量不仅取决于你是否分享，还取决于其他人是否分享。任何人的分享都会增加所有人（包括不分享者）的信息泄露量。
- 如果你**分享**，你会得到来自平台的补偿 p = {price:.4f}，但你的信息会从间接部分泄露变成完全泄露
- 如果你**不分享**，你可以保护未间接泄露的那部分信息，但代价是无法得到补偿
- **次模性**：分享的人越多，你再分享带来的边际泄露越小（基础泄露越高，边际泄露越低）
- 不分享也会有**基础泄露**（因为其他人分享会泄露你的信息），分享的真正成本是**边际泄露**带来的成本
- 补偿价格旨在覆盖你的边际隐私损失

**决策框架**：
- 隐私成本 = v × 边际信息泄露量
- 你需要权衡：补偿收益 p vs 隐私成本 v × 边际泄露量

请输出严格JSON：
{{
  "share": 0或1（0=不分享，1=分享），
  "reason": "简要说明你的决策理由（不超过150字）"
}}"""
            },
            
            "b.v5": {
                "system": system_prompt,
                "user_template": """# 场景：数据市场静态博弈（推断外部性）

你是用户 {user_id}，正在参与一个**一次性的数据市场决策**。

## 基本信息

**你的私有信息**：
- 你的隐私偏好：v[{user_id}] = {v_i:.3f}
- 平台给你的个性化报价：p[{user_id}] = {price:.4f}
  （注意：每个用户的报价可能不同）

**公共知识**（所有人都知道）：
- 用户总数：n = {n}
- 类型相关系数：ρ = {rho:.2f}
- 你的信息型与其他用户的信息相关，相关系数为 {rho:.2f}，代表其他用户的信息用于推断你的信息的能力。ρ为0时他人的信息完全无法推断你的信息，ρ为1时他人的信息可以完美推断你的信息（这种推断是相互的），ρ越高推断能力越强。
- 观测噪声：σ² = {sigma_noise_sq}
  观测噪声表示数据本身的不确定性。σ²越大，数据的噪声越大，平台从数据中提取有效信息的能力越弱，你的信息泄露程度越低；σ²越小，数据越准确，平台的推断越精确，信息泄露程度越高。
- 隐私偏好分布：所有用户的 v 均匀分布在 [{v_min}, {v_max}]
（你的 v = {v_i:.3f}，相对位置：{v_description}，属于{v_description}隐私偏好群体）

- **核心外部性**：泄露信息量不仅取决于你是否分享，还取决于其他人是否分享，任何人的分享都会带来泄露信息量增加。
- 如果你**分享**，你会得到来自平台的补偿p_i，但是会导致你的信息会从间接部分泄露变成完全泄露，你的单位信息成本为v_i。
- 如果你**不分享**，你就可以相应保护你未间接泄露的信息，但代价是无法得到补偿。
- 基础泄露越高（别人分享多），你分享的边际泄露越小。
- 不分享也会有**基础泄露**（因为其他人分享会泄露你的信息），分享的真正成本是**边际泄露**带来的成本，补偿价格旨在覆盖你的边际隐私损失。

## 你的任务

基于上述机制，在**不知道其他人具体决策**的情况下，决定是否分享数据。

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
            elif v_i < v_mean + 0.2:
                v_description = "中等"
            else:
                v_description = "偏高"
            
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
                v_description=v_description
            )
            return prompt
        else:
            # 使用默认提示词
            return super().build_user_decision_prompt(user_id, price)


class PromptExperimentController:
    """提示词实验控制器"""
    
    def __init__(self, 
                 model_name: str = "gpt-4.1-mini",
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
        
        # 初始化LLM客户端（使用配置文件中的配置，并覆盖temperature和max_tokens）
        llm_config = self.model_config.copy()
        llm_config["generate_args"] = llm_config.get("generate_args", {}).copy()
        llm_config["generate_args"]["temperature"] = 0.7
        llm_config["generate_args"]["max_tokens"] = 500
        
        llm_client = LLMClient(config=llm_config)
        
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
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", 
                        help="LLM模型名称 (默认: gpt-4.1-mini)")
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
