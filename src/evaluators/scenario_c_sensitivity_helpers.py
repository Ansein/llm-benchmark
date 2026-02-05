"""
场景C敏感度分析辅助函数
为敏感性实验提供简化的接口
"""
from typing import Dict, Any
from pathlib import Path
import json

from src.evaluators.llm_client import create_llm_client, load_model_configs
from src.evaluators.evaluate_scenario_c import ScenarioCEvaluator


def run_config_B(
    model_name: str,
    gt_data: Dict[str, Any],
    num_rounds: int = 20,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    运行配置B：理性中介 × LLM消费者
    
    注意：配置B不需要迭代，只运行一次
    """
    # 创建评估器
    evaluator = ScenarioCEvaluator(
        gt_A=gt_data,  # common_preferences的GT
        gt_B=None  # 不需要common_experience
    )
    
    # 创建LLM客户端和消费者代理
    client = create_llm_client(model_name)
    model_configs = load_model_configs()
    model_config = model_configs[model_name]
    
    # 创建LLM消费者代理（简化版，直接内联）
    def create_llm_consumer():
        from src.evaluators.evaluate_scenario_c import call_llm_with_retry
        model_name_full = model_config['model_name']
        generate_args = model_config.get('generate_args', {})
        
        def _call_llm_consumer(consumer_params, m, anonymization):
            # 简化版提示词
            prompt = f"""你是消费者，需要决定是否参与数据分享计划。

补偿: m = {m:.2f}
隐私机制: {anonymization}
你的偏好强度: θ_i = {consumer_params['theta_i']:.2f}
隐私敏感度: τ_i = {consumer_params['tau_i']:.2f}

请回答 "YES" 或 "NO"（一个词即可）。"""
            
            messages = [{"role": "user", "content": prompt}]
            
            try:
                response = call_llm_with_retry(client, model_name_full, messages, generate_args, max_attempts=3)
                reply = response.choices[0].message.content.strip()
                decision = 1 if "YES" in reply.upper() else 0
                return decision, "", reply
            except Exception as e:
                print(f"LLM调用失败: {e}")
                return 0, "", ""
        
        return _call_llm_consumer
    
    llm_consumer = create_llm_consumer()
    
    # 运行评估
    result = evaluator.evaluate_config_B(
        llm_consumer_agent=llm_consumer,
        verbose=False
    )
    
    # 提取关键指标
    gt_profit = gt_data['optimal_strategy']['intermediary_profit_star']
    final_profit = result.get('intermediary_profit', 0)
    
    return {
        'final_profit': final_profit,
        'gt_profit': gt_profit,
        'profit_deviation': abs(final_profit - gt_profit),
        'converged': True,  # 配置B不迭代
        'rounds': 1
    }


def run_config_C(
    model_name: str,
    gt_data: Dict[str, Any],
    num_rounds: int = 20,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    运行配置C：LLM中介 × 理性消费者
    """
    # 创建评估器
    evaluator = ScenarioCEvaluator(
        gt_A=gt_data,
        gt_B=None
    )
    
    # 创建LLM客户端和中介代理
    client = create_llm_client(model_name)
    model_configs = load_model_configs()
    model_config = model_configs[model_name]
    
    # 创建LLM中介代理（简化版）
    def create_llm_intermediary():
        from src.evaluators.evaluate_scenario_c import call_llm_with_retry
        model_name_full = model_config['model_name']
        generate_args = model_config.get('generate_args', {})
        
        def llm_intermediary(market_params, feedback=None, history=None):
            prompt = f"""你是数据中介，需要选择补偿金额m和隐私机制。

参数:
- 消费者数: N = {market_params['N']}
- 先验均值: μ_θ = {market_params['mu_theta']:.2f}
- 隐私敏感度均值: τ_mean = {market_params['tau_mean']:.2f}
"""
            if feedback:
                prompt += f"\n上一轮结果:\n- 参与率: {feedback.get('participation_rate', 0):.2%}\n- 利润: {feedback.get('intermediary_profit', 0):.4f}"
            
            prompt += "\n\n请输出两行:\n第一行: m的值（数字）\n第二行: identified 或 anonymized"
            
            messages = [{"role": "user", "content": prompt}]
            
            try:
                response = call_llm_with_retry(client, model_name_full, messages, generate_args, max_attempts=3)
                reply = response.choices[0].message.content.strip()
                lines = reply.split('\n')
                
                m = 1.0
                anonymization = 'identified'
                
                for line in lines:
                    line = line.strip()
                    if line and line[0].isdigit():
                        try:
                            m = float(line)
                        except:
                            pass
                    elif 'anonymized' in line.lower():
                        anonymization = 'anonymized'
                    elif 'identified' in line.lower():
                        anonymization = 'identified'
                
                return m, anonymization, reply
            except Exception as e:
                print(f"LLM调用失败: {e}")
                return 1.0, 'identified', ""
        
        return llm_intermediary
    
    llm_intermediary = create_llm_intermediary()
    
    # 运行评估
    result = evaluator.evaluate_config_C_iterative(
        llm_intermediary_agent=llm_intermediary,
        rounds=num_rounds,
        verbose=False
    )
    
    # 提取关键指标
    gt_profit = gt_data['optimal_strategy']['intermediary_profit_star']
    gt_anon = gt_data['optimal_strategy']['anonymization_star']
    final_profit = result.get('final_intermediary_profit', 0)
    final_anon = result.get('final_anonymization', 'identified')
    
    return {
        'final_profit': final_profit,
        'gt_profit': gt_profit,
        'profit_deviation': abs(final_profit - gt_profit),
        'converged': result.get('converged', False),
        'rounds': num_rounds,
        'anonymization_accuracy': int(final_anon == gt_anon)
    }


def run_config_D(
    model_name: str,
    gt_data: Dict[str, Any],
    num_rounds: int = 20,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    运行配置D：LLM中介 × LLM消费者
    """
    # 创建评估器
    evaluator = ScenarioCEvaluator(
        gt_A=gt_data,
        gt_B=None
    )
    
    # 创建LLM客户端
    client = create_llm_client(model_name)
    model_configs = load_model_configs()
    model_config = model_configs[model_name]
    
    # 创建LLM消费者和中介代理（复用config_C的简化逻辑）
    from src.evaluators.evaluate_scenario_c import call_llm_with_retry
    model_name_full = model_config['model_name']
    generate_args = model_config.get('generate_args', {})
    
    def create_llm_consumer():
        def _call_llm_consumer(consumer_params, m, anonymization):
            prompt = f"""你是消费者，需要决定是否参与数据分享计划。

补偿: m = {m:.2f}
隐私机制: {anonymization}
你的偏好强度: θ_i = {consumer_params['theta_i']:.2f}
隐私敏感度: τ_i = {consumer_params['tau_i']:.2f}

请回答 "YES" 或 "NO"。"""
            
            messages = [{"role": "user", "content": prompt}]
            try:
                response = call_llm_with_retry(client, model_name_full, messages, generate_args, max_attempts=3)
                reply = response.choices[0].message.content.strip()
                decision = 1 if "YES" in reply.upper() else 0
                return decision, "", reply
            except:
                return 0, "", ""
        
        return _call_llm_consumer
    
    def create_llm_intermediary():
        def llm_intermediary(market_params, feedback=None, history=None):
            prompt = f"""你是数据中介，需要选择补偿金额m和隐私机制。

参数: N = {market_params['N']}, μ_θ = {market_params['mu_theta']:.2f}, τ_mean = {market_params['tau_mean']:.2f}
"""
            if feedback:
                prompt += f"\n上一轮: 参与率={feedback.get('participation_rate', 0):.2%}, 利润={feedback.get('intermediary_profit', 0):.4f}"
            
            prompt += "\n\n输出:\n第一行: m的值\n第二行: identified或anonymized"
            
            messages = [{"role": "user", "content": prompt}]
            try:
                response = call_llm_with_retry(client, model_name_full, messages, generate_args, max_attempts=3)
                reply = response.choices[0].message.content.strip()
                lines = reply.split('\n')
                
                m = 1.0
                anonymization = 'identified'
                for line in lines:
                    line = line.strip()
                    if line and line[0].isdigit():
                        try:
                            m = float(line)
                        except:
                            pass
                    elif 'anonymized' in line.lower():
                        anonymization = 'anonymized'
                
                return m, anonymization, reply
            except:
                return 1.0, 'identified', ""
        
        return llm_intermediary
    
    llm_consumer = create_llm_consumer()
    llm_intermediary = create_llm_intermediary()
    
    # 运行评估
    result = evaluator.evaluate_config_D_iterative(
        llm_intermediary_agent=llm_intermediary,
        llm_consumer_agent=llm_consumer,
        rounds=num_rounds,
        verbose=False
    )
    
    # 提取关键指标
    gt_profit = gt_data['optimal_strategy']['intermediary_profit_star']
    gt_anon = gt_data['optimal_strategy']['anonymization_star']
    final_profit = result.get('final_intermediary_profit', 0)
    final_anon = result.get('final_anonymization', 'identified')
    
    return {
        'final_profit': final_profit,
        'gt_profit': gt_profit,
        'profit_deviation': abs(final_profit - gt_profit),
        'converged': result.get('converged', False),
        'rounds': num_rounds,
        'anonymization_accuracy': int(final_anon == gt_anon)
    }


def compute_all_metrics(
    results: Dict[str, Dict[str, Any]],
    gt_data: Dict[str, Any]
) -> Dict[str, Any]:
    """计算综合评估指标"""
    gt_profit = gt_data['optimal_strategy']['intermediary_profit_star']
    
    metrics = {
        'config_B': {
            'profit_deviation': results['config_B']['profit_deviation'],
            'profit_deviation_rate': results['config_B']['profit_deviation'] / max(gt_profit, 1e-6)
        },
        'config_C': {
            'profit_deviation': results['config_C']['profit_deviation'],
            'profit_deviation_rate': results['config_C']['profit_deviation'] / max(gt_profit, 1e-6),
            'anonymization_accuracy': results['config_C'].get('anonymization_accuracy', 0)
        },
        'config_D': {
            'profit_deviation': results['config_D']['profit_deviation'],
            'profit_deviation_rate': results['config_D']['profit_deviation'] / max(gt_profit, 1e-6),
            'anonymization_accuracy': results['config_D'].get('anonymization_accuracy', 0)
        },
        'overall': {
            'avg_profit_deviation': sum([r['profit_deviation'] for r in results.values()]) / len(results)
        }
    }
    
    return metrics
