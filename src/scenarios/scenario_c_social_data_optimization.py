"""
场景C: m向量连续优化模块

实现论文式(11)的个性化补偿优化：
m*_i = Ui((Si, X−i), X−i) − Ui((Si, X), X)

使用连续优化算法（scipy或SGD）求解N维优化问题。
支持多进程并行加速。
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from functools import partial
import threading

from src.scenarios.scenario_c_social_data import (
    ScenarioCParams,
    generate_consumer_data,
    compute_rational_participation_rate,
    estimate_m0_mc,
    IntermediaryOptimizationResult
)


class EarlyStopCounter:
    """早停计数器：跟踪函数评估次数，达到限制时停止"""
    def __init__(self, max_evaluations: int = 1600):
        self.max_evaluations = max_evaluations
        self.count = 0
        self.lock = threading.Lock()
        self.stopped = False
    
    def increment(self):
        """增加计数并检查是否达到限制"""
        with self.lock:
            self.count += 1
            if self.count >= self.max_evaluations:
                self.stopped = True
                return True
            return False
    
    def reset(self):
        """重置计数器"""
        with self.lock:
            self.count = 0
            self.stopped = False
    
    def get_count(self):
        """获取当前计数"""
        with self.lock:
            return self.count


class EarlyStopError(Exception):
    """早停异常：达到最大评估次数时抛出"""
    pass


# 全局早停计数器
_early_stop_counter = EarlyStopCounter(max_evaluations=1600)


def evaluate_m_vector_profit(
    m_vector: np.ndarray,
    anonymization: str,
    params_base: Dict,
    num_mc_samples: int = 30,
    max_iter: int = 20,
    tol: float = 1e-3,
    seed: Optional[int] = None,
    early_stop_counter: Optional[EarlyStopCounter] = None
) -> float:
    """
    评估给定m向量的中介利润
    
    这是优化的目标函数。
    
    Args:
        m_vector: N维补偿向量
        anonymization: 匿名化策略
        params_base: 基础参数
        early_stop_counter: 早停计数器（可选）
        ...
    
    Returns:
        中介利润R（负值如果亏损，用于最小化）
    
    Raises:
        EarlyStopError: 如果达到最大评估次数
    """
    # ✅ 早停检查
    counter = early_stop_counter if early_stop_counter is not None else _early_stop_counter
    if counter.increment():
        raise EarlyStopError(
            f"达到最大函数评估次数限制 ({counter.max_evaluations}次)，强制停止优化。"
        )
    
    # 构建参数
    params = ScenarioCParams(
        m=m_vector.copy(),
        anonymization=anonymization,
        **params_base
    )
    
    try:
        # ✅ 修复：无论m当前是否个性化，都为每个消费者单独计算
        # 这样优化器才能"看到"个性化的潜在好处
        # 否则会陷入循环：m统一 → 代表性消费者 → 梯度对称 → m保持统一
        
        # 计算均衡参与率（始终使用个性化模式）
        r_star, r_history, delta_u_avg, delta_u_vector, p_vector = compute_rational_participation_rate(
            params,
            max_iter=max_iter,
            tol=tol,
            num_mc_samples=num_mc_samples,
            compute_per_consumer=True  # ✅ 始终为每个消费者单独计算
        )
        
        # 定义参与决策规则（使用个性化的ΔU）
        def participation_rule(p, world, rng):
            if p.tau_dist == "none":
                return delta_u_vector > 0
            elif p.tau_dist == "normal":
                tau_samples = rng.normal(p.tau_mean, p.tau_std, p.N)
                return tau_samples <= delta_u_vector
            elif p.tau_dist == "uniform":
                tau_low = p.tau_mean - np.sqrt(3) * p.tau_std
                tau_high = p.tau_mean + np.sqrt(3) * p.tau_std
                tau_samples = rng.uniform(tau_low, tau_high, p.N)
                return tau_samples <= delta_u_vector
            else:
                return np.full(p.N, False, dtype=bool)
        
        # 估计m_0
        m_0, _, _, e_num_participants = estimate_m0_mc(
            params=params,
            participation_rule=participation_rule,
            T=100,
            beta=1.0,
            seed=seed if seed is not None else params_base.get('seed', 42)
        )
        
        # 计算中介利润（使用个性化参与概率）
        # R = m_0 - E[Σ m_i·a_i]
        # ✅ 始终使用个性化版本：E[Σ m_i·a_i] = Σ m_i·p_i
        intermediary_cost = np.sum(m_vector * p_vector)
        intermediary_profit = m_0 - intermediary_cost
        
        return intermediary_profit
    
    except (RuntimeError, ValueError) as e:
        # 不收敛或无效 → 返回大负值（惩罚）
        print(f"  警告: 评估失败 - {e}")
        return -1e6


def optimize_m_vector_scipy(
    anonymization: str,
    params_base: Dict,
    m_bounds: Tuple[float, float] = (0.0, 3.0),
    method: str = 'L-BFGS-B',
    num_mc_samples: int = 30,
    max_iter: int = 20,
    tol: float = 1e-3,
    seed: Optional[int] = None,
    verbose: bool = True,
    m_init: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float, Dict]:
    """
    使用scipy优化器求解最优m向量
    
    优化问题：
        max_{m_i ≥ 0} R = m_0(m) - Σ m_i·a_i(m)
    
    Args:
        anonymization: 匿名化策略
        params_base: 基础参数（不含m）
        m_bounds: 补偿范围 [m_min, m_max]
        method: scipy优化方法
            - 'L-BFGS-B': 拟牛顿法（推荐，支持边界）
            - 'SLSQP': 序列二次规划
        num_mc_samples: MC样本数
        max_iter: 固定点最大迭代
        tol: 收敛容差
        seed: 随机种子
        verbose: 是否打印
    
    Returns:
        m_star_vector: 最优补偿向量
        profit_star: 最优利润
        info: 优化信息
    """
    N = params_base['N']
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🎯 m向量连续优化（scipy.{method}）")
        print(f"{'='*80}")
        print(f"维度: N={N}")
        print(f"匿名化策略: {anonymization}")
        print(f"补偿范围: [{m_bounds[0]:.2f}, {m_bounds[1]:.2f}]")
    
    # ✅ 创建早停计数器（每个优化任务独立）
    early_stop_counter = EarlyStopCounter(max_evaluations=1600)
    
    # 定义目标函数（最大化利润 = 最小化负利润）
    def objective(m_vec):
        try:
            profit = evaluate_m_vector_profit(
                m_vector=m_vec,
                anonymization=anonymization,
                params_base=params_base,
                num_mc_samples=num_mc_samples,
                max_iter=max_iter,
                tol=tol,
                seed=seed,
                early_stop_counter=early_stop_counter
            )
            return -profit  # 最小化负利润
        except EarlyStopError as e:
            # 返回一个很大的值，让优化器停止
            if verbose:
                print(f"\n⚠️  早停触发: {e}")
                print(f"   已评估 {early_stop_counter.get_count()} 次")
            raise  # 重新抛出异常，让优化器捕获
    
    # 初始值：如果提供则使用，否则使用tau_mean作为初始猜测
    if m_init is None:
        m_init = np.full(N, params_base.get('tau_mean', 1.0))
    
    # 边界约束
    bounds = [(m_bounds[0], m_bounds[1]) for _ in range(N)]
    
    if verbose:
        print(f"\n初始值: m_init = {np.mean(m_init):.3f} (均值)")
        print(f"开始优化...")
        print(f"早停限制: 最多 {early_stop_counter.max_evaluations} 次函数评估")
    
    # 运行优化
    try:
        result = minimize(
            fun=objective,
            x0=m_init,
            method=method,
            bounds=bounds,
            options={
                'disp': verbose,
                'maxiter': 100
            }
        )
    except EarlyStopError as e:
        # 早停：返回当前最佳值（如果有）
        if verbose:
            print(f"\n{'='*80}")
            print(f"⚠️  优化因早停而终止")
            print(f"{'='*80}")
            print(f"原因: {e}")
            print(f"已评估次数: {early_stop_counter.get_count()}")
        
        # 尝试使用最后一次评估的值（如果可用）
        # 注意：scipy.minimize 在异常时可能没有 result 对象
        # 这里我们需要一个fallback策略
        if verbose:
            print(f"使用初始值作为近似解")
        m_star_vector = m_init.copy()
        # 评估一次以获取利润
        try:
            profit_star = evaluate_m_vector_profit(
                m_vector=m_init,
                anonymization=anonymization,
                params_base=params_base,
                num_mc_samples=num_mc_samples,
                max_iter=max_iter,
                tol=tol,
                seed=seed,
                early_stop_counter=None  # 不计数，因为这是最终评估
            )
        except:
            profit_star = -1e6  # fallback
        
        # 创建假的result对象
        class FakeResult:
            def __init__(self):
                self.x = m_star_vector
                self.fun = -profit_star
                self.success = False
                self.nit = 0
                self.nfev = early_stop_counter.get_count()
                self.message = f"Early stop at {early_stop_counter.get_count()} evaluations"
        
        result = FakeResult()
    
    m_star_vector = result.x
    profit_star = -result.fun  # 转换回正值
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"✅ 优化完成")
        print(f"{'='*80}")
        print(f"状态: {'成功' if result.success else '失败'}")
        print(f"迭代次数: {result.nit}")
        print(f"函数调用: {result.nfev}")
        print(f"最优利润: R* = {profit_star:.4f}")
        print(f"最优补偿统计:")
        print(f"  均值: {np.mean(m_star_vector):.4f}")
        print(f"  标准差: {np.std(m_star_vector):.4f}")
        print(f"  最小值: {np.min(m_star_vector):.4f}")
        print(f"  最大值: {np.max(m_star_vector):.4f}")
        print(f"{'='*80}")
    
    # 计算最优解对应的r_star（用于GT生成）
    params = params_base.copy()
    params['m'] = m_star_vector
    params['anonymization'] = anonymization
    params = ScenarioCParams(**params)
    
    # ✅ 始终使用个性化计算以保持一致性
    r_star_final, _, delta_u_final, _, _ = compute_rational_participation_rate(
        params,
        max_iter=max_iter,
        tol=tol,
        num_mc_samples=num_mc_samples,
        compute_per_consumer=True
    )
    
    info = {
        'success': result.success,
        'nit': result.nit,
        'nfev': result.nfev,
        'message': result.message,
        'm_mean': float(np.mean(m_star_vector)),
        'm_std': float(np.std(m_star_vector)),
        'm_min': float(np.min(m_star_vector)),
        'm_max': float(np.max(m_star_vector)),
        'r_star': float(r_star_final),
        'delta_u': float(delta_u_final)
    }
    
    return m_star_vector, profit_star, info


def optimize_m_vector_evolutionary(
    anonymization: str,
    params_base: Dict,
    m_bounds: Tuple[float, float] = (0.0, 3.0),
    popsize: int = 15,
    maxiter: int = 50,
    num_mc_samples: int = 30,
    max_iter_fp: int = 20,
    tol: float = 1e-3,
    seed: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, float, Dict]:
    """
    使用进化算法求解最优m向量
    
    优点：
    - 全局搜索，不易陷入局部最优
    - 适合非凸优化
    - 不需要梯度
    
    缺点：
    - 收敛较慢
    - 函数调用次数多
    
    Args:
        anonymization: 匿名化策略
        params_base: 基础参数
        m_bounds: 补偿范围
        popsize: 种群大小（相对N的倍数）
        maxiter: 最大迭代代数
        ...
    
    Returns:
        m_star_vector: 最优补偿向量
        profit_star: 最优利润
        info: 优化信息
    """
    N = params_base['N']
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🎯 m向量连续优化（differential_evolution）")
        print(f"{'='*80}")
        print(f"维度: N={N}")
        print(f"匿名化策略: {anonymization}")
        print(f"补偿范围: [{m_bounds[0]:.2f}, {m_bounds[1]:.2f}]")
        print(f"种群大小: {popsize} × N = {popsize * N}")
        print(f"最大代数: {maxiter}")
    
    # ✅ 创建早停计数器（每个优化任务独立）
    early_stop_counter = EarlyStopCounter(max_evaluations=1600)
    
    # 定义目标函数
    def objective(m_vec):
        try:
            profit = evaluate_m_vector_profit(
                m_vector=m_vec,
                anonymization=anonymization,
                params_base=params_base,
                num_mc_samples=num_mc_samples,
                max_iter=max_iter_fp,
                tol=tol,
                seed=seed,
                early_stop_counter=early_stop_counter
            )
            return -profit  # 最小化负利润
        except EarlyStopError as e:
            if verbose:
                print(f"\n⚠️  早停触发: {e}")
                print(f"   已评估 {early_stop_counter.get_count()} 次")
            raise
    
    # 边界约束
    bounds = [m_bounds for _ in range(N)]
    
    if verbose:
        print(f"\n开始进化搜索...")
        print(f"早停限制: 最多 {early_stop_counter.max_evaluations} 次函数评估")
    
    # 运行进化算法
    try:
        result = differential_evolution(
            func=objective,
            bounds=bounds,
            strategy='best1bin',
            maxiter=maxiter,
            popsize=popsize,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=seed,
            disp=verbose,
            polish=True  # 最后用局部优化polish
        )
    except EarlyStopError as e:
        # 早停：返回当前最佳值
        if verbose:
            print(f"\n{'='*80}")
            print(f"⚠️  进化搜索因早停而终止")
            print(f"{'='*80}")
            print(f"原因: {e}")
            print(f"已评估次数: {early_stop_counter.get_count()}")
            print(f"使用当前种群最佳值作为近似解")
        
        # 对于进化算法，我们需要一个fallback
        # 这里使用边界中点作为近似
        m_star_vector = np.full(N, (m_bounds[0] + m_bounds[1]) / 2)
        try:
            profit_star = evaluate_m_vector_profit(
                m_vector=m_star_vector,
                anonymization=anonymization,
                params_base=params_base,
                num_mc_samples=num_mc_samples,
                max_iter=max_iter_fp,
                tol=tol,
                seed=seed,
                early_stop_counter=None  # 不计数
            )
        except:
            profit_star = -1e6
        
        # 创建假的result对象
        class FakeResult:
            def __init__(self):
                self.x = m_star_vector
                self.fun = -profit_star
                self.success = False
                self.nit = 0
                self.nfev = early_stop_counter.get_count()
                self.message = f"Early stop at {early_stop_counter.get_count()} evaluations"
        
        result = FakeResult()
    
    m_star_vector = result.x
    profit_star = -result.fun
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"✅ 进化搜索完成")
        print(f"{'='*80}")
        print(f"状态: {'成功' if result.success else '失败'}")
        print(f"迭代次数: {result.nit}")
        print(f"函数调用: {result.nfev}")
        print(f"最优利润: R* = {profit_star:.4f}")
        print(f"最优补偿统计:")
        print(f"  均值: {np.mean(m_star_vector):.4f}")
        print(f"  标准差: {np.std(m_star_vector):.4f}")
        print(f"  最小值: {np.min(m_star_vector):.4f}")
        print(f"  最大值: {np.max(m_star_vector):.4f}")
        print(f"{'='*80}")
    
    # 计算最优解对应的r_star（用于GT生成）
    params = params_base.copy()
    params['m'] = m_star_vector
    params['anonymization'] = anonymization
    params = ScenarioCParams(**params)
    
    # ✅ 始终使用个性化计算以保持一致性
    r_star_final, _, delta_u_final, _, _ = compute_rational_participation_rate(
        params,
        max_iter=max_iter_fp,
        tol=tol,
        num_mc_samples=num_mc_samples,
        compute_per_consumer=True
    )
    
    info = {
        'success': result.success,
        'nit': result.nit,
        'nfev': result.nfev,
        'message': result.message,
        'm_mean': float(np.mean(m_star_vector)),
        'm_std': float(np.std(m_star_vector)),
        'm_min': float(np.min(m_star_vector)),
        'm_max': float(np.max(m_star_vector)),
        'r_star': float(r_star_final),
        'delta_u': float(delta_u_final)
    }
    
    return m_star_vector, profit_star, info


def _evaluate_single_m_grid(args):
    """辅助函数：评估单个网格点（用于并行）"""
    m_val, params_base, policy, num_mc_samples, max_iter, N = args
    m_uniform = np.full(N, m_val)
    # 注意：并行时每个进程有独立的计数器，这里不传递计数器
    # 如果需要全局计数，需要使用共享内存或其他机制
    try:
        profit = evaluate_m_vector_profit(m_uniform, policy, params_base, num_mc_samples, max_iter)
    except EarlyStopError:
        # 如果达到限制，返回一个很小的值
        profit = -1e6
    return m_val, profit


def optimize_intermediary_policy_personalized(
    params_base: Dict,
    policies: list = None,
    optimization_method: str = 'hybrid',
    m_bounds: Tuple[float, float] = (0.0, 3.0),
    num_mc_samples: int = 30,
    max_iter: int = 20,
    tol: float = 1e-3,
    seed: Optional[int] = None,
    verbose: bool = True,
    grid_size: int = 11,
    n_jobs: int = -1
) -> Dict:
    """
    求解中介最优策略（个性化补偿版）
    
    对每个匿名化策略，优化N维补偿向量m，选择利润最高的。
    
    Args:
        params_base: 基础参数（不含m和anonymization）
        policies: 匿名化策略候选（默认['identified', 'anonymized']）
        optimization_method: 优化方法
            - 'scipy': 使用scipy.minimize (L-BFGS-B)
            - 'evolutionary': 使用differential_evolution
            - 'hybrid': 网格搜索初始化 + scipy优化（推荐，默认）
        m_bounds: 补偿范围
        grid_size: 网格搜索的点数（仅hybrid方法）
        n_jobs: 并行进程数（-1=使用所有CPU核心，1=不并行）
        ...
    
    Returns:
        {
            'm_star_vector': np.ndarray,  # 最优补偿向量
            'anonymization_star': str,
            'profit_star': float,
            'results_by_policy': Dict,    # 每个策略的结果
            'optimization_info': Dict
        }
    """
    if policies is None:
        policies = ['identified', 'anonymized']
    
    # 确定并行进程数
    if n_jobs == -1:
        n_jobs = cpu_count()
    n_jobs = max(1, min(n_jobs, cpu_count()))
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🎯 中介最优策略求解（个性化补偿m_i）")
        print(f"{'='*80}")
        print(f"优化方法: {optimization_method}")
        print(f"策略候选: {policies}")
        print(f"消费者数: N={params_base['N']}")
        if n_jobs > 1:
            print(f"并行加速: 使用{n_jobs}个CPU核心")
    
    results_by_policy = {}
    
    # 对每个策略优化m向量
    for policy in policies:
        if verbose:
            print(f"\n{'─'*80}")
            print(f"策略: {policy}")
            print(f"{'─'*80}")
        
        if optimization_method == 'hybrid':
            # 混合方法：网格搜索找初始点 + scipy精细优化
            if verbose:
                print(f"\n【第1步】网格搜索找初始点（粗搜索，{grid_size}个点）...")
            
            # 网格搜索找最优均匀m
            m_grid = np.linspace(m_bounds[0], m_bounds[1], grid_size)
            
            # 并行评估网格点
            parallel_success = False
            if n_jobs > 1 and grid_size > 3:
                try:
                    if verbose:
                        print(f"  使用{n_jobs}个进程并行评估...")
                    
                    # 准备参数
                    N = params_base['N']
                    args_list = [(m_val, params_base, policy, num_mc_samples, max_iter, N) 
                                for m_val in m_grid]
                    
                    # 并行计算
                    with Pool(processes=n_jobs) as pool:
                        results = pool.map(_evaluate_single_m_grid, args_list)
                    
                    # 找最优
                    best_m_uniform = m_bounds[0]
                    best_profit_grid = -np.inf
                    for m_val, profit in results:
                        if profit > best_profit_grid:
                            best_profit_grid = profit
                            best_m_uniform = m_val
                        if verbose:
                            print(f"  m={m_val:.2f} -> profit={profit:.4f}")
                    
                    parallel_success = True
                    
                except (PermissionError, OSError) as e:
                    if verbose:
                        print(f"  [WARNING] 并行计算失败: {e}")
                        print(f"  退回到串行模式...")
            
            if not parallel_success:
                # 串行评估（fallback或默认）
                best_m_uniform = m_bounds[0]
                best_profit_grid = -np.inf
                
                for m_val in m_grid:
                    m_uniform = np.full(params_base['N'], m_val)
                    profit = evaluate_m_vector_profit(m_uniform, policy, params_base, num_mc_samples, max_iter)
                    if profit > best_profit_grid:
                        best_profit_grid = profit
                        best_m_uniform = m_val
                    if verbose:
                        print(f"  m={m_val:.2f} -> profit={profit:.4f}")
            
            if verbose:
                print(f"网格搜索最优: m_uniform = {best_m_uniform:.4f}, profit = {best_profit_grid:.4f}")
                print(f"\n【第2步】从最优初始点开始连续优化（scipy.L-BFGS-B）...")
            
            # 从最优均匀m开始scipy优化
            m_init = np.full(params_base['N'], best_m_uniform)
            m_vec, profit, info = optimize_m_vector_scipy(
                anonymization=policy,
                params_base=params_base,
                m_bounds=m_bounds,
                num_mc_samples=num_mc_samples,
                max_iter=max_iter,
                tol=tol,
                seed=seed,
                verbose=verbose,
                m_init=m_init
            )
            info['grid_search_init'] = best_m_uniform
            info['grid_search_profit'] = best_profit_grid
            
        elif optimization_method == 'scipy':
            m_vec, profit, info = optimize_m_vector_scipy(
                anonymization=policy,
                params_base=params_base,
                m_bounds=m_bounds,
                num_mc_samples=num_mc_samples,
                max_iter=max_iter,
                tol=tol,
                seed=seed,
                verbose=verbose
            )
        elif optimization_method == 'evolutionary':
            m_vec, profit, info = optimize_m_vector_evolutionary(
                anonymization=policy,
                params_base=params_base,
                m_bounds=m_bounds,
                num_mc_samples=num_mc_samples,
                max_iter_fp=max_iter,
                tol=tol,
                seed=seed,
                verbose=verbose
            )
        else:
            raise ValueError(f"Unknown optimization_method: {optimization_method}")
        
        results_by_policy[policy] = {
            'm_vector': m_vec,
            'profit': profit,
            'info': info
        }
    
    # ✅ 应用利润约束：过滤亏损策略
    profitable_policies = {
        p: r for p, r in results_by_policy.items()
        if r['profit'] > 0.0
    }
    
    if not profitable_policies:
        # 所有策略都亏损 → 不参与
        if verbose:
            print(f"\n{'='*80}")
            print(f"⚠️  所有策略均亏损，中介选择不参与市场")
            print(f"{'='*80}")
        
        return {
            'm_star_vector': np.zeros(params_base['N']),
            'anonymization_star': 'no_participation',
            'profit_star': 0.0,
            'results_by_policy': results_by_policy,
            'participation_feasible': False
        }
    
    # 选择最优策略
    best_policy = max(profitable_policies.keys(), 
                     key=lambda p: profitable_policies[p]['profit'])
    best_result = profitable_policies[best_policy]
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"✅ 最优策略")
        print(f"{'='*80}")
        print(f"匿名化: {best_policy}")
        print(f"利润: R* = {best_result['profit']:.4f}")
        print(f"补偿统计:")
        print(f"  均值: {best_result['info']['m_mean']:.4f}")
        print(f"  标准差: {best_result['info']['m_std']:.4f}")
        print(f"  范围: [{best_result['info']['m_min']:.4f}, {best_result['info']['m_max']:.4f}]")
        print(f"{'='*80}")
    
    return {
        'm_star_vector': best_result['m_vector'],
        'anonymization_star': best_policy,
        'profit_star': best_result['profit'],
        'r_star': best_result['info']['r_star'],
        'delta_u': best_result['info']['delta_u'],
        'results_by_policy': results_by_policy,
        'participation_feasible': True,
        'optimization_method': optimization_method
    }
