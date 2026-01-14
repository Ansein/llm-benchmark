"""
快速测试脚本
用最少的迭代次数测试评估系统是否正常工作
"""

from src.evaluators import create_llm_client, ScenarioAEvaluator, ScenarioBEvaluator


def test_scenario_a():
    """测试场景A"""
    print("\n" + "="*60)
    print("🧪 测试场景A")
    print("="*60)
    
    try:
        # 创建LLM客户端
        llm_client = create_llm_client("gpt-4.1-mini")
        print(f"✅ LLM客户端创建成功: {llm_client.config_name}")
        
        # 创建评估器
        evaluator = ScenarioAEvaluator(llm_client)
        print("✅ 评估器创建成功")
        
        # 运行评估（使用最小参数）
        print("\n开始评估（num_trials=1, max_iterations=2）...")
        results = evaluator.simulate_llm_equilibrium(
            num_trials=1,  # 只测试1次
            max_iterations=2  # 最多2轮迭代
        )
        
        # 打印摘要
        evaluator.print_evaluation_summary(results)
        
        # 保存结果
        evaluator.save_results(results, "test_eval_scenario_a.json")
        
        print("\n✅ 场景A测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 场景A测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scenario_b():
    """测试场景B"""
    print("\n" + "="*60)
    print("🧪 测试场景B")
    print("="*60)
    
    try:
        # 创建LLM客户端
        llm_client = create_llm_client("gpt-4.1-mini")
        print(f"✅ LLM客户端创建成功: {llm_client.config_name}")
        
        # 创建评估器
        evaluator = ScenarioBEvaluator(llm_client)
        print("✅ 评估器创建成功")
        
        # 运行评估（使用最小参数）
        print("\n开始评估（num_trials=1, max_iterations=2）...")
        results = evaluator.simulate_llm_equilibrium(
            num_trials=1,  # 只测试1次
            max_iterations=2  # 最多2轮迭代
        )
        
        # 打印摘要
        evaluator.print_evaluation_summary(results)
        
        # 保存结果
        evaluator.save_results(results, "test_eval_scenario_b.json")
        
        print("\n✅ 场景B测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 场景B测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "#"*60)
    print("🚀 开始测试评估系统")
    print("#"*60)
    
    # 测试场景A
    result_a = test_scenario_a()
    
    # 测试场景B
    result_b = test_scenario_b()
    
    # 总结
    print("\n" + "#"*60)
    print("📊 测试总结")
    print("#"*60)
    print(f"场景A: {'✅ 通过' if result_a else '❌ 失败'}")
    print(f"场景B: {'✅ 通过' if result_b else '❌ 失败'}")
    
    if result_a and result_b:
        print("\n🎉 所有测试通过！系统可以正常使用。")
        print("\n下一步：运行完整评估")
        print("  python run_evaluation.py --single --scenarios A --models gpt-4.1-mini")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")


if __name__ == "__main__":
    main()
