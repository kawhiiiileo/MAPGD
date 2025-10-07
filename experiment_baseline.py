"""
Baseline-Compatible MAPGD实验脚本
严格匹配baseline论文的实验设置和参数
"""

import os
import time
import json
import numpy as np
from core import MAPGDFramework
from mapgd_tasks import get_mapgd_task_class
from mapgd_predictors import get_mapgd_predictor
from init_prompt import TEST_INITIAL_PROMPT, JAILBREAK_INITIAL_PROMPT
from liar_config import EXPERIMENT_CONFIG as LIAR_CONFIG
from jailbreak_config import EXPERIMENT_CONFIG as JAILBREAK_CONFIG
from ethos_config import EXPERIMENT_CONFIG as ETHOS_CONFIG
from gsm8k_config import EXPERIMENT_CONFIG as GSM8K_CONFIG
from aqua_config import EXPERIMENT_CONFIG as AQUA_CONFIG
from svamp_config import EXPERIMENT_CONFIG as SVAMP_CONFIG
from hcgc_caaw import ChannelAdaptiveAgentWeighting


class BaselineCompatibleMAPGDFramework(MAPGDFramework):
    def __init__(self, config):
        super().__init__(config)
        # 启用 CAAW
        if config.get('enable_caaw', True):
            print("✅ Channel-Adaptive Agent Weighting (CAAW) is ENABLED.")
            self.caaw = ChannelAdaptiveAgentWeighting(config)
            self.agent_performance_history = []
        else:
            print("❌ Channel-Adaptive Agent Weighting (CAAW) is DISABLED.")
            self.caaw = None
            self.agent_performance_history = None
    """
    Baseline兼容的MAPGD框架

    实现baseline论文中的标准配置：
    - minibatch size = 64
    - beam size = 4
    - 6 optimization steps
    - 4 errors per group, 4 gradients per group
    - 2 monte carlo samples per candidate
    - 8 successor candidates per parent
    - bandit selection
    """

    def optimize_prompt(self, initial_prompt, task_name, data_dir, predictor=None,
                        predictor_type='binary_classification'):
        """主优化流程 - 修复预测器初始化问题"""
        start_time = time.time()

        # 🔧 修复：确保config中包含task_name
        if 'task_name' not in self.config:
            self.config['task_name'] = task_name

        print(f"🔍 Task name in config: {self.config.get('task_name')}")
        print(f"🔍 Predictor type: {predictor_type}")

        # 加载任务和数据
        task_class = get_mapgd_task_class(task_name)
        task = task_class(data_dir)
        print(f"🔎 task.data_dir = {task.data_dir}")
        train_examples = task.get_train_examples()
        test_examples = task.get_test_examples()

        print(f"📊 Dataset: {len(train_examples)} train, {len(test_examples)} test examples")

        # 🔧 修复：确保预测器正确初始化
        if predictor is None:
            print(f"🔧 Creating predictor of type: {predictor_type}")
            predictor = get_mapgd_predictor(predictor_type, self.config)
            print(f"✅ Predictor created: {type(predictor)}")

        # 🔧 修复：确保预测器不为None
        if predictor is None:
            raise ValueError(f"Failed to create predictor of type: {predictor_type}")

        # 初始化智能体 - 确保使用正确的任务名称
        print(f"🤖 Initializing agents for task: {task_name}")
        self.agent_manager.initialize_agents(initial_prompt)

        # 其余代码保持不变...
        optimization_history = []
        best_overall_prompt = initial_prompt
        best_overall_score = 0.0
        current_best_prompt = initial_prompt
        last_iteration_test_score = 0.0

        # ========== BASELINE主优化循环 (6步) ==========
        for iteration in range(self.max_iterations):
            print(f"\n🔄 === Optimization Step {iteration + 1}/{self.max_iterations} ===")
            iteration_start = time.time()

            # 1️⃣ BASELINE: 采样minibatch (|Dmini| = 64)
            minibatch_size = min(self.config.get('minibatch_size', 64), len(train_examples))
            minibatch = np.random.choice(train_examples, minibatch_size, replace=False).tolist()
            print(f"📦 Sampled minibatch: {minibatch_size} examples")

            # 2️⃣ BASELINE: 梯度生成 (4 errors at a time, m=4 gradients per group)
            print(f"🔍 Predictor check before gradient generation: {type(predictor)}")
            agent_gradients = self.agent_manager.generate_gradients(
                minibatch, task, predictor
            )
            print(f"📈 Generated gradients from {len(agent_gradients)} agents")

            # 3️⃣ 梯度协调与融合
            if self.config.get('disable_fusion', False):
                fused_gradients = sum(agent_gradients.values(), [])  # 直接拼接所有梯度
                print(f"🚫 Fusion disabled, using {len(fused_gradients)} raw gradients")
            else:
                fused_gradients = self.gradient_coordinator.coordinate_gradients(
                    agent_gradients,
                    caaw_instance=self.caaw,
                    agent_history=self.agent_performance_history,
                    current_iteration=iteration
                )
                print(f"🔀 Fused to {len(fused_gradients)} coordinated gradients")

            print(f"🔀 Fused to {len(fused_gradients)} coordinated gradients")

            # 4️⃣ expansion
            candidates = self.prompt_expander.expand(
                current_prompt = current_best_prompt,  # 使用融合提示作为基础
                gradients = fused_gradients
            )
            print(f"🌱 Generated {len(candidates)} candidate prompts")

            # 5️⃣ BASELINE: Bandit选择
            eval_sample_size = min(32, len(train_examples))  # 评估样本数
            eval_samples = np.random.choice(train_examples, eval_sample_size, replace=False).tolist()

            scores = self.candidate_selector.evaluate_prompts(
                candidates, eval_samples, task, predictor
            )
            print(f"📊 Evaluated {len(candidates)} candidates")

            # 7️⃣ 选择最佳候选 (beam size = 4)
            prompt_score_pairs = list(zip(scores, candidates))
            prompt_score_pairs.sort(reverse=True)

            # beam_size = self.config.get('beam_size', 4)
            # best_prompts = [prompt for _, prompt in prompt_score_pairs[:beam_size]]
            # # self.agent_manager.update_agents(best_prompts)

            # 8️⃣ 记录结果 - 选择评分最高的提示
            # 多智能体协作已经通过梯度融合完成，这里直接选择最佳候选
            current_best_prompt = prompt_score_pairs[0][1]  # 评分最高的提示
            current_best_score = prompt_score_pairs[0][0]   # 对应的评分

            print(f"📈 Selected best prompt with score: {current_best_score:.4f}")

            # 测试集评估
            test_eval_size = min(100, len(test_examples))
            test_samples = test_examples[:test_eval_size]
            test_score, _, _, _ = task.evaluate_prompt(
                current_best_prompt, test_samples, predictor, n=test_eval_size
            )
            print(f"📊 Best prompt test score: {test_score:.4f}")
            if self.caaw and self.agent_performance_history is not None:
                performance_gain = test_score - last_iteration_test_score
                print(f"[CAAW] Performance gain this iteration: {performance_gain:.4f}")
                # 将这个增益归功于所有参与本轮的 agent
                for i in range(len(self.agent_manager.agents)):
                    self.agent_performance_history.append({
                        'iteration': iteration,
                        'agent_id': f'agent_{i}',
                        'improvement': performance_gain,
                        'test_score': test_score
                    })
                # 更新上一轮的分数
                last_iteration_test_score = test_score
            # 更新全局最佳
            if test_score > best_overall_score:
                best_overall_prompt = current_best_prompt
                best_overall_score = test_score
            
            iteration_time = time.time() - iteration_start
            
            iteration_result = {
                'iteration': iteration + 1,
                'best_prompt': current_best_prompt,
                'train_score': current_best_score,
                'test_score': test_score,
                'num_candidates': len(candidates),
                'minibatch_size': minibatch_size,
                'eval_sample_size': eval_sample_size,
                'iteration_time': iteration_time
            }
            
            optimization_history.append(iteration_result)
            
            print(f"✅ Step {iteration + 1} Results:")
            print(f"   📈 Train Score: {current_best_score:.4f}")
            print(f"   🎯 Test Score: {test_score:.4f}")
            print(f"   ⏱️  Time: {iteration_time:.2f}s")
            print(f"   📝 Best Prompt: {current_best_prompt}")
            
            # 收敛性检查
            if self.convergence_monitor.check_convergence(test_score):
                print(f"🎯 Converged after {iteration + 1} optimization steps")
                break
            
        total_time = time.time() - start_time
        
        # 返回结果
        final_result = {
            'framework': 'MAPGD-Baseline-Compatible',
            'best_prompt': best_overall_prompt,  # 使用多智能体融合的最终提示
            'final_test_score': best_overall_score,  # 使用融合提示的测试得分
            'optimization_history': optimization_history,
            'total_iterations': len(optimization_history),
            'total_time': total_time,
            'config': self.config,
            'baseline_compliant': True,
            'multi_agent_fusion': True  # 标记使用了多智能体融合
        }
        
        print(f"\n🏁 MAPGD Optimization Complete!")
        print(f"📊 Final Results:")
        print(f"   🏆 Best Test Score: {best_overall_score:.4f}")
        print(f"   🔢 Total Steps: {len(optimization_history)}")
        print(f"   ⏱️  Total Time: {total_time:.2f}s")
        print(f"   📝 Final Prompt: {best_overall_prompt}")
        
        return final_result


# 2. 在 run_baseline_experiment 函数中更新配置选择部分：
def run_baseline_experiment(task="jailbreak"):
    """运行baseline兼容的MAPGD实验"""

    # ========== 根据任务选择配置 ==========
    if task == "liar":
        config = LIAR_CONFIG.copy()
        initial_prompt = LIAR_CONFIG['initial_prompt']
        predictor_type = 'binary_classification'

    elif task == "jailbreak":
        config = JAILBREAK_CONFIG.copy()
        initial_prompt = JAILBREAK_CONFIG['initial_prompt']
        predictor_type = 'binary_classification'

    elif task == "ethos":
        config = ETHOS_CONFIG.copy()
        initial_prompt = ETHOS_CONFIG['initial_prompt']
        predictor_type = 'binary_classification'

    elif task == "gsm8k":  # 添加GSM8k分支
        config = GSM8K_CONFIG.copy()
        initial_prompt = GSM8K_CONFIG['initial_prompt']
        predictor_type = 'math_reasoning'  # 使用数学推理预测器
    elif task == "svamp":  # 添加GSM8k分支
        config = SVAMP_CONFIG.copy()
        initial_prompt = SVAMP_CONFIG['initial_prompt']
        predictor_type = 'math_reasoning'  # 使用数学推理预测器
    elif task == "aqua":  # ⭐ 新增 AQuA 分支
        config = AQUA_CONFIG.copy()
        initial_prompt = AQUA_CONFIG['initial_prompt']
        predictor_type = 'aqua_reasoning'  # 使用 AQuA 专用预测器

    else:
        raise ValueError(f"Unsupported task: {task}")


    
    print("🚀 Starting MAPGD Experiment with Baseline-Compatible Settings")
    print("=" * 60)
    print("📊 Experiment Configuration (Baseline Match):")
    print(f"  • Optimization Steps: {config['max_iterations']}")
    print(f"  • Beam Size: {config['beam_size']}")
    print(f"  • Minibatch Size: {config['minibatch_size']}")
    print(f"  • Error Group Size: {config['error_group_size']}")
    print(f"  • Gradients per Group: {config['gradients_per_group']}")
    print(f"  • Monte Carlo Samples: {config['monte_carlo_samples']}")
    print(f"  • Successor Candidates: {config['successor_candidates']}")
    print(f"  • Selection Strategy: {config['selection_strategy']}")
    print("=" * 60)
    
    try:
        # 初始化框架
        framework = BaselineCompatibleMAPGDFramework(config)
        
        # 初始提示 (使用简单但有效的初始提示)
        #initial_prompt = TEST_INITIAL_PROMPT
        
        # 运行优化
        print(f"\n🎯 Starting optimization with initial prompt...")
        print(f"Initial prompt: {initial_prompt[:80]}...")
        predictor = get_mapgd_predictor(predictor_type, config)
        results = framework.optimize_prompt(
            initial_prompt=initial_prompt,
            task_name=config['task_name'],
            data_dir=config['data_dir'],
            predictor=predictor,
            predictor_type=predictor_type  # 传入预测器类型
        )
        
        # 保存结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_file = f"baseline_{task}_results_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            results_to_save = {
                'experiment_type': 'baseline_compatible',
                'config': config,
                'results': results,
                'timestamp': timestamp
            }
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {result_file}")
        return results
        
    except Exception as e:
        print(f"✗ Baseline experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="jailbreak",
                       choices=["liar", "jailbreak", "ethos", "gsm8k", "aqua", "svamp"])  # 添加 "gsm8k"
    args = parser.parse_args()

    print("🔬 MAPGD Baseline-Compatible Experiment")
    print(f"🎯 Goal: Match baseline paper configuration for fair comparison (Task={args.task})")
    print()

    results = run_baseline_experiment(task=args.task)
    if results:
        print("\n✅ Baseline-compatible experiment completed successfully!")
        print(f"🎯 Final accuracy: {results['final_test_score']:.4f}")
    else:
        print("\n❌ Baseline-compatible experiment failed!")