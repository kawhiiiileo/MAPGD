"""
AQuA-RAT数学推理选择题数据集实验配置
用于MAPGD框架的数学推理任务
"""
import os

# 数据配置
DATA_CONFIG = {
    'task_name': 'aqua',
    'data_dir': os.path.join(os.path.dirname(__file__), 'data/aqua'),
    'sentence_transformer_model': 'all-MiniLM-L6-v2',
}

# AQuA专用初始提示词
AQUA_INITIAL_PROMPT = """
# Task
Solve the math word problem and choose the correct answer from the given options.

# Instructions
1. Read the problem carefully
2. Analyze each option
3. Show your reasoning step by step
4. Select the correct answer (A, B, C, D, or E)

# Output Format
Show your reasoning and end with: Answer: [LETTER]
(For example: "Answer: A" or "Answer: B")

# Problem
{text}

# Options
{options}
"""

# MAPGD框架配置
MAPGD_CONFIG = {
    # 核心参数
    'max_iterations': 6,
    'beam_size': 4,
    'num_agents': 4,
    'minibatch_size': 32,
    'error_group_size': 4,
    'gradients_per_group': 4,
    'monte_carlo_samples': 2,
    'successor_candidates': 8,

    # 异步并行配置
    'async_mode': True,
    'async_evaluation': True,
    'max_concurrent_agents': 4,
    'max_concurrent_evaluators': 2,
    'random_agent_roles': False,

    # 梯度协调参数
    'disable_fusion': False,
    'fusion_method': 'semantic_clustering',
    'conflict_threshold': 0.3,
    'cluster_threshold': 0.7,
    'max_clusters': 5,
    'enable_hcgc': True,  # 启用/禁用 HCGC
    'margin_scale': 2.0,  # 角度边界的n参数(公式7)
    'clustering_temperature': 0.1,  # 聚类中softmax的温度参数τ

    # ✅ CAAW 配置
    'enable_caaw': True,  # 启用/禁用 CAAW
    'caaw_lambda': 1.0,  # 智能体加权的温度参数λ(公式11)
    'caaw_validation_samples': 20,  # 用于计算智能体增益的样本数
    # 提示扩展策略配置
    'expansion_strategy': 'beam_search',
    'mc_samples': 2,
    'diversity_threshold': 0.7,

    # GPU加速配置
    'use_gpu': True,
    'batch_size': 16,
    'max_workers': 2,
    'batch_encoding': True,
    'cache_enabled': True,

    # Bandit选择参数
    'selection_strategy': 'ucb',
    'evaluation_budget': 60,
    'min_evaluations_per_candidate': 3,
    'c': 1.0,
    'epsilon': 0.1,

    # 收敛参数
    'convergence_patience': 3,
    'convergence_threshold': 0.01,

    # LLM配置
    'temperature': 0.1,
    'model': 'gpt-4',
}

# 合并所有配置
EXPERIMENT_CONFIG = {
    **DATA_CONFIG,
    **MAPGD_CONFIG,
    'initial_prompt': AQUA_INITIAL_PROMPT
}