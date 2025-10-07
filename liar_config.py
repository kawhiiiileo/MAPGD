"""
Liar数据集实验配置
用于MAPGD框架的虚假信息检测任务，和论文对齐
"""
from init_prompt import LIAR_INITIAL_PROMPT

# 数据配置
DATA_CONFIG = {
    'task_name': 'liar',
    'data_dir': 'D:\研究生\espai\mapgd\data\hf_binary',
    'sentence_transformer_model': r'D:\研究生\espai\mapgd\all-MiniLM-L6-v2',
}

# MAPGD框架配置
MAPGD_CONFIG = {
    # 核心参数
    'max_iterations': 10,
    'beam_size': 4,
    'num_agents': 4,
    'minibatch_size': 150,
    'error_group_size': 4,
    'gradients_per_group': 4,
    'monte_carlo_samples': 2,
    'successor_candidates': 8,
    'enable_hcgc': True,  # 启用/禁用 HCGC
    'margin_scale': 2.0,  # 角度边界的n参数(公式7)
    'clustering_temperature': 0.1,  # 聚类中softmax的温度参数τ

    # ✅ CAAW 配置
    'enable_caaw': True,  # 启用/禁用 CAAW
    'caaw_lambda': 1,  # 智能体加权的温度参数λ(公式11)
    'caaw_validation_samples': 20,  # 用于计算智能体增益的样本数
    # 异步并行配置 - 新增性能优化
    'disable_fusion': False, # 禁用梯度融合
    'async_mode': True,  # 启用异步梯度生成
    'async_evaluation': True,  # 启用异步候选评估
    'max_concurrent_agents': 4,  # 并发智能体数量
    'max_concurrent_evaluators': 4,  # 并发评估生成数量
    # 梯度协调参数 - 使用真实的SentenceTransformer
    'fusion_method': 'semantic_clustering',
    'conflict_threshold': 0.3,
    'cluster_threshold': 0.7,
    'max_clusters': 5,
    "random_agent_roles": False, #无专业化智能体
    # 提示扩展策略配置 - 关键修复
    'expansion_strategy': 'beam_search',  # 'beam_search', 'monte_carlo', 'hybrid'
    'mc_samples': 2,  # Monte Carlo采样数量
    'diversity_threshold': 0.7,  # 多样性过滤阈值
    
    # GPU加速配置
    'use_gpu': True,
    'batch_size': 100,  # SentenceTransformer批量处理大小
    'max_workers': 4,  # 并行评估线程数
    'batch_encoding': True,  # 批量编码启用
    'cache_enabled': True,
    
    # Bandit选择参数
    'selection_strategy': 'ucb',  # ucb, thompson, epsilon_greedy
    'evaluation_budget': 80,
    'min_evaluations_per_candidate': 3,
    'c': 1.0,
    'epsilon': 0.1,
    
    # 收敛参数
    'convergence_patience': 3,
    'convergence_threshold': 0.01,
}


# 合并所有配置
EXPERIMENT_CONFIG = {
    **DATA_CONFIG,
    **MAPGD_CONFIG,
    'initial_prompt': LIAR_INITIAL_PROMPT
}
