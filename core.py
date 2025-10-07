# 把 core.py 顶部现有的 argparse.parse_args() 段替换为下面这段：

import argparse
import sys

# 当 core 被其他模块 import（比如 graph_experiment.py）时，
# 不要用 parse_args() 去消费外部传入的所有命令行参数，
# 否则会导致 import 时 argparse 报错（unrecognized arguments）。
# 使用 parse_known_args()：只解析我们关心的 --task，忽略未知参数。
from utils import load_config

DATA_CONFIG, EXPERIMENT_CONFIG = load_config()

import time
import random
import numpy as np
import re
from sklearn.cluster import KMeans
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from llm import call_openai
from mapgd_tasks import get_mapgd_task_class
from mapgd_predictors import get_mapgd_predictor
from utils import MAPGDUtils
from abc import abstractmethod, ABCMeta
from hcgc_caaw import HypersphereConstrainedGradientClustering

# Optional GPU support - graceful fallback if torch unavailable
try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    GPU_AVAILABLE = False


class MAPGDFramework(metaclass=ABCMeta):
    """
    MAPGD: Multi-Agent Prompt Gradient Descent Framework

    Implements collaborative prompt optimization through:
    1. Multi-agent exploration with specialized roles
    2. Semantic gradient coordination and fusion
    3. Beam search + Monte Carlo prompt expansion
    4. Bandit-based efficient candidate selection

    Based on the paper: "MAPGD: Multi-Agent Prompt Gradient Descent
    for Collaborative Prompt Optimization"
    """

    def __init__(self, config):
        # Core MAPGD components initialization
        self.config = config
        self.max_iterations = config.get('max_iterations', 10)

        # Multi-agent collaboration module
        self.agent_manager = AgentManager(config)

        # Gradient coordination module (semantic clustering + conflict resolution)
        self.gradient_coordinator = GradientCoordinator(config)

        # Collaborative prompt expansion (beam search + MC paraphrasing)
        self.prompt_expander = CollaborativePromptExpander(config)

        # Bandit-based candidate selection
        self.candidate_selector = BanditBasedSelector(config)

        # Convergence monitoring
        self.convergence_monitor = ConvergenceMonitor(config)

        print(f"MAPGD Framework initialized with {config.get('num_agents', 4)} specialized agents")

    @abstractmethod
    def optimize_prompt(self, initial_prompt, task_name, data_dir, predictor=None):
        """
        Main MAPGD optimization workflow

        Implements the collaborative gradient descent process:
        1. Multi-agent gradient generation (parallel exploration)
        2. Semantic gradient coordination and fusion
        3. Collaborative prompt expansion via beam search
        4. Bandit-based efficient candidate selection
        5. Agent synchronization and convergence monitoring

        Returns optimized prompt with performance history
        """
        pass


class SpecializedPromptAgent:
    """
    Specialized Prompt Agent for MAPGD Framework

    Each agent focuses on a specific aspect of prompt optimization:
    - instruction_specialist: Task description and instruction clarity
    - example_curator: Few-shot example selection and formatting
    - format_designer: Output format and structure design
    - style_optimizer: Language style and tone adjustment

    Implements role-specific gradient generation and prompt updating
    """

    #     AGENT_TYPES = {
    #     "instruction_specialist": {
    #         "focus": "Clarity and completeness of task instructions for report generation and data verification",
    #         "expertise": "Analyze and improve instructions to ensure clarity, completeness, and enforceability, with specific guidance for verifying input data authenticity and generating structured reports",
    #         "gradient_type": "instruction_gradient",
    #         "role_description": "Specializes in crafting clear, complete, and executable instructions for generating truthful, structured reports and rejecting false or exaggerated inputs. Instructions must include steps to verify input data (e.g., checking financial metrics, timelines, or scales for realism) and rules for producing professional business, financial, and legal reports.",
    #         "prompt": "You are an instruction specialist tasked with improving the clarity and completeness of task instructions for a model that generates truthful, structured reports and identifies false or exaggerated inputs. Your goal is to ensure the instructions are clear, enforceable, and include: 1) A step to verify input data authenticity by checking for realistic metrics (e.g., ARR in millions, not negative or absurd values), timelines (e.g., plausible years like 2025, not 11975), and scales (e.g., reasonable team sizes). 2) Rules for generating structured reports with Business Analysis, Financial Report, and Legal Prompt sections, ensuring professional and verifiable content. 3) Guidance to reject invalid inputs with a clear error message. Revise the instructions to be concise, unambiguous, and task-specific, incorporating examples like: Valid input ('$4.16M ARR') should produce a structured report; invalid input ('$-46,110,7-458 ARR') should return '<Error> Invalid input: Exaggerated or unverifiable claims detected.'"
    #     },
    #     "example_curator": {
    #         "focus": "Selection and formatting of representative examples for report generation and false data detection",
    #         "expertise": "Optimize the selection of diverse, representative examples from positive (truthful) and negative (false/exaggerated) samples, ensuring consistent formatting and clear distinction between valid and invalid inputs",
    #         "gradient_type": "example_gradient",
    #         "role_description": "Focuses on selecting and formatting diverse examples that demonstrate truthful report generation and false data rejection. Examples must include positive samples (e.g., realistic ARR, market size) and negative samples (e.g., exaggerated financials, absurd timelines), formatted consistently to guide the model in distinguishing valid from invalid inputs.",
    #         "prompt": "You are an example curator tasked with selecting and formatting representative examples to guide a model in generating truthful, structured reports and rejecting false or exaggerated inputs. Your goal is to: 1) Select diverse examples from positive samples (e.g., '$4.16M ARR', 'construction market valued at $50B') and negative samples (e.g., '$-46,110,7-458 ARR', '1050 FTEs by 11975'). 2) Ensure examples highlight key differences, such as realistic vs. absurd financial metrics, timelines, or scales. 3) Format examples consistently with fields: Section, Statement, Context, and Output (structured report for valid inputs, error message for invalid inputs). For example: Positive sample: 'Section: Key Insights, Statement: The company has demonstrated strong revenue growth with $4.16M in ARR, Context: SaaS startup report, Output: <Business Analysis>..., <Financial Report>..., <Legal Prompt>...'; Negative sample: 'Section: ARR, Statement: $-46,110,7-458, Context: ARR section, Output: <Error> Invalid input: Exaggerated or unverifiable claims detected.' Optimize the example set for diversity, representativeness, and clarity."
    #     },
    #     "format_designer": {
    #         "focus": "Design of structured output formats for reports and error messages",
    #         "expertise": "Design clear, consistent output templates for structured reports (Business Analysis, Financial Report, Legal Prompt) and error messages for invalid inputs",
    #         "gradient_type": "format_gradient",
    #         "role_description": "Specializes in designing clear, structured output templates for truthful report generation and false data rejection. Ensures outputs use consistent tags (<Business Analysis>, <Financial Report>, <Legal Prompt> for valid inputs; <Error> for invalid inputs) and are easy for the model to parse and generate.",
    #         "prompt": "You are a format designer tasked with creating clear, structured output templates for a model that generates truthful reports and rejects false or exaggerated inputs. Your goal is to: 1) Design a consistent output format for valid inputs, using tags: <Business Analysis>, <Financial Report>, <Legal Prompt>, with clear section boundaries and professional content. 2) Design an error format for invalid inputs: <Error> Invalid input: Exaggerated or unverifiable claims detected. 3) Ensure formats are parseable and unambiguous. For example: Valid input ('$4.16M ARR') should output: <Business Analysis> [Market analysis and recommendations] </Business Analysis> <Financial Report> [Financial impacts and risks] </Financial Report> <Legal Prompt> [Legal compliance analysis] </Legal Prompt>; Invalid input ('$-46,110,7-458 ARR') should output: <Error> Invalid input: Exaggerated or unverifiable claims detected. Optimize the format for clarity, consistency, and task alignment."
    #     },
    #     "style_optimizer": {
    #         "focus": "Professional language style and tone for report generation and error messages",
    #         "expertise": "Optimize language for professionalism, conciseness, and adaptability to business and legal contexts, ensuring error messages are clear and direct",
    #         "gradient_type": "style_gradient",
    #         "role_description": "Optimizes language style and tone for generating professional, truthful reports and clear error messages. Ensures reports use concise, industry-appropriate language for business, financial, and legal contexts, and error messages are direct and unambiguous.",
    #         "prompt": "You are a style optimizer tasked with refining the language style and tone for a model that generates truthful, structured reports and rejects false or exaggerated inputs. Your goal is to: 1) Ensure report sections (<Business Analysis>, <Financial Report>, <Legal Prompt>) use professional, concise, and industry-appropriate language for business and legal contexts (e.g., 'The $4.16M ARR reflects strong market traction' instead of vague or informal phrasing). 2) Ensure error messages for invalid inputs (e.g., '$-46,110,7-458 ARR') are clear, direct, and professional (e.g., '<Error> Invalid input: Exaggerated or unverifiable claims detected'). 3) Avoid jargon overload or overly complex language. Optimize the style for professionalism, clarity, and task-specific adaptation, ensuring alignment with examples like: Positive output: 'The company’s $4.16M ARR indicates robust growth'; Negative output: '<Error> Invalid input: Exaggerated or unverifiable claims detected.'"
    #     }
    # }
    BASE_AGENT_TYPES = {
        'instruction_specialist': {
            'focus': 'Clarity of task descriptions and instructions',
            'expertise': 'Analyze the completeness, clarity, and enforceability of instructions',
            'gradient_type': 'instruction_gradient',
            'role_description': 'Specializes in analyzing and improving task instructions, ensuring clarity, completeness, and executability'
        },
        'example_curator': {
            'focus': 'Few-shot example selection and formatting',
            'expertise': 'Optimize the representativeness, diversity, and format consistency of examples',
            'gradient_type': 'example_gradient',
            'role_description': 'Focuses on selecting representative, diverse examples and ensuring consistent formatting'
        },
        'format_designer': {
            'focus': 'Output format and structure design',
            'expertise': 'Design clear output templates and structured formats',
            'gradient_type': 'format_gradient',
            'role_description': 'Designs clear output templates and structured formats for better model understanding'
        },
        'style_optimizer': {
            'focus': 'Language style and tone adjustments',
            'expertise': 'Optimize the professionalism and adaptability of language expression',
            'gradient_type': 'style_gradient',
            'role_description': 'Optimizes language expression for professionalism and task-specific adaptation'
        },
        'generic': {
            'focus': 'General prompt improvement',
            'expertise': 'Provide general reasons and improvements without specialization',
            'gradient_type': 'generic_gradient',
            'role_description': 'General agent without specialization, explores improvements broadly'
        }
    }

    # 数学推理专用智能体类型 (适用于GSM8k等数学任务)
    MATH_AGENT_TYPES = {
        'reasoning_specialist': {
            'focus': 'Mathematical reasoning process and step-by-step logic',
            'expertise': 'Analyze and improve the logical flow, step clarity, and reasoning structure in mathematical problem solving',
            'gradient_type': 'reasoning_gradient',
            'role_description': 'Specializes in enhancing mathematical reasoning processes, ensuring clear step-by-step logic and proper problem decomposition'
        },
        'calculation_optimizer': {
            'focus': 'Numerical computation accuracy and methodology',
            'expertise': 'Optimize calculation methods, numerical precision, and computational approaches',
            'gradient_type': 'calculation_gradient',
            'role_description': 'Focuses on improving calculation accuracy, suggesting better computational methods and ensuring numerical correctness'
        },
        'problem_interpreter': {
            'focus': 'Word problem comprehension and key information extraction',
            'expertise': 'Enhance problem understanding, identify key variables, constraints, and mathematical relationships',
            'gradient_type': 'interpretation_gradient',
            'role_description': 'Specializes in interpreting math word problems, extracting relevant information, and identifying mathematical relationships'
        },
        'solution_formatter': {
            'focus': 'Mathematical solution presentation and answer format',
            'expertise': 'Design clear solution formats, ensure proper mathematical notation, and structure final answers',
            'gradient_type': 'format_gradient',
            'role_description': 'Optimizes mathematical solution presentation, ensures clear formatting and proper answer notation (e.g., #### format)'
        },
        'generic': {
            'focus': 'General mathematical prompt improvement',
            'expertise': 'Provide general mathematical reasoning improvements without specialization',
            'gradient_type': 'generic_gradient',
            'role_description': 'General mathematical agent without specialization, explores broad mathematical reasoning improvements'
        }
    }

    # 任务类型到智能体类型的映射
    TASK_AGENT_MAPPING = {
        'liar': BASE_AGENT_TYPES,
        'ethos': BASE_AGENT_TYPES,
        'jailbreak': BASE_AGENT_TYPES,
        'gsm8k': MATH_AGENT_TYPES,
        'aqua': MATH_AGENT_TYPES,
        'svamp': MATH_AGENT_TYPES,
        'classification': BASE_AGENT_TYPES,
        'default': BASE_AGENT_TYPES
    }

    def __init__(self, agent_type, task_name='default'):
        """
        初始化专业化智能体

        Args:
            agent_type: 智能体类型名称
            task_name: 任务名称，用于选择合适的智能体配置
        """
        self.task_name = task_name
        self.agent_type = agent_type

        # 根据任务类型选择智能体配置
        agent_types = self.TASK_AGENT_MAPPING.get(task_name, self.BASE_AGENT_TYPES)

        if agent_type in agent_types:
            self.config = agent_types[agent_type]
        else:
            # fallback：如果传入了未知类型，就默认用 generic
            print(f"Warning: Unknown agent type '{agent_type}' for task '{task_name}', using generic")
            self.config = agent_types.get('generic', self.BASE_AGENT_TYPES['generic'])

        # 智能体状态
        self.current_prompt = None
        self.gradient_history = []
        self.performance_memory = []
        self.specialization_confidence = 1.0

        print(f"Initialized {agent_type} for {task_name}: {self.config['role_description']}")

    @classmethod
    def get_available_agent_types(cls, task_name='default'):
        """获取指定任务可用的智能体类型"""
        agent_types = cls.TASK_AGENT_MAPPING.get(task_name, cls.BASE_AGENT_TYPES)
        return list(agent_types.keys())

    async def generate_specialized_gradient_async(self, prompt, error_examples, task):
        gradient_prompt = self._construct_gradient_prompt(prompt, error_examples, task)

        # 在线程池中执行LLM调用以避免阻塞
        loop = asyncio.get_event_loop()

        def sync_llm_call():
            return call_openai(
                prompt=gradient_prompt,
                system_prompt="You are a professional prompt word optimization expert."
            )

        # 异步执行LLM调用
        with ThreadPoolExecutor(max_workers=EXPERIMENT_CONFIG['max_concurrent_agents']) as executor:
            response = await loop.run_in_executor(executor, sync_llm_call)

        # 解析梯度响应
        gradient = self._parse_gradient_response(response)

        # 更新历史记录
        self.gradient_history.append({
            'prompt': prompt,
            'gradient': gradient,
            'context': error_examples,
            'timestamp': time.time(),
            'agent_type': self.agent_type
        })

        return gradient

    def generate_specialized_gradient(self, prompt, error_examples, task):
        gradient_prompt = self._construct_gradient_prompt(
            prompt, error_examples, task
        )

        # 调用LLM生成梯度
        response = call_openai(
            prompt=gradient_prompt,
            system_prompt="You are a professional prompt word optimization expert."
        )

        # 解析梯度响应
        gradient = self._parse_gradient_response(response)

        # 更新历史记录
        self.gradient_history.append({
            'prompt': prompt,
            'gradient': gradient,
            'context': error_examples,
            'timestamp': time.time()
        })

        return gradient

    def _construct_gradient_prompt(self, prompt, error_examples, task, num_feedbacks=4):
        """
        构建针对任务优化的梯度提示
        """
        # 格式化错误样本
        error_text = self._format_error_examples(error_examples)

        # 根据任务类型调整提示
        if self.task_name == 'gsm8k' or 'math' in self.task_name:
            task_context = "mathematical word problem solving"
        else:
            task_context = "text classification"

        base_prompt = f"""
        I'm trying to write a zero-shot {task_context} prompt.
        My current prompt is: "{prompt}"
        But this prompt gets the following examples wrong:
        {error_text}

        As an expert specialized in {self.config['expertise']}, please give {num_feedbacks} reasons 
        why the prompt could have gotten these examples wrong from a {self.config['focus']} perspective.
        Focus specifically on {task_context} requirements.

        Wrap each reason with <START> and <END>
        """
        return base_prompt

    def _format_error_examples(self, error_examples):
        """格式化错误样本"""
        texts = []
        if not error_examples:
            return "暂无错误样本"

        for example in error_examples:  # 只显示前3个后续可以更改
            text = example.get('text', '')
            texts.append(text)
        texts = np.random.choice(texts, EXPERIMENT_CONFIG['error_group_size'], replace=False) if len(texts) > \
                                                                                                 EXPERIMENT_CONFIG[
                                                                                                     'error_group_size'] else texts
        return "\n".join(texts)

    def _parse_gradient_response(self, response):
        """解析梯度响应 - 使用工具模块"""
        return MAPGDUtils.parse_gradient_response(response)


class GradientCoordinator:
    """
    Multi-Agent Gradient Coordinator for MAPGD

    Implements semantic gradient coordination through:
    1. Semantic vectorization using sentence embeddings
    2. Conflict detection via cosine similarity analysis
    3. Gradient clustering using K-means on semantic space
    4. Intelligent fusion of conflicting gradients via LLM

    Core innovation: Transforms textual gradients into semantic space
    for mathematical-like operations and conflict resolution
    """

    def __init__(self, config):
        self.fusion_method = config.get('fusion_method', 'semantic_clustering')
        self.conflict_threshold = config.get('conflict_threshold', 0.3)
        self.max_clusters = config.get('max_clusters', 5)

        # 使用共享的语义模型
        self.semantic_model = MAPGDUtils.get_semantic_model()
        # ✅ 添加 HCGC
        self.hcgc = HypersphereConstrainedGradientClustering(config)
        self.enable_hcgc = config.get('enable_hcgc', True)
        # ADDED: 在协调器中添加对 CAAW 的配置引用
        self.enable_caaw = config.get('enable_caaw', True)

    def coordinate_gradients(self, agent_gradients, caaw_instance=None, agent_history=None, current_iteration=0):
        """
        Main gradient coordination pipeline. Now with optional CAAW integration.
        """
        print(f"Coordinating gradients from {len(agent_gradients)} agents...")

        # 始终使用 HCGC 进行聚类
        print("[HCGC] Using hypersphere-constrained clustering...")

        # 1. 嵌入和聚类 (从 HCGC 逻辑中提取)
        gradient_vectors, metadata = self.hcgc._embed_to_hypersphere(agent_gradients)
        if len(gradient_vectors) == 0:
            return []

        conflicts = self.hcgc._detect_conflicts(gradient_vectors, metadata)
        n_clusters = min(len(gradient_vectors), self.max_clusters)
        cluster_labels, centroids = self.hcgc._cluster_gradients(gradient_vectors, n_clusters)
        cluster_labels = self.hcgc._apply_angular_margin(gradient_vectors, cluster_labels, centroids)
        clusters = self.hcgc._organize_clusters(metadata, cluster_labels, centroids)
        print(f"[HCGC] Formed {len(clusters)} coherent gradient clusters.")

        # 2. 检查是否启用 CAAW 并进行加权融合
        if self.enable_caaw and caaw_instance and agent_history is not None:
            print("[CAAW] Applying Channel-Adaptive Agent Weighting...")
            fused_gradients = self._fuse_clusters_with_caaw(
                clusters, conflicts, agent_gradients, caaw_instance, agent_history, current_iteration
            )
        else:
            # 回退到标准的（非加权）融合
            print("[Fusion] Using uniform (non-weighted) fusion.")
            fused_gradients = self.hcgc._fuse_clusters(clusters, conflicts)

        print(f"Fusion complete: generated {len(fused_gradients)} coordinated gradients.")
        return fused_gradients

    # ADDED: 新增一个专门用于 CAAW 加权融合的方法
    def _fuse_clusters_with_caaw(self, clusters, conflicts, original_agent_gradients, caaw, history, iteration):
        fused_gradients = []

        # 为本次迭代创建一个 agent -> gradients 的映射，以便在加权融合时查找
        agent_to_gradient_map = {}
        for agent_id, grads in original_agent_gradients.items():
            agent_to_gradient_map[agent_id] = [MAPGDUtils.extract_gradient_text(g) for g in grads]

        for cluster in clusters:
            if len(cluster['gradients']) == 1:
                fused_gradients.append(cluster['gradients'][0])
                continue

            # 1. 获取参与此集群的所有 agent
            agent_ids_in_cluster = list(cluster['agents'])

            # 2. 计算这些 agent 的权重
            agent_weights = caaw.compute_agent_weights(agent_ids_in_cluster, history, iteration)

            # 3. 使用 CAAW 的加权融合方法
            fused_gradient = caaw.weighted_fusion(
                gradients=cluster['gradients'],
                agent_weights=agent_weights,
                agent_to_gradient_map=agent_to_gradient_map
            )
            fused_gradients.append(fused_gradient)

        return fused_gradients

    def _vectorize_gradients(self, agent_gradients):
        """将文本梯度转换为语义向量 - 使用共享工具"""
        vectors = {}

        # 收集所有梯度文本用于批量处理
        all_texts = []
        text_to_agent = {}

        for agent_id, gradients in agent_gradients.items():
            for i, gradient in enumerate(gradients):
                gradient_text = MAPGDUtils.extract_gradient_text(gradient)
                all_texts.append(gradient_text)
                text_to_agent[len(all_texts) - 1] = (agent_id, i)

        # 批量编码所有文本（使用共享工具）
        print(f"Batch encoding {len(all_texts)} gradient texts...")
        all_vectors = MAPGDUtils.encode_texts(all_texts)

        # 将向量重新分配给对应的智能体
        for agent_id in agent_gradients.keys():
            vectors[agent_id] = []

        for idx, vector in enumerate(all_vectors):
            agent_id, _ = text_to_agent[idx]
            vectors[agent_id].append(vector)

        # 转换为numpy数组
        for agent_id in vectors:
            vectors[agent_id] = np.array(vectors[agent_id])

        return vectors

    def _detect_gradient_conflicts(self, gradient_vectors):
        """检测梯度方向冲突 - 使用共享工具"""
        conflicts = []

        agent_ids = list(gradient_vectors.keys())

        for i, agent_a in enumerate(agent_ids):
            for agent_b in agent_ids[i + 1:]:

                vectors_a = gradient_vectors[agent_a]
                vectors_b = gradient_vectors[agent_b]
                sim_mat = vectors_a @ vectors_b.T / (
                            np.linalg.norm(vectors_a, axis=1)[:, None] * np.linalg.norm(vectors_b, axis=1)[None, :])
                idx = np.where(sim_mat < -self.conflict_threshold)
                for i_a, i_b in zip(*idx):
                    conflicts.append({'agents': (agent_a, agent_b), 'indices': (i_a, i_b), 'sim': sim_mat[i_a, i_b]})
        return conflicts

    def _cluster_gradients(self, agent_gradients, gradient_vectors):
        """基于语义相似度聚类梯度"""
        # 将所有梯度向量合并
        all_vectors = []
        gradient_metadata = []

        for agent_id, gradients in agent_gradients.items():
            vectors = gradient_vectors[agent_id]
            for i, (gradient, vector) in enumerate(zip(gradients, vectors)):
                all_vectors.append(vector)
                gradient_metadata.append({
                    'agent_id': agent_id,
                    'gradient_idx': i,
                    'gradient': gradient
                })

        # K-means聚类
        n_clusters = min(len(all_vectors), 5)  # 最多5个集群
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(all_vectors)

        # 组织聚类结果
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_gradients = []
            cluster_agents = set()

            for i, label in enumerate(cluster_labels):
                if label == cluster_id:
                    metadata = gradient_metadata[i]
                    cluster_gradients.append(metadata['gradient'])
                    cluster_agents.add(metadata['agent_id'])

            clusters.append({
                'id': cluster_id,
                'gradients': cluster_gradients,
                'agents': list(cluster_agents),
                'centroid': kmeans.cluster_centers_[cluster_id]
            })

        return clusters

    def _fuse_gradient_clusters(self, clusters, conflicts):
        """融合梯度集群"""
        fused_gradients = []

        for cluster in clusters:
            if len(cluster['gradients']) == 1:
                # 单一梯度直接使用
                fused_gradients.append(cluster['gradients'][0])
            else:
                # 多梯度融合
                fusion_prompt = self._create_fusion_prompt(
                    cluster['gradients'], conflicts
                )

                # 使用LLM进行智能融合
                response = call_openai(
                    prompt=fusion_prompt,
                    system_prompt="You are a professional prompt word optimization expert who is good at integrating multiple improvement suggestions."
                )

                fused_gradient = self._parse_fusion_response(response)
                fused_gradients.append(fused_gradient)

        return fused_gradients

    def _create_fusion_prompt(self, gradients, conflicts, num_gradients=4):
        """创建梯度融合提示"""
        gradient_texts = []
        for i, gradient in enumerate(gradients):
            gradient_text = MAPGDUtils.extract_gradient_text(gradient)
            gradient_texts.append(f"建议{i + 1}: {gradient_text}")

        conflict_info = ""
        if conflicts:
            conflict_info = f"""

            The following potential conflicts have been detected and need to be resolved:
            {self._format_conflicts(conflicts)}
            """
        gradients = "\n".join(gradient_texts)
        fusion_prompt = f"""

        I need to combine the following multiple prompt improvement suggestions into {num_gradients} unified 
        and coherent prompt improvements:

        {gradients}
        {conflict_info}

        Wrap each unified and coherent prompt improvement with <START> and <END>
        """

        return fusion_prompt

    def _parse_fusion_response(self, response):
        """解析融合响应 - 使用工具模块"""
        return MAPGDUtils.parse_fusion_response(response)

    def _format_conflicts(self, conflicts):
        """格式化冲突信息 - 使用工具模块"""
        return MAPGDUtils.format_conflicts(conflicts)


class BanditBasedSelector:
    """基于多臂赌博机的高效候选选择器"""

    def __init__(self, config):
        self.config = config  # 保存配置以供后续使用
        self.selection_strategy = config.get('selection_strategy', 'ucb')  # 'ucb', 'thompson', 'epsilon_greedy'
        self.evaluation_budget = config.get('evaluation_budget', 50)
        self.min_evaluations = config.get('min_evaluations_per_candidate', 3)
        self.async_evaluation = config.get('async_evaluation', True)  # 启用异步评估
        self.max_concurrent_evaluations = config.get('max_concurrent_evaluations', 4)

    # def evaluate_prompts(self, candidate_prompts, training_data, task, predictor):
    #     if not candidate_prompts:
    #         return []
    #     print(f"Evaluating {len(candidate_prompts)} candidate prompts...")
    #     scores = []
    #     eval_sample_size = min(50, len(training_data))  # 适中的样本大小
    #     eval_data = random.sample(training_data, eval_sample_size)

    #     print(f"  Using {eval_sample_size} samples for evaluation")

    #     for i, candidate in enumerate(candidate_prompts):
    #         try:
    #             print(f"  Evaluating candidate {i+1}/{len(candidate_prompts)}...")
    #             score, _, _, _ = task.evaluate_prompt(
    #                 candidate, eval_data, predictor, n=eval_sample_size
    #             )
    #             scores.append(score)
    #             print(f"    Score: {score:.3f}")
    #         except Exception as e:
    #             print(f"    Failed: {e}")
    #             scores.append(0.0)
    #     print(f"Evaluation completed. Scores: {[f'{s:.3f}' for s in scores]}")
    #     return scores
    def evaluate_prompts(self, candidate_prompts, training_data, task, predictor):
        if not candidate_prompts:
            return []
        print(f"Evaluating {len(candidate_prompts)} candidate prompts...")
        scores = [0.0] * len(candidate_prompts)
        eval_sample_size = min(150, len(training_data))
        eval_data = random.sample(training_data, eval_sample_size)

        print(f"  Using {eval_sample_size} samples for evaluation")

        # 并发数从 liar_config.EXPERIMENT_CONFIG 读取
        max_workers = EXPERIMENT_CONFIG.get('max_concurrent_evaluators', 4)
        print(f"  Running up to {max_workers} concurrent evaluations")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._eval_single, candidate, eval_data, task, predictor, eval_sample_size
                ): idx
                for idx, candidate in enumerate(candidate_prompts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    score = future.result()
                    print(f"    Candidate {idx + 1}/{len(candidate_prompts)} Score: {score:.3f}")
                except Exception as e:
                    print(f"    Candidate {idx + 1} Failed: {e}")
                    score = 0.0
                scores[idx] = score

        print(f"Evaluation completed. Scores: {[f'{s:.3f}' for s in scores]}")
        return scores

    def _eval_single(self, candidate, eval_data, task, predictor, n):
        """Helper: evaluate one prompt returning only score."""
        score, _, _, _ = task.evaluate_prompt(candidate, eval_data, predictor, n=n)
        return score

    def select_top_candidates(self, candidate_prompts, task, training_data):
        """使用Bandit算法高效选择最佳候选"""
        if not candidate_prompts:
            return []

        if self.selection_strategy == 'ucb':
            return self._ucb_selection(candidate_prompts, task, training_data)
        elif self.selection_strategy == 'thompson':
            return self._thompson_sampling(candidate_prompts, task, training_data)
        else:
            return self._epsilon_greedy_selection(candidate_prompts, task, training_data)

    def _ucb_selection(self, candidates, task, training_data):
        print(f"UCB Selection: evaluating {len(candidates)} candidates")

        # UCB参数设置 (参考evaluators.py)
        c = self.config.get('c', 1.0)  # UCB置信参数
        num_rounds = min(20, self.evaluation_budget // len(candidates))  # 学术实验轮数
        samples_per_eval = min(10, len(training_data) // 4)  # 每轮评估样本数

        print(f"  UCB rounds: {num_rounds}, samples per eval: {samples_per_eval}")

        # 初始化UCB统计 (参考UCBBandits class)
        counts = np.zeros(len(candidates))
        scores = np.zeros(len(candidates))

        # UCB主循环 (简化版bandit algorithm)
        for round_idx in range(num_rounds):
            print(f"  Round {round_idx + 1}/{num_rounds}")

            # 计算UCB值选择候选 (基于evaluators.py的choose方法)
            if np.sum(counts) == 0:
                # 初始轮：随机选择
                selected_indices = list(range(len(candidates)))
            else:
                # UCB选择 (参考UCBBandits.choose)
                avg_scores = np.divide(scores, counts + 1e-3,
                                       out=np.zeros_like(scores), where=counts > 0)
                confidence = c * np.sqrt(np.log(round_idx + 1) / (counts + 1e-3))
                ucb_values = avg_scores + confidence

                # 选择top-k个候选进行评估
                k = min(len(candidates), 3)  # 限制并发评估数
                selected_indices = np.argsort(ucb_values)[::-1][:k]

            # 评估选中的候选 (简化评估逻辑)
            eval_data = random.sample(training_data, samples_per_eval)

            for idx in selected_indices:
                try:
                    # 直接使用task.evaluate_prompt (避免复杂的_evaluate_candidate)
                    score, _, _, _ = task.evaluate_prompt(
                        candidates[idx], eval_data, None, n=samples_per_eval
                    )

                    # 更新UCB统计 (参考UCBBandits.update)
                    counts[idx] += samples_per_eval
                    scores[idx] += score * samples_per_eval

                    avg_score = scores[idx] / counts[idx] if counts[idx] > 0 else 0
                    print(f"    Candidate {idx}: score={score:.3f}, avg={avg_score:.3f}")

                except Exception as e:
                    print(f"    Candidate {idx}: evaluation failed ({e})")
                    counts[idx] += 1  # 仍然计数，避免重复选择

        # 返回最终结果 (参考UCBBandits.get_scores)
        final_scores = np.divide(scores, counts, out=np.zeros_like(scores), where=counts > 0)

        # 选择top-k候选
        beam_size = self.config.get('beam_size', 3)
        top_indices = np.argsort(final_scores)[::-1][:beam_size]
        selected_candidates = [candidates[i] for i in top_indices]

        print(f"  UCB final scores: {[f'{final_scores[i]:.3f}' for i in top_indices]}")
        print(f"  Selected {len(selected_candidates)} candidates")

        return selected_candidates

    def _thompson_sampling(self, candidates, task, training_data):
        """Thompson采样策略 """
        print(f"Thompson Sampling: evaluating {len(candidates)} candidates")

        # 基于Beta分布的Thompson采样
        candidate_scores = []
        eval_sample_size = min(10, len(training_data))
        eval_data = random.sample(training_data, eval_sample_size)

        for i, candidate in enumerate(candidates):
            try:
                # 评估候选
                score, _, _, _ = task.evaluate_prompt(
                    candidate, eval_data, None, n=eval_sample_size
                )

                # Thompson采样：从Beta分布采样
                alpha = 1 + score * 10  # 成功次数
                beta = 1 + (1 - score) * 10  # 失败次数
                sampled_score = np.random.beta(alpha, beta)

                candidate_scores.append((sampled_score, candidate))
                print(f"  Candidate {i + 1}: score={score:.3f}, sampled={sampled_score:.3f}")

            except Exception as e:
                print(f"  Candidate {i + 1}: failed ({e})")
                candidate_scores.append((0.0, candidate))

        # 按采样分数排序，返回top-k
        candidate_scores.sort(reverse=True, key=lambda x: x[0])
        beam_size = self.config.get('beam_size', 3)
        return [candidate for _, candidate in candidate_scores[:beam_size]]

    def _epsilon_greedy_selection(self, candidates, task, training_data):
        """Epsilon贪心策略"""
        epsilon = 0.1  # 探索率
        print(f"Epsilon-Greedy (ε={epsilon}): evaluating {len(candidates)} candidates")

        beam_size = self.config.get('beam_size', 3)

        if random.random() < epsilon:
            # 探索：随机选择
            print("  Using exploration (random selection)")
            return random.sample(candidates, min(beam_size, len(candidates)))
        else:
            # 利用：贪心选择最佳
            print("  Using exploitation (greedy selection)")
            candidate_scores = []
            eval_sample_size = min(8, len(training_data))
            eval_data = random.sample(training_data, eval_sample_size)

            for i, candidate in enumerate(candidates):
                try:
                    score, _, _, _ = task.evaluate_prompt(
                        candidate, eval_data, None, n=eval_sample_size
                    )
                    candidate_scores.append((score, candidate))
                    print(f"  Candidate {i + 1}: score={score:.3f}")
                except Exception as e:
                    print(f"  Candidate {i + 1}: failed ({e})")
                    candidate_scores.append((0.0, candidate))

            # 返回top-k候选
            candidate_scores.sort(reverse=True, key=lambda x: x[0])
            return [candidate for _, candidate in candidate_scores[:beam_size]]


class AgentManager:
    """
    Multi-Agent Manager for MAPGD Framework

    Orchestrates collaboration between 4 specialized agents:
    - instruction_specialist: Optimizes task instructions
    - example_curator: Manages few-shot examples
    - format_designer: Structures output formats
    - style_optimizer: Refines language style

    Implements parallel gradient generation and agent synchronization
    """

    def __init__(self, config):
        self.config = config
        self.num_agents = config.get('num_agents', 4)
        # 🔧 修复：确保正确获取任务名称
        self.task_name = config.get('task_name', 'default')
        print(f"🔍 AgentManager initialized for task: {self.task_name}")  # 添加调试信息

        self.synchronization_strategy = config.get('sync_strategy', 'best_prompt_sharing')
        self.agents = []
        self.async_mode = config.get('async_mode', True)
        self.max_concurrent_agents = config.get('max_concurrent_agents', 4)

    def initialize_agents(self, initial_prompt):
        """初始化智能体 (根据任务类型动态选择专业化角色)"""

        # 根据任务获取可用的智能体类型
        available_agent_types = SpecializedPromptAgent.get_available_agent_types(self.task_name)

        random_roles = self.config.get("random_agent_roles", False)
        mode = "random" if random_roles else "specialized"

        print(f"Initializing {self.num_agents} {mode} agents for task '{self.task_name}'...")
        print(f"Available agent types: {available_agent_types}")

        for i in range(self.num_agents):
            if random_roles:
                # 随机选择一个角色
                agent_type = random.choice(available_agent_types)
            else:
                # 默认逻辑：循环分配角色
                agent_type = available_agent_types[i % len(available_agent_types)]

            # 创建智能体时传入任务名称
            agent = SpecializedPromptAgent(agent_type, task_name=self.task_name)
            agent.current_prompt = initial_prompt
            self.agents.append(agent)

            print(f"  Agent {i + 1}: {agent_type} - {agent.config['role_description']}")

        print(
            f"Agent initialization complete for {self.task_name}. Ready for {'async' if self.async_mode else 'sync'} optimization.")

    async def generate_gradients_async(self, training_data, task, predictor):
        """
        Asynchronous Parallel Gradient Generation - Core Performance Optimization
        All agents perform parallel analysis simultaneously, avoiding serial waiting:
        1. Parallel data sampling and error analysis
        2. Parallel gradient generation
        3. Unified result collection
        """

        print(f"Starting gradient generation from {len(self.agents)} agents...")
        start_time = time.time()

        async def generate_single_agent_gradient(agent_idx, agent):
            """单个智能体的异步梯度生成任务"""
            try:
                print(f"  Agent {agent_idx + 1} ({agent.agent_type}): starting analysis...")

                # 为每个智能体采样不同的数据子集，增加探索多样性
                sample_size = min(20, len(training_data))
                start_idx = (agent_idx * 10) % len(training_data)
                end_idx = min(start_idx + sample_size, len(training_data))

                if end_idx - start_idx < sample_size:
                    sample_data = random.sample(training_data, min(sample_size, len(training_data)))
                else:
                    sample_data = training_data[start_idx:end_idx]

                # 评估当前提示性能，获取错误样本
                score, texts, labels, preds = task.evaluate_prompt(
                    agent.current_prompt, sample_data, predictor, n=len(sample_data)
                )

                # 构造错误样本
                error_examples = []
                for text, label, pred in zip(texts, labels, preds):
                    if label != pred:
                        error_examples.append({
                            'text': text,
                            'true_label': label,
                            'predicted_label': pred
                        })
                # 异步生成梯度
                gradients = await agent.generate_specialized_gradient_async(
                    agent.current_prompt, error_examples, task
                )

                print(f"  Agent {agent_idx + 1} completed: {len(gradients)} gradients")
                return f'agent_{agent_idx}', gradients

            except Exception as e:
                print(f"  Agent {agent_idx + 1} failed: {e}")
                return f'agent_{agent_idx}', [f"Improve the prompt for better {agent.config['focus']}"]

        # 并行执行所有智能体的梯度生成
        tasks = [
            generate_single_agent_gradient(i, agent)
            for i, agent in enumerate(self.agents)
        ]

        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 整理结果
        agent_gradients = {}
        successful_agents = 0

        for result in results:
            if isinstance(result, tuple):
                agent_id, gradients = result
                agent_gradients[agent_id] = gradients
                successful_agents += 1
            else:
                print(f"  Task failed with exception: {result}")

        total_time = time.time() - start_time
        print(f"gradient generation completed:")
        print(f"  - {successful_agents}/{len(self.agents)} agents successful")
        print(f"  - Total time: {total_time:.2f}s")
        print(f"  - Average time per agent: {total_time / len(self.agents):.2f}s")

        return agent_gradients

    def generate_gradients(self, training_data, task, predictor):
        """
        智能体梯度生成入口 - 支持同步/异步模式切换

        根据配置选择执行模式：
        - async_mode=True: 并行异步执行所有智能体
        - async_mode=False: 串行同步执行（兼容模式）
        """

        if self.async_mode:
            # 异步模式：并行执行
            print("Using ASYNC mode for gradient generation...")
            return asyncio.run(self.generate_gradients_async(training_data, task, predictor))
        else:
            # 同步模式：串行执行（保持原有逻辑）
            print("Using SYNC mode for gradient generation...")
            return self._generate_gradients_sync(training_data, task, predictor)

    def _generate_gradients_sync(self, training_data, task, predictor):
        """
        同步版本梯度生成 - 保持向后兼容性
        """
        agent_gradients = {}

        print(f"Generating specialized gradients from {len(self.agents)} agents...")

        for i, agent in enumerate(self.agents):
            print(f"  Agent {i + 1} ({agent.agent_type}): analyzing prompt performance...")

            # Sample training data for agent evaluation
            sample_size = min(20, len(training_data))
            sample_data = random.sample(training_data, sample_size)

            # 使用当前提示评估性能，获取错误样本
            try:
                score, texts, labels, preds = task.evaluate_prompt(
                    agent.current_prompt, sample_data, predictor, n=len(sample_data)
                )

                # 构造错误样本
                error_examples = []
                for text, label, pred in zip(texts, labels, preds):
                    if label != pred:
                        error_examples.append({
                            'text': text,
                            'true_label': label,
                            'predicted_label': pred
                        })

                # 如果没有错误样本，随机选择一些样本
                if not error_examples:
                    error_examples = [
                        {
                            'text': ex.get('text', ''),
                            'true_label': ex.get('label', ''),
                            'predicted_label': ex.get('label', '')
                        }
                        for ex in sample_data[:3]
                    ]

                # 生成梯度
                gradients = agent.generate_specialized_gradient(
                    agent.current_prompt, error_examples, task
                )

                agent_gradients[f'agent_{i}'] = gradients

            except Exception as e:
                print(f"Error generating gradients for agent {i}: {e}")
                agent_gradients[f'agent_{i}'] = [f"Improve the prompt for better {agent.config['focus']}"]

        return agent_gradients

    def get_current_prompts(self):
        """获取当前提示"""
        return [agent.current_prompt for agent in self.agents]

    def update_agents(self, selected_prompts):
        """更新智能体状态"""
        for i, agent in enumerate(self.agents):
            if i < len(selected_prompts):
                agent.current_prompt = selected_prompts[i]


class CollaborativePromptExpander:
    """
    Collaborative Prompt Expansion for MAPGD

    Implements prompt space exploration through:
    1. Beam Search: Systematic expansion guided by semantic gradients
    2. Monte Carlo Paraphrasing: Stochastic variations for diversity
    3. Gradient-Prompt Fusion: Direct application of textual gradients
    4. Diversity Maximization: Ensuring exploration coverage

    Key innovation: Combines deterministic beam search with stochastic
    Monte Carlo sampling for balanced exploration-exploitation
    """

    def __init__(self, config):
        self.beam_size = config.get('beam_size', 3)
        self.mc_samples = config.get('mc_samples', 2)  # Monte Carlo samples per prompt
        self.diversity_threshold = config.get('diversity_threshold', 0.7)
        self.expansion_strategy = config.get('expansion_strategy',
                                             'beam_search')  # 'beam_search', 'monte_carlo', 'hybrid'

        # 使用共享的语义模型
        self.semantic_model = MAPGDUtils.get_semantic_model()
        if self.semantic_model:
            print(f"CollaborativePromptExpander: Using shared semantic model for diversity filtering")
        else:
            print(f"Warning: No semantic model available for diversity filtering")

    def expand(self, current_prompt, gradients):
        """
        Collaborative prompt expansion pipeline - Strategy-based approach:

        Strategy options:
        - 'beam_search': Systematic gradient application (deterministic)
        - 'monte_carlo': Stochastic paraphrasing variations (exploratory)
        - 'hybrid': Balanced combination of both approaches

        Returns: Expanded set of candidate prompts based on selected strategy
        """

        print(f"Expanding prompts with strategy: {self.expansion_strategy}")

        candidates = []

        if self.expansion_strategy == 'beam_search':
            # Strategy 1: Pure beam search with gradient application
            candidates = self._beam_search_expansion(current_prompt, gradients)
            print(f"Beam search strategy generated {len(candidates)} candidates")

        elif self.expansion_strategy == 'monte_carlo':
            # Strategy 2: Pure Monte Carlo paraphrasing
            candidates = self._monte_carlo_paraphrasing(current_prompt)
            print(f"Monte Carlo strategy generated {len(candidates)} variants")

        diverse_candidates = self._filter_for_diversity(candidates)
        print(f"Filtered to {len(diverse_candidates)} diverse candidates")

        return diverse_candidates[:self.beam_size * 2]  # Return reasonable number of candidates

    def _beam_search_expansion(self, current_prompt, gradients):
        """Systematic beam search guided by semantic gradients"""
        beam_candidates = []

        for gradient in gradients:
            # Apply gradient to generate new candidate
            expanded_prompt = self._apply_gradient(current_prompt, gradient)
            for i in range(len(expanded_prompt)):
                beam_candidates.append(expanded_prompt[i])

        return beam_candidates

    def _monte_carlo_paraphrasing(self, base_prompt):
        """Stochastic Monte Carlo paraphrasing for diversity"""
        mc_variants = []

        # Limit MC to top prompts
        for _ in range(self.mc_samples):
            variant = self._generate_stochastic_variant(base_prompt)
            mc_variants.append(variant[0])

        return mc_variants

    def _apply_gradient(self, prompt, gradient, steps_per_gradient=1):
        """Apply semantic gradient to prompt via LLM"""
        application_prompt = f"""
        I'm trying to write a zero-shot classifier.
        My current prompt is:
        "{prompt}"
        the gradient of this prompt is {gradient}
        Based on the above information, I wrote{steps_per_gradient} different improved prompts.
        The {steps_per_gradient} new prompts are wrapped with <START> and <END>:
        """

        response = call_openai(
            prompt=application_prompt,
            system_prompt="You are a professional expert in prompt word optimization, skilled in applying semantic gradients for precise improvements."
        )

        return MAPGDUtils.parse_gradient_response(response)

    def _generate_stochastic_variant(self, prompt):
        """Generate stochastic Monte Carlo variant"""
        # paraphrase_styles = [
        #     "更简洁清晰的表达方式",
        #     "更正式专业的语言风格",
        #     "更具体详细的指令描述",
        #     "更友好易懂的表达方式"
        # ]

        # style = random.choice(paraphrase_styles)

        variant_prompt = f"""
        Generate a variation of the following instruction while keeping the semantic meaning.

        Input: {prompt}
        Each output prompt is wrapped with <START> and <END>.
        """

        response = call_openai(
            prompt=variant_prompt,
            system_prompt=""
        )

        return MAPGDUtils.parse_gradient_response(response)

    def _filter_for_diversity(self, candidates):
        """Filter candidates for diversity using shared semantic tools"""
        return MAPGDUtils.filter_for_diversity(
            candidates,
            diversity_threshold=self.diversity_threshold,
            max_candidates=self.beam_size * 2
        )


class ConvergenceMonitor:
    """收敛监控器"""

    def __init__(self, config):
        self.convergence_threshold = config.get('convergence_threshold', 0.01)
        self.patience = config.get('patience', 3)
        self.performance_history = []
        self.no_improvement_count = 0

    def check_convergence(self, current_performance=None):
        """检查是否收敛"""
        if current_performance is not None:
            self.performance_history.append(current_performance)

        if len(self.performance_history) < 2:
            return False

        # 检查性能提升
        recent_improvement = (
                self.performance_history[-1] - self.performance_history[-2]
        )

        if recent_improvement < self.convergence_threshold:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0

        return self.no_improvement_count >= self.patience
