"""
MAPGD Framework Utilities

共享工具模块，包含所有重复使用的功能：
- 语义模型管理（单例模式）
- 文本处理和向量化
- 相似度计算
- LLM响应解析
"""

import numpy as np
import re
from sentence_transformers import SentenceTransformer
import re


import argparse

# 在 utils.py 中的 load_config 函数更新：

def load_config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--task',
        type=str,
        default='liar',
        choices=['liar', 'ethos', 'jailbreak', 'gsm8k', 'aqua', 'svamp'],  # 添加 'gsm8k'
        help="Configuration source to use (liar, ethos, jailbreak, gsm8k, aqua or svamp)"
    )
    args, _ = parser.parse_known_args()
    config_source = args.task or 'liar'

    if config_source == 'liar':
        from liar_config import DATA_CONFIG, EXPERIMENT_CONFIG
        print("Using liar_config for DATA_CONFIG and EXPERIMENT_CONFIG")
    elif config_source == 'ethos':
        from ethos_config import DATA_CONFIG, EXPERIMENT_CONFIG
        print("Using ethos_config for DATA_CONFIG and EXPERIMENT_CONFIG")
    elif config_source == 'jailbreak':
        from jailbreak_config import DATA_CONFIG, EXPERIMENT_CONFIG
        print("Using jailbreak_config for DATA_CONFIG and EXPERIMENT_CONFIG")
    elif config_source == 'gsm8k':  # 添加这个分支
        from gsm8k_config import DATA_CONFIG, EXPERIMENT_CONFIG
        print("Using gsm8k_config for DATA_CONFIG and EXPERIMENT_CONFIG")
    elif config_source == 'aqua':  # ⭐ 新增 AQuA 分支
        from aqua_config import DATA_CONFIG, EXPERIMENT_CONFIG
        print("Using aqua_config for DATA_CONFIG and EXPERIMENT_CONFIG")
    elif config_source == 'svamp':  # ✅ 添加这个分支
        from svamp_config import DATA_CONFIG, EXPERIMENT_CONFIG
        print("Using svamp_config for DATA_CONFIG and EXPERIMENT_CONFIG")
    else:
        from liar_config import DATA_CONFIG, EXPERIMENT_CONFIG
        print(f"Unknown config_source '{config_source}', falling back to liar_config")

    return DATA_CONFIG, EXPERIMENT_CONFIG

# Optional GPU support - graceful fallback if torch unavailable
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None
    GPU_AVAILABLE = False

from utils import load_config  # 如果你把 load_config 放在 utils.py


# 或者
# from utils.config_loader import load_config  # 如果你用文件夹结构

class SemanticModelManager:
    """语义模型管理器 - 单例模式"""

    _instance = None
    _semantic_model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self):
        """获取共享的语义模型实例"""
        if self._semantic_model is None:
            self._semantic_model = self._load_semantic_model()
        return self._semantic_model

    def _load_semantic_model(self):
        """加载语义模型"""
        try:
            DATA_CONFIG, _ = load_config()  # ✅ 动态获取 DATA_CONFIG
            if torch is not None:
                device = 'cuda' if GPU_AVAILABLE else 'cpu'
                model = SentenceTransformer(
                    DATA_CONFIG['sentence_transformer_model'],
                    device=device
                )
                print(f"SemanticModelManager: Loaded SentenceTransformer on {device}")
                return model
        except Exception as e:
            print(f"Warning: Failed to load SentenceTransformer: {e}")
            return None


class TextUtils:
    """文本处理工具类"""
    
    @staticmethod
    def extract_gradient_text(gradient):
        """提取梯度文本内容"""
        if isinstance(gradient, str):
            return gradient
        elif isinstance(gradient, list):
            return " ".join(gradient)
        elif isinstance(gradient, dict):
            return gradient.get('content', str(gradient))
        else:
            return str(gradient)
    
    @staticmethod
    def parse_llm_response_with_tags(response, tag_pattern):
        """解析带标签的LLM响应"""
        matches = re.findall(tag_pattern, response, re.DOTALL|re.IGNORECASE)
        if matches:
            return [match.strip() for match in matches]
        else:
            return [response.strip()]
    
    @staticmethod
    def format_conflict_info(conflicts):
        """格式化冲突信息"""
        if not conflicts:
            return "无冲突"
        
        conflict_texts = []
        for conflict in conflicts:
            sim_score = conflict.get('sim', conflict.get('similarity', 0))
            conflict_texts.append(
                f"智能体 {conflict['agents'][0]} 和 {conflict['agents'][1]} 的建议存在冲突 (相似度: {sim_score:.2f})"
            )
        
        return "\n".join(conflict_texts)


class VectorUtils:
    """向量处理工具类"""
    
    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        """计算余弦相似度"""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)
    
    @staticmethod
    def encode_texts(texts, fallback_dim=384):
        """批量编码文本为语义向量"""
        model_manager = SemanticModelManager()
        model = model_manager.get_model()
        
        if model is not None:
            return model.encode(texts, convert_to_numpy=True)
        else:
            # 如果没有模型，返回随机向量作为占位符
            print(f"Warning: Using random vectors as fallback for {len(texts)} texts")
            return np.random.rand(len(texts), fallback_dim)  # MiniLM-L6-v2 的向量维度
    
    @staticmethod
    def batch_cosine_similarity_matrix(vectors_a, vectors_b):
        """批量计算余弦相似度矩阵"""
        # 归一化向量
        norm_a = np.linalg.norm(vectors_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(vectors_b, axis=1, keepdims=True)
        
        # 避免除零
        norm_a = np.where(norm_a == 0, 1, norm_a)
        norm_b = np.where(norm_b == 0, 1, norm_b)
        
        normalized_a = vectors_a / norm_a
        normalized_b = vectors_b / norm_b
        
        # 计算相似度矩阵
        return normalized_a @ normalized_b.T


class ParsingUtils:
    """解析工具类"""
    
    @staticmethod
    def parse_gradient_response(response):
        """解析梯度响应"""
        pattern = r'<\s*START\s*>(.*?)<\s*END\s*>'
        results = TextUtils.parse_llm_response_with_tags(response, pattern)
        return results if results else [response.strip()]
    
    @staticmethod
    def parse_fusion_response(response):
        """解析融合响应"""
        pattern = r'<\s*START\s*>(.*?)<\s*END\s*>'
        results = TextUtils.parse_llm_response_with_tags(response, pattern)
        return results[0]


class DiversityFilter:
    """多样性过滤工具"""
    
    @staticmethod
    def filter_by_semantic_similarity(candidates, diversity_threshold=0.7, max_candidates=None):
        """基于语义相似度的多样性过滤"""
        if not candidates or len(candidates) == 1:
            return candidates
        
        diverse_candidates = [candidates[0]]  # 总是包含第一个候选
        
        # 获取语义模型
        model_manager = SemanticModelManager()
        model = model_manager.get_model()
        
        if model is not None:
            print(f"Using semantic similarity for diversity filtering...")
            
            # 批量编码候选文本
            candidate_embeddings = VectorUtils.encode_texts(candidates)
            
            for i, candidate in enumerate(candidates[1:], 1):
                candidate_embedding = candidate_embeddings[i]
                
                # 检查与已选择候选的语义相似度
                is_diverse = True
                for existing_candidate in diverse_candidates:
                    existing_idx = candidates.index(existing_candidate)
                    existing_embedding = candidate_embeddings[existing_idx]
                    
                    # 计算余弦相似度
                    similarity = VectorUtils.cosine_similarity(candidate_embedding, existing_embedding)
                    
                    if similarity > diversity_threshold:
                        is_diverse = False
                        break
                
                if is_diverse:
                    diverse_candidates.append(candidate)
                    
                # 限制候选数量
                if max_candidates and len(diverse_candidates) >= max_candidates:
                    break
                        
            print(f"Semantic diversity filtering: {len(diverse_candidates)}/{len(candidates)} candidates retained")
        else:
            print(f"No semantic model, using simple diversity filtering...")
            # 简单的去重过滤
            for candidate in candidates[1:]:
                if max_candidates and len(diverse_candidates) >= max_candidates:
                    break
                # 简单检查：避免完全相同的候选
                if candidate not in diverse_candidates:
                    diverse_candidates.append(candidate)
        
        return diverse_candidates


# 为了向后兼容，提供一个统一的工具接口
class MAPGDUtils:
    """MAPGD框架统一工具接口"""
    
    # 语义模型相关
    @staticmethod
    def get_semantic_model():
        """获取共享的语义模型"""
        return SemanticModelManager().get_model()
    
    # 文本处理相关
    @staticmethod
    def extract_gradient_text(gradient):
        """提取梯度文本"""
        return TextUtils.extract_gradient_text(gradient)
    
    @staticmethod
    def parse_llm_response_with_tags(response, tag_pattern):
        """解析LLM响应"""
        return TextUtils.parse_llm_response_with_tags(response, tag_pattern)
    
    @staticmethod
    def format_conflicts(conflicts):
        """格式化冲突信息"""
        return TextUtils.format_conflict_info(conflicts)
    
    # 向量计算相关
    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        """计算余弦相似度"""
        return VectorUtils.cosine_similarity(vec_a, vec_b)
    
    @staticmethod
    def encode_texts(texts):
        """编码文本"""
        return VectorUtils.encode_texts(texts)
    
    # 解析相关
    @staticmethod
    def parse_gradient_response(response):
        """解析梯度响应"""
        return ParsingUtils.parse_gradient_response(response)
    
    @staticmethod
    def parse_fusion_response(response):
        """解析融合响应"""
        return ParsingUtils.parse_fusion_response(response)
    
    # 多样性过滤相关
    @staticmethod
    def filter_for_diversity(candidates, diversity_threshold=0.7, max_candidates=None):
        """多样性过滤"""
        return DiversityFilter.filter_by_semantic_similarity(
            candidates, diversity_threshold, max_candidates
        )
