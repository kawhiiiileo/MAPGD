"""
MAPGD预测器 - 兼容baseline项目的预测接口
基于ProTeGi项目的predictors.py进行适配
"""

from abc import ABC, abstractmethod
from llm import call_openai


class MAPGDPredictor(ABC):
    """MAPGD预测器基类"""
    
    def __init__(self, config):
        self.config = config
        self.temperature = config.get('temperature', 0.0)

    @abstractmethod
    def inference(self, example, prompt):
        """进行推理预测"""
        pass


class MAPGDMathPredictor(MAPGDPredictor):
    """数学推理预测器 - 专门用于GSM8k等数学任务"""

    def inference(self, example, prompt):
        """对单个数学问题进行推理预测"""
        try:
            # 将问题文本插入到提示词中
            if '{text}' in prompt:
                formatted_prompt = prompt.replace('{text}', example['text'])
            elif '{{text}}' in prompt:
                formatted_prompt = prompt.replace('{{text}}', example['text'])
            else:
                # 如果没有占位符，直接拼接
                formatted_prompt = f"{prompt}\n\nProblem: {example['text']}"

            # 调用LLM进行数学推理
            response = call_openai(
                prompt=formatted_prompt,
                system_prompt="You are a helpful assistant for solving math word problems. Show your work step by step.",
                model=self.config.get('model', "gpt-4"),
                # 数学推理需要更确定性
            )

            # 返回完整的推理过程文本 (让任务类负责提取数值答案)
            return response.strip()

        except Exception as e:
            print(f"Math prediction error: {e}")
            return "Unable to solve this problem."

    def batch_inference(self, examples, prompt, max_examples=None):
        """批量数学推理预测"""
        if max_examples:
            examples = examples[:max_examples]

        predictions = []
        for example in examples:
            pred = self.inference(example, prompt)
            predictions.append(pred)

        return predictions


# 更新 get_mapgd_predictor 函数



class MAPGDBinaryPredictor(MAPGDPredictor):
    """二分类预测器
    categories = ['No', 'Yes']
    """
    def inference(self, example, prompt):
        """对单个样本进行二分类预测"""
        try:
            # 将文本插入到提示词中
            if '{text}' in prompt:
                formatted_prompt = prompt.replace('{text}', example['text'])
            elif '{{text}}' in prompt:
                formatted_prompt = prompt.replace('{{text}}', example['text'])
            else:
                # 如果没有占位符，直接拼接
                formatted_prompt = f"{prompt}\n\nText: {example['text']}"
            formatted_prompt += "\nOnly Answer Yes or No:"
            
            # 调用LLM进行预测
            response = call_openai(
                prompt=formatted_prompt,
                system_prompt="You are a helpful assistant for text classification.",
                #model="deepseek-chat"
                model="gpt-4"
            )
            
            # 解析响应
            response = response.strip().upper()
            
            # 判断是否为正类
            if any(pos_word in response for pos_word in ['YES', 'POSITIVE', 'TRUE', '1']):
                return 1
            else:
                return 0
                
        except Exception as e:
            print(f"Prediction error: {e}")
            # 默认返回负类
            return 0
    
    def batch_inference(self, examples, prompt, max_examples=None):
        """批量预测"""
        if max_examples:
            examples = examples[:max_examples]
        
        predictions = []
        for example in examples:
            pred = self.inference(example, prompt)
            predictions.append(pred)
        
        return predictions


"""
将以下代码添加到 mapgd_predictors.py 文件中
位置: 在 MAPGDMathPredictor 类之后, get_mapgd_predictor 函数之前
"""


class MAPGDAquaPredictor(MAPGDPredictor):
    """AQuA-RAT 数学推理选择题预测器"""

    def inference(self, example, prompt):
        """对单个数学选择题进行推理预测"""
        try:
            # 格式化选项
            options_text = "\n".join(example['options'])

            # 将问题和选项插入到提示词中
            if '{text}' in prompt and '{options}' in prompt:
                formatted_prompt = prompt.replace('{text}', example['text'])
                formatted_prompt = formatted_prompt.replace('{options}', options_text)
            elif '{{text}}' in prompt and '{{options}}' in prompt:
                formatted_prompt = prompt.replace('{{text}}', example['text'])
                formatted_prompt = formatted_prompt.replace('{{options}}', options_text)
            else:
                # 如果没有占位符,直接拼接
                formatted_prompt = f"{prompt}\n\nProblem: {example['text']}\n\nOptions:\n{options_text}"

            # 调用LLM进行数学推理
            response = call_openai(
                prompt=formatted_prompt,
                system_prompt="You are a helpful assistant for solving math word problems with multiple choice. Show your work step by step and select the correct answer.",
                model=self.config.get('model', "gpt-4"),
            )

            # 返回完整的推理过程文本 (让任务类负责提取答案字母)
            return response.strip()

        except Exception as e:
            print(f"AQuA prediction error: {e}")
            return "Unable to solve this problem."

    def batch_inference(self, examples, prompt, max_examples=None):
        """批量数学选择题推理预测"""
        if max_examples:
            examples = examples[:max_examples]

        predictions = []
        for example in examples:
            pred = self.inference(example, prompt)
            predictions.append(pred)

        return predictions


# ===== 更新 get_mapgd_predictor 函数 =====
# 将原有的 get_mapgd_predictor 函数替换为以下版本:

def get_mapgd_predictor(task_type, config, categories=None):
    """获取MAPGD预测器"""
    if task_type == 'binary_classification':
        return MAPGDBinaryPredictor(config)
    elif task_type == 'math_reasoning':
        return MAPGDMathPredictor(config)
    elif task_type == 'aqua_reasoning':  # 新增 AQuA 预测器
        return MAPGDAquaPredictor(config)
    else:
        raise ValueError(f'Unsupported task type: {task_type}')
