"""
MAPGD任务处理器 - 专门处理Liar数据集
用于虚假信息检测的二分类任务
"""
import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
class MAPGDDataProcessor(ABC):
    """MAPGD数据处理器基类，兼容baseline项目格式"""
    
    def __init__(self, data_dir, max_threads=1):
        self.data_dir = data_dir
        self.max_threads = max_threads

    @abstractmethod
    def get_train_examples(self):
        pass

    @abstractmethod
    def get_test_examples(self):
        pass

    @abstractmethod
    def evaluate_prompt(self, prompt, test_examples, predictor, n=100):
        pass

    @abstractmethod
    def stringify_prediction(self, pred):
        pass

class GSM8kTask:
    def __init__(self, data_path):
        self.data = self._load_data(data_path)

    def _load_data(self, data_path):
        # 加载 GSM8k 数据集 (json, csv 或其他格式)
        import json
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_prompt(self, prompt, examples, predictor, n=50):
        """使用 LLM + prompt 在 GSM8k 上做推理，并计算准确率"""
        correct = 0
        for ex in examples[:n]:
            question, answer = ex["question"], ex["answer"]

            # 调用预测器生成模型推理结果
            prediction = predictor.predict(prompt, question)

            # 简单匹配答案 (可以加更复杂的数字解析逻辑)
            if str(answer).strip() in prediction:
                correct += 1

        accuracy = correct / n
        return accuracy, correct, n, prediction



class MAPGDClassificationTask(MAPGDDataProcessor):
    """分类任务处理器"""
    
    def evaluate_prompt(self, prompt, test_examples, predictor, n=150):
        """评估提示词在测试集上的性能"""
        labels = []
        preds = []
        texts = []
        
        # 限制测试样本数量
        #test_sample = test_examples[:n] if len(test_examples) > n else test_examples
        test_sample = np.random.choice(test_examples, n, replace=False) if len(test_examples) > n else test_examples
        for ex in test_sample:
            try:
                pred = predictor.inference(ex, prompt)
                texts.append(ex['text'])
                labels.append(ex['label'])
                preds.append(pred)
            except Exception as e:
                # 如果预测失败，跳过这个样本
                continue
        
        if not labels:
            return 0.0, texts, labels, preds
            
        # 计算F1分数
        f1 = f1_score(labels, preds, average='micro')
        return f1, texts, labels, preds


class MAPGDBinaryClassificationTask(MAPGDClassificationTask):
    """二分类任务"""
    categories = ['No', 'Yes']

    def stringify_prediction(self, pred):
        return MAPGDBinaryClassificationTask.categories[pred]


class MAPGDLiarTask(MAPGDBinaryClassificationTask):
    """Liar数据集任务 - 专门用于虚假信息检测"""

    def __init__(self, data_dir=None):
        if data_dir is None:
            # 默认用项目目录下的 data/hf_binary
            data_dir = os.path.join(os.path.dirname(__file__), "data/hf_binary")
        super().__init__(data_dir)
        
    def get_train_examples(self):
        """加载训练数据"""
        try:
            exs = []
            with open(f'{self.data_dir}/train.jsonl', 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    row = json.loads(line.strip())
                    exs.append({'id': f'train-{i}', 'label': row['label'], 'text': row['text']})
            print(f"Loaded {len(exs)} training examples from liar dataset")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load Liar train data: {e}")
    
    def get_test_examples(self):
        """加载测试数据"""
        try:
            exs = []
            with open(f'{self.data_dir}/test.jsonl', 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    row = json.loads(line.strip())
                    exs.append({'id': f'test-{i}', 'label': row['label'], 'text': row['text']})
            print(f"Loaded {len(exs)} test examples from liar dataset")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load Liar test data: {e}")

class MAPGDJailbreakTask(MAPGDBinaryClassificationTask):
    """Jailbreak 数据集任务 (tsv 格式: [{"role": "user", "text": "xxx"}]\tlabel)"""

    def __init__(self, data_dir=None):
        """
        data_dir: 包含 train.tsv 和 test.tsv 的目录
        """
        data_dir = os.path.join(os.path.dirname(__file__), "data/jailbreak")
        super().__init__(data_dir)

    import json
    import os

    def _load_tsv(self, filepath, split_name):
        exs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    print(f"Skipping line {i}: empty line")
                    continue
                try:
                    # 尝试用 \t 或 \\t 分割
                    if "\t" in line:
                        text_json, label = line.rsplit("\t", 1)
                    elif "\\t" in line:
                        text_json, label = line.rsplit("\\t", 1)
                    else:
                        print(f"Skipping line {i}: no tab separator found: {repr(line)}")
                        continue
                    text_list = json.loads(text_json)
                    if not isinstance(text_list, list) or len(text_list) == 0:
                        print(f"Skipping line {i}: invalid JSON list: {text_json}")
                        continue
                    if "text" not in text_list[0]:
                        print(f"Skipping line {i}: missing 'text' field in JSON: {text_json}")
                        continue
                    text = text_list[0]["text"]
                    exs.append({
                        'id': f'{split_name}-{i}',
                        'label': int(label),
                        'text': text
                    })
                except Exception as e:
                    print(f"Skipping line {i} due to error: {e}, line: {repr(line)}")
                    continue
        print(f"Loaded {len(exs)} {split_name} examples from jailbreak dataset")
        return exs

    def get_train_examples(self):
        return self._load_tsv(os.path.join(self.data_dir, "train.tsv"), "train")

    def get_test_examples(self):
        return self._load_tsv(os.path.join(self.data_dir, "test.tsv"), "test")


class MAPGDEthosBinaryTask(MAPGDBinaryClassificationTask):
    """
    ETHOS Hate Speech 二分类任务
    数据格式: text ; hate_score
    """

    data_dir = os.path.join(os.path.dirname(__file__), "data/ethos")
    categories = ['No', 'Yes']

    def _load_data(self):
        # 读取 CSV
        df = pd.read_csv(os.path.join(self.data_dir, 'Ethos_Dataset_Binary.csv'),
                         sep=';', header=None, names=['text', 'score'])
        # 过滤掉 0 < score < 0.7 的样本，只保留极端样本
        df = df[(df['score'] <= 0) | (df['score'] >= 0.7)].reset_index(drop=True)
        return df

    def get_train_examples(self):
        df = self._load_data()
        exs = df.reset_index().to_dict('records')
        # 后面的样本作为训练集
        exs = [{'id': x['index'], 'text': x['text'], 'label': 1 if x['score'] > 0.4 else 0}
               for x in exs[200:]]
        print(f"Loaded {len(exs)} training examples")
        return exs

    def get_test_examples(self):
        df = self._load_data()
        exs = df.reset_index().to_dict('records')
        # 前 200 个样本作为测试集
        exs = [{'id': x['index'], 'text': x['text'], 'label': 1 if x['score'] > 0.4 else 0}
               for x in exs[:200]]
        print(f"Loaded {len(exs)} test examples")
        return exs


import re
import pandas as pd

# 在 mapgd_tasks.py 文件末尾添加以下代码：

import re
import pandas as pd


class MAPGDGsm8kTask(MAPGDDataProcessor):
    """
    GSM8k 数学推理任务
    数据格式: parquet文件，包含question和answer字段
    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data/gsm8k")
        super().__init__(data_dir)
        self.answer_pattern = re.compile(r'#### (-?\d+(?:\.\d+)?)')

    def _load_parquet_data(self, filename):
        """加载parquet文件"""
        filepath = os.path.join(self.data_dir, filename)
        try:
            df = pd.read_parquet(filepath)
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load {filename}: {e}")

    def _extract_numerical_answer(self, answer_text):
        """从答案文本中提取数值答案"""
        # GSM8k答案格式: "...explanation... #### 72"
        match = self.answer_pattern.search(answer_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _extract_prediction_answer(self, prediction_text):
        """从模型预测中提取数值答案（更鲁棒的版本）"""
        # 优先匹配 #### 格式
        match = self.answer_pattern.search(prediction_text)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass  # 如果转换失败，继续尝试其他方法

        # 如果 #### 格式失败，查找文本中所有的数字
        # 移除模型可能输出的逗号，例如 "1,000"
        prediction_text = prediction_text.replace(',', '')
        # 查找所有整数和浮点数
        numbers = re.findall(r'-?\d+\.\d+|-?\d+', prediction_text)

        if numbers:
            try:
                # 返回最后一个找到的数字，这在数学推理任务中通常是最终答案
                return float(numbers[-1])
            except (ValueError, IndexError):
                return None  # 如果最后一个元素不是有效数字

        return None  # 如果没有找到任何数字

    def get_train_examples(self):
        """加载训练数据 - 限制为200个样本"""
        try:
            df = self._load_parquet_data("train.parquet")
            exs = []
            for i, row in df.iterrows():
                answer_num = self._extract_numerical_answer(row['answer'])
                if answer_num is not None:  # 只保留能解析答案的样本
                    exs.append({
                        'id': f'train-{i}',
                        'text': row['question'],
                        'answer': row['answer'],
                        'numerical_answer': answer_num
                    })
                    # 限制训练样本数量为200
                    if len(exs) >= 200:
                        break
            print(f"Loaded {len(exs)} training examples from GSM8k dataset (limited to 200)")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load GSM8k train data: {e}")

    def get_test_examples(self):
        """加载测试数据 - 限制为100个样本"""
        try:
            df = self._load_parquet_data("test.parquet")
            exs = []
            for i, row in df.iterrows():
                answer_num = self._extract_numerical_answer(row['answer'])
                if answer_num is not None:  # 只保留能解析答案的样本
                    exs.append({
                        'id': f'test-{i}',
                        'text': row['question'],
                        'answer': row['answer'],
                        'numerical_answer': answer_num
                    })
                    # 限制测试样本数量为100
                    if len(exs) >= 100:
                        break
            print(f"Loaded {len(exs)} test examples from GSM8k dataset (limited to 100)")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load GSM8k test data: {e}")

    def evaluate_prompt(self, prompt, test_examples, predictor, n=50):
        """评估数学推理任务的性能"""
        correct = 0
        total = 0
        texts = []
        true_answers = []
        pred_answers = []

        # 限制测试样本数量 (数学推理比较耗时)
        test_sample = np.random.choice(test_examples, n, replace=False) if len(test_examples) > n else test_examples

        for ex in test_sample:
            try:
                # 调用预测器进行推理
                prediction_text = predictor.inference(ex, prompt)

                # 提取预测的数值答案
                pred_answer = self._extract_prediction_answer(prediction_text)
                true_answer = ex['numerical_answer']

                texts.append(ex['text'])
                true_answers.append(true_answer)
                pred_answers.append(pred_answer)

                # 检查答案是否正确 (允许小的浮点数误差)
                if pred_answer is not None and abs(pred_answer - true_answer) < 1e-6:
                    correct += 1

                total += 1

            except Exception as e:
                print(f"Error processing example {ex['id']}: {e}")
                continue

        accuracy = correct / total if total > 0 else 0.0
        print(f"GSM8k Accuracy: {correct}/{total} = {accuracy:.4f}")

        return accuracy, texts, true_answers, pred_answers

    def stringify_prediction(self, pred):
        """字符串化预测结果"""
        if pred is None:
            return "No answer"
        return str(pred)


"""
将以下代码添加到 mapgd_tasks.py 文件中
位置: 在 MAPGDGsm8kTask 类之后, get_mapgd_task_class 函数之前
"""




class MAPGDAquaTask(MAPGDDataProcessor):
    """
    AQuA-RAT 数学推理选择题任务
    数据格式: Parquet文件, 包含question, options, rationale, correct字段
    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data/aqua")
        super().__init__(data_dir)
        self.answer_pattern = re.compile(r'Answer\s*:\s*([A-E])', re.IGNORECASE)

    # 修改：将 _load_json_data 替换为 _load_parquet_data
    def _load_parquet_data(self, filename):
        """加载Parquet文件"""
        filepath = os.path.join(self.data_dir, filename)
        try:
            # 使用pandas读取parquet文件
            df = pd.read_parquet(filepath, engine='pyarrow')
            # 将DataFrame转换为字典列表，方便后续处理
            # to_dict('records') 会将每一行转为一个字典
            data = df.to_dict('records')
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load {filepath}: {e}")

    def _format_options(self, options):
        """格式化选项列表为字符串"""
        # Parquet中的options字段可能是字符串列表，逻辑保持不变
        return "\n".join(options)

    def _extract_answer_from_prediction(self, prediction_text):
        """从模型预测中提取答案字母"""
        # 此函数逻辑与文件格式无关，保持不变
        match = self.answer_pattern.search(prediction_text)
        if match:
            return match.group(1).upper()

        letters = re.findall(r'\b([A-E])\b', prediction_text.upper())
        if letters:
            return letters[-1]

        return None

    def get_train_examples(self):
        """加载训练数据 - 限制为200个样本"""
        try:
            # 修改：调用新的加载函数并使用.parquet扩展名
            data = self._load_parquet_data("train.parquet")
            exs = []
            for i, item in enumerate(data):
                exs.append({
                    'id': f'train-{i}',
                    'text': item['question'],
                    'options': item['options'],
                    'rationale': item.get('rationale', ''),  # 使用.get保证健壮性
                    'correct': item['correct']
                })
                if len(exs) >= 200:
                    break
            print(f"Loaded {len(exs)} training examples from AQuA dataset (limited to 200)")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load AQuA train data: {e}")

    def get_test_examples(self):
        """加载测试数据 - 限制为100个样本"""
        try:
            # 修改：调用新的加载函数并使用.parquet扩展名
            data = self._load_parquet_data("test.parquet")
            exs = []
            for i, item in enumerate(data):
                exs.append({
                    'id': f'test-{i}',
                    'text': item['question'],
                    'options': item['options'],
                    'rationale': item.get('rationale', ''),  # 使用.get保证健壮性
                    'correct': item['correct']
                })
                if len(exs) >= 100:
                    break
            print(f"Loaded {len(exs)} test examples from AQuA dataset (limited to 100)")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load AQuA test data: {e}")

    # evaluate_prompt 和 stringify_prediction 方法不需要修改，因为它们处理的是已经加载到内存中的数据
    # 所以这里省略了这两个方法的代码，你可以保持原样

    def evaluate_prompt(self, prompt, test_examples, predictor, n=50):
        """评估数学推理选择题任务的性能"""
        correct = 0
        total = 0
        texts = []
        true_answers = []
        pred_answers = []

        # 限制测试样本数量
        test_sample = np.random.choice(test_examples, n, replace=False).tolist() if len(test_examples) > n else test_examples

        for ex in test_sample:
            try:
                # 调用预测器进行推理
                prediction_text = predictor.inference(ex, prompt)
                pred_answer = self._extract_answer_from_prediction(prediction_text)
                true_answer = ex['correct']

                texts.append(ex['text'])
                true_answers.append(true_answer)
                pred_answers.append(pred_answer)

                if pred_answer is not None and pred_answer == true_answer:
                    correct += 1
                total += 1
            except Exception as e:
                print(f"Error processing example {ex.get('id', 'N/A')}: {e}")
                continue

        accuracy = correct / total if total > 0 else 0.0
        print(f"AQuA Accuracy: {correct}/{total} = {accuracy:.4f}")

        return accuracy, texts, true_answers, pred_answers

    def stringify_prediction(self, pred):
        """字符串化预测结果"""
        if pred is None:
            return "No answer"
        return str(pred)


import os
import re
import json
import numpy as np

class MAPGDSvampTask(MAPGDDataProcessor):
    """
    SVAMP数学推理任务
    数据格式: JSON文件，包含 Body, Question, Answer 等字段
    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data/svamp")
        super().__init__(data_dir)
        self.answer_pattern = re.compile(r'#### (-?\d+(?:\.\d+)?)')

    def _load_json_data(self, filename):
        """加载 JSON 文件（数组格式）"""
        filepath = os.path.join(self.data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)   # 直接读取整个数组
            return data
        except Exception as e:
            raise RuntimeError(f"Failed to load {filename}: {e}")

    def _extract_prediction_answer(self, prediction_text):
        """从模型预测中提取数值答案"""
        # 优先匹配 #### 格式
        match = self.answer_pattern.search(prediction_text)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass

        # 如果 #### 格式失败，查找文本中所有的数字
        prediction_text = prediction_text.replace(',', '')
        numbers = re.findall(r'-?\d+\.\d+|-?\d+', prediction_text)

        if numbers:
            try:
                return float(numbers[-1])
            except (ValueError, IndexError):
                return None

        return None

    def get_train_examples(self):
        """加载训练数据 - 限制为200个样本"""
        try:
            data = self._load_json_data("train.json")
            exs = []
            for i, item in enumerate(data):
                exs.append({
                    'id': item.get('ID', f'train-{i}'),
                    'text': item['Body'],
                    'question': item['Question'],
                    'answer': item['Answer'],
                    'numerical_answer': float(item['Answer']),
                    'type': item.get('Type', 'Unknown')
                })
                if len(exs) >= 200:
                    break
            print(f"Loaded {len(exs)} training examples from SVAMP dataset (limited to 200)")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load SVAMP train data: {e}")

    def get_test_examples(self):
        """加载测试数据 - 限制为100个样本"""
        try:
            data = self._load_json_data("test.json")
            exs = []
            for i, item in enumerate(data):
                exs.append({
                    'id': item.get('ID', f'test-{i}'),
                    'text': item['Body'],
                    'question': item['Question'],
                    'answer': item['Answer'],
                    'numerical_answer': float(item['Answer']),
                    'type': item.get('Type', 'Unknown')
                })
                if len(exs) >= 100:
                    break
            print(f"Loaded {len(exs)} test examples from SVAMP dataset (limited to 100)")
            return exs
        except Exception as e:
            raise RuntimeError(f"Failed to load SVAMP test data: {e}")

    def evaluate_prompt(self, prompt, test_examples, predictor, n=50):
        """评估数学推理任务的性能"""
        correct = 0
        total = 0
        texts = []
        true_answers = []
        pred_answers = []

        test_sample = np.random.choice(test_examples, n, replace=False) if len(test_examples) > n else test_examples

        for ex in test_sample:
            try:
                # 构造完整的问题文本 - 组合Body和Question
                ex_with_question = ex.copy()
                ex_with_question['text'] = f"{ex['text']} {ex['question']}"

                # 调用预测器进行推理
                prediction_text = predictor.inference(ex_with_question, prompt)

                # 提取预测的数值答案
                pred_answer = self._extract_prediction_answer(prediction_text)
                true_answer = ex['numerical_answer']

                texts.append(f"{ex['text']} {ex['question']}")
                true_answers.append(true_answer)
                pred_answers.append(pred_answer)

                # 检查答案是否正确
                if pred_answer is not None and abs(pred_answer - true_answer) < 1e-6:
                    correct += 1

                total += 1

            except Exception as e:
                print(f"Error processing example {ex['id']}: {e}")
                continue

        accuracy = correct / total if total > 0 else 0.0
        print(f"SVAMP Accuracy: {correct}/{total} = {accuracy:.4f}")

        return accuracy, texts, true_answers, pred_answers

    def stringify_prediction(self, pred):
        """字符串化预测结果"""
        if pred is None:
            return "No answer"
        return str(pred)
# ===== 更新 get_mapgd_task_class 函数 =====
# 将原有的 get_mapgd_task_class 函数替换为以下版本:

def get_mapgd_task_class(task_name):
    """获取MAPGD任务类"""
    if task_name == 'liar':
        return MAPGDLiarTask
    elif task_name == 'jailbreak':
        return MAPGDJailbreakTask
    elif task_name == 'ethos':
        return MAPGDEthosBinaryTask
    elif task_name == 'gsm8k':
        return MAPGDGsm8kTask
    elif task_name == 'aqua':  # 新增 AQuA 分支
        return MAPGDAquaTask
    elif task_name == 'svamp':  # ✅ 添加这一行
        return MAPGDSvampTask
    else:
        raise ValueError(f'Unsupported task: {task_name}')
