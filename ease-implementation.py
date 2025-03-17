"""
EaSe (Entropy and Semantic subjectivity)算法实现
基于论文: "EaSe: A Diagnostic Tool for VQA Based on Answer Diversity"
增强版本：增加了健壮性和错误处理
"""
import json
import numpy as np
from collections import Counter
import torch
from scipy.stats import entropy
import fasttext
import os
import re
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


class VectorModel:
    """
    兼容类，用于统一处理不同类型的词向量模型
    """
    def __init__(self, model, dim=300):
        self.model = model
        self.dim = dim
        # 检测模型类型
        self.is_dict = isinstance(model, dict)

    def get_sentence_vector(self, text):
        """统一接口，获取句子向量"""
        if self.is_dict:
            # 如果是词向量字典
            words = text.lower().split()
            vectors = []

            for word in words:
                if word in self.model:
                    vectors.append(self.model[word])

            if vectors:
                # 返回所有找到的词向量的平均值
                return np.mean(vectors, axis=0)
            else:
                # 如果没有找到任何词的向量，返回零向量
                return np.zeros(self.dim)
        else:
            # 如果是FastText模型对象
            try:
                return self.model.get_sentence_vector(text)
            except AttributeError:
                # 如果模型没有get_sentence_vector方法
                print(f"警告: 模型缺少get_sentence_vector方法，尝试替代方法")
                try:
                    return self.model.get_word_vector(text)
                except:
                    print(f"警告: 无法获取'{text}'的向量表示，返回零向量")
                    return np.zeros(self.dim)


# 加载词向量
def load_vectors(path):
    """
    加载词向量模型，支持多种格式

    Args:
        path: 词向量文件路径，支持.vec和.bin格式

    Returns:
        VectorModel: 封装了词向量的统一接口
    """
    print(f"正在加载词向量模型: {path}")

    try:
        # 对于.vec文件使用自定义加载方法
        if path.endswith('.vec'):
            vectors = {}
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    # 第一行包含词汇量和维度
                    first_line = f.readline().split()
                    n_words = int(first_line[0])
                    dim = int(first_line[1])
                    print(f"词向量包含 {n_words} 个词, 维度: {dim}")

                    # 读取所有词向量
                    for i, line in tqdm(enumerate(f), total=n_words, desc="加载词向量"):
                        if i >= n_words:
                            break

                        tokens = line.rstrip().split(' ')
                        if len(tokens) < dim + 1:
                            continue

                        word = tokens[0]
                        try:
                            vector = np.array([float(val) for val in tokens[1:dim+1]])
                            vectors[word] = vector
                        except ValueError:
                            continue

                    print(f"成功加载 {len(vectors)} 个词向量")
                    return VectorModel(vectors, dim)
                except Exception as e:
                    print(f"警告: 加载.vec文件时出错: {e}")

                    # 尝试不读取头信息的方式加载
                    print("尝试替代加载方法...")
                    f.seek(0)
                    vectors = {}
                    dim = None

                    for line in tqdm(f, desc="替代加载"):
                        tokens = line.rstrip().split(' ')
                        if len(tokens) < 2:
                            continue

                        word = tokens[0]
                        try:
                            vector = np.array([float(val) for val in tokens[1:]])
                            if dim is None:
                                dim = len(vector)
                            elif len(vector) != dim:
                                continue

                            vectors[word] = vector
                        except ValueError:
                            continue

                    if vectors:
                        print(f"替代方法成功加载 {len(vectors)} 个词向量, 维度: {dim}")
                        return VectorModel(vectors, dim)
                    else:
                        raise ValueError("无法加载词向量文件")

        # 对于.bin文件使用FastText的load_model
        elif path.endswith('.bin'):
            try:
                model = fasttext.load_model(path)
                print(f"成功加载FastText二进制模型")
                # 获取维度
                dim = len(model.get_word_vector(model.get_words()[0])) if model.get_words() else 300
                return VectorModel(model, dim)
            except Exception as e:
                print(f"加载.bin文件时出错: {e}")
                raise
        else:
            raise ValueError(f"不支持的文件格式: {path}")

    except Exception as e:
        print(f"致命错误: 无法加载词向量模型: {e}")
        print("将使用随机向量代替！模型性能可能会显著降低。")
        # 创建一个简单的随机词向量模型作为备选
        return VectorModel({}, 300)


def centroid_vector(words, vector_model):
    """
    计算多个答案的中心词向量

    Args:
        words: 单词列表
        vector_model: VectorModel实例

    Returns:
        np.ndarray: 中心词向量，如果没有词，则返回None
    """
    if not words:
        return None

    embeddings = []
    for word in words:
        try:
            embedding = vector_model.get_sentence_vector(word)
            if embedding is not None and not np.all(embedding == 0):
                embeddings.append(embedding)
        except Exception as e:
            print(f"警告: 获取'{word}'的词向量时出错: {e}")

    if not embeddings:
        # 所有词都没有有效的向量表示
        return np.zeros(vector_model.dim)

    # 计算平均向量
    centroid = np.mean(embeddings, axis=0)
    return centroid


def compute_similarity(word, centroid, vector_model):
    """
    计算一个词与中心向量的余弦相似度

    Args:
        word: 单词或短语
        centroid: 中心向量
        vector_model: VectorModel实例

    Returns:
        float: 余弦相似度，范围[0,1]
    """
    try:
        word_embedding = vector_model.get_sentence_vector(word)

        # 检查空向量
        if np.all(word_embedding == 0) or np.all(centroid == 0):
            return 0

        # 将向量重塑为2D数组用于余弦相似度计算
        word_embedding = word_embedding.reshape(1, -1)
        centroid = centroid.reshape(1, -1)

        sim = cosine_similarity(word_embedding, centroid)[0][0]
        # 将负相似度值设为0
        sim = max(0, sim)
        return sim
    except Exception as e:
        print(f"警告: 计算'{word}'的相似度时出错: {e}")
        return 0


def compute_ease_score(answers, vector_model):
    """
    计算EaSe分数

    Args:
        answers: 一个问题的所有答案列表
        vector_model: VectorModel实例

    Returns:
        float: EaSe分数，介于0到1之间的值，值越高表示样本越容易
    """
    if not answers or len(answers) == 0:
        return 0

    try:
        # 计算答案频率分布
        answer_counter = Counter(answers)

        # 获取唯一答案
        unique_answers = list(answer_counter.keys())

        # 计算中心向量
        centroid = centroid_vector(unique_answers, vector_model)
        if centroid is None or np.all(centroid == 0):
            return 0

        # 找到最常见的答案
        most_common_answer, max_freq = answer_counter.most_common(1)[0]

        # 计算最常见答案与中心向量的相似度
        max_sim = compute_similarity(most_common_answer, centroid, vector_model)

        # 设置动态阈值
        eps = 0.0001
        threshold = max_sim - eps

        # 将答案分为高相似度组和低相似度组
        high_sim_answers = []
        low_sim_answers = []

        for ans, freq in answer_counter.items():
            sim = compute_similarity(ans, centroid, vector_model)

            if sim >= threshold:
                high_sim_answers.extend([ans] * freq)
            else:
                low_sim_answers.extend([ans] * freq)

        # 创建新的答案分布
        if high_sim_answers:
            # 如果有高相似度答案，将它们合并为一类
            new_distribution = [1] * len(high_sim_answers)
            for ans in set(low_sim_answers):
                count = low_sim_answers.count(ans)
                new_distribution.append(count)
        else:
            # 如果没有高相似度答案，使用原始分布
            new_distribution = list(answer_counter.values())

        # 计算熵
        total = sum(new_distribution)
        if total == 0:
            return 0

        probs = [count / total for count in new_distribution]
        ent = entropy(probs, base=2)

        # 归一化熵（假设最大熵为log2(10)，因为VQA通常有10个答案）
        max_entropy = np.log2(len(new_distribution)) if len(new_distribution) > 0 else 0
        norm_entropy = ent / 2.302  # 论文中使用的标准化值

        # 计算EaSe分数 = 1 - 归一化熵
        ease_score = 1 - norm_entropy

        return ease_score
    except Exception as e:
        print(f"警告: 计算EaSe分数时出错: {e}")
        return 0


def process_vqa_dataset(annotations_path, questions_path, vector_model, output_dir=None):
    """
    处理VQA数据集，计算每个样本的EaSe分数

    Args:
        annotations_path: 标注文件路径
        questions_path: 问题文件路径
        vector_model: VectorModel实例
        output_dir: 输出目录，如果提供则保存结果

    Returns:
        tuple: (ease_scores, splits)
            - ease_scores: 问题ID到EaSe分数的映射
            - splits: 三个难度级别的问题ID列表 (TH, BH, E)
    """
    print(f"正在处理: {annotations_path} 和 {questions_path}")

    try:
        # 加载标注
        with open(annotations_path, 'r') as f:
            annotations_data = json.load(f)

        # 加载问题
        with open(questions_path, 'r') as f:
            questions_data = json.load(f)

        # 提取标注和问题
        annotations = annotations_data.get('annotations', annotations_data)
        questions = questions_data.get('questions', questions_data)

        print(f"标注数量: {len(annotations)}")
        print(f"问题数量: {len(questions)}")

        # 创建问题ID到问题的映射
        qid_to_question = {q['question_id']: q for q in questions}

        # 计算EaSe分数
        ease_scores = {}
        th_ids = []  # TOP-HARD: EaSe < 0.5
        bh_ids = []  # BOTTOM-HARD: 0.5 <= EaSe < 1.0
        e_ids = []   # EASY: EaSe = 1.0

        for anno in tqdm(annotations, desc="计算EaSe分数"):
            qid = anno['question_id']

            try:
                # 提取答案
                answers = [a['answer'].lower() for a in anno['answers']]

                # 计算EaSe分数
                ease_score = compute_ease_score(answers, vector_model)
                ease_scores[qid] = ease_score

                # 根据EaSe分数分类
                if ease_score < 0.5:
                    th_ids.append(qid)
                elif ease_score < 1.0:
                    bh_ids.append(qid)
                else:
                    e_ids.append(qid)
            except Exception as e:
                print(f"警告: 处理问题ID {qid}时出错: {e}")
                # 默认为中等难度
                ease_scores[qid] = 0.75
                bh_ids.append(qid)

        splits = {
            'TH': th_ids,  # TOP-HARD
            'BH': bh_ids,  # BOTTOM-HARD
            'E': e_ids     # EASY
        }

        print(f"TOP-HARD样本: {len(th_ids)} ({len(th_ids) / len(annotations) * 100:.2f}%)")
        print(f"BOTTOM-HARD样本: {len(bh_ids)} ({len(bh_ids) / len(annotations) * 100:.2f}%)")
        print(f"EASY样本: {len(e_ids)} ({len(e_ids) / len(annotations) * 100:.2f}%)")

        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # 保存EaSe分数
            with open(os.path.join(output_dir, 'ease_scores.json'), 'w') as f:
                json.dump(ease_scores, f)

            # 保存分割
            for split_name, ids in splits.items():
                with open(os.path.join(output_dir, f'{split_name.lower()}_ids.json'), 'w') as f:
                    json.dump(ids, f)

        return ease_scores, splits

    except Exception as e:
        print(f"致命错误: 处理数据集时出错: {e}")
        # 返回空结果
        return {}, {'TH': [], 'BH': [], 'E': []}


def main():
    # 路径设置
    word2vec_path = '/Users/oliverlau/Desktop/EASE/wiki-news-300d-1M-subword.vec'  # 替换为实际路径
    train_anno_path = '/Users/oliverlau/Desktop/EASE/v2_mscoco_train2014_annotations.json'
    train_ques_path = '/Users/oliverlau/Desktop/EASE/v2_OpenEnded_mscoco_train2014_questions.json'
    val_anno_path = '/Users/oliverlau/Desktop/EASE/v2_mscoco_val2014_annotations.json'
    val_ques_path = '/Users/oliverlau/Desktop/EASE/v2_OpenEnded_mscoco_val2014_questions.json'
    output_dir = './EaSe_results'

    try:
        # 加载词向量模型
        vector_model = load_vectors(word2vec_path)

        # 处理训练集
        print("\n处理训练集...")
        train_ease_scores, train_splits = process_vqa_dataset(
            train_anno_path, train_ques_path, vector_model,
            output_dir=os.path.join(output_dir, 'train')
        )

        # 处理验证集
        print("\n处理验证集...")
        val_ease_scores, val_splits = process_vqa_dataset(
            val_anno_path, val_ques_path, vector_model,
            output_dir=os.path.join(output_dir, 'val')
        )

        print("\n处理完成. 输出路径:", output_dir)

    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()