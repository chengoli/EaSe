"""
EaSe (Entropy and Semantic subjectivity)算法实现
基于论文: "EaSe: A Diagnostic Tool for VQA Based on Answer Diversity"
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

# 加载FastText词向量
def load_vectors(path):
    """
    加载预训练的FastText词向量模型
    """
    print(f"正在加载词向量模型: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到词向量文件: {path}")
    
    model = fasttext.load_model(path)
    print("词向量模型加载完成!")
    return model

def get_embedding(text, model):
    """
    获取文本的词向量表示
    """
    # 预处理文本
    text = text.lower()
    # 去除标点和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    
    # 使用FastText获取词向量
    embedding = model.get_sentence_vector(text)
    return embedding

def centroid_vector(words, model):
    """
    计算多个答案的中心词向量
    """
    if not words:
        return None
    
    embeddings = []
    for word in words:
        embedding = get_embedding(word, model)
        embeddings.append(embedding)
    
    # 计算平均向量
    centroid = np.mean(embeddings, axis=0)
    return centroid

def compute_similarity(word, centroid, model):
    """
    计算一个词与中心向量的余弦相似度
    """
    word_embedding = get_embedding(word, model)
    # 将向量重塑为2D数组用于余弦相似度计算
    word_embedding = word_embedding.reshape(1, -1)
    centroid = centroid.reshape(1, -1)
    
    sim = cosine_similarity(word_embedding, centroid)[0][0]
    # 将负相似度值设为0
    sim = max(0, sim)
    return sim

def compute_ease_score(answers, model):
    """
    计算EaSe分数
    
    参数:
    - answers: 一个问题的所有答案列表
    - model: FastText词向量模型
    
    返回:
    - ease_score: 一个介于0到1之间的值，值越高表示样本越容易
    """
    if not answers:
        return 0
    
    # 计算答案频率分布
    answer_counter = Counter(answers)
    
    # 获取唯一答案
    unique_answers = list(answer_counter.keys())
    
    # 计算中心向量
    centroid = centroid_vector(unique_answers, model)
    if centroid is None:
        return 0
    
    # 找到最常见的答案
    most_common_answer, max_freq = answer_counter.most_common(1)[0]
    
    # 计算最常见答案与中心向量的相似度
    max_sim = compute_similarity(most_common_answer, centroid, model)
    
    # 设置动态阈值
    eps = 0.0001
    threshold = max_sim - eps
    
    # 将答案分为高相似度组和低相似度组
    high_sim_answers = []
    low_sim_answers = []
    
    for ans, freq in answer_counter.items():
        sim = compute_similarity(ans, centroid, model)
        
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
    probs = [count / total for count in new_distribution]
    ent = entropy(probs, base=2)
    
    # 归一化熵（假设最大熵为log2(10)，因为VQA通常有10个答案）
    max_entropy = np.log2(len(new_distribution)) if len(new_distribution) > 0 else 0
    norm_entropy = ent / 2.302  # 论文中使用的标准化值
    
    # 计算EaSe分数 = 1 - 归一化熵
    ease_score = 1 - norm_entropy
    
    return ease_score

def process_vqa_dataset(annotations_path, questions_path, model, output_dir=None):
    """
    处理VQA数据集，计算每个样本的EaSe分数
    
    参数:
    - annotations_path: 标注文件路径
    - questions_path: 问题文件路径
    - model: FastText词向量模型
    - output_dir: 输出目录，如果提供则保存结果
    
    返回:
    - ease_scores: 问题ID到EaSe分数的映射
    - splits: 三个难度级别的问题ID列表 (TH, BH, E)
    """
    print(f"正在处理文件: {annotations_path} 和 {questions_path}")
    
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
        
        # 提取答案
        answers = [a['answer'].lower() for a in anno['answers']]
        
        # 计算EaSe分数
        ease_score = compute_ease_score(answers, model)
        ease_scores[qid] = ease_score
        
        # 根据EaSe分数分类
        if ease_score < 0.5:
            th_ids.append(qid)
        elif ease_score < 1.0:
            bh_ids.append(qid)
        else:
            e_ids.append(qid)
    
    splits = {
        'TH': th_ids,  # TOP-HARD
        'BH': bh_ids,  # BOTTOM-HARD
        'E': e_ids     # EASY
    }
    
    print(f"TOP-HARD样本数: {len(th_ids)} ({len(th_ids) / len(annotations) * 100:.2f}%)")
    print(f"BOTTOM-HARD样本数: {len(bh_ids)} ({len(bh_ids) / len(annotations) * 100:.2f}%)")
    print(f"EASY样本数: {len(e_ids)} ({len(e_ids) / len(annotations) * 100:.2f}%)")
    
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

def main():
    # 参数设置
    word2vec_path = '/path/to/wiki-news-300d-1M-subword.vec'  # 替换为实际路径
    train_anno_path = '/path/to/v2_mscoco_train2014_annotations.json'
    train_ques_path = '/path/to/v2_OpenEnded_mscoco_train2014_questions.json'
    val_anno_path = '/path/to/v2_mscoco_val2014_annotations.json'
    val_ques_path = '/path/to/v2_OpenEnded_mscoco_val2014_questions.json'
    output_dir = './EaSe_results'
    
    # 加载词向量模型
    model = load_vectors(word2vec_path)
    
    # 处理训练集
    print("\n处理训练集...")
    train_ease_scores, train_splits = process_vqa_dataset(
        train_anno_path, train_ques_path, model, 
        output_dir=os.path.join(output_dir, 'train')
    )
    
    # 处理验证集
    print("\n处理验证集...")
    val_ease_scores, val_splits = process_vqa_dataset(
        val_anno_path, val_ques_path, model,
        output_dir=os.path.join(output_dir, 'val')
    )
    
    print("\n处理完成！结果已保存到", output_dir)

if __name__ == "__main__":
    main()
