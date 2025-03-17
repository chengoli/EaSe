"""
VQA数据集的EaSe评分分析和可视化
基于论文: "EaSe: A Diagnostic Tool for VQA Based on Answer Diversity"
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
import os
from tqdm import tqdm

def analyze_question_types(questions_path, annotations_path, ease_scores_path, output_dir=None):
    """
    分析不同问题类型与EaSe分数的关系
    
    参数:
    - questions_path: 问题文件路径
    - annotations_path: 标注文件路径
    - ease_scores_path: EaSe分数文件路径
    - output_dir: 输出目录
    """
    # 加载问题
    with open(questions_path, 'r') as f:
        questions = json.load(f)['questions']
    
    # 加载标注
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)['annotations']
    
    # 加载EaSe分数
    with open(ease_scores_path, 'r') as f:
        ease_scores = json.load(f)
    
    # 创建问题ID到问题的映射
    qid_to_question = {q['question_id']: q['question'] for q in questions}
    
    # 定义问题类型的关键词
    question_types = {
        'what': [],
        'where': [],
        'when': [],
        'who': [],
        'why': [],
        'how': [],
        'yes_no': [],
        'number': []
    }
    
    # 分析每个问题
    for anno in annotations:
        qid = anno['question_id']
        q_text = qid_to_question[qid].lower()
        ease_score = ease_scores.get(str(qid), ease_scores.get(qid, 0))
        
        # 根据首词或内容判断问题类型
        if q_text.startswith('what'):
            question_types['what'].append(ease_score)
        elif q_text.startswith('where'):
            question_types['where'].append(ease_score)
        elif q_text.startswith('when'):
            question_types['when'].append(ease_score)
        elif q_text.startswith('who'):
            question_types['who'].append(ease_score)
        elif q_text.startswith('why'):
            question_types['why'].append(ease_score)
        elif q_text.startswith('how'):
            question_types['how'].append(ease_score)
        elif q_text.startswith('is') or q_text.startswith('are') or q_text.startswith('can') or q_text.startswith('does'):
            question_types['yes_no'].append(ease_score)
        elif any(word in q_text for word in ['many', 'much', 'count', 'number']):
            question_types['number'].append(ease_score)
        else:
            # 对于其他类型的问题，可以进一步分析或归类为其他
            pass
    
    # 计算每种问题类型的平均EaSe分数
    type_stats = {}
    for q_type, scores in question_types.items():
        if scores:
            type_stats[q_type] = {
                'count': len(scores),
                'mean_ease': np.mean(scores),
                'median_ease': np.median(scores),
                'std_ease': np.std(scores)
            }
    
    # 创建分布图
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 绘制平均EaSe分数的条形图
        plt.figure(figsize=(12, 6))
        types = list(type_stats.keys())
        means = [type_stats[t]['mean_ease'] for t in types]
        
        sns.barplot(x=types, y=means)
        plt.title('不同问题类型的平均EaSe分数')
        plt.xlabel('问题类型')
        plt.ylabel('平均EaSe分数')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'question_type_ease.png'))
        
        # 为每种问题类型绘制EaSe分数分布图
        plt.figure(figsize=(12, 8))
        for i, (q_type, scores) in enumerate(question_types.items()):
            if scores:
                plt.subplot(3, 3, i+1)
                sns.histplot(scores, kde=True, bins=20)
                plt.title(f'{q_type} (n={len(scores)})')
                plt.xlabel('EaSe分数')
                plt.ylabel('频率')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'question_type_distributions.png'))
    
    return type_stats

def analyze_confidence_ease_correlation(annotations_path, ease_scores_path, output_dir=None):
    """
    分析人类信心分数与EaSe分数之间的相关性
    
    参数:
    - annotations_path: 标注文件路径
    - ease_scores_path: EaSe分数文件路径
    - output_dir: 输出目录
    """
    # 加载标注
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)['annotations']
    
    # 加载EaSe分数
    with open(ease_scores_path, 'r') as f:
        ease_scores = json.load(f)
    
    # 提取信心分数和EaSe分数
    confidence_ease_pairs = []
    
    for anno in annotations:
        qid = anno['question_id']
        ease_score = ease_scores.get(str(qid), ease_scores.get(qid, 0))
        
        # 获取标注者的信心分数（如果有）
        confidence_scores = []
        for answer in anno['answers']:
            if 'answer_confidence' in answer:
                # 将文本信心转换为数值
                conf_text = answer['answer_confidence']
                if conf_text == 'yes':
                    confidence_scores.append(1.0)
                elif conf_text == 'maybe':
                    confidence_scores.append(0.5)
                elif conf_text == 'no':
                    confidence_scores.append(0.0)
        
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            confidence_ease_pairs.append((avg_confidence, ease_score))
    
    # 转换为DataFrame
    df = pd.DataFrame(confidence_ease_pairs, columns=['confidence', 'ease'])
    
    # 计算相关性
    correlation = df.corr(method='spearman').loc['confidence', 'ease']
    
    # 创建散点图
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        # 添加一些抖动以更好地显示分布
        sns.regplot(x='confidence', y='ease', data=df, scatter_kws={'alpha': 0.3})
        plt.title(f'信心分数与EaSe分数的关系 (Spearman ρ = {correlation:.3f})')
        plt.xlabel('平均信心分数')
        plt.ylabel('EaSe分数')
        plt.savefig(os.path.join(output_dir, 'confidence_ease_correlation.png'))
        
        # 分组绘制箱线图
        plt.figure(figsize=(10, 6))
        df['confidence_bin'] = pd.cut(df['confidence'], bins=[0, 0.25, 0.5, 0.75, 1.0], 
                                       labels=['[0-0.25)', '[0.25-0.5)', '[0.5-0.75)', '[0.75-1.0]'])
        sns.boxplot(x='confidence_bin', y='ease', data=df)
        plt.title('不同信心水平下的EaSe分数分布')
        plt.xlabel('信心分数区间')
        plt.ylabel('EaSe分数')
        plt.savefig(os.path.join(output_dir, 'confidence_ease_boxplot.png'))
    
    return correlation, df

def analyze_model_performance_by_ease(predictions_path, ease_scores_path, output_dir=None):
    """
    分析模型在不同EaSe分数区间的表现
    
    参数:
    - predictions_path: 模型预测文件路径
    - ease_scores_path: EaSe分数文件路径
    - output_dir: 输出目录
    """
    # 加载模型预测
    with open(predictions_path, 'r') as f:
        predictions = json.load(f)
    
    # 加载EaSe分数
    with open(ease_scores_path, 'r') as f:
        ease_scores = json.load(f)
    
    # 将预测结果转换为问题ID到正确率的映射
    qid_to_accuracy = {}
    
    # 预测格式可能因模型而异，此处使用通用格式
    if isinstance(predictions, list):
        for pred in predictions:
            if 'question_id' in pred and 'score' in pred:
                qid_to_accuracy[pred['question_id']] = pred['score']
    elif isinstance(predictions, dict):
        qid_to_accuracy = predictions
    
    # 整合EaSe分数和模型正确率
    ease_accuracy_pairs = []
    
    for qid, ease_score in ease_scores.items():
        if str(qid) in qid_to_accuracy or qid in qid_to_accuracy:
            accuracy = qid_to_accuracy.get(str(qid), qid_to_accuracy.get(qid, 0))
            ease_accuracy_pairs.append((ease_score, accuracy))
    
    # 转换为DataFrame
    df = pd.DataFrame(ease_accuracy_pairs, columns=['ease', 'accuracy'])
    
    # 计算相关性
    correlation = df.corr(method='spearman').loc['ease', 'accuracy']
    
    # 按EaSe分数分组计算平均准确率
    df['ease_bin'] = pd.cut(df['ease'], bins=10)
    grouped = df.groupby('ease_bin')['accuracy'].mean().reset_index()
    
    # 创建可视化
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 散点图
        plt.figure(figsize=(10, 6))
        sns.regplot(x='ease', y='accuracy', data=df, scatter_kws={'alpha': 0.3})
        plt.title(f'EaSe分数与模型准确率的关系 (Spearman ρ = {correlation:.3f})')
        plt.xlabel('EaSe分数')
        plt.ylabel('模型准确率')
        plt.savefig(os.path.join(output_dir, 'ease_accuracy_correlation.png'))
        
        # 分组条形图
        plt.figure(figsize=(12, 6))
        sns.barplot(x='ease_bin', y='accuracy', data=grouped)
        plt.title('不同EaSe分数区间的平均模型准确率')
        plt.xlabel('EaSe分数区间')
        plt.ylabel('平均准确率')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ease_accuracy_grouped.png'))
    
    return correlation, df

def main():
    # 参数设置
    questions_path = '/path/to/v2_OpenEnded_mscoco_val2014_questions.json'
    annotations_path = '/path/to/v2_mscoco_val2014_annotations.json'
    ease_scores_path = './EaSe_results/val/ease_scores.json'
    predictions_path = '/path/to/model_predictions.json'
    output_dir = './analysis_results'
    
    # 分析问题类型
    print("分析问题类型与EaSe分数的关系...")
    type_stats = analyze_question_types(
        questions_path, annotations_path, ease_scores_path, 
        output_dir=os.path.join(output_dir, 'question_types')
    )
    
    # 分析信心-EaSe相关性
    print("分析信心分数与EaSe分数的相关性...")
    correlation, df = analyze_confidence_ease_correlation(
        annotations_path, ease_scores_path,
        output_dir=os.path.join(output_dir, 'confidence')
    )
    print(f"信心分数与EaSe分数的Spearman相关系数: {correlation:.3f}")
    
    # 分析模型表现
    if os.path.exists(predictions_path):
        print("分析模型表现与EaSe分数的关系...")
        correlation, df = analyze_model_performance_by_ease(
            predictions_path, ease_scores_path,
            output_dir=os.path.join(output_dir, 'model_performance')
        )
        print(f"EaSe分数与模型准确率的Spearman相关系数: {correlation:.3f}")
    
    print("分析完成！结果已保存到", output_dir)

if __name__ == "__main__":
    main()
