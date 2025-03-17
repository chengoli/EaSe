"""
运行LXMERT模型训练的示例脚本
包括数据准备和模型训练的完整流程
"""

import os
import subprocess
from datetime import datetime

# 设置工作目录和所需文件的路径
WORK_DIR = "/path/to/your/working/directory"  # 修改为您的工作目录
DATA_DIR = os.path.join(WORK_DIR, "data")
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
LOGS_DIR = os.path.join(WORK_DIR, "logs")

# 创建所需目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# 日志文件
log_file = os.path.join(LOGS_DIR, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# 数据路径
VQA_DATA = {
    "train_questions": os.path.join(DATA_DIR, "v2_OpenEnded_mscoco_train2014_questions.json"),
    "train_annotations": os.path.join(DATA_DIR, "v2_mscoco_train2014_annotations.json"),
    "val_questions": os.path.join(DATA_DIR, "v2_OpenEnded_mscoco_val2014_questions.json"),
    "val_annotations": os.path.join(DATA_DIR, "v2_mscoco_val2014_annotations.json"),
}

FEATURE_DIR = os.path.join(DATA_DIR, "image_features")
EASE_RESULTS_DIR = os.path.join(OUTPUT_DIR, "EaSe_results")
LXMERT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "lxmert_ease_output")

# 步骤1: 计算EaSe分数和分割
def run_ease_computation():
    print("步骤1: 计算EaSe分数和数据分割...")
    
    # 构建命令
    command = f"""
    python semantic_subjectivity.py \\
        --word2vec {os.path.join(DATA_DIR, "wiki-news-300d-1M-subword.vec")} \\
        --train_annotations {VQA_DATA['train_annotations']} \\
        --train_questions {VQA_DATA['train_questions']} \\
        --val_annotations {VQA_DATA['val_annotations']} \\
        --val_questions {VQA_DATA['val_questions']} \\
        --output_dir {EASE_RESULTS_DIR}
    """
    
    # 运行命令
    print(command)
    try:
        subprocess.run(command, shell=True, check=True)
        print("EaSe计算完成！")
    except subprocess.CalledProcessError as e:
        print(f"EaSe计算失败: {e}")
        return False
    
    return True

# 步骤2: 训练LXMERT模型
def train_lxmert(strategy="HF"):
    print(f"步骤2: 使用{strategy}策略训练LXMERT模型...")
    
    # 构建命令
    command = f"""
    python fine_tuning_script.py \\
        --train_questions {VQA_DATA['train_questions']} \\
        --train_annotations {VQA_DATA['train_annotations']} \\
        --val_questions {VQA_DATA['val_questions']} \\
        --val_annotations {VQA_DATA['val_annotations']} \\
        --image_features_path {FEATURE_DIR} \\
        --th_ids_path {os.path.join(EASE_RESULTS_DIR, "train/th_ids.json")} \\
        --bh_ids_path {os.path.join(EASE_RESULTS_DIR, "train/bh_ids.json")} \\
        --e_ids_path {os.path.join(EASE_RESULTS_DIR, "train/e_ids.json")} \\
        --output_dir {os.path.join(LXMERT_OUTPUT_DIR, strategy)} \\
        --batch_size 16 \\
        --learning_rate 5e-5 \\
        --training_strategy {strategy} \\
        --num_train_epochs 3 \\
        --epochs_th 1 \\
        --epochs_bh 1 \\
        --epochs_e 1
    """
    
    # 运行命令
    print(command)
    try:
        with open(log_file, "a") as f:
            subprocess.run(command, shell=True, check=True, stdout=f, stderr=subprocess.STDOUT)
        print(f"LXMERT模型训练完成！日志保存在 {log_file}")
    except subprocess.CalledProcessError as e:
        print(f"LXMERT模型训练失败: {e}")
        return False
    
    return True

# 步骤3: 评估不同策略的效果
def evaluate_strategies():
    print("步骤3: 评估不同训练策略的效果...")
    
    # 运行分析脚本
    command = f"""
    python case_study_analysis.py \\
        --val_questions {VQA_DATA['val_questions']} \\
        --val_annotations {VQA_DATA['val_annotations']} \\
        --ease_scores_path {os.path.join(EASE_RESULTS_DIR, "val/ease_scores.json")} \\
        --predictions_hf {os.path.join(LXMERT_OUTPUT_DIR, "HF/predictions_best.json")} \\
        --predictions_all {os.path.join(LXMERT_OUTPUT_DIR, "ALL/predictions_best.json")} \\
        --predictions_th {os.path.join(LXMERT_OUTPUT_DIR, "TH/predictions_best.json")} \\
        --predictions_th_bh {os.path.join(LXMERT_OUTPUT_DIR, "TH+BH/predictions_best.json")} \\
        --output_dir {os.path.join(OUTPUT_DIR, "analysis")}
    """
    
    # 运行命令
    print(command)
    try:
        subprocess.run(command, shell=True, check=True)
        print("策略评估完成！")
    except subprocess.CalledProcessError as e:
        print(f"策略评估失败: {e}")
        return False
    
    return True

# 主流程
def main():
    print("开始VQA模型训练流程...")
    
    # 步骤1: 计算EaSe分数
    if not run_ease_computation():
        print("由于EaSe计算失败，流程终止。")
        return
    
    # 步骤2: 训练LXMERT模型（多种策略）
    strategies = ["HF", "ALL", "TH", "TH+BH"]
    for strategy in strategies:
        if not train_lxmert(strategy):
            print(f"{strategy}策略训练失败，继续下一个。")
    
    # 步骤3: 评估不同策略的效果
    if not evaluate_strategies():
        print("策略评估失败。")
    
    print("VQA模型训练流程完成！")

if __name__ == "__main__":
    main()
