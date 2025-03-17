"""
使用EaSe策略对LXMERT模型进行微调
实现基于论文中的HardFirst (HF)训练策略，先使用困难样本，再逐步加入更容易的样本
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import argparse
from transformers import LxmertTokenizer, LxmertForQuestionAnswering, AdamW
from transformers import get_linear_schedule_with_warmup
import logging
import random
from collections import Counter
import matplotlib.pyplot as plt

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lxmert_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="使用EaSe困难度指标微调LXMERT模型")
    
    # 数据路径
    parser.add_argument("--train_questions", default="/Users/oliverlau/Desktop/EASE/v2_OpenEnded_mscoco_train2014_questions.json", type=str)
    parser.add_argument("--train_annotations", default="/Users/oliverlau/Desktop/EASE/v2_mscoco_train2014_annotations.json", type=str)
    parser.add_argument("--val_questions", default="/Users/oliverlau/Desktop/EASE/v2_OpenEnded_mscoco_val2014_questions.json", type=str)
    parser.add_argument("--val_annotations", default="/Users/oliverlau/Desktop/EASE/v2_mscoco_val2014_annotations.json", type=str)
    #parser.add_argument("--image_features_path", default="", type=str)
    
    # EaSe分割文件
    parser.add_argument("--th_ids_path", default="./EaSe_results/train/th_ids.json", type=str)
    parser.add_argument("--bh_ids_path", default="./EaSe_results/train/bh_ids.json", type=str)
    parser.add_argument("--e_ids_path", default="./EaSe_results/train/e_ids.json", type=str)
    
    # 输出路径
    parser.add_argument("--output_dir", default="./lxmert_ease_output", type=str)
    
    # 训练参数
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--max_seq_length", default=20, type=int)
    
    # 训练策略
    parser.add_argument("--training_strategy", default="HF", type=str, 
                      choices=["HF", "ALL", "TH", "TH+BH"], 
                      help="HF: HardFirst策略，ALL: 所有数据，TH: 仅困难样本，TH+BH: 困难+中等难度样本")
    
    # 阶段训练参数（适用于HF策略）
    parser.add_argument("--epochs_th", default=2, type=int, help="阶段1：困难样本训练的轮数")
    parser.add_argument("--epochs_bh", default=2, type=int, help="阶段2：中等难度样本训练的轮数")
    parser.add_argument("--epochs_e", default=1, type=int, help="阶段3：简单样本训练的轮数")
    
    args = parser.parse_args()
    return args

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 加载并处理VQA数据
class VQADataset(Dataset):
    def __init__(self, questions_path, annotations_path, image_features_path, tokenizer, max_seq_length=20):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.image_features_path = image_features_path
        
        # 加载问题
        with open(questions_path, 'r') as f:
            questions_data = json.load(f)
            self.questions = questions_data['questions']
        
        # 加载标注
        with open(annotations_path, 'r') as f:
            annotations_data = json.load(f)
            self.annotations = annotations_data['annotations']
        
        # 确保问题和标注顺序一致
        self.qid_to_idx = {q['question_id']: i for i, q in enumerate(self.questions)}
        self.sorted_annotations = []
        for q in self.questions:
            for a in self.annotations:
                if a['question_id'] == q['question_id']:
                    self.sorted_annotations.append(a)
                    break
        
        assert len(self.questions) == len(self.sorted_annotations)
        
        # 构建答案词表（基于所有训练答案）
        self.ans2label = {}
        self.label2ans = []
        
        # 收集所有答案
        all_answers = []
        for ann in self.sorted_annotations:
            answers = [a['answer'] for a in ann['answers']]
            all_answers.extend(answers)
        
        # 获取最常见的答案
        answer_counter = Counter(all_answers)
        top_answers = [a[0] for a in answer_counter.most_common(3000)]  # 取前3000个答案
        
        # 创建答案映射
        self.ans2label = {a: i for i, a in enumerate(top_answers)}
        self.label2ans = top_answers
        
        logger.info(f"加载了 {len(self.questions)} 个问题和标注")
        logger.info(f"构建了包含 {len(self.label2ans)} 个答案的词表")
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        annotation = self.sorted_annotations[idx]
        
        # 问题ID和图像ID
        question_id = question['question_id']
        image_id = question['image_id']
        
        # 问题文本
        question_text = question['question']
        
        # 对问题进行tokenize处理
        inputs = self.tokenizer(
            question_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # 从图像特征文件中加载特征
        try:
            # 实际应用中，您需要根据您的特征存储格式修改此部分
            features_path = os.path.join(self.image_features_path, f"COCO_{image_id}.npz")
            if os.path.exists(features_path):
                npz_file = np.load(features_path)
                visual_feats = npz_file['features']
                visual_pos = npz_file['boxes']
            else:
                # 如果找不到特征，使用零张量代替
                visual_feats = np.zeros((36, 2048), dtype=np.float32)
                visual_pos = np.zeros((36, 4), dtype=np.float32)
                logger.warning(f"找不到图像特征: {features_path}")
        except Exception as e:
            logger.error(f"加载图像特征时出错 {image_id}: {str(e)}")
            visual_feats = np.zeros((36, 2048), dtype=np.float32)
            visual_pos = np.zeros((36, 4), dtype=np.float32)
        
        # 将可变长度特征裁剪或填充到固定长度
        visual_feats = torch.tensor(visual_feats, dtype=torch.float)
        visual_pos = torch.tensor(visual_pos, dtype=torch.float)
        
        # 获取最常见的答案
        answers = [a['answer'] for a in annotation['answers']]
        answer_counter = Counter(answers)
        most_common_answer = answer_counter.most_common(1)[0][0]
        
        # 将答案转换为标签
        if most_common_answer in self.ans2label:
            label = self.ans2label[most_common_answer]
        else:
            # 对于不在词表中的答案，使用一个特殊标签
            label = -100
        
        return {
            'question_id': question_id,
            'input_ids': inputs.input_ids.squeeze(0),
            'attention_mask': inputs.attention_mask.squeeze(0),
            'visual_feats': visual_feats,
            'visual_pos': visual_pos,
            'label': torch.tensor(label, dtype=torch.long)
        }

# 评估函数
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_qids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            visual_feats = batch['visual_feats'].to(device)
            visual_pos = batch['visual_pos'].to(device)
            labels = batch['label'].to(device)
            question_ids = batch['question_id']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_feats=visual_feats,
                visual_pos=visual_pos,
                labels=labels
            )
            
            logits = outputs.question_answering_score
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_qids.extend(question_ids)
    
    # 计算准确率
    correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l and l != -100)
    total = sum(1 for l in all_labels if l != -100)
    accuracy = correct / total if total > 0 else 0
    
    # 保存预测结果
    predictions = [{"question_id": qid, "predicted": pred} for qid, pred in zip(all_qids, all_preds)]
    
    return accuracy, predictions

# 训练函数
def train(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"训练轮次 {epoch}")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        visual_feats = batch['visual_feats'].to(device)
        visual_pos = batch['visual_pos'].to(device)
        labels = batch['label'].to(device)
        
        model.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_feats=visual_feats,
            visual_pos=visual_pos,
            labels=labels
        )
        
        loss = outputs.loss
        epoch_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = epoch_loss / len(dataloader)
    return avg_loss

# 实现HardFirst训练策略
def train_hard_first(args, model, train_dataset, val_dataset, device):
    # 加载EaSe分割的问题ID
    with open(args.th_ids_path, 'r') as f:
        th_ids = set(json.load(f))
    
    with open(args.bh_ids_path, 'r') as f:
        bh_ids = set(json.load(f))
    
    with open(args.e_ids_path, 'r') as f:
        e_ids = set(json.load(f))
    
    # 创建每个难度级别的数据集索引
    th_indices = [i for i, item in enumerate(train_dataset.questions) if item['question_id'] in th_ids]
    bh_indices = [i for i, item in enumerate(train_dataset.questions) if item['question_id'] in bh_ids]
    e_indices = [i for i, item in enumerate(train_dataset.questions) if item['question_id'] in e_ids]
    
    logger.info(f"TH样本: {len(th_indices)} ({len(th_indices)/len(train_dataset)*100:.2f}%)")
    logger.info(f"BH样本: {len(bh_indices)} ({len(bh_indices)/len(train_dataset)*100:.2f}%)")
    logger.info(f"E样本: {len(e_indices)} ({len(e_indices)/len(train_dataset)*100:.2f}%)")
    
    # 创建验证数据加载器
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 阶段1：仅用TH样本训练
    logger.info("阶段1: 使用TH样本训练")
    th_dataset = Subset(train_dataset, th_indices)
    th_dataloader = DataLoader(
        th_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(th_dataloader) * args.epochs_th
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )
    
    best_accuracy = 0
    losses_th = []
    
    for epoch in range(args.epochs_th):
        avg_loss = train(model, th_dataloader, optimizer, scheduler, device, epoch + 1)
        losses_th.append(avg_loss)
        
        # 评估
        accuracy, predictions = evaluate(model, val_dataloader, device)
        logger.info(f"TH训练后 轮次 {epoch + 1}/{args.epochs_th}: 损失 = {avg_loss:.4f}, 准确率 = {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 保存模型
            model_path = os.path.join(args.output_dir, "model_th_best.pt")
            torch.save(model.state_dict(), model_path)
            # 保存预测结果
            with open(os.path.join(args.output_dir, "predictions_th_best.json"), 'w') as f:
                json.dump(predictions, f)
    
    # 保存最后一个TH训练模型
    model_path = os.path.join(args.output_dir, "model_th_final.pt")
    torch.save(model.state_dict(), model_path)
    
    # 阶段2：使用TH+BH样本训练
    logger.info("阶段2: 使用TH+BH样本训练")
    th_bh_indices = th_indices + bh_indices
    th_bh_dataset = Subset(train_dataset, th_bh_indices)
    th_bh_dataloader = DataLoader(
        th_bh_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(th_bh_dataloader) * args.epochs_bh
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )
    
    losses_bh = []
    
    for epoch in range(args.epochs_bh):
        avg_loss = train(model, th_bh_dataloader, optimizer, scheduler, device, epoch + 1)
        losses_bh.append(avg_loss)
        
        # 评估
        accuracy, predictions = evaluate(model, val_dataloader, device)
        logger.info(f"TH+BH训练后 轮次 {epoch + 1}/{args.epochs_bh}: 损失 = {avg_loss:.4f}, 准确率 = {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 保存模型
            model_path = os.path.join(args.output_dir, "model_th_bh_best.pt")
            torch.save(model.state_dict(), model_path)
            # 保存预测结果
            with open(os.path.join(args.output_dir, "predictions_th_bh_best.json"), 'w') as f:
                json.dump(predictions, f)
    
    # 保存最后一个TH+BH训练模型
    model_path = os.path.join(args.output_dir, "model_th_bh_final.pt")
    torch.save(model.state_dict(), model_path)
    
    # 阶段3：使用全部样本训练
    logger.info("阶段3: 使用所有样本训练")
    all_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(all_dataloader) * args.epochs_e
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )
    
    losses_all = []
    
    for epoch in range(args.epochs_e):
        avg_loss = train(model, all_dataloader, optimizer, scheduler, device, epoch + 1)
        losses_all.append(avg_loss)
        
        # 评估
        accuracy, predictions = evaluate(model, val_dataloader, device)
        logger.info(f"全部样本训练后 轮次 {epoch + 1}/{args.epochs_e}: 损失 = {avg_loss:.4f}, 准确率 = {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 保存模型
            model_path = os.path.join(args.output_dir, "model_all_best.pt")
            torch.save(model.state_dict(), model_path)
            # 保存预测结果
            with open(os.path.join(args.output_dir, "predictions_all_best.json"), 'w') as f:
                json.dump(predictions, f)
    
    # 保存最终模型
    model_path = os.path.join(args.output_dir, "model_final.pt")
    torch.save(model.state_dict(), model_path)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    x1 = list(range(1, args.epochs_th + 1))
    x2 = list(range(args.epochs_th + 1, args.epochs_th + args.epochs_bh + 1))
    x3 = list(range(args.epochs_th + args.epochs_bh + 1, args.epochs_th + args.epochs_bh + args.epochs_e + 1))
    
    plt.plot(x1, losses_th, 'r-', label='TH')
    plt.plot(x2, losses_bh, 'g-', label='TH+BH')
    plt.plot(x3, losses_all, 'b-', label='ALL')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss by Stage')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'training_loss.png'))
    
    return best_accuracy

# 标准训练策略（不使用HardFirst）
def train_standard(args, model, train_dataset, val_dataset, device):
    # 根据训练策略选择数据
    if args.training_strategy == "TH":
        # 仅使用困难样本
        with open(args.th_ids_path, 'r') as f:
            th_ids = set(json.load(f))
        indices = [i for i, item in enumerate(train_dataset.questions) if item['question_id'] in th_ids]
        subset = Subset(train_dataset, indices)
        logger.info(f"使用TH样本: {len(indices)}/{len(train_dataset)} ({len(indices)/len(train_dataset)*100:.2f}%)")
    
    elif args.training_strategy == "TH+BH":
        # 使用困难和中等样本
        with open(args.th_ids_path, 'r') as f:
            th_ids = set(json.load(f))
        with open(args.bh_ids_path, 'r') as f:
            bh_ids = set(json.load(f))
        
        indices = [i for i, item in enumerate(train_dataset.questions) 
                  if item['question_id'] in th_ids or item['question_id'] in bh_ids]
        subset = Subset(train_dataset, indices)
        logger.info(f"使用TH+BH样本: {len(indices)}/{len(train_dataset)} ({len(indices)/len(train_dataset)*100:.2f}%)")
    
    else:  # "ALL"
        # 使用所有样本
        subset = train_dataset
        logger.info(f"使用所有样本: {len(train_dataset)}")
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )
    
    # 开始训练
    best_accuracy = 0
    losses = []
    
    for epoch in range(args.num_train_epochs):
        avg_loss = train(model, train_dataloader, optimizer, scheduler, device, epoch + 1)
        losses.append(avg_loss)
        
        # 评估
        accuracy, predictions = evaluate(model, val_dataloader, device)
        logger.info(f"轮次 {epoch + 1}/{args.num_train_epochs}: 损失 = {avg_loss:.4f}, 准确率 = {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # 保存最佳模型
            model_path = os.path.join(args.output_dir, "model_best.pt")
            torch.save(model.state_dict(), model_path)
            # 保存预测结果
            with open(os.path.join(args.output_dir, "predictions_best.json"), 'w') as f:
                json.dump(predictions, f)
    
    # 保存最终模型
    model_path = os.path.join(args.output_dir, "model_final.pt")
    torch.save(model.state_dict(), model_path)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.num_train_epochs + 1), losses, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss ({args.training_strategy})')
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'training_loss.png'))
    
    return best_accuracy

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载预训练的模型和分词器
    logger.info("加载LXMERT模型和分词器...")
    tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    model = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased")
    model.to(device)
    
    # 加载数据集
    logger.info("加载训练和验证数据集...")
    train_dataset = VQADataset(
        args.train_questions,
        args.train_annotations,
        args.image_features_path,
        tokenizer,
        max_seq_length=args.max_seq_length
    )
    
    val_dataset = VQADataset(
        args.val_questions,
        args.val_annotations,
        args.image_features_path,
        tokenizer,
        max_seq_length=args.max_seq_length
    )
    
    # 配置模型输出层
    num_answers = len(train_dataset.label2ans)
    logger.info(f"调整模型输出层为 {num_answers} 个答案类别...")
    model.resize_num_qa_labels(num_answers)
    
    # 根据训练策略选择训练函数
    if args.training_strategy == "HF":
        logger.info("使用HardFirst训练策略...")
        best_accuracy = train_hard_first(args, model, train_dataset, val_dataset, device)
    else:
        logger.info(f"使用标准训练策略: {args.training_strategy}")
        best_accuracy = train_standard(args, model, train_dataset, val_dataset, device)
    
    logger.info(f"训练完成！最佳验证准确率: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()
