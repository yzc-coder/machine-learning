# -*- coding: utf-8 -*-
"""
BERT 欺诈检测分类器训练模块
=============================
使用预训练的中文 BERT 模型 (bert-base-chinese) 对欺诈通话文本进行二分类。
包含模型训练、验证、保存和加载功能。
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class BertFraudClassifier:
    """
    基于 BERT 的欺诈检测分类器
    """
    def __init__(self, model_name='bert-base-chinese', num_labels=2, max_length=512, device=None):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"使用设备: {self.device}")
        print(f"加载预训练模型: {model_name}")

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        self.model.to(self.device)

    def train(self, train_loader, val_loader=None, epochs=5, lr=2e-5, warmup_steps=0, 
              save_path='bert_fraud_model', patience=2):
        """
        训练 BERT 分类器
        """
        optimizer = AdamW(self.model.parameters(), lr=lr, eps=1e-8)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        best_val_acc = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_acc': [], 'val_f1': []}

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')

            for batch in progress_bar:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            print(f"Epoch {epoch+1} - 平均训练损失: {avg_loss:.4f}")

            # 验证阶段
            if val_loader:
                val_acc, val_f1, val_report = self.evaluate(val_loader)
                history['val_acc'].append(val_acc)
                history['val_f1'].append(val_f1)
                print(f"Epoch {epoch+1} - 验证准确率: {val_acc:.4f}, F1: {val_f1:.4f}")

                # 早停与模型保存
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    self.save_model(save_path)
                    print(f"  模型已保存 (最佳验证准确率: {best_val_acc:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  早停触发，在第 {epoch+1} 轮停止训练")
                        break

        # 保存训练历史
        with open(f'{save_path}_history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")
        return history

    def evaluate(self, data_loader):
        """
        评估模型性能
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc='评估中'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        report = classification_report(all_labels, all_preds, target_names=['非欺诈', '欺诈'], zero_division=0)

        return acc, f1, {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'report': report
        }

    def predict(self, texts):
        """
        对单条或多条文本进行预测
        返回预测标签和置信度分数
        """
        self.model.eval()
        if isinstance(texts, str):
            texts = [texts]

        results = []
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0][pred].item()

                results.append({
                    'prediction': pred,
                    'label': '欺诈' if pred == 1 else '非欺诈',
                    'confidence': confidence,
                    'probabilities': probs[0].cpu().numpy().tolist()
                })

        return results if len(results) > 1 else results[0]

    def predict_proba(self, texts):
        """
        返回概率分数（用于对抗攻击中的重要性计算）
        """
        self.model.eval()
        if isinstance(texts, str):
            texts = [texts]

        all_probs = []
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                all_probs.append(probs[0].cpu().numpy())

        return np.array(all_probs)

    def save_model(self, path):
        """保存模型和分词器"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"模型已保存至: {path}")

    def load_model(self, path):
        """加载已保存的模型和分词器"""
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()
        print(f"模型已从 {path} 加载")


if __name__ == '__main__':
    # 简单测试
    from data_loader import DataProcessor

    processor = DataProcessor()
    processor.load_data()
    train_df, test_df = processor.preprocess()

    classifier = BertFraudClassifier()

    train_loader, test_loader, train_data, test_data = processor.create_dataloaders(
        classifier.tokenizer, batch_size=8, max_length=256
    )

    # 快速训练（可调整 epochs）
    history = classifier.train(
        train_loader, test_loader,
        epochs=3, lr=2e-5,
        save_path='bert_fraud_model'
    )

    # 最终评估
    acc, f1, metrics = classifier.evaluate(test_loader)
    print(f"\n最终测试结果:")
    print(f"准确率: {acc:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print(metrics['report'])

    # 预测示例
    sample_text = train_data['cleaned_text'].iloc[0]
    result = classifier.predict(sample_text)
    print(f"\n预测示例:")
    print(f"文本: {sample_text[:100]}...")
    print(f"预测: {result['label']} (置信度: {result['confidence']:.4f})")
