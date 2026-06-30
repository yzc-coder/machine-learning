# -*- coding: utf-8 -*-
"""
主运行脚本
==========
整合所有实验模块，一键运行完整的欺诈通话检测对抗攻击实验流程。

实验流程：
  1. 数据加载与预处理
  2. BERT 分类器训练（若已训练则加载）
  3. TextFooler 对抗攻击实验
  4. Fraud-R1 LLM 检测实验
  5. PromptAttack 攻击策略实验
  6. 综合评估与报告生成

使用方法：
  python main.py           # 运行完整实验
  python main.py --skip-train   # 跳过模型训练（使用已有模型）
  python main.py --quick         # 快速模式（仅少量样本）
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

from data_loader import DataProcessor
from train_bert import BertFraudClassifier
from textfooler_attack import TextFoolerAttack
from fraud_r1 import FraudR1Simulator
from prompt_attack import PromptAttack, evaluate_attack_effectiveness
from evaluate import ExperimentEvaluator


def run_full_experiment(skip_train=False, quick_mode=False):
    """
    运行完整实验流程
    """
    evaluator = ExperimentEvaluator()

    # ============================================================
    # 第1步：数据加载与预处理
    # ============================================================
    print("\n" + "=" * 60)
    print("第1步：数据加载与预处理")
    print("=" * 60)

    processor = DataProcessor()
    processor.load_data()
    train_df, test_df = processor.preprocess()
    train_data, test_data = processor.get_fraud_binary_data()
    stats = processor.get_statistics()

    evaluator.add_result('dataset_stats', {
        'train_size': len(train_data),
        'test_size': len(test_data),
        'fraud_ratio': f"{train_data['binary_label'].mean()*100:.1f}%"
    })

    # 快速模式下减少样本
    if quick_mode:
        print("\n[快速模式] 使用少量样本进行实验...")
        train_data = train_data.sample(n=min(500, len(train_data)), random_state=42)
        test_data = test_data.sample(n=min(100, len(test_data)), random_state=42)
        print(f"快速模式: 训练 {len(train_data)} 条, 测试 {len(test_data)} 条")

    # ============================================================
    # 第2步：BERT 分类器训练
    # ============================================================
    print("\n" + "=" * 60)
    print("第2步：BERT 欺诈检测分类器")
    print("=" * 60)

    model_path = 'bert_fraud_model'

    if skip_train and os.path.exists(model_path):
        print("跳过训练，加载已保存的模型...")
        classifier = BertFraudClassifier()
        classifier.load_model(model_path)
    else:
        classifier = BertFraudClassifier(max_length=256)

        train_loader, test_loader, _, _ = processor.create_dataloaders(
            classifier.tokenizer,
            batch_size=8 if not quick_mode else 4,
            max_length=256
        )

        epochs = 1 if quick_mode else 5
        print(f"开始训练 ({epochs} epochs)...")
        history = classifier.train(
            train_loader, test_loader,
            epochs=epochs, lr=2e-5,
            save_path=model_path,
            patience=2
        )

        # 评估
        acc, f1, metrics = classifier.evaluate(test_loader)

    # 在完整测试集上评估
    _, full_test_loader, _, _ = processor.create_dataloaders(
        classifier.tokenizer, batch_size=16, max_length=256
    )
    acc, f1, metrics = classifier.evaluate(full_test_loader)

    print(f"\nBERT 基线性能:")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  精确率: {metrics['precision']:.4f}")
    print(f"  召回率: {metrics['recall']:.4f}")
    print(f"  F1 分数: {metrics['f1']:.4f}")

    evaluator.add_result('bert_baseline', {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'confusion_matrix': metrics['confusion_matrix']
    })

    # ============================================================
    # 第3步：TextFooler 对抗攻击实验
    # ============================================================
    print("\n" + "=" * 60)
    print("第3步：TextFooler 对抗攻击实验")
    print("=" * 60)

    attacker = TextFoolerAttack(classifier.model, classifier.tokenizer, classifier.device)

    # 选取欺诈样本进行攻击
    fraud_samples = test_data[test_data['binary_label'] == 1]
    sample_size = min(50 if not quick_mode else 10, len(fraud_samples))
    attack_samples = fraud_samples.sample(n=sample_size, random_state=42)

    attack_texts = attack_samples['cleaned_text'].tolist()
    attack_labels = attack_samples['binary_label'].tolist()

    print(f"对 {len(attack_texts)} 条欺诈样本执行 TextFooler 攻击...")
    results, stats = attacker.batch_attack(attack_texts, attack_labels)

    print(f"\nTextFooler 攻击结果:")
    print(f"  总样本: {stats['total']}")
    print(f"  攻击成功: {stats['success']}")
    print(f"  成功率: {stats['success_rate']:.2%}")
    print(f"  平均修改词数: {stats['avg_changes']:.1f}")
    print(f"  平均语义相似度: {stats['avg_similarity']:.4f}")

    evaluator.add_result('textfooler', {
        'total': stats['total'],
        'success': stats['success'],
        'success_rate': stats['success_rate'],
        'avg_changes': stats['avg_changes'],
        'avg_similarity': stats['avg_similarity'],
    })

    # ============================================================
    # 第4步：Fraud-R1 实验
    # ============================================================
    print("\n" + "=" * 60)
    print("第4步：Fraud-R1 大模型实验")
    print("=" * 60)

    simulator = FraudR1Simulator()

    # 在测试集上运行
    test_texts = test_data['cleaned_text'].tolist()
    test_labels = test_data['binary_label'].tolist()

    if quick_mode:
        test_texts = test_texts[:100]
        test_labels = test_labels[:100]

    summary = simulator.run_experiment(test_texts, test_labels)

    print(f"\nFraud-R1 实验结果:")
    exp_names = {
        'direct': '直接判断      ',
        'rewrite_1': '一次改写(紧迫感)',
        'rewrite_2': '两次改写(+协商感)',
        'rewrite_3': '三次改写(+威胁感)',
    }
    for key, name in exp_names.items():
        s = summary[key]
        print(f"  {name}: 准确率 {s['accuracy_pct']} ({s['correct']}/{s['total']})")

    evaluator.add_result('fraud_r1', summary)

    # ============================================================
    # 第5步：PromptAttack 攻击策略实验
    # ============================================================
    print("\n" + "=" * 60)
    print("第5步：PromptAttack 策略实验")
    print("=" * 60)

    prompt_attacker = PromptAttack()

    # 对欺诈样本应用三种策略
    fraud_samples_pa = fraud_samples.sample(n=min(30 if not quick_mode else 10, len(fraud_samples)), random_state=42)
    pa_texts = fraud_samples_pa['cleaned_text'].tolist()
    pa_labels = fraud_samples_pa['binary_label'].tolist()

    print(f"对 {len(pa_texts)} 条欺诈样本应用 PromptAttack 策略...")

    # 准备攻击结果
    attack_results_list = []
    for i, (text, label) in enumerate(zip(pa_texts, pa_labels)):
        modified = prompt_attacker.apply_combined(text)
        attack_results_list.append((text, modified))

    # 评估攻击效果
    pa_summary = evaluate_attack_effectiveness(attack_results_list, classifier, pa_labels)

    print(f"\nPromptAttack 攻击效果:")
    pa_names = {
        'remove_keywords': '策略1-去关键词    ',
        'naturalize': '策略2-自然化      ',
        'victim_perspective': '策略3-受害者视角  ',
    }
    for key, name in pa_names.items():
        s = pa_summary[key]
        print(f"  {name}: 攻击成功率 {s['success_rate']:.2%}, 平均置信度下降 {s['avg_conf_drop']:.4f}")

    evaluator.add_result('prompt_attack', pa_summary)

    # ============================================================
    # 第6步：综合评估与报告
    # ============================================================
    print("\n" + "=" * 60)
    print("第6步：综合评估与报告生成")
    print("=" * 60)

    evaluator.generate_comparison_table()
    evaluator.generate_summary_report()
    evaluator.save_all_results()

    print("\n" + "=" * 60)
    print("实验全部完成！")
    print(f"结果文件保存在 results/ 目录下")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='欺诈通话检测对抗攻击实验')
    parser.add_argument('--skip-train', action='store_true', help='跳过模型训练')
    parser.add_argument('--quick', action='store_true', help='快速模式（少量样本）')

    args = parser.parse_args()

    print("=" * 60)
    print("欺诈通话检测 - 对抗攻击实验系统")
    print("=" * 60)
    print(f"配置: skip_train={args.skip_train}, quick={args.quick}")
    print("")

    run_full_experiment(skip_train=args.skip_train, quick_mode=args.quick)
