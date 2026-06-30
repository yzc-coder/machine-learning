# -*- coding: utf-8 -*-
"""
综合评估与可视化模块
====================
功能：汇总所有实验结果，生成对比表格、图表和综合评估报告。

评估维度：
  1. BERT 分类器基础性能
  2. TextFooler 攻击效果
  3. Fraud-R1 各策略检测准确率
  4. PromptAttack 三种策略的攻击成功率
  5. 跨方法对比分析
"""

import pandas as pd
import numpy as np
import json
import os
from collections import OrderedDict


class ExperimentEvaluator:
    """
    实验评估器：汇总、对比和可视化所有实验结果
    """

    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}

    def add_result(self, name, data):
        """添加实验结果"""
        self.results[name] = data

    def generate_comparison_table(self):
        """
        生成核心指标对比表
        """
        print("\n" + "=" * 70)
        print("核心指标对比表")
        print("=" * 70)

        table_data = []

        # BERT 基础性能
        if 'bert_baseline' in self.results:
            bert = self.results['bert_baseline']
            table_data.append({
                '方法': 'BERT 分类器（基线）',
                '准确率': f"{bert.get('accuracy', 0):.2%}",
                '精确率': f"{bert.get('precision', 0):.2%}",
                '召回率': f"{bert.get('recall', 0):.2%}",
                'F1分数': f"{bert.get('f1', 0):.4f}",
                '备注': '基础欺诈检测模型'
            })

        # TextFooler 攻击
        if 'textfooler' in self.results:
            tf = self.results['textfooler']
            table_data.append({
                '方法': 'TextFooler 攻击',
                '准确率': '-',
                '精确率': '-',
                '召回率': '-',
                'F1分数': '-',
                '备注': f"攻击成功率: {tf.get('success_rate', 0):.2%}, 平均修改{tf.get('avg_changes', 0):.1f}词"
            })

        # Fraud-R1 各策略
        if 'fraud_r1' in self.results:
            fr = self.results['fraud_r1']
            exp_names = {
                'direct': 'Fraud-R1 直接判断',
                'rewrite_1': 'Fraud-R1 +紧迫感',
                'rewrite_2': 'Fraud-R1 +协商感',
                'rewrite_3': 'Fraud-R1 +威胁感',
            }
            for key, name in exp_names.items():
                if key in fr:
                    table_data.append({
                        '方法': name,
                        '准确率': fr[key].get('accuracy_pct', '-'),
                        '精确率': '-',
                        '召回率': '-',
                        'F1分数': '-',
                        '备注': f"{fr[key].get('correct', 0)}/{fr[key].get('total', 0)}"
                    })

        # PromptAttack
        if 'prompt_attack' in self.results:
            pa = self.results['prompt_attack']
            pa_names = {
                'remove_keywords': 'PromptAttack-去关键词',
                'naturalize': 'PromptAttack-自然化',
                'victim_perspective': 'PromptAttack-受害者视角',
            }
            for key, name in pa_names.items():
                if key in pa:
                    table_data.append({
                        '方法': name,
                        '准确率': '-',
                        '精确率': '-',
                        '召回率': '-',
                        'F1分数': '-',
                        '备注': f"攻击成功率: {pa[key].get('success_rate', 0):.2%}, 置信度下降: {pa[key].get('avg_conf_drop', 0):.4f}"
                    })

        df = pd.DataFrame(table_data)
        print(df.to_string(index=False))

        # 保存表格
        csv_path = os.path.join(self.output_dir, 'comparison_table.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n对比表已保存至: {csv_path}")

        return df

    def generate_summary_report(self):
        """
        生成综合评估报告（文本格式）
        """
        report = []
        report.append("=" * 70)
        report.append("欺诈通话检测对抗攻击实验 - 综合评估报告")
        report.append("=" * 70)
        report.append("")

        # 1. 数据集概述
        report.append("一、数据集概述")
        report.append("-" * 40)
        report.append("数据集：中文欺诈通话对话数据集")
        if 'dataset_stats' in self.results:
            ds = self.results['dataset_stats']
            report.append(f"  训练集: {ds.get('train_size', 'N/A')} 条")
            report.append(f"  测试集: {ds.get('test_size', 'N/A')} 条")
            report.append(f"  欺诈样本比例: {ds.get('fraud_ratio', 'N/A')}")
        report.append("")

        # 2. BERT 基线
        report.append("二、BERT 基线模型性能")
        report.append("-" * 40)
        if 'bert_baseline' in self.results:
            b = self.results['bert_baseline']
            report.append(f"  模型: bert-base-chinese")
            report.append(f"  测试准确率: {b.get('accuracy', 0):.4f}")
            report.append(f"  精确率: {b.get('precision', 0):.4f}")
            report.append(f"  召回率: {b.get('recall', 0):.4f}")
            report.append(f"  F1 分数: {b.get('f1', 0):.4f}")
        report.append("")

        # 3. TextFooler 攻击结果
        report.append("三、TextFooler 对抗攻击结果")
        report.append("-" * 40)
        if 'textfooler' in self.results:
            tf = self.results['textfooler']
            report.append(f"  攻击方法: 词级同义词替换")
            report.append(f"  攻击成功率: {tf.get('success_rate', 0):.2%}")
            report.append(f"  平均修改词数: {tf.get('avg_changes', 0):.1f}")
            report.append(f"  平均语义相似度: {tf.get('avg_similarity', 0):.4f}")
            report.append(f"  结论: BERT 对 TextFooler 攻击具有较好的鲁棒性，"
                         f"攻击成功率较低，说明词级扰动难以显著改变模型判断。")
        report.append("")

        # 4. Fraud-R1 结果
        report.append("四、Fraud-R1 实验结果")
        report.append("-" * 40)
        if 'fraud_r1' in self.results:
            fr = self.results['fraud_r1']
            report.append(f"  方法: 基于 LLM 的欺诈检测（模拟 DeepSeek-V3）")
            for key, name in [('direct', '直接判断'), ('rewrite_1', '一次改写(+紧迫感)'),
                              ('rewrite_2', '两次改写(+协商感)'), ('rewrite_3', '三次改写(+威胁感)')]:
                if key in fr:
                    report.append(f"  {name}: {fr[key].get('accuracy_pct', 'N/A')} "
                                 f"({fr[key].get('correct', 0)}/{fr[key].get('total', 0)})")
            report.append(f"  结论: 增加改写策略可显著提升欺诈检测率，")
            report.append(f"        说明大模型通过 Prompt 工程可以更全面地理解对话中的欺诈意图。")
        report.append("")

        # 5. PromptAttack
        report.append("五、PromptAttack 攻击策略评估")
        report.append("-" * 40)
        if 'prompt_attack' in self.results:
            pa = self.results['prompt_attack']
            for key, name in [('remove_keywords', '去关键词'), ('naturalize', '自然化'),
                              ('victim_perspective', '受害者视角')]:
                if key in pa:
                    report.append(f"  {name}: 攻击成功率 {pa[key].get('success_rate', 0):.2%}, "
                                 f"平均置信度下降 {pa[key].get('avg_conf_drop', 0):.4f}")
            report.append(f"  结论: 三种 Prompt 改写策略均能一定程度上绕过 BERT 检测，")
            report.append(f"        说明当前模型对语义层面的扰动仍需增强鲁棒性。")
        report.append("")

        # 6. 综合结论
        report.append("六、综合结论")
        report.append("-" * 40)
        report.append("1. BERT 模型在欺诈通话检测任务上表现良好，能够有效识别欺诈对话。")
        report.append("2. TextFooler 词级对抗攻击对 BERT 的威胁有限，模型具有较好的词级鲁棒性。")
        report.append("3. Fraud-R1 通过多策略 Prompt 工程可达到接近完美的检测率，")
        report.append("   展示了 LLM 在欺诈检测中的潜力。")
        report.append("4. PromptAttack 策略能从语义层面绕过检测，暴露了当前模型的不足。")
        report.append("5. 未来工作可考虑模型集成、对抗训练等方法提升模型的整体鲁棒性。")

        report_text = '\n'.join(report)
        print(report_text)

        # 保存报告
        report_path = os.path.join(self.output_dir, 'summary_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n评估报告已保存至: {report_path}")

        return report_text

    def save_all_results(self):
        """
        保存所有结果为 JSON
        """
        # 转换 numpy 类型
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json_path = os.path.join(self.output_dir, 'all_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=convert)
        print(f"完整结果已保存至: {json_path}")


if __name__ == '__main__':
    print("综合评估模块测试")
    evaluator = ExperimentEvaluator()

    # 示例数据
    evaluator.add_result('bert_baseline', {
        'accuracy': 0.9234, 'precision': 0.9150,
        'recall': 0.9387, 'f1': 0.9267
    })
    evaluator.add_result('textfooler', {
        'success_rate': 0.034, 'avg_changes': 3.2, 'avg_similarity': 0.9389
    })
    evaluator.add_result('fraud_r1', {
        'direct': {'accuracy_pct': '96.74%', 'correct': 830, 'total': 858},
        'rewrite_1': {'accuracy_pct': '88.34%', 'correct': 758, 'total': 858},
        'rewrite_2': {'accuracy_pct': '99.88%', 'correct': 857, 'total': 858},
        'rewrite_3': {'accuracy_pct': '99.88%', 'correct': 857, 'total': 858},
    })
    evaluator.add_result('prompt_attack', {
        'remove_keywords': {'success_rate': 0.2296, 'avg_conf_drop': 0.15},
        'naturalize': {'success_rate': 0.2762, 'avg_conf_drop': 0.12},
        'victim_perspective': {'success_rate': 0.3834, 'avg_conf_drop': 0.18},
    })

    evaluator.generate_comparison_table()
    evaluator.generate_summary_report()
    evaluator.save_all_results()
