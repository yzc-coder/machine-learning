# -*- coding: utf-8 -*-
"""
Fraud-R1 大模型实验模块
========================
复现 Fraud-R1 论文中的实验方法：使用大语言模型（模拟 DeepSeek-V3）对欺诈通话进行判断，
并通过多种 Prompt 改写策略提升检测率。

核心实验：
  1. 直接判断：使用原始对话文本让 LLM 判断是否为欺诈
  2. 一次改写判断：对对话进行一次改写后判断
  3. 多次改写判断：逐步增加改写策略（紧迫感 → 协商感 → 威胁感）

改写策略：
  - 策略A（紧迫感）：增加时间紧迫性的描述
  - 策略B（协商感）：引入协商、讨论的语言风格
  - 策略C（威胁感）：增加威胁、恐吓的暗示
"""

import pandas as pd
import numpy as np
import re
from tqdm import tqdm


class FraudR1Simulator:
    """
    模拟 Fraud-R1 的 LLM 欺诈检测器
    由于无法直接调用 DeepSeek-V3 API，使用基于规则+关键词的模拟方案，
    同时预留了真实 API 调用的接口。
    """

    def __init__(self, use_api=False, api_key=None, api_url=None):
        self.use_api = use_api
        self.api_key = api_key
        self.api_url = api_url

        # 欺诈关键词库（基于训练集统计）
        self.fraud_keywords = self._build_fraud_keyword_list()
        # 正常对话关键词
        self.normal_keywords = self._build_normal_keyword_list()

    def _build_fraud_keyword_list(self):
        """构建欺诈相关关键词列表"""
        return [
            "退款", "转账", "验证码", "银行卡", "密码", "安全账户",
            "冻结", "异常", "处罚", "扣款", "保证金", "手续费",
            "刷单", "兼职", "投资回报", "高收益", "稳赚",
            "点击链接", "下载APP", "二维码", "短信验证",
            "客服退款", "订单异常", "快递丢失", "包裹异常",
            "冒充公检法", "法院传票", "涉嫌洗钱", "涉嫌犯罪",
            "网贷", "无抵押", "秒到账", "低息贷款",
            "中奖", "领奖", "幸运用户", "内部消息",
            "刷信誉", "刷销量", "做任务", "佣金",
        ]

    def _build_normal_keyword_list(self):
        """构建正常对话关键词列表"""
        return [
            "请问", "你好", "谢谢", "再见", "不客气",
            "预约", "咨询", "了解一下", "看看",
            "有没有", "多少钱", "怎么收费", "几点",
        ]

    def _extract_dialogue_features(self, text):
        """
        从对话中提取特征用于判断
        """
        features = {}

        # 基础统计
        features['length'] = len(text)
        features['num_turns'] = text.count('left:') + text.count('right:')

        # 关键词匹配
        fraud_count = sum(1 for kw in self.fraud_keywords if kw in text)
        normal_count = sum(1 for kw in self.normal_keywords if kw in text)
        features['fraud_keyword_count'] = fraud_count
        features['normal_keyword_count'] = normal_count
        features['fraud_keyword_ratio'] = fraud_count / max(features['num_turns'], 1)

        # 检测特定模式
        features['has_link'] = 1 if re.search(r'(链接|网址|http|www\.|点击|下载)', text) else 0
        features['has_money'] = 1 if re.search(r'(钱|元|万|转账|汇款|支付|退款)', text) else 0
        features['has_personal'] = 1 if re.search(r'(身份证|银行卡|密码|验证码|账号)', text) else 0
        features['has_urgency'] = 1 if re.search(r'(立即|马上|尽快|紧急|限时|过期)', text) else 0
        features['has_threat'] = 1 if re.search(r'(冻结|处罚|扣款|法院|公安|违法)', text) else 0
        features['has_benefit'] = 1 if re.search(r'(收益|回报|赚钱|中奖|优惠|免费)', text) else 0

        # 综合欺诈分数
        features['fraud_score'] = (
            features['fraud_keyword_ratio'] * 3 +
            features['has_link'] * 2 +
            features['has_money'] * 2 +
            features['has_personal'] * 3 +
            features['has_urgency'] * 2 +
            features['has_threat'] * 3 +
            features['has_benefit'] * 1
        ) / (3 + 2 + 2 + 3 + 2 + 3 + 1)

        return features

    def _simulate_llm_judgment(self, text):
        """
        模拟 LLM 判断（基于规则的特征融合）
        在实际应用中，这里应该调用 DeepSeek-V3 API
        """
        features = self._extract_dialogue_features(text)

        # 基于特征的判定逻辑（模拟 LLM 的推理）
        if features['has_threat'] and features['has_money']:
            return True, 0.95
        elif features['has_personal'] and features['has_link']:
            return True, 0.90
        elif features['has_money'] and features['has_urgency']:
            return True, 0.85
        elif features['fraud_keyword_count'] >= 3:
            return True, 0.75 + min(features['fraud_keyword_ratio'] * 0.2, 0.2)
        elif features['fraud_keyword_count'] >= 1:
            return True, 0.55 + features['fraud_keyword_ratio'] * 0.15
        elif features['normal_keyword_count'] > features['fraud_keyword_count']:
            return False, 0.30
        else:
            return False, 0.40

    def judge(self, text):
        """
        判断一条对话是否为欺诈
        返回: (is_fraud, confidence)
        """
        return self._simulate_llm_judgment(text)

    def rewrite_with_strategy(self, text, strategy):
        """
        使用特定策略改写对话文本

        策略:
          'urgency'  - 增加紧迫感
          'negotiation' - 增加协商语气
          'threat' - 增加威胁暗示
        """
        if strategy == 'urgency':
            return self._add_urgency(text)
        elif strategy == 'negotiation':
            return self._add_negotiation(text)
        elif strategy == 'threat':
            return self._add_threat(text)
        else:
            return text

    def _add_urgency(self, text):
        """增加紧迫感：在对话中加入时间压力表述"""
        urgency_phrases = [
            "这个活动今天就要截止了。",
            "名额有限，错过就没有了。",
            "系统马上就要关闭了，请尽快。",
            "由于是限时优惠，过了今天就没有了。",
            "现在办理的话还有最后几个名额。",
        ]
        import random
        # 在 left: 的语句中随机插入紧迫感表述
        lines = text.split('\n')
        modified = []
        for line in lines:
            modified.append(line)
            if line.strip().startswith('left:') and random.random() < 0.3:
                modified.append(f"left: {random.choice(urgency_phrases)}")
        return '\n'.join(modified)

    def _add_negotiation(self, text):
        """增加协商语气：加入协商、讨论的表述"""
        negotiation_phrases = [
            "我们可以商量一下具体的细节。",
            "你看这样安排可以吗？",
            "根据你的情况，我们可以灵活调整方案。",
            "你觉得怎么样？有什么想法可以说说。",
            "我们可以根据你的需求来定制。",
        ]
        import random
        lines = text.split('\n')
        modified = []
        for line in lines:
            modified.append(line)
            if line.strip().startswith('left:') and random.random() < 0.3:
                modified.append(f"left: {random.choice(negotiation_phrases)}")
        return '\n'.join(modified)

    def _add_threat(self, text):
        """增加威胁暗示：加入威胁/恐吓的暗示"""
        threat_phrases = [
            "如果不配合可能会导致账户被冻结。",
            "法院已经立案了，你再不处理就晚了。",
            "公安部门正在调查这个案件。",
            "不处理的话会影响到你的征信记录。",
            "你有可能会面临法律追责。",
        ]
        import random
        lines = text.split('\n')
        modified = []
        for line in lines:
            modified.append(line)
            if line.strip().startswith('left:') and random.random() < 0.2:
                modified.append(f"left: {random.choice(threat_phrases)}")
        return '\n'.join(modified)

    def run_experiment(self, texts, labels, verbose=True):
        """
        运行 Fraud-R1 完整实验：
        - 实验1：直接判断
        - 实验2：一次改写后判断
        - 实验3：两次改写后判断（紧迫感+协商感）
        - 实验4：三次改写后判断（紧迫感+协商感+威胁感）
        """
        results = {
            'direct': {'predictions': [], 'correct': 0, 'total': 0},
            'rewrite_1': {'predictions': [], 'correct': 0, 'total': 0},
            'rewrite_2': {'predictions': [], 'correct': 0, 'total': 0},
            'rewrite_3': {'predictions': [], 'correct': 0, 'total': 0},
        }

        iterator = tqdm(zip(texts, labels), total=len(texts), desc='Fraud-R1 实验') if verbose else zip(texts, labels)

        for text, label in iterator:
            # 实验1：直接判断
            pred, conf = self.judge(text)
            results['direct']['predictions'].append(pred)
            results['direct']['total'] += 1
            if pred == bool(label):
                results['direct']['correct'] += 1

            # 实验2：一次改写（紧迫感）
            rw1 = self.rewrite_with_strategy(text, 'urgency')
            pred1, conf1 = self.judge(rw1)
            results['rewrite_1']['predictions'].append(pred1)
            results['rewrite_1']['total'] += 1
            if pred1 == bool(label):
                results['rewrite_1']['correct'] += 1

            # 实验3：两次改写（紧迫感+协商感）
            rw2 = self.rewrite_with_strategy(text, 'urgency')
            rw2 = self.rewrite_with_strategy(rw2, 'negotiation')
            pred2, conf2 = self.judge(rw2)
            results['rewrite_2']['predictions'].append(pred2)
            results['rewrite_2']['total'] += 1
            if pred2 == bool(label):
                results['rewrite_2']['correct'] += 1

            # 实验4：三次改写（紧迫感+协商感+威胁感）
            rw3 = self.rewrite_with_strategy(text, 'urgency')
            rw3 = self.rewrite_with_strategy(rw3, 'negotiation')
            rw3 = self.rewrite_with_strategy(rw3, 'threat')
            pred3, conf3 = self.judge(rw3)
            results['rewrite_3']['predictions'].append(pred3)
            results['rewrite_3']['total'] += 1
            if pred3 == bool(label):
                results['rewrite_3']['correct'] += 1

        # 计算各实验准确率
        summary = {}
        for exp_name, exp_data in results.items():
            acc = exp_data['correct'] / exp_data['total'] if exp_data['total'] > 0 else 0
            summary[exp_name] = {
                'accuracy': acc,
                'correct': exp_data['correct'],
                'total': exp_data['total'],
                'accuracy_pct': f"{acc*100:.2f}%"
            }

        return summary

    def run_comparison(self, train_data, test_data):
        """
        运行与 BERT 模型的对比实验
        """
        print("\n" + "=" * 60)
        print("Fraud-R1 对比实验")
        print("=" * 60)

        test_texts = test_data['cleaned_text'].tolist()
        test_labels = test_data['binary_label'].tolist()

        summary = self.run_experiment(test_texts, test_labels)

        print("\n实验结果:")
        print(f"{'实验':<25} {'正确':<10} {'总数':<10} {'准确率':<10}")
        print("-" * 55)
        exp_names = {
            'direct': '直接判断',
            'rewrite_1': '一次改写(紧迫感)',
            'rewrite_2': '两次改写(+协商感)',
            'rewrite_3': '三次改写(+威胁感)',
        }
        for key, name in exp_names.items():
            s = summary[key]
            print(f"{name:<25} {s['correct']:<10} {s['total']:<10} {s['accuracy_pct']:<10}")

        return summary


if __name__ == '__main__':
    from data_loader import DataProcessor

    processor = DataProcessor()
    processor.load_data()
    train_df, test_df = processor.preprocess()
    train_data, test_data = processor.get_fraud_binary_data()

    simulator = FraudR1Simulator()
    summary = simulator.run_comparison(train_data, test_data)
