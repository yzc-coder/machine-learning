# -*- coding: utf-8 -*-
"""
TextFooler 对抗攻击模块
========================
实现 TextFooler 词级对抗攻击算法（Jin et al., 2020）。
核心思想：通过计算每个词对模型预测的重要性，按重要性降序用同义词替换关键词，
        在不改变原意的前提下使模型产生错误预测。

算法流程：
  步骤1：计算词重要性分数（通过逐个删除单词观察模型预测概率变化）
  步骤2：对高重要性词，使用同义词库按序替换，直到模型预测翻转
  步骤3：若单词替换不足以翻转，迭代替换多个高重要性词
"""

import torch
import numpy as np
import jieba
import re
from tqdm import tqdm
from difflib import SequenceMatcher


class TextFoolerAttack:
    """
    TextFooler 对抗攻击实现
    """
    def __init__(self, model, tokenizer, device=None, max_perturb_ratio=0.3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_perturb_ratio = max_perturb_ratio  # 最大修改比例
        self.synonym_dict = self._build_synonym_dict()

    def _build_synonym_dict(self):
        """
        构建中文同义词词典（简化版，实际应用中可使用更完整的同义词库）
        这里构建一个针对欺诈通话场景的定制同义词库
        """
        synonyms = {
            # 高频词 - 欺诈相关
            "资金": ["款项", "费用", "金额", "钱款"],
            "贷款": ["借款", "信贷", "融资", "放款"],
            "产品": ["商品", "货物", "物品", "制品"],
            "信息": ["讯息", "资讯", "数据", "消息"],
            "抵押": ["质押", "担保", "抵押担保"],
            "客户": ["用户", "客人", "顾客", "委托人"],
            "手机": ["移动电话", "电话", "行动电话"],
            "下载": ["获取", "安装", "装载"],
            "链接": ["地址", "网址", "连接"],
            "退款": ["退费", "退还", "退回款项"],
            "安全": ["保险", "可靠", "稳妥"],
            "活动": ["促销", "优惠", "项目"],
            "问题": ["疑问", "困扰", "难题"],
            "处理": ["解决", "应对", "处置"],
            "账户": ["账号", "户头"],
            "平台": ["系统", "网站", "渠道"],
            "服务": ["业务", "协助", "支援"],
            "投资": ["理财", "出资", "投入"],
            "收益": ["回报", "利润", "获利"],
            "申请": ["办理", "申办", "提出"],
            "验证": ["核实", "确认", "校验"],
            "优惠": ["折扣", "减免", "特惠"],
            "系统": ["平台", "程序", "软件"],
            "记录": ["档案", "记载", "登记"],
            "订单": ["定单", "单据"],
            "提醒": ["通知", "告知", "提示"],
            "发送": ["传送", "发出", "寄送"],
            "立即": ["马上", "即刻", "迅速"],
            "紧急": ["紧迫", "急切", "紧要"],
            "需要": ["须要", "需求", "要求"],
            # 更多同义词
            "回复": ["答复", "回应", "反馈"],
            "询问": ["咨询", "查问", "访问"],
            "了解": ["知晓", "清楚", "理解"],
            "帮助": ["协助", "帮忙", "支持"],
            "机会": ["时机", "机遇", "良机"],
            "方案": ["计划", "安排", "方法"],
            "程序": ["软件", "应用", "工具"],
            "电话": ["座机", "热线", "专线"],
            "专员": ["客服", "代表", "员工"],
            "留意": ["注意", "留心", "关注"],
            "保护": ["保障", "维护", "守护"],
            "权益": ["权利", "利益", "福利"],
        }
        return synonyms

    def _tokenize_words(self, text):
        """
        使用 jieba 分词，保留标点和位置信息
        """
        # 先用正则分离中文和非中文字符
        words = list(jieba.cut(text))
        # 过滤纯空格
        words = [w for w in words if w.strip()]
        return words

    def _compute_word_importance(self, text, original_prob, target_label):
        """
        步骤1：计算每个词的重要性分数
        通过逐个删除词语，观察模型对目标类别预测概率的变化
        """
        words = self._tokenize_words(text)
        importance_scores = []

        for i, word in enumerate(words):
            # 创建删除该词后的文本
            modified_words = words[:i] + words[i+1:]
            modified_text = ''.join(modified_words)

            if len(modified_text.strip()) == 0:
                importance_scores.append((word, i, 0))
                continue

            # 获取新预测概率
            prob = self._get_prediction_prob(modified_text, target_label)
            importance = original_prob - prob  # 概率下降越多，该词越重要
            importance_scores.append((word, i, importance))

        # 按重要性降序排列
        importance_scores.sort(key=lambda x: x[2], reverse=True)
        return importance_scores

    def _get_prediction_prob(self, text, target_label):
        """
        获取模型对目标标签的预测概率
        """
        self.model.eval()
        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            return probs[0][target_label].item()

    def _semantic_similarity(self, text1, text2):
        """
        计算两个文本的语义相似度（使用编辑距离比）
        """
        return SequenceMatcher(None, text1, text2).ratio()

    def _get_synonyms(self, word):
        """
        获取词语的同义词列表
        """
        if word in self.synonym_dict:
            return self.synonym_dict[word]
        # 对于不在词典中的词，返回空列表（即无法替换）
        return []

    def attack(self, text, true_label, target_label=None):
        """
        执行 TextFooler 攻击

        参数:
          text: 原始文本
          true_label: 真实标签
          target_label: 目标标签（攻击后的目标类别），默认为翻转标签

        返回:
          dict: 包含攻击结果、修改后的文本、修改次数等信息
        """
        if target_label is None:
            target_label = 1 - true_label

        original_prob = self._get_prediction_prob(text, target_label)

        # 检查原始预测是否正确（如果已经是目标标签，无需攻击）
        orig_pred = 1 if self._get_prediction_prob(text, 1) > 0.5 else 0
        if orig_pred == target_label:
            return {
                'success': True,
                'perturbed_text': text,
                'num_changes': 0,
                'original_prob': original_prob,
                'final_prob': original_prob,
                'similarity': 1.0,
                'changes': []
            }

        # 步骤1：计算词重要性
        importance_scores = self._compute_word_importance(text, original_prob, target_label)

        # 步骤2：按重要性顺序尝试替换
        words = self._tokenize_words(text)
        max_changes = int(len(words) * self.max_perturb_ratio)
        changes = []

        for word, idx, importance in importance_scores:
            if len(changes) >= max_changes:
                break

            # 只考虑有正重要性的词（删除后概率下降）
            if importance <= 0:
                continue

            synonyms = self._get_synonyms(word)
            if not synonyms:
                continue

            best_synonym = None
            best_prob = float('inf')
            best_similarity = 0

            for syn in synonyms:
                modified_words = words.copy()
                modified_words[idx] = syn
                modified_text = ''.join(modified_words)

                prob = self._get_prediction_prob(modified_text, target_label)
                sim = self._semantic_similarity(text, modified_text)

                # 选择使目标概率上升最多且相似度最高的同义词
                if prob > best_prob - 0.01 and sim > best_similarity:
                    best_synonym = syn
                    best_prob = prob
                    best_similarity = sim

            if best_synonym:
                words[idx] = best_synonym
                changes.append({
                    'position': idx,
                    'original': word,
                    'replacement': best_synonym,
                    'importance': importance,
                    'prob_after_replace': best_prob,
                    'similarity': best_similarity
                })

                # 检查是否攻击成功（预测翻转到目标标签）
                current_text = ''.join(words)
                current_prob = self._get_prediction_prob(current_text, target_label)
                if current_prob > 0.5:
                    return {
                        'success': True,
                        'perturbed_text': current_text,
                        'num_changes': len(changes),
                        'original_prob': original_prob,
                        'final_prob': current_prob,
                        'similarity': self._semantic_similarity(text, current_text),
                        'changes': changes
                    }

        # 攻击未成功
        final_text = ''.join(words) if changes else text
        return {
            'success': False,
            'perturbed_text': final_text,
            'num_changes': len(changes),
            'original_prob': original_prob,
            'final_prob': self._get_prediction_prob(final_text, target_label),
            'similarity': self._semantic_similarity(text, final_text),
            'changes': changes
        }

    def batch_attack(self, texts, labels, verbose=True):
        """
        批量执行 TextFooler 攻击
        """
        results = []
        success_count = 0
        iterator = tqdm(zip(texts, labels), total=len(texts), desc='TextFooler 攻击') if verbose else zip(texts, labels)

        for text, label in iterator:
            result = self.attack(text, label)
            results.append(result)
            if result['success']:
                success_count += 1

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({
                    '成功率': f'{success_count}/{len(results)}',
                    '修改数': f'{np.mean([r["num_changes"] for r in results]):.1f}'
                })

        return results, {
            'total': len(results),
            'success': success_count,
            'success_rate': success_count / len(results) if results else 0,
            'avg_changes': np.mean([r['num_changes'] for r in results]) if results else 0,
            'avg_similarity': np.mean([r['similarity'] for r in results]) if results else 0,
        }


if __name__ == '__main__':
    from train_bert import BertFraudClassifier
    from data_loader import DataProcessor
    import pandas as pd

    # 加载数据和模型
    processor = DataProcessor()
    processor.load_data()
    train_df, test_df = processor.preprocess()
    train_data, test_data = processor.get_fraud_binary_data()

    # 加载模型（需要先训练或已有保存的模型）
    try:
        classifier = BertFraudClassifier()
        classifier.load_model('bert_fraud_model')

        # 初始化攻击器
        attacker = TextFoolerAttack(classifier.model, classifier.tokenizer, classifier.device)

        # 取少量样本测试
        sample_texts = test_data['cleaned_text'].tolist()[:10]
        sample_labels = test_data['binary_label'].tolist()[:10]

        print("开始 TextFooler 攻击测试...")
        results, stats = attacker.batch_attack(sample_texts, sample_labels)

        print(f"\n攻击结果:")
        print(f"  总样本: {stats['total']}")
        print(f"  成功: {stats['success']}")
        print(f"  成功率: {stats['success_rate']:.2%}")
        print(f"  平均修改词数: {stats['avg_changes']:.1f}")
        print(f"  平均语义相似度: {stats['avg_similarity']:.4f}")

        # 展示一个攻击成功案例
        for r in results:
            if r['success'] and r['num_changes'] > 0:
                print(f"\n攻击成功案例:")
                print(f"  修改词数: {r['num_changes']}")
                print(f"  语义相似度: {r['similarity']:.4f}")
                for c in r['changes']:
                    print(f"    '{c['original']}' → '{c['replacement']}' (重要性={c['importance']:.6f})")
                break

    except Exception as e:
        print(f"请先训练模型: python train_bert.py")
        print(f"错误: {e}")
