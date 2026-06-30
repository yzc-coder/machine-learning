# -*- coding: utf-8 -*-
"""
PromptAttack 策略攻击模块
===========================
实现三种 Prompt 改写攻击策略，用于绕过欺诈检测模型：

  策略1（去关键词）：移除或替换高风险欺诈词汇，降低模型对欺诈特征的敏感度
  策略2（自然语言润色）：改善语言流畅度、引入日常用语，使对话听起来更自然
  策略3（协商语气）：增加协商/受害者视角的表述，降低对抗性

同时对比这三种策略对 BERT 分类器和 Fraud-R1 模拟检测器的影响。
"""

import re
import random
import numpy as np
from tqdm import tqdm


class PromptAttack:
    """
    Prompt 改写攻击 - 通过文本改写绕过欺诈检测
    """

    def __init__(self):
        # 高风险欺诈关键词（容易被检测的关键词）
        self.high_risk_words = [
            "转账", "汇款", "退款", "验证码", "银行卡号", "密码",
            "安全账户", "冻结", "扣款", "保证金", "手续费",
            "刷单", "高收益", "稳赚", "内部消息", "中奖",
            "点击链接", "下载APP", "二维码", "短信验证",
            "涉嫌", "违法", "公安", "法院", "通缉",
        ]

        # 欺诈词的委婉替代
        self.euphemism_map = {
            "转账": "处理", "汇款": "调度", "退款": "返还",
            "验证码": "编号", "密码": "识别号", "银行卡号": "卡片号码",
            "安全账户": "指定账户", "保证金": "预付金", "手续费": "服务费",
            "刷单": "推广", "高收益": "较好回报", "中奖": "获得礼品",
        }

        # 自然化替换短语
        self.natural_phrases = [
            "是这样的，", "就是说，", "你看啊，", "其实吧，",
            "说实话，", "跟你说啊，", "反正就是，", "讲真的，",
            "对了，", "那个，", "顺便说一下，",
        ]

        # 协商/受害者视角短语
        self.victim_phrases = [
            "我自己也是被这个坑过的，",
            "作为过来人我跟你说，",
            "我也是最近才知道的，",
            "我之前也遇到类似的情况，",
            "我们这种普通人不注意就会吃亏，",
            "我理解你的顾虑，",
            "我自己也是这么过来的，",
        ]

    def strategy_remove_keywords(self, text):
        """
        策略1：移除/替换高风险关键词
        目标是降低分类器对欺诈关键词的注意力
        """
        modified = text
        changes = []

        for word in self.high_risk_words:
            if word in modified:
                replacement = self.euphemism_map.get(word, "")
                if replacement:
                    modified = modified.replace(word, replacement)
                    changes.append((word, replacement))

        return modified, {'strategy': 'remove_keywords', 'changes': len(changes), 'details': changes}

    def strategy_naturalize(self, text):
        """
        策略2：自然语言润色
        在对话中插入日常口语表达，使对话更自然
        """
        lines = text.split('\n')
        modified_lines = []
        changes = 0
        rng = random.Random(42)  # 固定种子保证可重复

        for line in lines:
            modified_lines.append(line)
            # 在 left: 的语句后面随机插入自然化短语
            if line.strip().startswith('left:') and rng.random() < 0.4:
                phrase = rng.choice(self.natural_phrases)
                # 插入到 left: 前缀之后
                prefix_end = line.find(':', 5)
                if prefix_end > 0:
                    new_line = line[:prefix_end+1] + ' ' + phrase + line[prefix_end+1:]
                    modified_lines[-1] = new_line
                    changes += 1

        return '\n'.join(modified_lines), {'strategy': 'naturalize', 'changes': changes}

    def strategy_victim_perspective(self, text):
        """
        策略3：协商/受害者语气
        增加展示弱势和理解的表述，降低对抗性感知
        """
        lines = text.split('\n')
        modified_lines = []
        changes = 0
        rng = random.Random(123)  # 固定种子

        for line in lines:
            if line.strip().startswith('left:') and rng.random() < 0.35:
                phrase = rng.choice(self.victim_phrases)
                prefix_end = line.find(':', 5)
                if prefix_end > 0:
                    new_line = line[:prefix_end+1] + ' ' + phrase + line[prefix_end+1:]
                    modified_lines.append(new_line)
                    changes += 1
                    continue
            modified_lines.append(line)

        return '\n'.join(modified_lines), {'strategy': 'victim_perspective', 'changes': changes}

    def apply_combined(self, text):
        """
        组合应用三种策略
        """
        results = {}
        original_text = text

        # 策略1
        t1, info1 = self.strategy_remove_keywords(original_text)
        results['remove_keywords'] = (t1, info1)

        # 策略2
        t2, info2 = self.strategy_naturalize(original_text)
        results['naturalize'] = (t2, info2)

        # 策略3
        t3, info3 = self.strategy_victim_perspective(original_text)
        results['victim_perspective'] = (t3, info3)

        return results


def evaluate_attack_effectiveness(attack_results, classifier, original_labels):
    """
    评估三种攻击策略对分类器的效果

    返回每种策略下：
    - 攻击成功率（原被检测为欺诈的样本，攻击后变为非欺诈的比例）
    - 平均置信度变化
    """
    from difflib import SequenceMatcher

    results_summary = {
        'remove_keywords': {'success': 0, 'total_fraud': 0, 'conf_changes': []},
        'naturalize': {'success': 0, 'total_fraud': 0, 'conf_changes': []},
        'victim_perspective': {'success': 0, 'total_fraud': 0, 'conf_changes': []},
    }

    for i, (text, label) in enumerate(attack_results):
        if label != 1:  # 只关注欺诈样本
            continue

        original_pred = classifier.predict(text)
        orig_conf = original_pred['confidence'] if original_pred['prediction'] == 1 else 1 - original_pred['confidence']

        for strategy, (modified_text, info) in attack_results[i][1].items():
            new_pred = classifier.predict(modified_text)
            new_conf = new_pred['confidence'] if new_pred['prediction'] == 1 else 1 - new_pred['confidence']

            results_summary[strategy]['total_fraud'] += 1
            results_summary[strategy]['conf_changes'].append(orig_conf - new_conf)

            # 攻击成功：原预测为欺诈，现在预测为非欺诈
            if original_pred['prediction'] == 1 and new_pred['prediction'] == 0:
                results_summary[strategy]['success'] += 1

    # 计算统计
    for strategy in results_summary:
        total = results_summary[strategy]['total_fraud']
        if total > 0:
            results_summary[strategy]['success_rate'] = results_summary[strategy]['success'] / total
        else:
            results_summary[strategy]['success_rate'] = 0
        changes = results_summary[strategy]['conf_changes']
        results_summary[strategy]['avg_conf_drop'] = np.mean(changes) if changes else 0

    return results_summary


if __name__ == '__main__':
    print("PromptAttack 模块测试")
    print("=" * 60)

    # 测试文本
    test_text = """left: 喂，你好，这边是深圳电讯客服中心，我是客服专员李明。
right: 你好，有什么事吗？
left: 我们注意到你最近在我们平台购买了一部手机，但是根据我们的系统记录，这部手机似乎出现了问题。
right: 真的吗？那怎么办？
left: 别担心，我们有一个解决方案。为了确保你的权益，我们需要你点击一个链接来验证你的订单信息。
right: 好的，那链接怎么给我？
left: 我可以发一个短信给你，里面有一个链接，你点击后按照提示操作就可以解决这个问题了。
right: 行，你发短信给我吧。"""

    attacker = PromptAttack()

    print("\n原始文本:")
    print(test_text[:200] + "...")

    # 测试各策略
    t1, info1 = attacker.strategy_remove_keywords(test_text)
    print(f"\n策略1 - 去关键词 ({info1['changes']} 处修改):")
    print(t1[:200] + "...")

    t2, info2 = attacker.strategy_naturalize(test_text)
    print(f"\n策略2 - 自然化 ({info2['changes']} 处修改):")
    print(t2[:200] + "...")

    t3, info3 = attacker.strategy_victim_perspective(test_text)
    print(f"\n策略3 - 受害者视角 ({info3['changes']} 处修改):")
    print(t3[:200] + "...")
