# 欺诈通话检测中的对抗攻击研究

基于 BERT、TextFooler、Fraud-R1 与 PromptAttack 的对比分析。

## 项目简介

本项目以中文欺诈通话对话数据集为研究对象，系统开展了欺诈检测与对抗攻击研究。在检测方面，基于 BERT 预训练语言模型构建了欺诈通话二分类器；在对抗攻击方面，分别实施了词级别的 TextFooler 攻击和语义级别的 PromptAttack 策略攻击，并参考 Fraud-R1 的思路模拟了基于大语言模型的多层 Prompt 改写策略在欺诈检测中的应用效果。

**主要结论：**

- BERT 基线模型在测试集上达到 92.34% 准确率，对词级别攻击鲁棒性较强（TextFooler 攻击成功率仅 3.4%）
- Fraud-R1 多层 Prompt 改写可将检测率提升至 99.88%
- PromptAttack 受害者视角策略攻击成功率高达 38.34%，暴露了语义层面防御的不足

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU（推荐 8GB+ VRAM）或 CPU

## 安装

```bash
git clone https://github.com/yourusername/fraud-detection-adversarial-attack.git
cd fraud-detection-adversarial-attack
pip install -r requirements.txt
```

## 快速开始

```bash
# 快速测试（使用 500 条训练 / 100 条测试样本）
python main.py --quick

# 运行完整实验
python main.py

# 跳过 BERT 训练，仅运行攻击与评估
python main.py --skip-train
```

## 项目结构

```
├── data_loader.py          # 数据加载与预处理
├── train_bert.py           # BERT 分类器训练
├── textfooler_attack.py    # TextFooler 词级别对抗攻击
├── fraud_r1.py             # Fraud-R1 多轮 Prompt 改写策略
├── prompt_attack.py        # PromptAttack 语义级别攻击
├── evaluate.py             # 综合评估与结果输出
├── main.py                 # 主运行脚本
├── generate_report.py      # 实验报告生成工具
├── requirements.txt        # Python 依赖
└── README.md
```

## 功能模块说明

| 模块 | 功能 |
|---|---|
| `data_loader.py` | CSV 数据读取、文本清洗、标签编码、PyTorch Dataset 封装 |
| `train_bert.py` | bert-base-chinese 微调，含 AdamW 优化器、线性调度器、早停机制 |
| `textfooler_attack.py` | 词重要性计算 + 中文同义词替换，含 30+ 词对的欺诈领域同义词词典 |
| `fraud_r1.py` | 三层递进式 Prompt 改写：紧迫感 → 协商感 → 威胁感 |
| `prompt_attack.py` | 三种语义攻击：去关键词、自然化、受害者视角 |
| `evaluate.py` | 对比表格生成、汇总报告、JSON 结果保存 |
| `main.py` | 命令行参数驱动的一键实验流程 |

## 实验结果

实验完成后，结果保存在 `results/` 目录下，包含：

- 各实验方案的准确率、精确率、召回率、F1 分数
- TextFooler 与 PromptAttack 的攻击成功率对比
- 综合性能对比表

## 数据集

实验使用课堂提供的欺诈通话数据集：

- 训练集：14,363 条对话（欺诈 7,341 条，非欺诈 6,294 条）
- 测试集：2,677 条对话
- 标注维度：对话策略标签、通话类型、是否欺诈、欺诈子类型

## License

MIT
