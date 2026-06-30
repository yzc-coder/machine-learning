# -*- coding: utf-8 -*-
"""
数据加载与预处理模块
====================
功能：加载欺诈通话数据集，进行数据清洗、标签编码、文本预处理，
      并将数据划分为训练集和测试集。
"""

import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class FraudCallDataset(Dataset):
    """
    欺诈通话数据集类，用于 PyTorch DataLoader
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class DataProcessor:
    """
    数据处理器：加载、清洗、预处理欺诈通话数据
    """
    def __init__(self, data_dir='欺诈通话数据集'):
        self.data_dir = data_dir
        self.label_encoders = {}
        self.train_df = None
        self.test_df = None

    def clean_text(self, text):
        """
        清洗对话文本
        - 去除"音频内容："前缀
        - 保留 left:/right: 对话结构
        - 去除多余空白
        """
        if pd.isna(text):
            return ""
        text = str(text)
        # 去除音频内容标记
        text = re.sub(r'音频内容[：:]\s*', '', text)
        # 规范化空白
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def load_data(self):
        """
        加载训练集和测试集
        """
        train_path = os.path.join(self.data_dir, '训练集结果.csv')
        test_path = os.path.join(self.data_dir, '测试集结果.csv')

        self.train_df = pd.read_csv(train_path, encoding='utf-8-sig')
        self.test_df = pd.read_csv(test_path, encoding='utf-8-sig')

        print(f"训练集大小: {len(self.train_df)}")
        print(f"测试集大小: {len(self.test_df)}")

        return self.train_df, self.test_df

    def preprocess(self):
        """
        完整预处理流程
        """
        if self.train_df is None:
            self.load_data()

        # 清洗文本
        self.train_df['cleaned_text'] = self.train_df['specific_dialogue_content'].apply(self.clean_text)
        self.test_df['cleaned_text'] = self.test_df['specific_dialogue_content'].apply(self.clean_text)

        # 过滤空文本
        self.train_df = self.train_df[self.train_df['cleaned_text'].str.len() > 0].reset_index(drop=True)
        self.test_df = self.test_df[self.test_df['cleaned_text'].str.len() > 0].reset_index(drop=True)

        # 编码 is_fraud 标签
        le_fraud = LabelEncoder()
        # 处理 NaN 值，统一标记为 "Unknown"
        self.train_df['is_fraud'] = self.train_df['is_fraud'].fillna('Unknown')
        self.test_df['is_fraud'] = self.test_df['is_fraud'].fillna('Unknown')

        all_fraud_labels = pd.concat([self.train_df['is_fraud'], self.test_df['is_fraud']])
        le_fraud.fit(all_fraud_labels)
        self.label_encoders['is_fraud'] = le_fraud

        self.train_df['fraud_label'] = le_fraud.transform(self.train_df['is_fraud'])
        self.test_df['fraud_label'] = le_fraud.transform(self.test_df['is_fraud'])

        print(f"is_fraud 标签映射: {dict(zip(le_fraud.classes_, le_fraud.transform(le_fraud.classes_)))}")

        # 编码 fraud_type
        le_ftype = LabelEncoder()
        all_ftypes = pd.concat([self.train_df['fraud_type'].fillna('Non-fraud'), 
                                self.test_df['fraud_type'].fillna('Non-fraud')])
        le_ftype.fit(all_ftypes)
        self.label_encoders['fraud_type'] = le_ftype

        self.train_df['fraud_type_label'] = le_ftype.transform(self.train_df['fraud_type'].fillna('Non-fraud'))
        self.test_df['fraud_type_label'] = le_ftype.transform(self.test_df['fraud_type'].fillna('Non-fraud'))

        # 编码 interaction_strategy
        le_strategy = LabelEncoder()
        all_strategies = pd.concat([self.train_df['interaction_strategy'], self.test_df['interaction_strategy']])
        le_strategy.fit(all_strategies.astype(str))
        self.label_encoders['interaction_strategy'] = le_strategy

        print(f"欺诈类型 ({len(le_ftype.classes_)} 类): {le_ftype.classes_.tolist()}")
        print(f"对话策略 ({len(le_strategy.classes_)} 类): {le_strategy.classes_.tolist()}")

        return self.train_df, self.test_df

    def get_fraud_binary_data(self):
        """
        获取二分类（欺诈/非欺诈）数据，去除 Unknown 标签
        """
        train = self.train_df[self.train_df['is_fraud'].isin(['True', 'False'])].copy()
        test = self.test_df[self.test_df['is_fraud'].isin(['True', 'False'])].copy()

        train['binary_label'] = (train['is_fraud'] == 'True').astype(int)
        test['binary_label'] = (test['is_fraud'] == 'True').astype(int)

        return train, test

    def create_dataloaders(self, tokenizer, batch_size=16, max_length=512):
        """
        创建训练和测试 DataLoader
        """
        train, test = self.get_fraud_binary_data()

        train_dataset = FraudCallDataset(
            train['cleaned_text'].tolist(),
            train['binary_label'].tolist(),
            tokenizer,
            max_length
        )
        test_dataset = FraudCallDataset(
            test['cleaned_text'].tolist(),
            test['binary_label'].tolist(),
            tokenizer,
            max_length
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"训练批次: {len(train_loader)}, 测试批次: {len(test_loader)}")

        return train_loader, test_loader, train, test

    def load_textfooler_data(self):
        """
        加载 TextFooler 攻击后的数据（Excel 文件）
        """
        xlsx_path = 'TextFooler攻击后新的欺诈通话数据.xlsx'
        if os.path.exists(xlsx_path):
            df = pd.read_excel(xlsx_path)
            col_orig = df.columns[0]
            col_attacked = df.columns[1]
            # 清洗
            df['original_clean'] = df[col_orig].apply(self.clean_text)
            df['attacked_clean'] = df[col_attacked].apply(self.clean_text)
            print(f"TextFooler 数据加载完成: {len(df)} 对样本")
            return df
        else:
            print(f"警告: 未找到 {xlsx_path}")
            return None

    def get_statistics(self):
        """
        输出数据集统计信息
        """
        print("\n" + "=" * 60)
        print("数据集统计信息")
        print("=" * 60)

        train, test = self.get_fraud_binary_data()

        print(f"\n训练集: {len(train)} 条")
        print(f"  - 欺诈: {(train['binary_label']==1).sum()} 条 ({(train['binary_label']==1).mean()*100:.1f}%)")
        print(f"  - 非欺诈: {(train['binary_label']==0).sum()} 条 ({(train['binary_label']==0).mean()*100:.1f}%)")

        print(f"\n测试集: {len(test)} 条")
        print(f"  - 欺诈: {(test['binary_label']==1).sum()} 条 ({(test['binary_label']==1).mean()*100:.1f}%)")
        print(f"  - 非欺诈: {(test['binary_label']==0).sum()} 条 ({(test['binary_label']==0).mean()*100:.1f}%)")

        # 对话长度统计
        lens = self.train_df['cleaned_text'].apply(len)
        print(f"\n对话长度统计 (训练集):")
        print(f"  - 平均: {lens.mean():.0f} 字符")
        print(f"  - 中位数: {lens.median():.0f} 字符")
        print(f"  - 最大: {lens.max()} 字符")
        print(f"  - 最小: {lens.min()} 字符")

        return train, test


if __name__ == '__main__':
    processor = DataProcessor()
    processor.load_data()
    train_df, test_df = processor.preprocess()
    processor.get_statistics()

    # 加载 TextFooler 数据
    tf_data = processor.load_textfooler_data()
    if tf_data is not None:
        print(f"\nTextFooler 攻击样本预览:")
        print(f"原始: {tf_data.iloc[0]['original_clean'][:200]}...")
        print(f"攻击: {tf_data.iloc[0]['attacked_clean'][:200]}...")
