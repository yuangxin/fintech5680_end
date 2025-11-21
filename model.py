
import torch
from torch import nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
# 只导入需要的transformers模块，避免TensorFlow冲突
from transformers import BertTokenizer, BertForSequenceClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class StockLSTMRegressor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        """
        LSTM模型用于股价预测
        Args:
            input_size: 输入特征数量（涨幅 + 情感指标 = 2）
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            output_size: 输出大小（1表示预测下个月的涨幅）
            dropout: dropout率
        """
        super(StockLSTMRegressor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Linear layer for final prediction
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        # sequence_length = 4 (过去4个月)
        # input_size = 2 (涨幅 + 情感指标)

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]

        # Dropout
        out = self.dropout(last_output)

        # Linear layer
        out = self.linear(out)

        return out

def calculate_sentiment_score(headlines, finbert_model, tokenizer):
    # 参数验证
    if finbert_model is None:
        print("❌ FinBERT模型为None")
        return 0.0
    
    if tokenizer is None:
        print("❌ Tokenizer为None")
        return 0.0

    if not headlines or len(headlines) == 0:
        print("⚠️ 没有新闻标题用于情感分析")
        return 0.0

    sentiment_scores = []
    
    try:
        device = next(finbert_model.parameters()).device
    except Exception as e:
        print(f"❌ 获取模型设备失败: {e}")
        device = torch.device("cpu")
        
    print(f"📝 正在分析 {len(headlines)} 个新闻标题的情感")

    for headline in headlines:
        try:
            # 使用FinBERT计算情感得分
            encoded = tokenizer.encode_plus(
                headline,
                add_special_tokens=True,
                max_length=64,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # 移动到正确的设备
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            with torch.no_grad():
                outputs = finbert_model(input_ids=input_ids, attention_mask=attention_mask)

                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]

                # FinBERT输出: [negative, neutral, positive]
                # 转换为连续的情感得分: positive - negative
                weighted_score = probabilities[0] - probabilities[1]
                sentiment_scores.append(weighted_score)

        except Exception as e:
            print(f"处理标题时出错: {headline[:50]}... 错误: {e}")
            # 出错时使用中性得分
            sentiment_scores.append(0.0)

    return float(np.mean(sentiment_scores)) if sentiment_scores else 0.0
