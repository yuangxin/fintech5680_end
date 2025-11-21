import numpy as np
import os
import torch
import calendar
import requests

# 全局变量（会被 streamlit 应用设置）
API_KEY = "l7eyBqnxp9XsobMxIBIVHx69zqRlY5rc"
base_url = "https://api.massive.com/v2/reference/news"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_monthly_returns(prices):
    """计算月度涨幅"""
    if not prices or len(prices) < 2:
        return []
    
    returns = []
    for i in range(1, len(prices)):
        monthly_return = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(monthly_return)
    return returns
