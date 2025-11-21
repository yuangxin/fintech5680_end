
import torch
import numpy as np
import os
import calendar
import requests
from date_utils import get_first_last_days
from data_module import get_stock_price_history
from model import StockLSTMRegressor, calculate_sentiment_score

# Global variables with fallback configuration
API_KEY = None
base_url = "https://api.massive.com/v2/reference/news"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try to load API key from multiple sources
try:
    from config import NEWS_API_KEY
    API_KEY = NEWS_API_KEY
except ImportError:
    API_KEY = "l7eyBqnxp9XsobMxIBIVHx69zqRlY5rc"

print(f"🔑 News API Key Status: {'✅ Loaded' if API_KEY else '❌ Missing'}")

def calculate_monthly_returns(prices):
    """计算月度涨幅"""
    if not prices or len(prices) < 2:
        return []
    
    returns = []
    for i in range(1, len(prices)):
        monthly_return = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(monthly_return)
    return returns

def predict_stock_end_to_end(ticker_symbol: str, model_path: str = "model/LSTM_FINTECH.pth"):
    result = {
        'ticker': ticker_symbol,
        'success': False,
        'predicted_return': None,
        'predicted_price': None,
        'current_price': None,
        'monthly_data': {},
        'error': None
    }
    
    try:
        # 检查必要的全局变量
        import builtins
        if not hasattr(builtins, 'finbert_model') or builtins.finbert_model is None:
            raise ValueError("FinBERT模型未正确加载")
        if not hasattr(builtins, 'tokenizer') or builtins.tokenizer is None:
            raise ValueError("Tokenizer未正确加载")
        
        finbert_model = builtins.finbert_model
        tokenizer = builtins.tokenizer
        print("📅 第1步：获取日期范围")
        start_date, end_date, target_months = get_first_last_days()
        
        print(f"\n📰 第2步：搜索 {ticker_symbol} 的新闻数据")
        news_by_month = {}
        
        for month_label in target_months:
            year, month = month_label.split('-')
            last_day = calendar.monthrange(int(year), int(month))[1]
            
            params = {
                "ticker": ticker_symbol,
                "published_utc.gte": f"{year}-{month}-01T00:00:00Z",
                "published_utc.lte": f"{year}-{month}-{last_day:02d}T23:59:59Z",
                "limit": 30,
                "order": "descending",
                "sort": "published_utc",
            }
            
            headers = {"Authorization": f"Bearer {API_KEY}"}
            resp = requests.get(base_url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            
            news_by_month[month_label] = {
                'count': data.get("count", 0),
                'articles': data.get("results", [])
            }
            
        
        total_news = sum(data['count'] for data in news_by_month.values())
        
        print(f"\n💭 第3步：计算情感得分")
        monthly_sentiment_scores = {}
        
        for month_label, month_data in news_by_month.items():
            articles = month_data['articles']
            
            headlines = []
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                headline = title if title else description
                if headline:
                    headlines.append(headline)
            
            try:
                sentiment_score = calculate_sentiment_score(headlines, finbert_model, tokenizer)
                if sentiment_score is None:
                    sentiment_score = 0.0
            except Exception as e:
                print(f"计算 {month_label} 情感得分时出错: {e}")
                sentiment_score = 0.0
            
            monthly_sentiment_scores[month_label] = sentiment_score
            

        print(f"\n📈 第4步：获取 {ticker_symbol} 股价数据")
        stock_prices = get_stock_price_history(ticker_symbol, start_date, end_date)
        
        if not stock_prices or stock_prices is None:
            raise ValueError(f"无法获取 {ticker_symbol} 的股价数据")
        
        if len(stock_prices) < 5:
            raise ValueError(f"股价数据不足，需要5个月数据，获得 {len(stock_prices)} 个月")
        
        # 第5步：计算月度涨幅
        print(f"\n📊 第5步：计算月度涨幅")
        monthly_returns = calculate_monthly_returns(stock_prices)
        
        if len(monthly_returns) != 4:
            raise ValueError(f"期望4个月涨幅数据，实际获得 {len(monthly_returns)} 个")
        
        # 第6步：准备LSTM输入数据
        print(f"\n🔧 第6步：准备LSTM输入数据")
        sentiment_values = list(monthly_sentiment_scores.values())
        
        # 创建特征矩阵
        features = []
        for i, month in enumerate(target_months):
            month_features = [
                monthly_returns[i],  # 月度涨幅
                sentiment_values[i]  # 情感得分
            ]
            features.append(month_features)
        
        features_array = np.array(features, dtype=np.float32)
        
        # 第7步：加载LSTM模型
        print(f"\n🤖 第7步：加载LSTM模型")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        lstm_model = StockLSTMRegressor(
            input_size=2, hidden_size=64, num_layers=2, output_size=1, dropout=0.2
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            lstm_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            lstm_model.load_state_dict(checkpoint)
        
        lstm_model = lstm_model.to(device)
        lstm_model.eval()
        
        # 第8步：进行预测
        print(f"\n🔮 第8步：进行股价预测")
        input_tensor = torch.from_numpy(features_array).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = lstm_model(input_tensor)
            predicted_return = prediction.item()
        
        # 计算预测价格
        current_price = stock_prices[-1]
        predicted_price = current_price * (1 + predicted_return)
        
        # 保存结果
        result.update({
            'success': True,
            'predicted_return': predicted_return,
            'predicted_price': predicted_price,
            'current_price': current_price,
            'monthly_data': {
                'months': target_months,
                'returns': monthly_returns,
                'sentiment_scores': sentiment_values,
                'prices': stock_prices
            },
            'news_by_month': news_by_month
        })
        
        # 第9步：显示预测结果
        print(f"\n🎯预测结果")
        print("=" * 60)
        print(f"📊 股票代码: {ticker_symbol}")
        print(f"📈 预测涨幅: {predicted_return:.4f} ({predicted_return*100:.2f}%)")
        
        if predicted_return > 0:
            print("✅ 模型预测股价将上涨")
        else:
            print("⚠️ 模型预测股价将下跌")
        
        print(f"\n📋 输入特征详情:")
        for i, month in enumerate(target_months):
            print(f"   {month}: 涨幅={monthly_returns[i]:+.4f}, 情感得分={sentiment_values[i]:+.4f}")
            
    except Exception as e:
        error_msg = f"预测过程中出错: {str(e)}"
        print(f"❌ {error_msg}")
        result['error'] = error_msg
        import traceback
        traceback.print_exc()
    
    return result
