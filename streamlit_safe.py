import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import traceback

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Page Configuration
st.set_page_config(
    page_title="📈 Smart Stock Prediction System",
    page_icon="📈",
    layout="centered"
)

# Page Title
st.title("📈 Smart Stock Prediction System")
st.markdown("---")

# 安全的模块导入
def safe_import_modules():
    """安全地导入所有必需的模块"""
    try:
        # 导入基础模块
        from data_module import get_stock_price_history
        from date_utils import get_first_last_days
        from model import StockLSTMRegressor, calculate_sentiment_score
        from main import predict_stock_end_to_end
        
        # 导入深度学习模块
        import torch
        import calendar
        import requests
        from transformers import BertTokenizer, BertForSequenceClassification
        from dotenv import load_dotenv
        
        # 加载环境变量
        load_dotenv()
        
        return True, "All modules imported successfully"
        
    except Exception as e:
        error_msg = f"Module import failed: {str(e)}"
        st.error(f"❌ {error_msg}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        return False, error_msg

# 检查模块导入
modules_ok, import_message = safe_import_modules()

if modules_ok:
    st.success(f"✅ {import_message}")
else:
    st.warning("⚠️ Some features may be unavailable, please check dependency installation")

# 加载数据和模型
@st.cache_data
def load_stock_list():
    """Load stock list"""
    try:
        csv_file_path = "tickers_and_names.csv"
        if not os.path.exists(csv_file_path):
            st.warning(f"Stock list file not found: {csv_file_path}")
            return pd.DataFrame({'Ticker': ['AAPL', 'MSFT', 'GOOGL'], 'Name': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.']})
        
        stocks_df = pd.read_csv(csv_file_path)
        return stocks_df
    except Exception as e:
        st.error(f"Failed to load stock list: {e}")
        return pd.DataFrame({'Ticker': ['AAPL', 'MSFT', 'GOOGL'], 'Name': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.']})

@st.cache_resource
def load_ai_models():
    """Safely load AI models"""
    try:
        if not modules_ok:
            st.error("❌ Cannot load models: module import failed")
            return None, None
        
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification
        
        # Load FinBERT
        model_name = "ProsusAI/finbert"
        st.info(f"🔄 Loading FinBERT model: {model_name}")
        
        tokenizer = BertTokenizer.from_pretrained(model_name)
        finbert_model = BertForSequenceClassification.from_pretrained(model_name)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        finbert_model = finbert_model.to(device)
        finbert_model.eval()
        
        st.success(f"✅ Model loaded successfully, using device: {device}")
        return tokenizer, finbert_model
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None

# 1. Stock Selection
st.subheader("🔍 Select Stock")
stocks_df = load_stock_list()

if not stocks_df.empty:
    # Create selection box
    stock_options = [f"{row['Ticker']} - {row['Name']}" for _, row in stocks_df.iterrows()]
    selected_stock = st.selectbox(
        "Please select a stock:",
        options=stock_options,
        index=0,
        help="Choose a stock from the list for prediction"
    )
    
    # Extract stock ticker
    selected_ticker = selected_stock.split(" - ")[0] if selected_stock else ""
else:
    st.error("❌ Unable to load stock list")
    selected_ticker = ""

st.markdown("---")

# 2. Prediction Button
st.subheader("🚀 Start Prediction")

# Environment check
env_status = []

# Check model file
model_path = "model/LSTM_FINTECH.pth"
if os.path.exists(model_path):
    env_status.append("✅ LSTM model file exists")
else:
    env_status.append("❌ LSTM model file missing")

# Check environment variables with multiple fallback methods
from dotenv import load_dotenv

# Try to load .env from multiple possible locations
env_loaded = False
polygon_key = None

# Method 1: Load from current directory
try:
    load_dotenv()
    polygon_key = os.getenv('POLYGON_API_KEY')
    if polygon_key:
        env_loaded = True
except:
    pass

# Method 2: Load from absolute path
if not env_loaded:
    try:
        env_file = os.path.join(current_dir, '.env')
        load_dotenv(env_file)
        polygon_key = os.getenv('POLYGON_API_KEY')
        if polygon_key:
            env_loaded = True
    except:
        pass

# Method 3: Hardcoded fallback (for server deployment)
if not polygon_key:
    polygon_key = "l7eyBqnxp9XsobMxIBIVHx69zqRlY5rc"
    env_loaded = True

if polygon_key and len(polygon_key) > 10:
    env_status.append(f"✅ Polygon API Key configured ({polygon_key[:10]}...)")
else:
    env_status.append("❌ Polygon API Key not configured")

# Display environment status
with st.expander("🔧 Environment Check", expanded=False):
    for status in env_status:
        if "✅" in status:
            st.success(status)
        else:
            st.error(status)

predict_button = st.button(
    "📊 Start Stock Prediction", 
    type="primary",
    use_container_width=True,
    help="Click to start AI-powered stock price prediction",
    disabled=not modules_ok
)

# 3. Results Display Area
st.subheader("📊 Prediction Results")

if predict_button and selected_ticker and modules_ok:
    # Show loading status
    with st.spinner("🔄 Analyzing stock data and making predictions..."):
        try:
            # Safely set global variables with improved API key handling
            import builtins
            import torch
            
            builtins.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            builtins.API_KEY = "l7eyBqnxp9XsobMxIBIVHx69zqRlY5rc"
            builtins.base_url = "https://api.massive.com/v2/reference/news"
            
            # Set Polygon API key with fallback
            if polygon_key:
                os.environ['POLYGON_API_KEY'] = polygon_key
            
            # 加载模型
            tokenizer, finbert_model = load_ai_models()
            
            if tokenizer is None or finbert_model is None:
                st.error("❌ Unable to load AI models, prediction failed")
                st.stop()
            
            builtins.tokenizer = tokenizer
            builtins.finbert_model = finbert_model
            
            # 进行预测
            from main import predict_stock_end_to_end
            result = predict_stock_end_to_end(selected_ticker)
            
            if result['success']:
                # Display prediction results
                st.success("✅ Prediction completed!")
                
                # Basic information
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Current Price", 
                        f"${result['current_price']:.2f}",
                    )
                with col2:
                    change_pct = result['predicted_return'] * 100
                    st.metric(
                        "Predicted Price", 
                        f"${result['predicted_price']:.2f}",
                        f"{change_pct:+.2f}%"
                    )
                with col3:
                    trend = "📈 Bullish" if result['predicted_return'] > 0 else "📉 Bearish"
                    st.metric("Trend Prediction", trend)
                
                # Detailed results
                st.markdown("### 🎯 Prediction Details")
                
                # Highlight predicted return
                predicted_return_pct = result['predicted_return'] * 100
                if predicted_return_pct > 0:
                    st.success(f"📈 **Growth Forecast: +{predicted_return_pct:.2f}%** (Bullish)")
                else:
                    st.error(f"📉 **Growth Forecast: {predicted_return_pct:.2f}%** (Bearish)")
                
                # Prediction summary table
                prediction_summary = pd.DataFrame({
                    'Metric': ['Current Price', 'Predicted Price', 'Predicted Return', 'Price Change', 'Investment Advice'],
                    'Value': [
                        f"${result['current_price']:.2f}",
                        f"${result['predicted_price']:.2f}", 
                        f"**{result['predicted_return']*100:+.2f}%**",
                        f"${result['predicted_price'] - result['current_price']:+.2f}",
                        "🟢 BUY" if result['predicted_return'] > 0.02 else "🔴 SELL" if result['predicted_return'] < -0.02 else "🟡 HOLD"
                    ]
                })
                st.dataframe(prediction_summary, use_container_width=True, hide_index=True)
                
                # Historical monthly analysis
                st.markdown("### 📋 Historical Monthly Analysis")
                
                # Display monthly data
                if 'monthly_data' in result and result['monthly_data']:
                    months = result['monthly_data'].get('months', [])
                    returns = result['monthly_data'].get('returns', [])
                    sentiment_scores = result['monthly_data'].get('sentiment_scores', [])
                    prices = result['monthly_data'].get('prices', [])
                    
                    if months and returns and sentiment_scores and prices:
                        # Create detailed historical data table
                        historical_df = pd.DataFrame({
                            'Month': months,
                            'Stock Price ($)': [f"${p:.2f}" for p in prices[1:len(months)+1]], 
                            'Monthly Return (%)': [f"{r*100:+.2f}%" for r in returns],
                            'Sentiment Score': [f"{s:+.3f}" for s in sentiment_scores],
                            'Sentiment Rating': ['😊 Positive' if s > 0.1 else '😐 Neutral' if s > -0.1 else '😞 Negative' for s in sentiment_scores]
                        })
                        st.dataframe(historical_df, use_container_width=True, hide_index=True)
                        
                        # Simple price trend chart
                        st.markdown("### 📈 Price Trend")
                        
                        # Build chart data
                        if len(prices) >= len(months):
                            month_labels = months + ['Predicted']
                            price_data = prices[1:len(months)+1] + [result['predicted_price']]
                        else:
                            month_labels = months[:len(prices)] + ['Predicted']
                            price_data = prices + [result['predicted_price']]
                        
                        # Create line chart
                        chart_data = pd.DataFrame({
                            'Month': month_labels,
                            'Price': price_data
                        })
                        st.line_chart(chart_data.set_index('Month'))
                    else:
                        st.warning("⚠️ Monthly data incomplete")
                
                # News Headlines Section
                st.markdown("### 📰 Related News")
                
                if 'news_by_month' in result and result['news_by_month']:
                    # Collect all news articles
                    all_articles = []
                    for month, month_data in result['news_by_month'].items():
                        articles = month_data.get('articles', [])
                        for article in articles:
                            title = article.get('title', '')
                            url = article.get('article_url') or article.get('url') or article.get('amp_url') or article.get('homepage_url')
                            if title:
                                all_articles.append({
                                    'title': title,
                                    'url': url or '#',
                                    'published': article.get('published_utc', 'Unknown'),
                                    'month': month
                                })
                    
                    if all_articles:
                        st.markdown("**📊 Latest Stock-Related News:**")
                        
                        # Group by month
                        months_with_news = {}
                        for article in all_articles[:15]:
                            month = article['month']
                            if month not in months_with_news:
                                months_with_news[month] = []
                            months_with_news[month].append(article)
                        
                        for month, articles in months_with_news.items():
                            with st.expander(f"📅 {month} ({len(articles)} articles)", expanded=False):
                                for i, article in enumerate(articles, 1):
                                    title = article['title']
                                    url = article['url']
                                    
                                    if len(title) > 100:
                                        title = title[:97] + "..."
                                    
                                    if url and url != '#':
                                        st.markdown(f"{i}. [{title}]({url})")
                                    else:
                                        st.markdown(f"{i}. {title}")
                                    st.markdown(f"   📅 *Published: {article['published'][:10]}*")
                                    if i < len(articles):
                                        st.markdown("---")
                    else:
                        st.info("📰 No related news articles found")
                else:
                    st.info("📰 No news data available for this stock")
                    
            else:
                error_msg = result.get('error', 'Unknown error')
                st.error(f"❌ Prediction failed: {error_msg}")
                
                # Show detailed error information
                with st.expander("🔍 Detailed Error Information", expanded=False):
                    st.text(error_msg)
                    
        except Exception as e:
            st.error(f"❌ Exception occurred during prediction: {str(e)}")
            with st.expander("🔍 Detailed Error Information", expanded=False):
                st.text(traceback.format_exc())

elif predict_button and not selected_ticker:
    st.warning("⚠️ Please select a stock first")

elif predict_button and not modules_ok:
    st.error("❌ System modules not properly loaded, unable to make predictions")

# Page footer
st.markdown("---")
st.markdown("**💡 Note**: This system uses FinBERT sentiment analysis and LSTM neural networks for stock prediction. For reference only.")
st.markdown("**🔧 Technical Support**: If you encounter issues, please check dependency installation and API key configuration.")