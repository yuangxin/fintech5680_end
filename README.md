# 📈 Smart Stock Prediction System

## 🌐 Live Demo
**Access the application here:** https://fintech5680end-hz5t7pjv8wfhniwcyxtnge.streamlit.app/

---

## 📋 Project Overview

The Smart Stock Prediction System is an AI-powered financial application that combines **sentiment analysis** and **time series prediction** to forecast stock price movements. The system integrates **FinBERT** (Financial BERT) for news sentiment analysis with a custom **LSTM neural network** for price prediction, creating a comprehensive solution for financial market analysis.

---

## 🏗️ Model Architecture

### 1. **Dual-Model Architecture**

The system employs a sophisticated two-stage prediction pipeline:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   News Data     │───▶│   FinBERT        │───▶│  Sentiment      │
│   Collection    │    │   Sentiment      │    │  Scores         │
└─────────────────┘    │   Analysis       │    └─────────────────┘
                       └──────────────────┘             │
┌─────────────────┐                                     │
│   Stock Price   │                                     ▼
│   History       │───────────────────────────────▶┌─────────────────┐
└─────────────────┘                                │   LSTM Neural   │
                                                   │   Network       │
                                                   │   Predictor     │
                                                   └─────────────────┘
                                                            │
                                                            ▼
                                                   ┌─────────────────┐
                                                   │   Stock Price   │
                                                   │   Prediction    │
                                                   └─────────────────┘
```

### 2. **FinBERT Sentiment Analysis Model**

- **Base Model**: ProsusAI/finbert
- **Architecture**: BERT-based transformer (110M parameters)
- **Training**: Pre-trained on financial texts and news
- **Output**: Sentiment scores (-1 to +1) for financial headlines
- **Input Processing**: 
  - Tokenization with max length of 64 tokens
  - Special token handling ([CLS], [SEP])
  - Attention mask generation

### 3. **LSTM Price Prediction Model**

```python
class StockLSTMRegressor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(StockLSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)
```

**Model Specifications:**
- **Input Features**: 2 dimensions (monthly returns + sentiment scores)
- **Hidden Units**: 64 LSTM cells per layer
- **Layers**: 2 stacked LSTM layers
- **Dropout**: 20% for regularization
- **Output**: Single value (predicted monthly return)
- **Sequence Length**: 4 months of historical data

---

## 🧠 Core Algorithms & Principles

### 1. **Sentiment-Driven Price Prediction**

The system operates on the **Efficient Market Hypothesis** extension that incorporates behavioral finance principles:

```
Monthly_Return(t+1) = f(Historical_Returns(t-3:t), Sentiment_Scores(t-3:t))
```

**Mathematical Foundation:**
- **Feature Engineering**: Combines quantitative (price returns) and qualitative (sentiment) data
- **Temporal Dependencies**: LSTM captures long-term dependencies in sequential data
- **Non-linear Mapping**: Neural network learns complex relationships between inputs and outputs

### 2. **Multi-Source Data Fusion**

**News Sentiment Aggregation:**
```python
sentiment_score = Σ(finbert_output_i) / n_articles
where finbert_output_i ∈ [-1, +1]
```

**Feature Matrix Construction:**
```
Input_Matrix = [
    [return_t-3, sentiment_t-3],
    [return_t-2, sentiment_t-2], 
    [return_t-1, sentiment_t-1],
    [return_t,   sentiment_t]
]
```

### 3. **Prediction Pipeline**

1. **Data Collection Phase**:
   - Fetch 4 months of historical stock prices (Polygon.io API)
   - Collect news articles for each month (Financial news API)

2. **Feature Extraction Phase**:
   - Calculate monthly returns: `(P_t - P_t-1) / P_t-1`
   - Process news headlines through FinBERT
   - Aggregate sentiment scores per month

3. **Prediction Phase**:
   - Construct 4x2 feature matrix
   - Pass through trained LSTM model
   - Generate next month's expected return

---

## ⚙️ Implementation Features

### 1. **Real-time Data Processing**
- **Stock Price API**: Polygon.io integration for live market data
- **News Feed**: Multi-source financial news aggregation
- **Caching**: Streamlit caching for model and data optimization

### 2. **Robust Error Handling**
- **API Fallbacks**: Multiple data source redundancy
- **Model Loading**: Safe model checkpointing and loading
- **Input Validation**: Comprehensive data quality checks

### 3. **Interactive Web Interface**
- **Stock Selection**: Dropdown menu with major stocks
- **Real-time Predictions**: On-demand model inference
- **Visualization**: Historical trends and prediction charts
- **News Integration**: Related financial news display

### 4. **Model Performance Features**
- **Device Optimization**: Automatic GPU/CPU selection
- **Memory Management**: Efficient tensor operations
- **Batch Processing**: Optimized inference pipeline

---

## 🚀 System Capabilities

### **Core Functionalities**

1. **📊 Stock Price Prediction**
   - Predicts next month's stock price movement
   - Provides percentage change estimates
   - Generates buy/sell/hold recommendations

2. **📰 Sentiment Analysis**
   - Processes financial news headlines
   - Quantifies market sentiment impact
   - Correlates news sentiment with price movements

3. **📈 Technical Analysis**
   - Historical price trend analysis
   - Monthly return calculations
   - Volatility assessment

4. **🎯 Investment Guidance**
   - Risk-based investment recommendations
   - Trend identification (Bullish/Bearish)
   - Confidence indicators

### **Advanced Features**

- **Multi-timeframe Analysis**: Historical monthly data examination
- **News Impact Scoring**: Quantitative news sentiment impact
- **Interactive Dashboards**: Real-time prediction interface
- **Model Explainability**: Feature importance visualization

---

## 🛠️ Technical Stack

### **Backend Technologies**
- **Deep Learning**: PyTorch for neural network implementation
- **NLP**: Hugging Face Transformers (FinBERT)
- **Data Processing**: Pandas, NumPy for numerical computations
- **APIs**: Polygon.io (stock data), Financial news APIs

### **Frontend & Deployment**
- **Web Framework**: Streamlit for interactive UI
- **Visualization**: Plotly for dynamic charts
- **Deployment**: Streamlit Cloud hosting
- **Configuration**: Environment-based API key management

### **Model Architecture**
```
Dependencies:
├── PyTorch >= 1.9.0          # Neural network framework
├── Transformers >= 4.0.0     # FinBERT implementation  
├── Streamlit >= 1.28.0       # Web application framework
├── Pandas >= 1.3.0           # Data manipulation
├── NumPy >= 1.21.0           # Numerical computations
└── Requests >= 2.25.0        # API communications
```

---

## 🚦 Quick Start Guide

### **1. Local Development**
```bash
# Clone repository
git clone <repository-url>
cd fintech-stock-prediction

# Install dependencies  
pip install -r requirements.txt

# Launch application
streamlit run streamlit_server.py
```

### **2. Server Deployment**
```bash
# Upload required files:
# - streamlit_server.py
# - main.py, model.py, data_module.py, date_utils.py
# - config.py, requirements.txt
# - model/LSTM_FINTECH.pth
# - tickers_and_names.csv

# Install dependencies
pip install -r requirements.txt

# Start server
streamlit run streamlit_server.py --server.port 8501
```

### **3. Access Application**
- **Local**: http://localhost:8501
- **Live Demo**: https://fintech5680end-hz5t7pjv8wfhniwcyxtnge.streamlit.app/

---

## 📊 Model Performance & Validation

### **Training Specifications**
- **Dataset**: 4-month rolling windows of stock data + news sentiment
- **Validation**: Time-series cross-validation
- **Metrics**: Mean Absolute Error (MAE), Directional Accuracy
- **Optimization**: Adam optimizer with learning rate scheduling

### **Architecture Justification**
- **LSTM Selection**: Captures temporal dependencies in financial time series
- **FinBERT Integration**: Financial domain-specific sentiment understanding
- **Feature Engineering**: Combines quantitative and qualitative market signals
- **Ensemble Approach**: Leverages both technical and fundamental analysis

---

## 🔬 Research Contributions

This system demonstrates several key innovations in financial machine learning:

1. **Multi-Modal Financial Prediction**: Successfully integrates textual sentiment with numerical price data
2. **Real-time Inference Pipeline**: Provides practical deployment of academic research
3. **Domain-Specific NLP**: Utilizes financial BERT for accurate sentiment quantification
4. **End-to-end System**: Complete pipeline from data collection to user interface

---

## 📈 Future Enhancements

- **Enhanced Models**: Integration of transformer-based time series models
- **Additional Features**: Technical indicators, macro-economic data
- **Multi-asset Support**: Extension to cryptocurrency, commodities, forex
- **Advanced Analytics**: Risk metrics, portfolio optimization features

---

## 📞 Technical Support

For technical issues or questions regarding the model architecture and implementation:

- **Live Application**: https://fintech5680end-hz5t7pjv8wfhniwcyxtnge.streamlit.app/
- **Model Type**: LSTM + FinBERT Ensemble
- **Framework**: PyTorch + Streamlit
- **Deployment**: Streamlit Cloud Platform

---

*This system represents a practical implementation of modern AI techniques in financial markets, combining deep learning with natural language processing for comprehensive market analysis.*