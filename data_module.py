import requests
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Try multiple methods to get API key
POLYGON_API_KEY = None

# Method 1: Environment variable
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')

# Method 2: Config file fallback
if not POLYGON_API_KEY:
    try:
        from config import POLYGON_API_KEY
    except ImportError:
        pass

# Method 3: Hardcoded fallback for server deployment
if not POLYGON_API_KEY:
    POLYGON_API_KEY = "l7eyBqnxp9XsobMxIBIVHx69zqRlY5rc"

print(f"🔑 Polygon API Key Status: {'✅ Loaded (' + str(len(POLYGON_API_KEY)) + ' chars)' if POLYGON_API_KEY else '❌ Missing'}")

# Monthly Stock Price History Data
def get_stock_price_history(ticker: str, start_date: str, end_date: str):
  try:
      if not POLYGON_API_KEY:
          print("❌ POLYGON_API_KEY 未设置")
          return None
      
      # Polygon.io API URL for stock data
      url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/month/{start_date}/{end_date}?apiKey={POLYGON_API_KEY}'
      
      print(f"请求URL: {url}")
      print(f"API密钥: {POLYGON_API_KEY[:10] if POLYGON_API_KEY else 'None'}...")  # 只显示前10个字符

      # Make the GET request with timeout
      response = requests.get(url, timeout=30)

      # Check if the request was successful
      if response.status_code == 200:
          # Parse the JSON data
          data = response.json()
          print(f"API响应状态: 成功")
          print(f"API响应数据: {data}")
          
          # 检查是否有results
          if 'results' in data and data['results'] and len(data['results']) > 0:
              prices = []
              for result in data['results']:
                  if 'vw' in result and result['vw'] is not None:
                      prices.append(float(result['vw']))
                  elif 'c' in result and result['c'] is not None:
                      prices.append(float(result['c']))  # 使用收盘价作为备选
              
              if prices:
                  print(f"✅ 成功获取 {len(prices)} 个月的股价数据")
                  return prices
              else:
                  print(f"❌ 结果中没有有效的价格数据")
                  return None
          else:
              print(f"❌ API返回了空结果或无results字段: {data}")
              return None
      else:
          print(f'❌ API请求失败: HTTP {response.status_code}')
          print(f'响应内容: {response.text}')
          return None
          
  except Exception as e:
      print(f"❌ 获取股价数据时发生异常: {str(e)}")
      import traceback
      traceback.print_exc()
      return None
