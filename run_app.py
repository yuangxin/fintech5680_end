"""
智能股票预测系统启动脚本

使用方法:
1. 确保已安装所需依赖: pip install streamlit plotly torch transformers pandas requests python-dotenv
2. 在 main.ipynb 中运行所有 %%writefile 单元格来创建模块文件
3. 运行此脚本: python run_app.py
"""

import subprocess
import sys
import os

def check_dependencies():
    """检查必要的依赖包"""
    required_packages = [
        'streamlit', 'plotly', 'torch', 'transformers', 
        'pandas', 'requests', 'python-dotenv', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))  # 处理包名差异
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - 未安装")
    
    if missing_packages:
        print(f"\n📦 正在安装缺失的包...")
        for package in missing_packages:
            try:
                import subprocess
                result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                      capture_output=True, text=True, check=True)
                print(f"✅ {package} 安装成功")
            except subprocess.CalledProcessError as e:
                print(f"❌ {package} 安装失败: {e}")
                return False
    
    return True

def check_files():
    """检查必要的文件是否存在"""
    required_files = [
        'streamlit_simple.py',
        'tickers_and_names.csv',
        'model/LSTM_FINTECH.pth'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} - 文件不存在")
    
    # 检查模块文件（应该通过 main.ipynb 创建）
    module_files = [
        'data_module.py', 
        'date_utils.py',
        'model.py',
        'main.py'
    ]
    
    missing_modules = []
    for file_path in module_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            missing_modules.append(file_path)
            print(f"❌ {file_path} - 模块文件不存在")
    
    if missing_modules:
        print(f"\n⚠️ 请在 main.ipynb 中运行所有 %%writefile 单元格来创建模块文件")
        return False
    
    if missing_files:
        print(f"\n⚠️ 缺失必要文件，请检查项目结构")
        return False
    
    return True

def run_streamlit():
    """启动 Streamlit 应用"""
    try:
        print("\n🚀 启动 Streamlit 应用...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_simple.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
    except KeyboardInterrupt:
        print("\n👋 应用已停止")

def main():
    """主函数"""
    print("📈 智能股票预测系统 - 启动检查")
    print("=" * 50)
    
    print("\n1. 检查 Python 依赖包...")
    if not check_dependencies():
        print("\n❌ 依赖包检查失败，请先安装必要的包")
        return
    
    print("\n2. 检查项目文件...")
    if not check_files():
        print("\n❌ 文件检查失败，请确保所有必要文件存在")
        return
    
    print("\n✅ 所有检查通过！")
    print("\n" + "=" * 50)
    
    # 启动应用
    run_streamlit()

if __name__ == "__main__":
    main()