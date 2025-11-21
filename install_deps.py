"""
快速安装脚本 - 安装必要的依赖包
"""

import subprocess
import sys

def install_packages():
    """安装必要的包"""
    packages = [
        'streamlit', 
        'plotly', 
        'python-dotenv',
        'torch',
        'transformers',
        'pandas', 
        'requests', 
        'numpy'
    ]
    
    print("📦 正在安装必要的依赖包...")
    print("=" * 50)
    
    for package in packages:
        print(f"\n🔧 安装 {package}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, check=True)
            
            print(f"✅ {package} 安装成功")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} 安装失败")
            print(f"错误信息: {e.stderr}")
    
    print("\n" + "=" * 50)
    print("📋 安装完成！现在可以运行 python run_app.py")

if __name__ == "__main__":
    install_packages()