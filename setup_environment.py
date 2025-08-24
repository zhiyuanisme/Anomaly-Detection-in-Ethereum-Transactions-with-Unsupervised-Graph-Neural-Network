#!/usr/bin/env python3
"""
环境设置和依赖检查脚本
Environment Setup and Dependency Check Script

运行此脚本来验证项目所需的依赖是否正确安装
Run this script to verify that project dependencies are correctly installed
"""

import sys
import subprocess
import importlib
import pkg_resources

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本 / Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - 版本符合要求")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - 需要Python 3.7+")
        return False

def check_dependencies():
    """检查核心依赖"""
    print("\n📦 检查核心依赖 / Checking core dependencies...")
    
    core_deps = [
        'torch',
        'torch_geometric', 
        'networkx',
        'numpy',
        'sklearn',
        'matplotlib',
        'tqdm',
        'pandas'
    ]
    
    missing_deps = []
    
    for dep in core_deps:
        try:
            if dep == 'sklearn':
                import sklearn
            elif dep == 'torch_geometric':
                import torch_geometric
            else:
                importlib.import_module(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - 未安装")
            missing_deps.append(dep)
    
    return missing_deps

def install_requirements():
    """安装requirements.txt中的依赖"""
    print("\n🔧 安装依赖 / Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖安装完成 / Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ 依赖安装失败 / Failed to install dependencies")
        return False

def check_gpu_availability():
    """检查GPU可用性"""
    print("\n🎮 检查GPU可用性 / Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用 - {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA不可用，将使用CPU模式")
    except ImportError:
        print("⚠️  无法检查GPU - PyTorch未安装")

def main():
    """主函数"""
    print("🚀 以太坊交易异常检测项目环境检查")
    print("🚀 Ethereum Transaction Anomaly Detection Environment Check")
    print("=" * 60)
    
    # 检查Python版本
    if not check_python_version():
        print("\n❌ Python版本不符合要求，请升级到Python 3.7+")
        return
    
    # 检查依赖
    missing_deps = check_dependencies()
    
    if missing_deps:
        print(f"\n⚠️  发现缺失的依赖: {missing_deps}")
        response = input("是否自动安装缺失的依赖? (y/n): ")
        if response.lower() in ['y', 'yes', '是']:
            if install_requirements():
                print("\n🔄 重新检查依赖...")
                missing_deps = check_dependencies()
    
    # 检查GPU
    check_gpu_availability()
    
    # 最终报告
    print("\n" + "=" * 60)
    if not missing_deps:
        print("🎉 环境检查完成！所有依赖都已正确安装。")
        print("🎉 Environment check complete! All dependencies are correctly installed.")
        print("\n📝 接下来可以运行:")
        print("📝 Next, you can run:")
        print("   - python check_training_data.py")
        print("   - jupyter notebook capstone2.ipynb")
    else:
        print(f"❌ 仍有缺失的依赖: {missing_deps}")
        print("❌ Some dependencies are still missing")
        print("请手动安装: pip install " + " ".join(missing_deps))

if __name__ == "__main__":
    main()