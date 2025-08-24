# 🎯 Anomaly Detection in Ethereum Transactions with Unsupervised Graph Neural Network
# 以太坊交易中的无监督图神经网络异常检测

## 📋 项目概述 / Project Overview

**English:**
This repository implements an unsupervised anomaly detection system for Ethereum transactions using Graph Neural Networks (GNNs). The project combines GraphSAGE embeddings with AutoEncoder reconstruction and One-Class SVM for detecting suspicious/phishing nodes in Ethereum transaction networks.

**中文:**
本项目实现了一个基于图神经网络的以太坊交易无监督异常检测系统。项目结合了GraphSAGE嵌入、自编码器重构和单类支持向量机来检测以太坊交易网络中的可疑/钓鱼节点。

## 🚀 技术栈 / Tech Stack

- **图神经网络 / Graph Neural Networks**: GraphSAGE, GAT (Graph Attention Network)
- **异常检测 / Anomaly Detection**: AutoEncoder + One-Class SVM (OCSVM)
- **深度学习框架 / Deep Learning**: PyTorch, PyTorch Geometric
- **数据处理 / Data Processing**: NetworkX, NumPy, scikit-learn
- **可视化 / Visualization**: Matplotlib, tqdm

## 📂 项目结构 / Project Structure

```
├── README.md                                     # 项目说明
├── requirements.txt                              # Python依赖
├── capstone2.ipynb                              # 主要分析笔记本
├── GraphSGAE+autoencoder+ocsvm_rw.ipynb        # GraphSAGE + AutoEncoder + OCSVM 完整流程
├── supergat_autoencoder_ocsvm_splite_trainset.ipynb # SuperGAT变体实现
├── graphsage_frontend.py                        # GraphSAGE前端实现
├── check_training_data.py                       # 训练数据格式检查工具
├── create_supergat_compat_files.py             # SuperGAT兼容文件创建
├── global_performance_summary.json              # 全局性能摘要
├── *.png                                        # 分析结果可视化图表
├── *.npy                                        # 预计算的Top-K结果
└── *.csv                                        # 性能分析结果
```

## 🔧 安装和设置 / Installation and Setup

### 环境要求 / Requirements
- Python 3.7+
- CUDA compatible GPU (optional, CPU also supported)

### 安装依赖 / Install Dependencies
```bash
pip install -r requirements.txt
```

## 📊 数据集 / Dataset

**训练集 / Training Set:**
- `dataset_k-hop2/training_subgraph_random_walk_1.pkl`
- `dataset_k-hop2/training_subgraph_random_walk_2.pkl` 
- `dataset_k-hop2/training_subgraph_random_walk_3.pkl`

**测试集 / Test Set:**
- `dataset_k-hop2/realistic_test_graph.pkl`

## 🏃‍♂️ 使用方法 / Usage

### 1. 数据检查 / Data Verification
```bash
python check_training_data.py
```

### 2. 运行主要分析 / Run Main Analysis
打开并运行 / Open and run:
- `capstone2.ipynb` - 主要分析流程
- `GraphSGAE+autoencoder+ocsvm_rw.ipynb` - 完整的GraphSAGE实现

### 3. GraphSAGE前端 / GraphSAGE Frontend
```bash
python graphsage_frontend.py
```

## 📈 模型架构 / Model Architecture

### GraphSAGE + AutoEncoder + OCSVM Pipeline:

1. **图嵌入 / Graph Embedding**: 使用GraphSAGE学习节点嵌入
2. **重构 / Reconstruction**: AutoEncoder重构节点特征
3. **异常检测 / Anomaly Detection**: One-Class SVM检测异常
4. **评估 / Evaluation**: Precision@K, Top-K异常节点检测

## 🎯 性能指标 / Performance Metrics

- **总节点数 / Total Nodes**: 84,285
- **异常节点数 / Anomaly Nodes**: 3,637
- **异常比例 / Anomaly Ratio**: 4.32%
- **评估指标 / Metrics**: Precision@K, F1-Score, ROC-AUC

## 📊 结果分析 / Results Analysis

项目包含多种分析可视化：
- 全局性能分析图表
- Top-K混淆矩阵分析
- GraphSAGE vs SuperGAT性能对比
- 异常节点检测效果评估

## 🤝 贡献 / Contributing

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证 / License

请参考项目根目录下的LICENSE文件。

## 📧 联系方式 / Contact

如有问题或建议，请通过GitHub Issues联系。