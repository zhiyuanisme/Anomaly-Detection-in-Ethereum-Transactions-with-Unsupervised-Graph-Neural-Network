# 数据集说明 / Dataset Instructions

## 📊 数据集要求 / Dataset Requirements

本项目需要以下数据文件才能正常运行：
The following data files are required for this project to run:

### 训练数据 / Training Data
```
dataset_k-hop2/
├── training_subgraph_random_walk_1.pkl
├── training_subgraph_random_walk_2.pkl
├── training_subgraph_random_walk_3.pkl
└── realistic_test_graph.pkl
```

### 数据格式 / Data Format

**训练子图 / Training Subgraphs:**
- 格式: Python pickle文件 (.pkl)
- 内容: NetworkX图对象或PyTorch Geometric Data对象
- 用途: 无监督训练GraphSAGE模型

**测试图 / Test Graph:**
- 格式: Python pickle文件 (.pkl) 
- 内容: 包含节点标签的真实以太坊交易网络
- 用途: 评估异常检测性能

### 获取数据集 / Obtaining the Dataset

1. **原始数据源 / Original Data Source:**
   - 以太坊交易数据可从公开的区块链浏览器获取
   - Ethereum transaction data can be obtained from public blockchain explorers

2. **数据预处理 / Data Preprocessing:**
   - 构建交易网络图
   - 执行随机游走采样
   - 标记已知的钓鱼/可疑地址

3. **检查数据 / Check Data:**
   ```bash
   python check_training_data.py
   ```

### 数据集创建指南 / Dataset Creation Guide

如果您需要创建自己的数据集，请参考以下步骤：

1. **收集以太坊交易数据**
2. **构建交易网络图**
3. **执行k-hop采样和随机游走**
4. **保存为pickle格式**

### 注意事项 / Notes

- 数据文件由于大小限制未包含在仓库中
- 请确保数据文件放置在正确的目录结构中
- 运行前请使用 `check_training_data.py` 验证数据格式

**Data files are not included in the repository due to size constraints**
**Please ensure data files are placed in the correct directory structure**
**Use `check_training_data.py` to verify data format before running**