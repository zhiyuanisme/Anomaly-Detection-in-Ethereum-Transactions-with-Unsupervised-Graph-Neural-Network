#!/usr/bin/env python3
"""检查训练子集数据格式的简单脚本"""

import pickle
import os

def check_training_data():
    """检查训练子集数据的格式"""
    
    training_files = [
        'dataset_k-hop2/training_subgraph_random_walk_1.pkl',
        'dataset_k-hop2/training_subgraph_random_walk_2.pkl', 
        'dataset_k-hop2/training_subgraph_random_walk_3.pkl'
    ]
    
    for i, file_path in enumerate(training_files, 1):
        if os.path.exists(file_path):
            print(f"\n=== 训练子集 {i} ===")
            print(f"文件: {file_path}")
            
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                print(f"数据类型: {type(data)}")
                print(f"是否为元组: {isinstance(data, tuple)}")
                
                if hasattr(data, '__len__'):
                    print(f"数据长度: {len(data)}")
                
                if isinstance(data, tuple):
                    print(f"元组长度: {len(data)}")
                    for j, item in enumerate(data):
                        print(f"  元组[{j}]类型: {type(item)}")
                        
                        # 检查是否是PyG数据
                        if hasattr(item, 'x'):
                            print(f"    节点特征形状: {item.x.shape}")
                        if hasattr(item, 'edge_index'):
                            print(f"    边索引形状: {item.edge_index.shape}")
                            print(f"    节点数: {item.x.shape[0] if hasattr(item, 'x') else 'Unknown'}")
                            print(f"    边数: {item.edge_index.shape[1] if hasattr(item, 'edge_index') else 'Unknown'}")
                        
                        # 检查是否是NetworkX图
                        if hasattr(item, 'nodes'):
                            print(f"    NetworkX节点数: {len(item.nodes())}")
                        if hasattr(item, 'edges'):
                            print(f"    NetworkX边数: {len(item.edges())}")
                        
                        # 如果是节点ID列表
                        if isinstance(item, (list, set)) and len(item) > 0:
                            print(f"    列表/集合长度: {len(item)}")
                            print(f"    前5个元素: {list(item)[:5]}")
                
                elif hasattr(data, 'nodes'):
                    # 直接是NetworkX图
                    print(f"NetworkX图 - 节点数: {len(data.nodes())}, 边数: {len(data.edges())}")
                elif hasattr(data, 'x'):
                    # 直接是PyG数据
                    print(f"PyG数据 - 节点数: {data.x.shape[0]}, 边数: {data.edge_index.shape[1]}")
                
            except Exception as e:
                print(f"❌ 加载失败: {e}")
        else:
            print(f"\n=== 训练子集 {i} ===")
            print(f"文件不存在: {file_path}")

if __name__ == "__main__":
    check_training_data()
