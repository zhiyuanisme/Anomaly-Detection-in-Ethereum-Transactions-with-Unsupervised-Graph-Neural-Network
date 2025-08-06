import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import json
from datetime import datetime
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# 设置页面配置
st.set_page_config(
    page_title="🎯 GraphSAGE异常检测系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .status-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .status-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa8;
        color: #856404;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .status-error {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown("""
<div class="main-header">
    <h1>🎯 GraphSAGE + AutoEncoder + OCSVM 异常检测系统</h1>
    <p>无监督图神经网络异常检测 | Top-K性能分析 | 混淆矩阵可视化</p>
</div>
""", unsafe_allow_html=True)

# 侧边栏配置
st.sidebar.header("🔧 系统配置")

# 数据路径配置
data_paths = {
    "训练数据目录": "dataset_k-hop2/",
    "结果保存目录": "quick_ocsvm_results/",
    "模型保存目录": "graphsage_autoencoder_models/"
}

st.sidebar.subheader("📂 数据路径")
for name, path in data_paths.items():
    st.sidebar.text_input(name, value=path, key=f"path_{name}")

# 模型参数配置
st.sidebar.subheader("🎛️ 模型参数")
ocsvm_nu = st.sidebar.slider("OCSVM Nu参数", 0.01, 0.2, 0.05, 0.01)
ocsvm_kernel = st.sidebar.selectbox("OCSVM核函数", ["rbf", "linear", "poly", "sigmoid"])
top_k_values = st.sidebar.multiselect("Top-K分析", [50, 100, 200, 500, 1000], default=[50, 100, 200, 500])

# 主要功能函数
@st.cache_data
def load_results_data(results_dir="quick_ocsvm_results"):
    """加载异常检测结果数据"""
    try:
        # 加载主要结果
        with open(os.path.join(results_dir, "quick_ocsvm_results.pkl"), 'rb') as f:
            main_results = pickle.load(f)
        
        # 加载扩展Top-K结果
        extended_results = None
        if os.path.exists(os.path.join(results_dir, "extended_topk_results.pkl")):
            with open(os.path.join(results_dir, "extended_topk_results.pkl"), 'rb') as f:
                extended_results = pickle.load(f)
        
        # 加载混淆矩阵结果
        confusion_results = None
        if os.path.exists(os.path.join(results_dir, "confusion_matrix_results.pkl")):
            with open(os.path.join(results_dir, "confusion_matrix_results.pkl"), 'rb') as f:
                confusion_results = pickle.load(f)
        
        return main_results, extended_results, confusion_results
    except Exception as e:
        st.error(f"❌ 数据加载失败: {e}")
        return None, None, None

@st.cache_data
def load_csv_results(results_dir="quick_ocsvm_results"):
    """加载CSV格式的结果文件"""
    csv_data = {}
    try:
        # 加载不同Top-K的CSV文件
        for k in [50, 100, 200, 500]:
            csv_file = os.path.join(results_dir, f"top_{k}_results.csv")
            if os.path.exists(csv_file):
                csv_data[k] = pd.read_csv(csv_file)
        
        # 加载混淆矩阵报告
        cm_file = os.path.join(results_dir, "confusion_matrix_report.csv")
        if os.path.exists(cm_file):
            csv_data['confusion_matrix'] = pd.read_csv(cm_file)
        
        return csv_data
    except Exception as e:
        st.error(f"❌ CSV数据加载失败: {e}")
        return {}

def create_performance_overview(main_results, extended_results):
    """创建性能概览"""
    st.subheader("📊 性能概览")
    
    if main_results is None:
        st.warning("⚠️ 请先加载结果数据")
        return
    
    # 主要指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Top-100精确度</h3>
            <h2 style="color: #667eea;">{:.1%}</h2>
            <p>异常检测准确率</p>
        </div>
        """.format(main_results.get('precision_100', 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 检测异常数</h3>
            <h2 style="color: #28a745;">{}</h2>
            <p>Top-100中真实异常</p>
        </div>
        """.format(main_results.get('top_100_anomaly_count', 0)), unsafe_allow_html=True)
    
    with col3:
        separation_ratio = main_results.get('error_stats', {}).get('separation_ratio', 0)
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 分离度</h3>
            <h2 style="color: #fd7e14;">{:.1f}x</h2>
            <p>异常/正常误差比</p>
        </div>
        """.format(separation_ratio), unsafe_allow_html=True)
    
    with col4:
        total_nodes = main_results.get('true_labels_info', {}).get('total_nodes', 0)
        st.markdown("""
        <div class="metric-card">
            <h3>📊 总节点数</h3>
            <h2 style="color: #6f42c1;">{:,}</h2>
            <p>测试图节点总数</p>
        </div>
        """.format(total_nodes), unsafe_allow_html=True)

def create_topk_analysis(extended_results, csv_data):
    """创建Top-K分析"""
    st.subheader("🏆 Top-K性能分析")
    
    if extended_results is None:
        st.warning("⚠️ 请先加载扩展结果数据")
        return
    
    # Top-K性能对比表格
    if 'topk_detailed_results' in extended_results:
        topk_data = extended_results['topk_detailed_results']
        
        # 创建性能对比表格
        performance_data = []
        for k in sorted(topk_data.keys()):
            performance_data.append({
                'Top-K': f"Top-{k}",
                'Precision': f"{topk_data[k]['precision']:.4f}",
                'Recall': f"{topk_data[k]['recall']:.4f}",
                'F1-Score': f"{topk_data[k]['f1']:.4f}",
                'Coverage': f"{topk_data[k]['coverage']:.4f}",
                '检测异常数': topk_data[k]['anomaly_count']
            })
        
        df_performance = pd.DataFrame(performance_data)
        st.dataframe(df_performance, use_container_width=True)
        
        # Top-K性能趋势图
        col1, col2 = st.columns(2)
        
        with col1:
            # Precision趋势
            k_values = sorted(topk_data.keys())
            precisions = [topk_data[k]['precision'] for k in k_values]
            recalls = [topk_data[k]['recall'] for k in k_values]
            f1_scores = [topk_data[k]['f1'] for k in k_values]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=k_values, y=precisions, mode='lines+markers', 
                                   name='Precision', line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(x=k_values, y=recalls, mode='lines+markers', 
                                   name='Recall', line=dict(color='red', width=3)))
            fig.add_trace(go.Scatter(x=k_values, y=f1_scores, mode='lines+markers', 
                                   name='F1-Score', line=dict(color='green', width=3)))
            
            fig.update_layout(title="Top-K性能趋势", xaxis_title="K值", yaxis_title="性能指标",
                            height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 检测异常数量柱状图
            anomaly_counts = [topk_data[k]['anomaly_count'] for k in k_values]
            
            fig = px.bar(x=[f"Top-{k}" for k in k_values], y=anomaly_counts,
                        title="Top-K检测异常数量", labels={'x': 'Top-K', 'y': '异常节点数'})
            fig.update_traces(marker_color='lightcoral')
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

def create_confusion_matrix_analysis(confusion_results):
    """创建混淆矩阵分析"""
    st.subheader("📊 混淆矩阵分析")
    
    if confusion_results is None:
        st.warning("⚠️ 请先加载混淆矩阵数据")
        return
    
    # 混淆矩阵指标表格
    if 'confusion_matrices' in confusion_results:
        cm_data = confusion_results['confusion_matrices']
        
        # 创建混淆矩阵表格
        cm_table_data = []
        for k in sorted(cm_data.keys()):
            cm = cm_data[k]
            cm_table_data.append({
                'Top-K': f"Top-{k}",
                'TP': cm['tp'],
                'FP': cm['fp'],
                'TN': cm['tn'],
                'FN': cm['fn'],
                'Precision': f"{cm['precision']:.4f}",
                'Recall': f"{cm['recall']:.4f}",
                'Specificity': f"{cm['specificity']:.4f}",
                'F1-Score': f"{cm['f1_score']:.4f}"
            })
        
        df_cm = pd.DataFrame(cm_table_data)
        st.dataframe(df_cm, use_container_width=True)
        
        # 混淆矩阵热力图
        st.subheader("🔥 混淆矩阵热力图")
        
        # 选择要显示的K值
        selected_k = st.selectbox("选择Top-K值", sorted(cm_data.keys()), index=1)
        
        if selected_k in cm_data:
            cm_matrix = cm_data[selected_k]['confusion_matrix']
            
            # 创建热力图
            fig = px.imshow(cm_matrix, 
                          labels=dict(x="预测标签", y="真实标签", color="节点数量"),
                          x=['预测正常', '预测异常'],
                          y=['真实正常', '真实异常'],
                          color_continuous_scale='Blues',
                          title=f"Top-{selected_k} 混淆矩阵")
            
            # 添加数值标注
            for i in range(len(cm_matrix)):
                for j in range(len(cm_matrix[0])):
                    fig.add_annotation(x=j, y=i, text=str(cm_matrix[i][j]),
                                     showarrow=False, font=dict(color="black", size=16))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def create_detailed_results_view(csv_data):
    """创建详细结果查看"""
    st.subheader("📋 详细结果查看")
    
    if not csv_data:
        st.warning("⚠️ 请先加载CSV数据")
        return
    
    # 选择要查看的Top-K结果
    available_k = [k for k in [50, 100, 200, 500] if k in csv_data]
    if not available_k:
        st.error("❌ 没有找到Top-K结果文件")
        return
    
    selected_k = st.selectbox("选择Top-K结果", available_k, key="detailed_k")
    
    if selected_k in csv_data:
        df = csv_data[selected_k]
        
        # 结果统计
        col1, col2, col3 = st.columns(3)
        with col1:
            correct_count = df['is_correct'].sum()
            st.metric("✅ 正确检测", f"{correct_count}/{len(df)}")
        with col2:
            precision = correct_count / len(df)
            st.metric("🎯 精确度", f"{precision:.1%}")
        with col3:
            avg_score = df['anomaly_score'].mean()
            st.metric("📈 平均异常分数", f"{avg_score:.6f}")
        
        # 过滤选项
        st.subheader("🔍 结果过滤")
        col1, col2 = st.columns(2)
        
        with col1:
            show_correct_only = st.checkbox("只显示正确检测")
            show_top_n = st.slider("显示前N个结果", 10, len(df), min(50, len(df)))
        
        with col2:
            label_filter = st.selectbox("标签过滤", ["全部", "异常", "正常"])
        
        # 应用过滤
        filtered_df = df.head(show_top_n).copy()
        
        if show_correct_only:
            filtered_df = filtered_df[filtered_df['is_correct'] == True]
        
        if label_filter == "异常":
            filtered_df = filtered_df[filtered_df['true_label'] == 1]
        elif label_filter == "正常":
            filtered_df = filtered_df[filtered_df['true_label'] == 0]
        
        # 添加标签名称和预测结果列
        filtered_df['标签名称'] = filtered_df['true_label'].map({0: '正常', 1: '异常'})
        filtered_df['预测结果'] = filtered_df['is_correct'].map({True: '✅ 正确', False: '❌ 错误'})
        
        # 重命名列
        display_df = filtered_df[['rank', 'node_id', 'anomaly_score', 'reconstruction_error', 
                                '标签名称', '预测结果']].copy()
        display_df.columns = ['排名', '节点ID', '异常分数', '重构误差', '真实标签', '预测结果']
        
        # 显示表格
        st.dataframe(display_df, use_container_width=True)
        
        # 下载按钮
        csv_string = display_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载过滤结果",
            data=csv_string,
            file_name=f"top_{selected_k}_filtered_results.csv",
            mime="text/csv"
        )

def create_error_distribution_analysis(main_results):
    """创建误差分布分析"""
    st.subheader("📊 重构误差分布分析")
    
    if main_results is None:
        st.warning("⚠️ 请先加载结果数据")
        return
    
    error_stats = main_results.get('error_stats', {})
    
    # 误差统计
    col1, col2, col3 = st.columns(3)
    
    with col1:
        normal_mean = error_stats.get('normal_mean', 0)
        st.metric("🔵 正常节点平均误差", f"{normal_mean:.6f}")
    
    with col2:
        anomaly_mean = error_stats.get('anomaly_mean', 0)
        st.metric("🔴 异常节点平均误差", f"{anomaly_mean:.6f}")
    
    with col3:
        separation_ratio = error_stats.get('separation_ratio', 0)
        st.metric("📊 分离度", f"{separation_ratio:.2f}x")
    
    # 分离度分析
    if separation_ratio > 100:
        st.markdown("""
        <div class="status-success">
            <strong>🌟 完美分离!</strong> 异常节点和正常节点的重构误差有明显区别，模型性能优秀。
        </div>
        """, unsafe_allow_html=True)
    elif separation_ratio > 10:
        st.markdown("""
        <div class="status-success">
            <strong>✅ 良好分离!</strong> 异常节点和正常节点有较好的区分度。
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-warning">
            <strong>⚠️ 分离度较低!</strong> 可能需要调整模型参数或特征工程。
        </div>
        """, unsafe_allow_html=True)

def create_model_info():
    """创建模型信息面板"""
    st.subheader("🔬 技术架构")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🧠 GraphSAGE自编码器**
        - 输入维度: 4 (优化后特征)
        - 隐藏维度: 128
        - 潜在维度: 64
        - Dropout: 0.2
        - 激活函数: ReLU
        """)
        
        st.markdown("""
        **🎯 OCSVM异常检测**
        - 核函数: RBF
        - Nu参数: 0.05
        - Gamma: auto
        - 特征: 仅重构误差 (最优方法)
        """)
    
    with col2:
        st.markdown("""
        **📊 数据集信息**
        - 训练集: 3个随机游走子图
        - 测试集: 真实网络图
        - 标签: 钓鱼/可疑 vs 正常节点
        - 异常比例: ~4.3%
        """)
        
        st.markdown("""
        **🚀 核心优势**
        - 100% Top-100精确度
        - 无监督学习
        - 可扩展性强
        - 重构误差最纯净异常信号
        """)

def main():
    """主函数"""
    # 数据加载状态
    st.sidebar.subheader("📊 数据状态")
    
    # 检查数据文件是否存在
    results_dir = st.session_state.get('path_结果保存目录', 'quick_ocsvm_results')
    
    data_status = {}
    required_files = [
        'quick_ocsvm_results.pkl',
        'extended_topk_results.pkl',
        'confusion_matrix_results.pkl',
        'top_100_results.csv'
    ]
    
    for file in required_files:
        file_path = os.path.join(results_dir, file)
        data_status[file] = os.path.exists(file_path)
        
        status_icon = "✅" if data_status[file] else "❌"
        st.sidebar.text(f"{status_icon} {file}")
    
    all_files_exist = all(data_status.values())
    
    if all_files_exist:
        st.sidebar.markdown("""
        <div class="status-success">
            ✅ 所有数据文件就绪
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div class="status-warning">
            ⚠️ 部分数据文件缺失
        </div>
        """, unsafe_allow_html=True)
    
    # 加载数据按钮
    if st.sidebar.button("🔄 加载/刷新数据", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # 主内容区域
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 性能概览", "🏆 Top-K分析", "📊 混淆矩阵", "📋 详细结果", "🔬 技术信息"
    ])
    
    # 加载数据
    main_results, extended_results, confusion_results = load_results_data(results_dir)
    csv_data = load_csv_results(results_dir)
    
    with tab1:
        create_performance_overview(main_results, extended_results)
        create_error_distribution_analysis(main_results)
    
    with tab2:
        create_topk_analysis(extended_results, csv_data)
    
    with tab3:
        create_confusion_matrix_analysis(confusion_results)
    
    with tab4:
        create_detailed_results_view(csv_data)
    
    with tab5:
        create_model_info()
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🎯 GraphSAGE异常检测系统 | 基于图神经网络的无监督异常检测 | 
        <a href="https://github.com/your-repo" style="color: #667eea;">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
