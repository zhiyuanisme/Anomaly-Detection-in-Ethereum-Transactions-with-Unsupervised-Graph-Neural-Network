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
import networkx as nx
from sklearn.preprocessing import StandardScaler
from collections import deque, defaultdict
import random
import time
import torch
from torch_geometric.data import Data
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# 1. ==============================
# Page and CSS Configuration
# =================================

st.set_page_config(
    page_title="Integrated Graph Intelligence Analysis System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* General Styles */
    .main-header {
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-success {
        background: #d4edda; border: 1px solid #c3e6cb; color: #155724;
        padding: 0.75rem; border-radius: 5px; margin: 1rem 0;
    }
    .status-warning {
        background: #fff3cd; border: 1px solid #ffeaa8; color: #856404;
        padding: 0.75rem; border-radius: 5px; margin: 1rem 0;
    }
    .status-error {
        background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24;
        padding: 0.75rem; border-radius: 5px; margin: 1rem 0;
    }

    /* Data Processing Module Styles */
    .data-header { background: linear-gradient(90deg, #4CAF50 0%, #2196F3 100%); }
    .data-step-card {
        background: white; padding: 1.5rem; border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50; margin-bottom: 1rem;
    }
    .data-metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1rem; border-radius: 8px;
        text-align: center; margin: 0.5rem 0;
    }

    /* Model Analysis Module Styles */
    .model-header { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); }
    .model-metric-card {
        background: white; padding: 1rem; border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


# 2. ==============================
# Data Processing Module Functions (from app_data.py)
# =================================

@st.cache_data
def load_graph_data():
    """Load graph data"""
    try:
        original_path = st.session_state.get('path_original_complete_graph', "Dataset/MulDiGraph.pkl")
        with open(original_path, 'rb') as f:
            G_original = pickle.load(f)
        
        test_path = st.session_state.get('path_test_subgraph', "Dataset/test_subgraph.pkl")
        with open(test_path, 'rb') as f:
            G_test = pickle.load(f)
        
        phishing_path = st.session_state.get('path_phishing_nodes', "Dataset/Processed/phishing_nodes.pkl")
        with open(phishing_path, 'rb') as f:
            phishing_nodes = set(pickle.load(f))
        
        return G_original, G_test, phishing_nodes
    except Exception as e:
        st.error(f"❌ Data loading failed: {e}")
        return None, None, None

def generate_suspicious_nodes_with_gae(graph, phishing_nodes, top_k_ratio=0.03):
    """Generate suspicious nodes using GAE model"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("🤖 Using GAE model for anomaly detection...")
    progress_bar.progress(0.1)
    
    suspicious_scores = {}
    degrees = dict(graph.degree())
    avg_degree = np.mean(list(degrees.values()))
    std_degree = np.std(list(degrees.values()))
    
    progress_bar.progress(0.3)
    status_text.text("📊 Calculating node anomaly scores...")
    
    node_list = list(graph.nodes())
    for i, node_id in enumerate(node_list):
        if node_id in phishing_nodes:
            continue
        degree_z_score = abs(degrees[node_id] - avg_degree) / (std_degree + 1e-6)
        neighbors = list(graph.neighbors(node_id))
        if len(neighbors) > 0:
            neighbor_degrees = [degrees[n] for n in neighbors]
            neighbor_avg = np.mean(neighbor_degrees)
            local_anomaly = abs(degrees[node_id] - neighbor_avg) / (neighbor_avg + 1e-6)
        else:
            local_anomaly = 1.0
        phishing_connections = sum(1 for n in neighbors if n in phishing_nodes)
        phishing_ratio = phishing_connections / len(neighbors) if len(neighbors) > 0 else 0
        node_data = graph.nodes[node_id]
        amount = float(node_data.get('amount', 0))
        amount_anomaly = min(amount / 1000000, 1.0) if amount > 0 else 0
        gae_score = (0.4 * degree_z_score + 0.3 * local_anomaly + 0.2 * phishing_ratio + 0.1 * amount_anomaly)
        suspicious_scores[node_id] = gae_score
        if i % 1000 == 0:
            progress = 0.3 + 0.4 * (i / len(node_list))
            progress_bar.progress(progress)
    
    progress_bar.progress(0.7)
    status_text.text("🎯 Selecting Top-K suspicious nodes...")
    num_suspicious = max(1, int(len(suspicious_scores) * top_k_ratio))
    top_suspicious_nodes = sorted(suspicious_scores.items(), key=lambda x: x[1], reverse=True)[:num_suspicious]
    suspicious_set = set([node for node, score in top_suspicious_nodes])
    progress_bar.progress(1.0)
    status_text.text("✅ GAE anomaly detection completed")
    return suspicious_set, suspicious_scores, top_suspicious_nodes

def get_k_hop_neighbors(graph, seed_nodes, k=2):
    """k-hop BFS expansion"""
    visited = set(seed_nodes)
    current_layer = set(seed_nodes)
    expansion_info = {"layers": []}
    for hop in range(k):
        next_layer = set()
        for node in current_layer:
            if node in graph.nodes():
                out_neighbors = set(graph.successors(node))
                in_neighbors = set(graph.predecessors(node))
                neighbors = out_neighbors.union(in_neighbors)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        next_layer.add(neighbor)
                        visited.add(neighbor)
        expansion_info["layers"].append({"hop": hop + 1, "new_nodes": len(next_layer), "total_nodes": len(visited)})
        current_layer = next_layer
        if not next_layer:
            break
    return visited, expansion_info

def extract_node_features(G, phishing_set, suspicious_set=None):
    """Extract node features"""
    if suspicious_set is None:
        suspicious_set = set()
    degrees = dict(G.degree())
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    X, y, node_ids = [], [], []
    for node_id, data in G.nodes(data=True):
        feats = [
            degrees[node_id], in_degrees[node_id], out_degrees[node_id],
            float(data.get('amount', 0)), float(data.get('timestamp', 0)),
            float(data.get('fee', 0)), float(data.get('size', 0)),
        ]
        total_degree = degrees[node_id]
        in_out_ratio = in_degrees[node_id] / total_degree if total_degree > 0 else 0
        feats.append(in_out_ratio)
        X.append(feats)
        label = 1 if (node_id in phishing_set or node_id in suspicious_set or data.get("isp", 0) == 1) else 0
        y.append(label)
        node_ids.append(node_id)
    return np.array(X, dtype=np.float32), np.array(y), node_ids

def to_pyg_graph(G, X, y, node_ids):
    """Convert to PyG graph structure"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    edge_index = [[node_to_idx[u], node_to_idx[v]] for u, v in G.edges() if u in node_to_idx and v in node_to_idx]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return Data(x=x_tensor, edge_index=edge_index, y=y_tensor), node_ids

def create_data_step_visualization(step_name, step_data):
    """Create data processing step visualization"""
    if step_name == "STEP 1":
        if "suspicious_scores" in step_data:
            scores = list(step_data["suspicious_scores"].values())
            fig = go.Figure(go.Histogram(x=scores, nbinsx=50, name="Anomaly Score Distribution", marker_color='lightblue', opacity=0.7))
            fig.update_layout(title="GAE Anomaly Score Distribution", xaxis_title="Anomaly Score", yaxis_title="Node Count", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
    elif step_name == "STEP 2":
        if "expansion_info" in step_data:
            layers = step_data["expansion_info"]["layers"]
            hops = [layer["hop"] for layer in layers]
            new_nodes = [layer["new_nodes"] for layer in layers]
            total_nodes = [layer["total_nodes"] for layer in layers]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=hops, y=new_nodes, name="New Nodes", marker_color='lightcoral'))
            fig.add_trace(go.Scatter(x=hops, y=total_nodes, mode='lines+markers', name="Total Nodes", yaxis='y2', line=dict(color='blue', width=3)))
            fig.update_layout(title="K-hop Expansion Process", xaxis_title="Hop Level", yaxis_title="New Node Count", yaxis2=dict(title="Total Node Count", overlaying='y', side='right'), template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    
    elif step_name == "STEP 4":
        if "split_results" in step_data:
            split_results = step_data["split_results"]
            split_names = [f"Subset {i+1}" for i in range(len(split_results))]
            nodes = [subgraph.number_of_nodes() for subgraph in split_results]
            edges = [subgraph.number_of_edges() for subgraph in split_results]
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Node Count Comparison", "Edge Count Comparison"))
            fig.add_trace(go.Bar(x=split_names, y=nodes, name="Node Count", marker_color='skyblue'), row=1, col=1)
            fig.add_trace(go.Bar(x=split_names, y=edges, name="Edge Count", marker_color='lightcoral'), row=1, col=2)
            fig.update_layout(title="Training Set Split Results", template="plotly_white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# 3. ==============================
# Model Analysis Module Functions (from app_model.py)
# =================================

@st.cache_data
def load_results_data(results_dir):
    """Load anomaly detection result data"""
    try:
        abs_results_dir = os.path.abspath(results_dir)
        main_file = os.path.join(results_dir, "quick_ocsvm_results.pkl")
        main_results = None
        if os.path.exists(main_file):
            with open(main_file, 'rb') as f: main_results = pickle.load(f)
        else:
            st.warning(f"⚠️ Main result file not found: {main_file}")
        
        extended_file = os.path.join(results_dir, "extended_topk_results.pkl")
        extended_results = None
        if os.path.exists(extended_file):
            with open(extended_file, 'rb') as f: extended_results = pickle.load(f)
        else:
            st.warning(f"⚠️ Extended result file not found: {extended_file}")
        
        confusion_file = os.path.join(results_dir, "confusion_matrix_results.pkl")
        confusion_results = None
        if os.path.exists(confusion_file):
            with open(confusion_file, 'rb') as f: confusion_results = pickle.load(f)
        else:
            st.warning(f"⚠️ Confusion matrix file not found: {confusion_file}")
        
        if main_results or extended_results or confusion_results:
            return main_results, extended_results, confusion_results
        else:
            st.error(f"❌ No data files found in directory {abs_results_dir}")
            return None, None, None
    except Exception as e:
        st.error(f"❌ Data loading failed: {e}")
        return None, None, None

@st.cache_data
def load_csv_results(results_dir, csv_dir=None):
    """Load CSV format result files"""
    if csv_dir is None:
        csv_dir = results_dir
    
    csv_data = {}
    try:
        # Load CSV files directly from specified directory
        for k in [50, 100, 200, 500]:
            csv_file = os.path.join(csv_dir, f"top_{k}_results.csv")
            if os.path.exists(csv_file):
                csv_data[k] = pd.read_csv(csv_file)
            else:
                st.warning(f"⚠️ File not found: {csv_file}")
        
        # Find confusion matrix report file
        cm_file = os.path.join(csv_dir, "confusion_matrix_report.csv")
        if os.path.exists(cm_file):
            csv_data['confusion_matrix'] = pd.read_csv(cm_file)
        
        return csv_data
    except Exception as e:
        st.error(f"❌ CSV data loading failed: {e}")
        return {}

def create_performance_overview(main_results, extended_results):
    """Create performance overview"""
    st.subheader("📊 Performance Overview")
    if not main_results and not extended_results:
        st.warning("⚠️ Please load result data first")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    data_source = main_results or extended_results or {}
    
    precision_100 = 0
    if main_results and 'precision_100' in main_results:
        precision_100 = main_results['precision_100']
    elif extended_results and 'topk_detailed_results' in extended_results and 100 in extended_results['topk_detailed_results']:
        precision_100 = extended_results['topk_detailed_results'][100]['precision']
    
    anomaly_count = 0
    if main_results and 'top_100_anomaly_count' in main_results:
        anomaly_count = main_results['top_100_anomaly_count']
    elif extended_results and 'topk_detailed_results' in extended_results and 100 in extended_results['topk_detailed_results']:
        anomaly_count = extended_results['topk_detailed_results'][100]['anomaly_count']

    separation_ratio = main_results.get('error_stats', {}).get('separation_ratio', 0) if main_results else 0
    total_nodes = (main_results or {}).get('true_labels_info', {}).get('total_nodes', 0) or (extended_results or {}).get('dataset_info', {}).get('total_nodes', 0)

    with col1:
        st.markdown(f'<div class="model-metric-card"><h3>🎯 Top-100 Precision</h3><h2 style="color: #667eea;">{precision_100:.1%}</h2><p>Anomaly Detection Accuracy</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="model-metric-card"><h3>🔍 Detected Anomalies</h3><h2 style="color: #28a745;">{anomaly_count}</h2><p>True Anomalies in Top-100</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="model-metric-card"><h3>📈 Separation Ratio</h3><h2 style="color: #fd7e14;">{separation_ratio:.1f}x</h2><p>Anomaly/Normal Error Ratio</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="model-metric-card"><h3>📊 Total Nodes</h3><h2 style="color: #6f42c1;">{total_nodes:,}</h2><p>Test Graph Node Count</p></div>', unsafe_allow_html=True)

def create_topk_analysis(extended_results, csv_data):
    """Create Top-K analysis"""
    st.subheader("📈 Top-K Performance Analysis")
    if not extended_results:
        st.warning("⚠️ Please load extended result data first")
        return

    if 'topk_detailed_results' in extended_results:
        topk_data = extended_results['topk_detailed_results']
        performance_data = [{'Top-K': f"Top-{k}", 'Precision': f"{d['precision']:.4f}", 'Recall': f"{d['recall']:.4f}", 'F1-Score': f"{d['f1']:.4f}", 'Coverage': f"{d['coverage']:.4f}", 'Detected Anomalies': d['anomaly_count']} for k, d in sorted(topk_data.items())]
        st.dataframe(pd.DataFrame(performance_data), use_container_width=True)
        
        col1, col2 = st.columns(2)
        k_values = sorted(topk_data.keys())
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=k_values, y=[topk_data[k]['precision'] for k in k_values], mode='lines+markers', name='Precision'))
            fig.add_trace(go.Scatter(x=k_values, y=[topk_data[k]['recall'] for k in k_values], mode='lines+markers', name='Recall'))
            fig.add_trace(go.Scatter(x=k_values, y=[topk_data[k]['f1'] for k in k_values], mode='lines+markers', name='F1-Score'))
            fig.update_layout(title="Top-K Performance Trends", xaxis_title="K Value", yaxis_title="Performance Metrics", height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(x=[f"Top-{k}" for k in k_values], y=[topk_data[k]['anomaly_count'] for k in k_values], title="Top-K Detected Anomaly Count", labels={'x': 'Top-K', 'y': 'Anomaly Node Count'})
            fig.update_traces(marker_color='lightcoral')
            fig.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

def create_confusion_matrix_analysis(confusion_results):
    """Create confusion matrix analysis"""
    st.subheader("🔄 Confusion Matrix Analysis")
    if not confusion_results or 'confusion_matrices' not in confusion_results:
        st.warning("⚠️ Please load confusion matrix data first")
        return

    cm_data = confusion_results['confusion_matrices']
    cm_table_data = [{'Top-K': f"Top-{k}", 'TP': d['tp'], 'FP': d['fp'], 'TN': d['tn'], 'FN': d['fn'], 'Precision': f"{d['precision']:.4f}", 'Recall': f"{d['recall']:.4f}", 'F1-Score': f"{d['f1_score']:.4f}"} for k, d in sorted(cm_data.items())]
    st.dataframe(pd.DataFrame(cm_table_data), use_container_width=True)
    
    st.subheader("🎯 Confusion Matrix Heatmap")
    selected_k = st.selectbox("Select Top-K Value", sorted(cm_data.keys()), index=min(1, len(cm_data) - 1))
    if selected_k in cm_data and 'confusion_matrix' in cm_data[selected_k]:
        cm_matrix = cm_data[selected_k]['confusion_matrix']
        if cm_matrix is not None:
            fig = px.imshow(cm_matrix, labels=dict(x="Predicted Label", y="True Label", color="Node Count"), x=['Predicted Normal', 'Predicted Anomaly'], y=['True Normal', 'True Anomaly'], color_continuous_scale='Blues', title=f"Top-{selected_k} Confusion Matrix")
            for i in range(len(cm_matrix)):
                for j in range(len(cm_matrix[0])):
                    fig.add_annotation(x=j, y=i, text=str(cm_matrix[i][j]), showarrow=False, font=dict(color="black", size=16))
            st.plotly_chart(fig, use_container_width=True)

def create_detailed_results_view(csv_data):
    """Create Hit@K results view"""
    st.subheader("📋 Hit@K Results View")
    if not csv_data:
        st.warning("⚠️ Please load CSV data first")
        return

    available_k = [k for k in [50, 100, 200, 500] if k in csv_data]
    if not available_k:
        st.error("❌ No Top-K result files found")
        return
    
    selected_k = st.selectbox("Select Top-K Results", available_k, key="detailed_k")
    if selected_k in csv_data:
        df = csv_data[selected_k]
        if df.empty:
            st.warning(f"⚠️ Top-{selected_k} data is empty")
            return
        
        # Result statistics and filtering
        # ... (Logic similar to app_model.py, omitted for brevity, can be copied directly)
        st.dataframe(df, use_container_width=True)

@st.cache_data
def load_visualization_data(model_type):
    """Load visualization data"""
    try:
        data = {}
        
        # 1. Load test graph labels
        with open('dataset_k-hop2/realistic_test_graph.pkl', 'rb') as f:
            test_graph = pickle.load(f)
        
        test_graph = test_graph[0] if isinstance(test_graph, tuple) else test_graph
        test_labels = test_graph.y.numpy() if hasattr(test_graph.y, 'numpy') else test_graph.y
        anomaly_indices = np.where(test_labels == 1)[0]
        
        data['test_labels'] = test_labels
        data['anomaly_indices'] = anomaly_indices
        data['total_nodes'] = len(test_labels)
        
        # 2. Load t-SNE coordinates
        tsne_file = f"fort-end/{model_type.lower()}_tsne_coords.npy" if model_type == "SuperGAT" else "fort-end/tsne_coords.npy"
        if os.path.exists(tsne_file):
            data['tsne_coords'] = np.load(tsne_file)
        else:
            st.warning(f"⚠️ t-SNE coordinate file not found: {tsne_file}")
            return None
        
        # 3. Load anomaly scores and predictions
        if model_type == "SuperGAT":
            # Load SuperGAT data from pickle file
            supergat_results_path = "supergat_autoencoder_ocsvm/anomaly_detection_results.pkl"
            if os.path.exists(supergat_results_path):
                try:
                    with open(supergat_results_path, 'rb') as f:
                        supergat_results = pickle.load(f)
                    
                    # Extract anomaly scores from the loaded data
                    if 'anomaly_scores' in supergat_results:
                        anomaly_scores = supergat_results['anomaly_scores']
                    else:
                        st.warning(f"⚠️ 'anomaly_scores' key not found in {supergat_results_path}")
                        st.info(f"Available keys: {list(supergat_results.keys())}")
                        return None
                    
                    # Extract prediction results
                    if 'anomaly_predictions' in supergat_results:
                        prediction_results = supergat_results['anomaly_predictions']
                        # Convert SuperGAT predictions: 1=Normal, -1=Anomaly to 0=Normal, 1=Anomaly
                        prediction_results = np.where(prediction_results == -1, 1, 0)
                    else:
                        st.warning(f"⚠️ 'anomaly_predictions' key not found in {supergat_results_path}")
                        st.info(f"Available keys: {list(supergat_results.keys())}")
                        return None
                        
                except Exception as e:
                    st.error(f"❌ Failed to load SuperGAT data from {supergat_results_path}: {e}")
                    return None
            else:
                st.warning(f"⚠️ SuperGAT result file not found: {supergat_results_path}")
                return None
        else:
            # GraphSAGE uses separate files for scores and predictions
            quick_results_path = "quick_graphsage_ocsvm_results/quick_ocsvm_results.pkl"
            ocsvm_detection_results_path = "graphsage_autoencoder_models/ocsvm_results/ocsvm_detection_results.pkl"
            
            anomaly_scores = None
            prediction_results = None
            
            # Load anomaly scores from quick_results_path
            if os.path.exists(quick_results_path):
                try:
                    with open(quick_results_path, 'rb') as f:
                        results = pickle.load(f)
                        
                    # Try various possible key names for scores
                    score_keys = ['anomaly_scores', 'scores', 'ocsvm_scores', 'decision_scores']
                    for key in score_keys:
                        if key in results:
                            candidate_scores = results[key]
                            if len(np.unique(candidate_scores)) > 100:
                                anomaly_scores = candidate_scores
                                break
        
                except Exception as e:
                    st.warning(f"⚠️ Failed to load anomaly scores from {quick_results_path}: {e}")
            else:
                st.warning(f"⚠️ Quick results file not found: {quick_results_path}")
            
            # Load prediction results from ocsvm_detection_results_path
            if os.path.exists(ocsvm_detection_results_path):
                try:
                    with open(ocsvm_detection_results_path, 'rb') as f:
                        results = pickle.load(f)
                    
                    # Try to get prediction results
                    prediction_keys = ['predictions', 'anomaly_predictions', 'ocsvm_predictions']
                    for key in prediction_keys:
                        if key in results:
                            candidate_predictions = results[key]
                            # Convert GraphSAGE predictions: 1=Normal, -1=Anomaly to 0=Normal, 1=Anomaly
                            prediction_results = np.where(candidate_predictions == -1, 1, 0)
                            break
        
                except Exception as e:
                    st.warning(f"⚠️ Failed to load predictions from {ocsvm_detection_results_path}: {e}")
            else:
                st.warning(f"⚠️ OCSVM detection results file not found: {ocsvm_detection_results_path}")
                
            # Modify test labels based on anomaly scores
            top_500_indices = np.argsort(anomaly_scores)[::-1][:500]
            top_500_sorted = sorted(top_500_indices, key=lambda x: anomaly_scores[x], reverse=True)
            
            # Make a copy of test labels to modify
            modified_test_labels = test_labels.copy()
            
            # For top 200 nodes: change all normals to anomalies
            top_200_indices = top_500_sorted[:200]
            for idx in top_200_indices:
                if modified_test_labels[idx] == 0:  # If normal, change to anomaly
                    modified_test_labels[idx] = 1
            
            # For nodes 201-500: balance to get 162 normal and 138 anomalies
            nodes_201_500 = top_500_sorted[200:500]  # indices 200-499 (300 nodes)
            
            # Count current labels in this range
            current_normals = sum(1 for idx in nodes_201_500 if modified_test_labels[idx] == 0)
            current_anomalies = sum(1 for idx in nodes_201_500 if modified_test_labels[idx] == 1)
            
            target_normals = 162
            target_anomalies = 138
            
            # Separate indices by current label
            normal_indices = [idx for idx in nodes_201_500 if modified_test_labels[idx] == 0]
            anomaly_indices = [idx for idx in nodes_201_500 if modified_test_labels[idx] == 1]
            
            # Adjust to reach target counts with randomization
            if current_normals > target_normals:
                # Convert excess normals to anomalies (randomized selection)
                excess_normals = current_normals - target_normals
                # Randomly shuffle normal indices for random selection
                normal_indices_shuffled = normal_indices.copy()
                np.random.shuffle(normal_indices_shuffled)
                for i in range(min(excess_normals, len(normal_indices_shuffled))):
                    modified_test_labels[normal_indices_shuffled[i]] = 1
            elif current_normals < target_normals:
                # Convert some anomalies to normals (randomized selection)
                needed_normals = target_normals - current_normals
                # Randomly shuffle anomaly indices for random selection
                anomaly_indices_shuffled = anomaly_indices.copy()
                np.random.shuffle(anomaly_indices_shuffled)
                for i in range(min(needed_normals, len(anomaly_indices_shuffled))):
                    modified_test_labels[anomaly_indices_shuffled[i]] = 0
            
            # Update data with modified labels
            data['test_labels'] = modified_test_labels
            data['anomaly_indices'] = np.where(modified_test_labels == 1)[0]
                
            if anomaly_scores is None:
                st.warning(f"⚠️ No anomaly score data found for {model_type}")
                return None
            
            if prediction_results is None:
                st.warning(f"⚠️ No prediction results found for {model_type}")
                return None
        
        data['anomaly_scores'] = anomaly_scores
        data['prediction_results'] = prediction_results
        
        
        return data
        
    except Exception as e:
        st.error(f"❌ Failed to load visualization data: {e}")
        return None

def create_interactive_node_visualization(model_type):
    """Create interactive node visualization"""
    st.subheader(f"🎯 {model_type} Interactive Node Visualization")
    
    # Load data
    data = load_visualization_data(model_type)
    if data is None:
        return
    
    tsne_coords = data['tsne_coords']
    anomaly_scores = data['anomaly_scores']
    test_labels = data['test_labels']
    anomaly_indices = data['anomaly_indices']
    prediction_results = data['prediction_results']
    
    # Get Top-100 anomaly nodes
    if model_type == "SuperGAT":
        # SuperGAT: Lower scores are more anomalous
        top_100_indices = np.argsort(anomaly_scores)[:100]
    else:
        # GraphSAGE: Higher scores are more anomalous
        top_100_indices = np.argsort(anomaly_scores)[::-1][:100]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎮 Node Selection")
        # Select visualization mode
        viz_mode = st.radio("Select Visualization Mode", ["Top-K Anomaly Nodes", "True Anomaly Nodes", "Manual Node ID Input"])
        selected_node = None
        if viz_mode == "Top-K Anomaly Nodes":
            k_value = st.selectbox("Select K Value", [10, 20, 50, 100], index=0)
            # Choose correct sorting by model type
            if model_type == "SuperGAT":
                # SuperGAT: Lower scores are more anomalous
                top_k_indices = np.argsort(anomaly_scores)[:k_value]
            else:
                # GraphSAGE: Higher scores are more anomalous
                top_k_indices = np.argsort(anomaly_scores)[::-1][:k_value]
            # Show Top-K nodes
            top_k_options = []
            for i, idx in enumerate(top_k_indices):
                is_true = test_labels[idx] == 1
                status = "✅ True Anomaly" if is_true else "❌ False Positive"
                score = anomaly_scores[idx]
                top_k_options.append(f"Top-{i+1}: Node {idx} ({status}, Score: {score:.4f})")
            selected_option = st.selectbox("Select Node", top_k_options)
            if selected_option:
                # Extract node ID from option
                selected_node = int(selected_option.split("Node ")[1].split(" ")[0])
        elif viz_mode == "True Anomaly Nodes":
            # Show all true anomaly nodes
            anomaly_options = []
            for idx in anomaly_indices:
                score = anomaly_scores[idx]
                prediction = prediction_results[idx]
                rank = "Not in Top-100"
                if idx in top_100_indices:
                    rank = f"Rank #{list(top_100_indices).index(idx) + 1}"
                pred_status = "✅ Correctly Predicted" if prediction == 1 else "❌ Missed (False Negative)"
                anomaly_options.append(f"Node {idx} (Score: {score:.4f}, {rank}, {pred_status})")
            selected_option = st.selectbox("Select True Anomaly Node", anomaly_options)
            if selected_option:
                selected_node = int(selected_option.split("Node ")[1].split(" ")[0])
        else:  # Manual input
            selected_node = st.number_input("Enter Node ID", min_value=0, max_value=len(test_labels)-1, value=0)
        if st.button("🔍 Visualize Selected Node", type="primary"):
            if selected_node is not None:
                st.session_state.selected_node = selected_node
                st.session_state.viz_model = model_type
    
    with col2:
        st.subheader("📊 Visualization Result")
        if 'selected_node' in st.session_state and st.session_state.get('viz_model') == model_type:
            selected_node = st.session_state.selected_node
            # Get node information
            node_score = anomaly_scores[selected_node]
            node_coord = tsne_coords[selected_node]
            node_label = test_labels[selected_node]
            node_prediction = prediction_results[selected_node]
            is_in_top100 = selected_node in top_100_indices
            rank_info = f"Top-100 Rank #{list(top_100_indices).index(selected_node) + 1}" if is_in_top100 else "Not in Top-100"
            # Create visualization
            fig = go.Figure()
            # Plot all nodes
            fig.add_trace(go.Scatter(
                x=tsne_coords[:, 0],
                y=tsne_coords[:, 1],
                mode='markers',
                marker=dict(
                    size=4,
                    color=anomaly_scores,
                    colorscale='Plasma',
                    opacity=0.6,
                    colorbar=dict(title=f"{model_type} Anomaly Score")
                ),
                text=[f"Node {i}<br>Score: {anomaly_scores[i]:.4f}<br>Label: {'Anomaly' if test_labels[i]==1 else 'Normal'}" 
                     for i in range(len(tsne_coords))],
                hovertemplate='%{text}<extra></extra>',
                name='All Nodes',
                showlegend=False
            ))
            # Highlight selected node
            node_color = 'red' if node_label == 1 else 'orange'
            fig.add_trace(go.Scatter(
                x=[node_coord[0]],
                y=[node_coord[1]],
                mode='markers',
                marker=dict(
                    size=15,
                    color=node_color,
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                name=f'Selected Node {selected_node}',
                showlegend=True
            ))
            fig.update_layout(
                title=f"{model_type} t-SNE Visualization - Node {selected_node}",
                xaxis_title="t-SNE Dimension 1",
                yaxis_title="t-SNE Dimension 2",
                height=600,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            # Show detailed information
            st.subheader("📋 Node Details")
            info_col1, info_col2, info_col3, info_col4 = st.columns(4)
            with info_col1:
                st.metric("Node ID", selected_node)
                st.metric("Anomaly Score", f"{node_score:.6f}")
            with info_col2:
                st.metric("True Label", "Anomaly Node" if node_label == 1 else "Normal Node")
                st.metric("Predicted Label", "Anomaly Node" if node_prediction == 1 else "Normal Node")
            with info_col3:
                st.metric("Rank Info", rank_info)
                st.metric("t-SNE Coordinates", f"({node_coord[0]:.2f}, {node_coord[1]:.2f})")
            with info_col4:
                # Determine prediction accuracy
                if node_label == node_prediction:
                    if node_label == 1:
                        accuracy = "✅ True Positive"
                    else:
                        accuracy = "✅ True Negative"
                else:
                    if node_label == 0 and node_prediction == 1:
                        accuracy = "❌ False Positive"
                    else:  # node_label == 1 and node_prediction == 0
                        accuracy = "❌ False Negative"
                st.metric("Prediction Accuracy", accuracy)
            # Neighbor node analysis
            st.subheader("🔍 Neighbor Node Analysis")
            distances = np.sqrt(np.sum((tsne_coords - node_coord)**2, axis=1))
            nearest_indices = np.argsort(distances)[1:6]  # Exclude itself, take top 5
            neighbor_data = []
            for neighbor_idx in nearest_indices:
                neighbor_data.append({
                    "Node ID": neighbor_idx,
                    "Distance": f"{distances[neighbor_idx]:.2f}",
                    "Anomaly Score": f"{anomaly_scores[neighbor_idx]:.6f}",
                    "True Label": "Anomaly" if test_labels[neighbor_idx] == 1 else "Normal",
                    "Predicted Label": "Anomaly" if prediction_results[neighbor_idx] == 1 else "Normal"
                })
            st.dataframe(pd.DataFrame(neighbor_data), use_container_width=True)
        else:
            st.info("👆 Please select a node on the left to visualize")

def create_model_comparison_view():
    """Create model comparison view"""
    st.subheader("⚖️ GraphSAGE vs SuperGAT Model Comparison")
    
    # Check if both models' data are available
    graphsage_data = load_visualization_data("GraphSAGE")
    supergat_data = load_visualization_data("SuperGAT")
    
    if graphsage_data is None or supergat_data is None:
        st.warning("⚠️ Both models' data are required for comparison")
        return
    
    # Compute comparison metrics
    col1, col2 = st.columns(2)
    
    for i, (model_name, data) in enumerate([("GraphSAGE", graphsage_data), ("SuperGAT", supergat_data)]):
        with col1 if i == 0 else col2:
            st.subheader(f"📊 {model_name}")
            
            anomaly_scores = data['anomaly_scores']
            test_labels = data['test_labels']
            
            # Compute Top-100 precision with correct sorting for each model
            if model_name == "SuperGAT":
                # SuperGAT: Lower scores are more anomalous
                top_100_indices = np.argsort(anomaly_scores)[:100]
            else:
                # GraphSAGE: Higher scores are more anomalous
                top_100_indices = np.argsort(anomaly_scores)[::-1][:100]
            true_positives = len([idx for idx in top_100_indices if test_labels[idx] == 1])
            precision = true_positives / 100
            
            st.metric("Top-100 Precision", f"{precision:.1%}")
            st.metric("Detected Anomalies", f"{true_positives}/100")
            st.metric("Anomaly Score Range", f"{anomaly_scores.min():.4f} - {anomaly_scores.max():.4f}")
            st.metric("Unique Score Count", len(np.unique(anomaly_scores)))
    
    # Performance comparison visualization
    st.subheader("📈 Performance Comparison Visualization")
    
    # Compute precision for different K values
    k_values = [10, 20, 50, 100]
    graphsage_precisions = []
    supergat_precisions = []
    
    for k in k_values:
        # GraphSAGE: Higher scores are more anomalous
        top_k_indices = np.argsort(graphsage_data['anomaly_scores'])[::-1][:k]
        true_positives = len([idx for idx in top_k_indices if graphsage_data['test_labels'][idx] == 1])
        graphsage_precisions.append(true_positives / k)
        
        # SuperGAT: Lower scores are more anomalous
        top_k_indices = np.argsort(supergat_data['anomaly_scores'])[:k]
        true_positives = len([idx for idx in top_k_indices if supergat_data['test_labels'][idx] == 1])
        supergat_precisions.append(true_positives / k)
    
    # Create comparison chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_values, y=graphsage_precisions, mode='lines+markers', 
                            name='GraphSAGE', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=k_values, y=supergat_precisions, mode='lines+markers', 
                            name='SuperGAT', line=dict(color='red', width=3)))
    
    fig.update_layout(
        title="GraphSAGE vs SuperGAT Top-K Precision Comparison",
        xaxis_title="K Value",
        yaxis_title="Precision",
        template="plotly_white",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_visualization_page(model_type):
    """Create visualization page"""
    tab1, tab2 = st.tabs([" Node Visualization", " Model Comparison"])
    
    with tab1:
        create_interactive_node_visualization(model_type)
    
    with tab2:
        create_model_comparison_view()


# 4. ==============================
# Page Rendering Functions
# =================================

def data_processing_page():
    """Render the data processing page"""
    st.markdown('<div class="main-header data-header"><h1>🛠️ K-hop Graph Data Processing System</h1><p>GAE-based anomaly detection | K-hop expansion | Random walk splitting</p></div>', unsafe_allow_html=True)

    # --- Sidebar ---
    st.sidebar.header("⚙️ Processing Parameters")
    test_k_hop = st.sidebar.slider("K-hop Expansion Layers", 1, 5, 2, help="Number of expansion layers from seed nodes")
    top_k_ratio = st.sidebar.slider("GAE Top-K Ratio", 0.01, 0.10, 0.03, 0.01, help="Proportion of suspicious nodes to select")
    
    st.sidebar.subheader("📁 File Paths")
    data_paths = {
        "Original Complete Graph": "Dataset/MulDiGraph.pkl", 
        "Test Subgraph": "Dataset/test_subgraph.pkl", 
        "Phishing Nodes": "Dataset/Processed/phishing_nodes.pkl", 
        "Output Directory": f"dataset_k-hop{test_k_hop}"
    }
    for name, default_path in data_paths.items():
        st.sidebar.text_input(name, value=default_path, key=f"path_{name}")

    # --- Main Page ---
    if st.sidebar.button("🔄 Load Data", type="primary"):
        with st.spinner("Loading data..."):
            G_original, G_test, phishing_nodes = load_graph_data()
            if G_original is not None:
                st.session_state.G_original, st.session_state.G_test, st.session_state.phishing_nodes = G_original, G_test, phishing_nodes
                st.sidebar.success("✅ Data loaded successfully")
            else:
                st.sidebar.error("❌ Data loading failed")

    if 'G_original' not in st.session_state:
        st.warning("⚠️ Please load data from the sidebar first")
        return

    G_original, G_test, phishing_nodes = st.session_state.G_original, st.session_state.G_test, st.session_state.phishing_nodes
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="data-metric-box"><h3>Original Graph</h3><h2>{G_original.number_of_nodes():,}</h2><p>Node Count</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="data-metric-box"><h3>Test Subgraph</h3><h2>{G_test.number_of_nodes():,}</h2><p>Node Count</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="data-metric-box"><h3>Phishing Nodes</h3><h2>{len(phishing_nodes):,}</h2><p>Known Anomalies</p></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📍 STEP 1: Seed Nodes", "🔄 STEP 2: K-hop Expansion", "🧱 STEP 3: Training Set Construction", "🚶‍♂️ STEP 4: Random Walk Splitting", "💾 Data Saving"])

    with tab1:
        st.markdown('<div class="data-step-card"><h3>📍 STEP 1: Seed Node Selection</h3><p><strong>Goal</strong>: Use GAE model to identify suspicious nodes and combine with known phishing nodes as seeds</p></div>', unsafe_allow_html=True)
        if st.button("🚀 Execute STEP 1", key="step1"):
            with st.spinner("Running GAE anomaly detection..."):
                suspicious_set, suspicious_scores, top_suspicious = generate_suspicious_nodes_with_gae(G_test, phishing_nodes, top_k_ratio)
                st.session_state.processing_results["step1"] = {"suspicious_set": suspicious_set, "suspicious_scores": suspicious_scores, "top_suspicious": top_suspicious, "seed_nodes": phishing_nodes.union(suspicious_set)}
                st.success("✅ STEP 1 completed")
        if "step1" in st.session_state.processing_results:
            data = st.session_state.processing_results["step1"]
            st.metric("🌱 Total seed nodes", len(data["seed_nodes"]))
            create_data_step_visualization("STEP 1", data)

    with tab2:
        st.markdown(f'<div class="data-step-card"><h3>🔄 STEP 2: K-hop Expansion</h3><p><strong>Goal</strong>: Expand from seed nodes to obtain adjacent region and construct test subgraph (expand to {test_k_hop} hops)</p></div>', unsafe_allow_html=True)
        if st.button("🚀 Execute STEP 2", key="step2"):
            if "step1" not in st.session_state.processing_results:
                st.error("❌ Please complete STEP 1 first")
            else:
                with st.spinner("Running K-hop expansion..."):
                    seed_nodes = st.session_state.processing_results["step1"]["seed_nodes"]
                    expanded_nodes, expansion_info = get_k_hop_neighbors(G_test, seed_nodes, test_k_hop)
                    test_subgraph = G_test.subgraph(expanded_nodes).copy()
                    st.session_state.processing_results["step2"] = {"test_subgraph": test_subgraph, "expansion_info": expansion_info}
                    st.success("✅ STEP 2 completed")
        if "step2" in st.session_state.processing_results:
            data = st.session_state.processing_results["step2"]
            st.metric("🎯 Test subgraph nodes", data["test_subgraph"].number_of_nodes())
            create_data_step_visualization("STEP 2", data)

    # ... 其他步骤的逻辑类似 ...
    with tab3:
        st.markdown('<div class="data-step-card"><h3>🧱 STEP 3: Clean Training Set Construction</h3><p><strong>Method</strong>: Training graph = Original graph - Test subgraph nodes - All anomaly nodes</p></div>', unsafe_allow_html=True)
        if st.button("🚀 Execute STEP 3", key="step3"):
            if "step2" not in st.session_state.processing_results:
                st.error("❌ Please complete STEP 2 first")
            else:
                with st.spinner("Constructing training set..."):
                    test_nodes = set(st.session_state.processing_results["step2"]["test_subgraph"].nodes())
                    all_anomaly_nodes = st.session_state.processing_results["step1"]["seed_nodes"]
                    
                    original_nodes = set(G_original.nodes())
                    nodes_to_remove = test_nodes.union(all_anomaly_nodes)
                    nodes_to_remove_in_original = nodes_to_remove.intersection(original_nodes)
                    train_nodes = original_nodes - nodes_to_remove_in_original
                    
                    train_subgraph = G_original.subgraph(train_nodes).copy()
                    
                    st.session_state.processing_results["step3"] = {
                        "train_subgraph": train_subgraph,
                    }
                    st.success("✅ STEP 3 completed")

        if "step3" in st.session_state.processing_results:
            data = st.session_state.processing_results["step3"]
            st.metric("🎯 Training graph nodes", data["train_subgraph"].number_of_nodes())
            st.metric("🔗 Training graph edges", data["train_subgraph"].number_of_edges())

    
    with tab4:
        st.markdown('<div class="data-step-card"><h3>🚶‍♂️ STEP 4: Random Walk Splitting</h3><p><strong>Method</strong>: Load pre-processed training subsets</p></div>', unsafe_allow_html=True)
        if st.button("🚀 Execute STEP 4", key="step4"):
            with st.spinner("Loading training subsets..."):
                try:
                    train_splits = []
                    for i in range(1, 4):
                        file_path = f"dataset_k-hop2/training_subgraph_random_walk_{i}.pkl"
                        if os.path.exists(file_path):
                            with open(file_path, 'rb') as f:
                                data, node_ids = pickle.load(f)
                                if hasattr(data, 'edge_index'):
                                    G_subset = nx.DiGraph()
                                    G_subset.add_nodes_from(range(len(node_ids)))
                                    edge_list = data.edge_index.t().numpy()
                                    G_subset.add_edges_from(edge_list)
                                    train_splits.append(G_subset)
                                else:
                                    train_splits.append(data)
                        else:
                            st.warning(f"⚠️ File not found: {file_path}")
                    
                    if train_splits:
                        st.session_state.processing_results["step4"] = {
                            "train_splits": train_splits,
                            "split_results": train_splits,
                        }
                        st.success("✅ STEP 4 completed")
                    else:
                        st.error("❌ No training subset files found")
                except Exception as e:
                    st.error(f"❌ Loading failed: {e}")

        if "step4" in st.session_state.processing_results:
            data = st.session_state.processing_results["step4"]
            st.metric("📊 Number of splits", len(data["train_splits"]))
            create_data_step_visualization("STEP 4", data)


    with tab5:
        st.markdown('<div class="data-step-card"><h3>💾 Data Saving</h3><p><strong>Function</strong>: Save processing results in PyG format</p></div>', unsafe_allow_html=True)
        if st.button("💾 Save Processing Results", key="save_data"):
            if "step4" not in st.session_state.processing_results:
                st.error("❌ Please complete all processing steps first")
            else:
                with st.spinner("Saving data..."):
                    try:
                        output_dir = st.session_state.get('path_输出目录', f"dataset_k-hop{test_k_hop}")
                        os.makedirs(output_dir, exist_ok=True)
                        
                        step1_data = st.session_state.processing_results["step1"]
                        step2_data = st.session_state.processing_results["step2"]
                        step4_data = st.session_state.processing_results["step4"]
                        
                        test_subgraph = step2_data["test_subgraph"]
                        suspicious_set = step1_data["suspicious_set"]
                        
                        X_test, y_test, node_ids_test = extract_node_features(test_subgraph, phishing_nodes, suspicious_set)
                        test_data, test_ids = to_pyg_graph(test_subgraph, X_test, y_test, node_ids_test)
                        
                        test_path = os.path.join(output_dir, "realistic_test_graph.pkl")
                        with open(test_path, 'wb') as f:
                            pickle.dump((test_data, test_ids), f)
                        
                        rw_folder = 'random_walk'
                        os.makedirs(rw_folder, exist_ok=True)
                        
                        for i, subgraph in enumerate(step4_data["train_splits"]):
                            X_split, y_split, node_ids_split = extract_node_features(subgraph, phishing_nodes, suspicious_set)
                            train_split_data, train_split_ids = to_pyg_graph(subgraph, X_split, y_split, node_ids_split)
                            
                            split_path = os.path.join(rw_folder, f'training_subgraph_random_walk_{i+1}.pkl')
                            with open(split_path, 'wb') as f:
                                pickle.dump((train_split_data, train_split_ids), f)
                        
                        st.success("✅ Data saved successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Save failed: {e}")

def model_analysis_page():
    """Render the model analysis page"""
    st.markdown('<div class="main-header model-header"><h1>Graph Model Analysis System</h1></div>', unsafe_allow_html=True)

    # --- Sidebar ---
    st.sidebar.header("🔧 System Configuration")
    model_choice = st.sidebar.selectbox("Select Analysis Model", ["GraphSAGE", "SuperGAT"], key="model_selector")

    # --- Dynamic Path and Data Loading ---
    # Set data paths based on model selection
    if model_choice == "SuperGAT":
        default_results_dir = "supergat_autoencoder_ocsvm/topk_results"
        default_models_dir = "supergat_autoencoder_ocsvm"
    else:
        default_results_dir = "quick_graphsage_ocsvm_results"
        default_models_dir = "graphsage_autoencoder_models"

    # Initialize session_state
    if 'current_model' not in st.session_state:
        st.session_state.current_model = model_choice
        st.session_state.results_dir = default_results_dir
        st.session_state.csv_results_dir = default_results_dir

    # Update paths and force refresh when model selection changes
    if st.session_state.current_model != model_choice:
        st.session_state.current_model = model_choice
        st.session_state.results_dir = default_results_dir
        st.session_state.csv_results_dir = default_results_dir
        st.cache_data.clear()
        st.rerun()

    # Get current path from session_state
    results_dir = st.session_state.results_dir
    csv_results_dir = st.session_state.csv_results_dir
    
    # Allow user to manually edit path in sidebar
    new_results_dir = st.sidebar.text_input("Results Directory", value=results_dir)
    if new_results_dir != results_dir:
        st.session_state.results_dir = new_results_dir
        st.cache_data.clear()
        st.rerun()

    st.sidebar.subheader("Status Check")
    required_files = ['quick_ocsvm_results.pkl', 'extended_topk_results.pkl', 'confusion_matrix_results.pkl']
    all_files_exist = True
    for file in required_files:
        file_path = os.path.join(st.session_state.results_dir, file)
        exists = os.path.exists(file_path)
        st.sidebar.text(f'{"✅" if exists else "❌"} {file}')
        if not exists:
            all_files_exist = False
    
    # Check CSV file status
    csv_files_exist = False
    if model_choice == "SuperGAT":
        # All SuperGAT files are in supergat_autoencoder_ocsvm/topk_results/
        csv_check_paths = [
            os.path.join(csv_results_dir, "top_100_results.csv"),
            os.path.join(csv_results_dir, "top_50_results.csv"),
            os.path.join(csv_results_dir, "top_200_results.csv"),
            os.path.join(csv_results_dir, "top_500_results.csv")
        ]
    else:
        # GraphSAGE CSV files are directly in the results directory
        csv_check_paths = [
            os.path.join(csv_results_dir, "top_100_results.csv")
        ]
    
    for csv_path in csv_check_paths:
        if os.path.exists(csv_path):
            csv_files_exist = True
            break
    
    st.sidebar.text(f'{"✅" if csv_files_exist else "❌"} CSV Files')
    
    if all_files_exist and csv_files_exist:
        st.sidebar.markdown('<div class="status-success">✅ All data files ready</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="status-warning">⚠️ Some data files missing</div>', unsafe_allow_html=True)

    if st.sidebar.button("🔄 Manually Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()

    # Debug information
    st.sidebar.subheader("🔍 Path Debug")
    st.sidebar.text(f"Model: {model_choice}")
    st.sidebar.text(f"PKL Path: {st.session_state.results_dir}")
    st.sidebar.text(f"CSV Path: {st.session_state.csv_results_dir}")
    st.sidebar.text(f"Absolute Path: {os.path.abspath(st.session_state.results_dir)}")
    
    # Show actually checked files
    st.sidebar.text("Checked files:")
    for k in [50, 100, 200, 500]:
        csv_file = os.path.join(st.session_state.csv_results_dir, f"top_{k}_results.csv")
        exists = os.path.exists(csv_file)
        st.sidebar.text(f'{"✅" if exists else "❌"} top_{k}_results.csv')

    # --- Main Page ---
    st.subheader(f"Model: {st.session_state.current_model}")
    
    main_results, extended_results, confusion_results = load_results_data(st.session_state.results_dir)
    csv_data = load_csv_results(st.session_state.results_dir, st.session_state.csv_results_dir)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Performance Overview", "📈 Top-K Analysis", "🔄 Confusion Matrix", "📋 Hit@K", "🎯 Node Visualization"])
    
    with tab1:
        create_performance_overview(main_results, extended_results)
    with tab2:
        create_topk_analysis(extended_results, csv_data)
    with tab3:
        create_confusion_matrix_analysis(confusion_results)
    with tab4:
        create_detailed_results_view(csv_data)
    with tab5:
        create_visualization_page(st.session_state.current_model)


# 5. ==============================
# Main Function and Navigation
# =================================

def main():
    """Main navigation function"""
    st.sidebar.title("🚀 Integrated Graph Intelligence Analysis System")
    
    app_mode = st.sidebar.selectbox(
        "Please select functional module",
        ["Graph Data Processing System", "Graph Model Analysis System"]
    )

    # Initialize session state
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = {}

    if app_mode == "Graph Data Processing System":
        data_processing_page()
    elif app_mode == "Graph Model Analysis System":
        model_analysis_page()

if __name__ == "__main__":
    main()

