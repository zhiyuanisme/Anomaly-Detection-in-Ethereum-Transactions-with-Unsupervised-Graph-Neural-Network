# Anomaly Detection in Ethereum Transactions with Unsupervised Graph Neural Network

This repository contains an implementation of anomaly detection in Ethereum transactions using unsupervised graph neural networks. The project combines graph-based machine learning approaches with autoencoder and One-Class SVM techniques for detecting suspicious transaction patterns.

## Project Structure

```
.
├── src/                    # Source code
│   ├── check_training_data.py
│   ├── create_supergat_compat_files.py
│   └── graphsage_frontend.py
├── notebooks/              # Jupyter notebooks for experiments and analysis
│   ├── GraphSGAE+autoencoder+ocsvm_rw.ipynb
│   ├── capstone2.ipynb
│   └── supergat_autoencoder_ocsvm_splite_trainset.ipynb
├── results/                # Analysis results and visualizations
│   ├── *.png              # Performance analysis charts
│   ├── *.csv              # Performance results data
│   ├── *.json             # Summary results
│   └── *.npy              # Ranking indices and model outputs
├── data/                   # Dataset storage (add your data files here)
├── models/                 # Trained model storage
└── docs/                   # Documentation
```

## Key Components

### Source Code (`src/`)
- **check_training_data.py**: Data validation and preprocessing utilities
- **create_supergat_compat_files.py**: SuperGAT compatibility file generation
- **graphsage_frontend.py**: GraphSAGE model frontend implementation

### Notebooks (`notebooks/`)
- **GraphSGAE+autoencoder+ocsvm_rw.ipynb**: Graph Spatial Graph Autoencoder with autoencoder and OCSVM
- **capstone2.ipynb**: Main capstone project implementation
- **supergat_autoencoder_ocsvm_splite_trainset.ipynb**: SuperGAT with autoencoder and OCSVM on split training set

### Results (`results/`)
Contains performance analysis visualizations, confusion matrices, and model evaluation results including:
- Global performance analysis charts
- Top-K confusion matrix analysis
- Comprehensive performance results
- Ranking indices for different K values (100, 500, 1000, 2000, 5000, 10000)

## Features

- **Graph Neural Networks**: Implementation of GraphSAGE and SuperGAT architectures
- **Unsupervised Learning**: Autoencoder-based anomaly detection
- **One-Class SVM**: Additional classification layer for anomaly detection
- **Performance Analysis**: Comprehensive evaluation metrics and visualizations
- **Scalable Architecture**: Support for large-scale Ethereum transaction graphs

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/zhiyuanisme/Anomaly-Detection-in-Ethereum-Transactions-with-Unsupervised-Graph-Neural-Network.git
   cd Anomaly-Detection-in-Ethereum-Transactions-with-Unsupervised-Graph-Neural-Network
   ```

2. **Set up your environment**:
   - Install required dependencies for graph neural networks (PyTorch Geometric, NetworkX, etc.)
   - Add your dataset files to the `data/` directory

3. **Run the analysis**:
   - Explore the Jupyter notebooks in the `notebooks/` directory
   - Run the source code in the `src/` directory for specific functionalities

## Usage

### Data Preparation
Place your Ethereum transaction data in the `data/` directory. The expected format should be compatible with graph neural network processing.

### Training Models
Use the notebooks to train and evaluate different model configurations:
- GraphSGAE + Autoencoder + OCSVM
- SuperGAT + Autoencoder + OCSVM

### Evaluation
Results and performance metrics are automatically saved to the `results/` directory, including:
- Performance analysis charts
- Confusion matrices
- Global ranking indices

## Models

The project implements several model architectures:
- **GraphSAGE**: Graph Sample and Aggregate for inductive learning
- **SuperGAT**: Self-supervised Graph Attention Networks
- **Autoencoder**: For unsupervised anomaly detection
- **One-Class SVM**: For binary classification of anomalies

## Results

The project includes comprehensive analysis results showing:
- Model performance across different configurations
- Top-K analysis for different threshold values
- Confusion matrix analysis for classification performance
- Global ranking systems for anomaly scoring

## Contributing

Feel free to contribute to this project by:
- Adding new model architectures
- Improving existing implementations
- Adding more comprehensive evaluation metrics
- Enhancing documentation

## License

This project is available for research and educational purposes.