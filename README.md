# miDiscover

**Production-Ready Sparse Autoencoder Training for Mechanistic Interpretability**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A comprehensive, production-ready platform for training Sparse Autoencoders (SAEs) on neural network activations, specifically designed for mechanistic interpretability research. Built with modern software engineering practices and proven to achieve research-quality results.

## 🎯 Why miDiscover?

**Research-Proven Results:**
- ✅ **25.7% sparsity** achieved on Microsoft Phi-2 activations
- ✅ **20,480 interpretable features** discovered per training run
- ✅ **100% Neuronpedia compatibility** for sharing with the research community
- ✅ **13-minute training pipeline** with multi-GPU optimization

**Production Architecture:**
- 🏗️ **Modular codebase** with clean separation of concerns
- 🐳 **Docker containerization** for reproducible environments
- 📊 **Database integration** for experiment tracking and collaboration
- 🌐 **REST API** for programmatic access and integration
- 🔧 **Comprehensive tooling** from training to interpretation to deployment

## 🚀 Quick Start

### Prerequisites

- **Hardware:** NVIDIA GPU with 16GB+ VRAM (dual-GPU setup recommended)
- **Software:** Python 3.10+, CUDA 11.8+, Docker (optional)
- **Storage:** 100GB+ free space for model outputs

### Installation

```bash
# Clone the repository
git clone https://github.com/Onegaishimas/midiscover.git
cd midiscover

# Create and activate virtual environment
python -m venv sae_env
source sae_env/bin/activate  # Windows: sae_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Your First SAE Training

```bash
# Quick demo (2-3 minutes)
python src/core/sae_train.py --max-samples 100 --epochs 1

# Research-quality training (10-15 minutes)
python src/core/sae_train.py --max-samples 1000 --epochs 3 --l1-coef 1e-3

# High-quality research (30-45 minutes)
python src/core/sae_train.py --max-samples 2000 --epochs 5 --l1-coef 1e-3 --use-db
```

### Interpret Your Results

```bash
# Test what concepts your features detect
python src/core/feature_interpreter.py ./data/sae_outputs \
  --custom-texts data/feature_find_txt_modules/emotions.txt --top-features 10

# Verify Neuronpedia compatibility
python scripts/utility/sae_verification.py ./data/sae_outputs --visualize --package
```

## 📁 Project Structure

```
miDiscover/
├── src/                          # Core source code
│   ├── core/                     # SAE training and analysis
│   │   ├── sae_train.py         # Main training pipeline
│   │   ├── feature_interpreter.py # Concept testing framework
│   │   └── db_utils.py          # Database integration
│   ├── api/                      # REST API server
│   │   └── server.py            # FastAPI implementation
│   └── utils/                    # Utilities and helpers
├── scripts/                      # Command-line tools
│   └── utility/                 # Analysis and maintenance scripts
├── data/                         # Data and configurations
│   ├── feature_find_txt_modules/ # Concept testing text files
│   ├── sae_outputs/             # Generated outputs (git-ignored)
│   └── init_scripts/            # Database initialization
├── tests/                        # Unit tests
├── docs/                         # Documentation
└── examples/                     # Usage examples
```

## 🔬 Key Features

### Systematic Feature Discovery
- **Multi-layer analysis** across transformer architectures
- **Concept testing framework** with 10 pre-built linguistic modules
- **Automated interpretation** of discovered features
- **Quality metrics** ensuring research-grade sparsity levels

### Database-Driven Experiment Tracking
- **PostgreSQL integration** for storing training runs and results
- **Comprehensive metrics** including loss curves and feature statistics
- **Collaborative research** with shared experiment databases
- **Data visualization** and analysis tools

### Production-Ready Architecture
- **Multi-GPU optimization** with automatic memory management
- **Containerized deployment** with Docker and docker-compose
- **REST API** for programmatic access and integration
- **Comprehensive error handling** and logging

### Research Community Integration
- **Neuronpedia compatibility** for sharing discovered features
- **Standardized outputs** following research best practices
- **Reproducible experiments** with detailed configuration tracking
- **Open research** enabling community collaboration

## 🎛️ Configuration Options

### Training Parameters

```bash
# Model and layer selection
--model-name microsoft/phi-2              # Target model
--layer-name model.layers.16.mlp.fc2      # Target layer

# Training configuration  
--max-samples 1000                        # Training dataset size
--epochs 3                                # Training epochs
--batch-size 32                           # Batch size
--l1-coef 1e-3                           # Sparsity coefficient

# Architecture options
--expansion-factor 8                      # SAE size multiplier (d_hidden = d_model * factor)

# Database and logging
--use-db                                  # Enable PostgreSQL logging
--wandb                                   # Enable Weights & Biases tracking
```

### Multi-Layer Analysis

```bash
# Compare features across transformer layers
python src/core/sae_train.py --layer-name model.layers.8.mlp.fc2 --max-samples 1000 --epochs 3
python src/core/sae_train.py --layer-name model.layers.16.mlp.fc2 --max-samples 1000 --epochs 3  
python src/core/sae_train.py --layer-name model.layers.24.mlp.fc2 --max-samples 1000 --epochs 3
```

### Hyperparameter Optimization

```bash
# Higher sparsity (fewer active features)
python src/core/sae_train.py --l1-coef 2e-3 --max-samples 1000 --epochs 3

# Lower sparsity (more comprehensive feature coverage)
python src/core/sae_train.py --l1-coef 5e-4 --max-samples 1000 --epochs 3
```

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# Start complete stack (PostgreSQL + API + pgAdmin)
docker-compose up -d

# Access points:
# - API Server: http://localhost:8000
# - Database: localhost:5432
# - pgAdmin: http://localhost:5050
```

### Manual Docker Build

```bash
# Build image
docker build -t midiscover:latest .

# Run training with GPU support
docker run --gpus all -v $(pwd)/data/sae_outputs:/app/data/sae_outputs \
  midiscover:latest python src/core/sae_train.py --max-samples 1000 --epochs 3
```

## 📊 Understanding Your Results

### Quality Indicators

**Research-Quality Results:**
- ✅ **Sparsity:** 20-30% inactive features
- ✅ **Reconstruction Loss:** < 0.02
- ✅ **Training Time:** 10-20 minutes for 1000 samples
- ✅ **Active Features:** 70-80% of total features

### Generated Outputs

After training, find these files in `data/sae_outputs/`:

```
sae_outputs/
├── sae_model.pt                  # Complete trained SAE (for inference)
├── feature_analysis.json         # 20,480 feature statistics
├── neuronpedia_metadata.json     # Ready for Neuronpedia upload
├── decoder_weights.npy           # Feature direction vectors
├── top_100_features.json         # Most active features summary
└── visualizations/               # Feature plots and analysis charts
```

### Concept Testing Results

```bash
# Test different types of language concepts
python src/core/feature_interpreter.py ./data/sae_outputs \
  --custom-texts data/feature_find_txt_modules/formal.txt \
  --top-features 20

# Available concept modules:
# emotions.txt, formal.txt, conversational.txt, technical.txt,
# temporal.txt, numbers.txt, questions.txt, commands.txt
```

## 🔧 Advanced Usage

### Database-Enabled Research

```bash
# Set up PostgreSQL (using Docker)
docker-compose up -d postgres

# Train with experiment tracking
python src/core/sae_train.py --max-samples 1000 --epochs 3 --use-db

# Explore training history
python scripts/utility/query_training_runs.py recent
python scripts/utility/explore_features.py list
```

### API Server

```bash
# Start the REST API
python src/api/server.py

# API endpoints available at http://localhost:8000/docs
# - POST /train: Start SAE training
# - GET /features/{run_id}: Get discovered features
# - POST /interpret: Interpret features with custom text
```

### Batch Processing

```bash
# Multiple experiments with different hyperparameters
for l1 in 5e-4 1e-3 2e-3; do
  python src/core/sae_train.py --l1-coef $l1 --max-samples 1000 --epochs 3 --use-db
done

# Multi-layer analysis
for layer in 8 12 16 20 24; do
  python src/core/sae_train.py --layer-name model.layers.${layer}.mlp.fc2 \
    --max-samples 1000 --epochs 3 --use-db
done
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
pytest tests/

# Format code
black src/ scripts/

# Type checking
mypy src/
```

### Adding New Features

1. **Concept Modules:** Add new text files to `data/feature_find_txt_modules/`
2. **Analysis Tools:** Extend `scripts/utility/` with new analysis scripts
3. **Model Support:** Add new architectures in `src/core/sae_train.py`
4. **API Endpoints:** Extend `src/api/server.py` with new functionality

## 📚 Documentation

- **[miDiscovery_Doc.md](miDiscovery_Doc.md):** Complete technical documentation and implementation details
- **[API Documentation](http://localhost:8000/docs):** Interactive API documentation (when server is running)
- **[Database Schema](data/init_scripts/):** PostgreSQL schema and setup scripts
- **[Docker Guide](Dockerfile):** Containerization and deployment details

## 🎓 Research Applications

### Mechanistic Interpretability
- **Feature discovery** across transformer layers
- **Circuit analysis** using discovered features
- **Concept understanding** through systematic testing
- **Safety research** through interpretable representations

### Academic Use
- **Reproducible experiments** with detailed configuration tracking
- **Collaborative research** through shared databases
- **Publication-ready results** with comprehensive documentation
- **Community sharing** via Neuronpedia integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [SAELens](https://github.com/jbloomAus/SAELens) and [TransformerLens](https://github.com/neelnanda-io/TransformerLens)
- Compatible with [Neuronpedia](https://neuronpedia.org) for research sharing
- Inspired by mechanistic interpretability research from Anthropic, OpenAI, and the broader research community

## 📞 Support

- **Documentation:** See [miDiscovery_Doc.md](miDiscovery_Doc.md) for detailed technical information
- **Issues:** Report bugs and request features via GitHub Issues
- **Discussions:** Join our community discussions for help and collaboration

---

**Ready to discover interpretable features in neural networks?**

```bash
git clone https://github.com/Onegaishimas/midiscover.git && cd midiscover
python src/core/sae_train.py --max-samples 1000 --epochs 3
```

Start your journey into mechanistic interpretability today! 🧠🔍