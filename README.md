# 🚀 Build LLM from Scratch - GPT-2 Implementation

A comprehensive implementation of GPT-2 architecture built from scratch for educational and research purposes. This project demonstrates the complete pipeline for training a Large Language Model, from data preprocessing to model training with modern MLOps practices.

## 🎯 Project Overview

This repository contains a complete implementation of a GPT-2 style transformer model, including:

- **Custom GPT-2 Architecture**: Built from scratch using PyTorch
- **Advanced Data Pipeline**: Optimized dataset loading and preprocessing
- **Multiple Training Strategies**: Support for both HuggingFace and custom training loops
- **MLOps Integration**: Weights & Biases tracking and experiment management
- **Production-Ready Code**: Modular, well-documented, and scalable implementation

## 🏗️ Architecture

### Model Configuration (GPT-124M)
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 128,    # Context length
    "emb_dim": 64,           # Embedding dimension
    "n_heads": 8,            # Number of attention heads
    "n_layers": 8,           # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False,       # Query-Key-Value bias
    "max_length": 128,       # Maximum sequence length
    "output_dimension": 64,  # Output dimension
    "batch_size": 2          # Batch size
}
```

### Key Features
- **Multi-Head Self-Attention**: Efficient attention mechanism implementation
- **Positional Encoding**: Learnable position embeddings
- **Layer Normalization**: Pre-norm architecture for stable training
- **Dropout Regularization**: Configurable dropout for generalization
- **Causal Masking**: Proper autoregressive generation support

## 📊 Dataset Support

### Supported Data Sources
1. **Local Text Files** (.txt format)
2. **HuggingFace Datasets** (with streaming support)
   - FineWeb (`HuggingFaceFW/fineweb`)
   - OpenWebText
   - Custom datasets

### Data Processing Features
- **Memory-Efficient Streaming**: Handle large datasets without OOM
- **Flexible Tokenization**: Support for multiple tokenization strategies
- **Dataset Merging**: Combine multiple data sources
- **Chunking & Overlap**: Intelligent text segmentation
- **Input-Target Pair Creation**: Both implicit and explicit shifting strategies

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/build-llm-from-scratch-gpt2.git
cd build-llm-from-scratch-gpt2

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e .
```

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install tiktoken wandb
pip install numpy matplotlib tqdm
```

## 🚀 Quick Start

### 1. Data Preparation
```python
from datasets import load_dataset

# Load from HuggingFace
dataset = load_huggingface_dataset(
    dataset_name="HuggingFaceFW/fineweb",
    name="sample-10BT",
    num_samples=10000
)

# Or load local text files
dataset = load_txt_file("path/to/your/text.txt")
```

### 2. Model Training
```python
# Initialize model with configuration
model = GPTModel(GPT_CONFIG_124M)

# Setup training
trainer = GPTTrainer(
    model=model,
    config=training_config,
    dataset=dataset
)

# Start training with W&B tracking
trainer.train()
```

### 3. Text Generation
```python
# Generate text
generated_text = model.generate(
    prompt="The future of AI is",
    max_length=100,
    temperature=0.8
)
print(generated_text)
```

## 📈 Training Strategies

### 1. HuggingFace Integration (Recommended)
- **Implicit Shifting**: Standard HuggingFace approach
- **Trainer API**: Built-in optimization and logging
- **Model Hub**: Easy model sharing and deployment

### 2. Custom Training Loop
- **Explicit Shifting**: Manual input-target pair creation
- **Fine-grained Control**: Custom loss functions and optimizers
- **Research Flexibility**: Easy experimentation with new techniques

## 🔧 Configuration

### Training Configuration
```python
training_config = {
    "learning_rate": 5e-4,
    "batch_size": 8,
    "num_epochs": 10,
    "warmup_steps": 1000,
    "weight_decay": 0.01,
    "gradient_clipping": 1.0,
    "save_steps": 1000,
    "eval_steps": 500
}
```

### Data Configuration
```python
data_config = {
    "max_length": 512,
    "stride": 256,
    "preprocessing_workers": 4,
    "streaming": True,
    "shuffle": True,
    "seed": 42
}
```

## 📊 Monitoring & Logging

### Weights & Biases Integration
- **Real-time Metrics**: Loss, perplexity, learning rate
- **Model Artifacts**: Automatic model versioning
- **Hyperparameter Tracking**: Complete experiment reproducibility
- **Custom Dashboards**: Visualize training progress

### Key Metrics Tracked
- Training/Validation Loss
- Perplexity
- Learning Rate Schedule
- Gradient Norms
- Model Parameters
- Dataset Statistics

## 🧪 Experiments & Results

### Baseline Results
| Model Size | Parameters | Dataset | Perplexity | Training Time |
|------------|------------|---------|------------|---------------|
| GPT-124M   | 124M       | FineWeb | TBD        | TBD           |

### Ablation Studies
- Effect of model size on performance
- Impact of different tokenization strategies
- Comparison of training strategies (implicit vs explicit)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/
isort src/

# Type checking
mypy src/
```

## 📚 Educational Resources

### Learning Path
1. **Transformer Architecture**: Understanding attention mechanisms
2. **Tokenization**: BPE, WordPiece, and SentencePiece
3. **Training Dynamics**: Learning rates, optimization, and regularization
4. **Scaling Laws**: Model size vs performance relationships
5. **Evaluation Metrics**: Perplexity, BLEU, and human evaluation

### Recommended Reading
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the original GPT-2 architecture
- HuggingFace for the transformers library and datasets
- The open-source ML community for inspiration and tools

## 📞 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **LinkedIn**: [Your LinkedIn Profile]
- **Twitter**: [@yourusername]

---

⭐ **Star this repository if you find it helpful!** ⭐

## 🔄 Project Status

- ✅ Core GPT-2 implementation
- ✅ Data pipeline and preprocessing
- ✅ Training infrastructure
- ✅ W&B integration
- 🚧 Model evaluation suite
- 🚧 Inference optimization
- 📋 Multi-GPU training support
- 📋 Model deployment tools

---

*Built with ❤️ for the ML community*