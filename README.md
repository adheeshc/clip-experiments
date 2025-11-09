# CLIP Experiments

A comprehensive exploration of OpenAI's CLIP (Contrastive Language-Image Pre-training) and Apple's MobileCLIP models, featuring experiments on vision-language understanding, performance optimization, and production-ready implementations.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
  - [CLIP Paper Exploration](#clip-paper-exploration-clip_paper_explorationpy)
  - [CLIP Optimization](#clip-optimization-clip_optimizationpy)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Available CLIP Models](#available-clip-models)
- [Datasets](#datasets)
- [Key Insights](#key-insights)
- [Performance Benchmarks](#performance-benchmarks)
- [Best Practices for Production](#best-practices-for-production)
- [Use Cases](#use-cases)
- [Quick Start](#quick-start)
- [Research Papers](#research-papers)
- [References](#references)

## Overview

This repository contains two main components:

1. **CLIP Paper Exploration** (`clip_paper_exploration.py`): Deep dive into CLIP's capabilities, zero-shot classification, embedding space analysis, and prompt engineering
2. **CLIP Optimization** (`clip_optimization.py`): Production-ready optimizations including FP16 quantization, text caching, batch processing, and model variants benchmarking

## Features

### CLIP Paper Exploration (`clip_paper_exploration.py`)

Nine comprehensive experiment modules exploring CLIP's capabilities:

#### 1. Model Exploration
- Loads and analyzes different CLIP model variants
- Displays model parameters, input resolution, context length, and vocabulary size

#### 2. Zero-Shot Classification
- Demonstrates basic zero-shot image classification
- Uses natural language descriptions to classify images without fine-tuning

#### 3. CIFAR-10 Zero-Shot Classification
- Tests CLIP's zero-shot capabilities on the CIFAR-10 dataset
- Evaluates performance across 10 object categories

#### 4. MNIST Zero-Shot Classification
- Evaluates CLIP on handwritten digit recognition
- Compares different prompt engineering strategies for digit classification

#### 5. Text-to-Image Retrieval
- Retrieves relevant images based on text queries
- Demonstrates CLIP's multimodal search capabilities

#### 6. Embedding Space Analysis
- Explores the structure of CLIP's embedding space
- Analyzes intra-category and inter-category similarities
- Examines semantic relationships between concepts

#### 7. Performance Benchmarking
- Measures inference latency for image and text encoding
- Tracks throughput and GPU memory usage
- Provides timing statistics for optimization insights

#### 8. Temperature Scaling Analysis
- Analyzes the effect of temperature on similarity distributions
- Demonstrates how temperature affects model confidence

#### 9. Prompt Engineering Deep Dive
- Systematic evaluation of different prompt templates
- Compares simple vs. detailed vs. contextual prompts
- Ranks prompt strategies by classification accuracy

### CLIP Optimization (`clip_optimization.py`)

Production-ready optimizations for deploying CLIP at scale:

#### 1. Text Embedding Cache
- Caches frequently-used text embeddings to avoid recomputation
- Provides cache statistics (hits, misses, hit rate)
- Supports saving/loading cache to disk
- **Speedup**: Up to 99x for repeated queries

#### 2. FP16 Quantization
- Automatic mixed precision (AMP) for faster inference
- Benchmarks FP32 vs FP16 across different batch sizes
- Minimal accuracy loss (<0.00003 difference in similarity scores)
- **Note**: Speedup varies by GPU architecture; modern GPUs with Tensor Cores see greater benefits

#### 3. Batch Processing
- Optimized batch processing for image encoding
- Benchmarks different batch sizes (1, 4, 8, 16, 32, 64)
- Maximizes GPU utilization and throughput
- **Speedup**: Up to 12.88x compared to single-image processing (batch size 64)

#### 4. Model Variants Comparison
- Benchmarks ResNet (RN50, RN101) vs Vision Transformer (ViT-B/16, ViT-B/32, ViT-L/14)
- Compares inference speed and throughput across models
- Helps select optimal model for latency/accuracy tradeoffs

#### 5. OptimizedClip Class
- Combines all optimizations into a single, easy-to-use class
- Configurable FP16, text caching, and batch processing
- Production-ready API for image classification and search
- Supports both CPU and CUDA devices

#### 6. Image Search System
- Efficient semantic image search using optimized CLIP
- Scales to thousands of images with fast retrieval
- Demonstrates production deployment patterns

## Project Structure

```
clip-experiments/
├── clip_paper_exploration.py  # CLIP research & experiments
├── clip_optimization.py       # Production optimizations
├── base_repos/
│   ├── CLIP/                  # OpenAI CLIP repository
│   └── ml-mobileclip/         # Apple MobileCLIP repository
├── data/
│   ├── clip_test_images/      # Sample images for testing
│   ├── cifar-10-batches-py/   # CIFAR-10 dataset
│   └── MNIST/                 # MNIST dataset
└── papers/
    ├── clip-paper.pdf         # CLIP research paper
    ├── mobileclip-v1.pdf      # MobileCLIP v1 paper
    └── mobileclip-v2.pdf      # MobileCLIP v2 paper
```

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended) or CPU

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd clip-experiments
```

2. Install dependencies:
```bash
pip install torch torchvision
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

3. Install additional requirements for base repositories:
```bash
pip install -r base_repos/CLIP/requirements.txt
pip install -r base_repos/ml-mobileclip/requirements.txt
```

## Usage

### Running Exploration Experiments

Run specific experiments by uncommenting the desired function calls in `clip_paper_exploration.py`:

```python
if __name__ == "__main__":
    # Uncomment the experiments you want to run

    # model_exploration()
    # zero_shot_classification()
    # cifar_zero_shot()
    # mnist_zero_shot()
    # text_to_image_retrieval()
    # embedding_space_analysis()
    # performance_benchmark()
    # temperature_scaling_analysis()
    # prompt_engineering_deep_dive()
```

Then run:
```bash
python clip_paper_exploration.py
```

### Running Optimization Benchmarks

Run optimization benchmarks by uncommenting the desired function calls in `clip_optimization.py`:

```python
if __name__ == "__main__":
    # demo_text_caching()           # Text embedding cache demo
    # fp16_quantization_benchmark()  # FP16 vs FP32 comparison
    # batch_processing_benchmark()   # Batch size optimization
    # model_variants_benchmark()     # Compare model variants
    # combined_optimizations()       # All optimizations together
    # image_search()                 # Production image search
```

Then run:
```bash
python clip_optimization.py
```

## Available CLIP Models

The repository supports multiple CLIP model variants:

| Model | Type | Performance | Best For |
|-------|------|-------------|----------|
| ViT-B/32 | Vision Transformer | 1265 img/sec | Speed (fastest) |
| RN50 | ResNet | 902 img/sec | Balanced |
| RN101 | ResNet | 622 img/sec | Balanced |
| ViT-B/16 | Vision Transformer | 432 img/sec | Better accuracy |
| ViT-L/14 | Vision Transformer | 106 img/sec | Best accuracy |
| ViT-L/14@336px | Vision Transformer | ~80 img/sec (est.) | Highest accuracy |

*Performance numbers based on batch size 16 benchmarks on the test hardware*

## Datasets

The experiments use the following datasets:
- **CIFAR-10**: 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **MNIST**: Handwritten digits (0-9)
- **Custom Test Images**: Various images for demonstration purposes

Datasets are automatically downloaded when running the experiments for the first time.

## Key Insights

### From Exploration Experiments

- **Prompt Engineering**: Detailed prompts like "a photo of a {class}" generally outperform simple class names
- **Temperature Scaling**: Lower temperatures make predictions more confident, higher temperatures create more uniform distributions
- **Embedding Space**: CLIP creates semantically meaningful embeddings where similar concepts cluster together
- **Zero-Shot Performance**: CLIP achieves competitive accuracy on CIFAR-10 without any fine-tuning

### From Optimization Experiments

- **Text Caching**: Repeated queries benefit massively from caching (35x measured speedup, up to 100x theoretical with 99% hit rate)
- **FP16 Quantization**: Minimal accuracy loss (0.00003 difference in similarity scores); speedup depends on GPU architecture
- **Batch Processing**: Larger batch sizes dramatically improve throughput (up to 12.88x improvement with batch size 64)
- **Model Selection**: ViT-B/32 is fastest (1265 img/sec), ViT-L/14 for maximum accuracy (106 img/sec)
- **Image Search**: Efficient semantic search at 25ms per query over 1000 images
- **Production Ready**: Combined optimizations enable real-time inference with <100ms latency per image

## Performance Benchmarks

**Benchmark Hardware Specifications:**
- **Laptop**: Acer Predator PH16-71
- **CPU**: Intel Core i7-13700HX (13th Gen, 16 cores, 24 threads @ 2.3 GHz base)
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
- **RAM**: 16 GB
- **OS**: Windows 11 (64-bit)
- **Python**: 3.10
- **CUDA**: Enabled

### Text Caching Performance

| Metric | Without Cache | With Cache | Speedup |
|--------|--------------|------------|---------|
| 100 text encodings | 204ms | 6ms | **35.39x** |
| Cache hit rate | N/A | 99% | - |
| Theoretical speedup | N/A | - | **100x** |

### FP16 vs FP32 (ViT-L/14)

| Batch Size | FP32 Time (s) | FP16 Time (s) | Speedup | Memory (GB) |
|------------|---------------|---------------|---------|-------------|
| 1 | 0.787 | 0.779 | 1.01x | 0.951 |
| 8 | 3.984 | 3.877 | 1.03x | 0.955 |
| 32 | 17.127 | 17.147 | 1.00x | 0.970 |
| 64 | 34.613 | 34.739 | 1.00x | 0.989 |

**Accuracy Check:**
- FP32 Similarity: 0.189209
- FP16 Similarity: 0.189237
- Difference: **0.000028** (negligible)

**Note on FP16 Performance:** The RTX 4060 Laptop GPU has Tensor Cores that support FP16, but the minimal speedup observed (~1.01x) may be due to:
- Model architecture (ViT-L/14) not fully utilizing Tensor Cores
- Memory bandwidth limitations on laptop GPU
- CUDA/PyTorch version optimization
- Other GPUs (especially datacenter GPUs like A100/V100) may see 1.5-2x improvements

### Batch Processing Throughput (ViT-B/32)

| Batch Size | Time (s) | Images/sec | Speedup |
|------------|----------|------------|---------|
| 1 | 0.677 | 147.68 | 1.18x |
| 4 | 0.215 | 465.36 | 3.73x |
| 8 | 0.101 | 993.95 | 7.96x |
| 16 | 0.081 | 1234.08 | 9.88x |
| 32 | 0.077 | 1304.32 | 10.45x |
| 64 | 0.062 | **1607.65** | **12.88x** |

### Model Variants Performance (Batch Size 16, 30 runs)

| Model | Time (s) | Images/sec | Relative Speed |
|-------|----------|------------|----------------|
| ViT-B/32 | 0.379 | **1265.44** | Fastest |
| RN50 | 0.532 | 901.86 | Fast |
| RN101 | 0.772 | 621.67 | Medium |
| ViT-B/16 | 1.112 | 431.84 | Slow |
| ViT-L/14 | 4.525 | 106.08 | Slowest (best accuracy) |

### Production Image Search

| Metric | Value |
|--------|-------|
| Images indexed | 1,000 |
| Average search time | **25ms per query** |
| Top-k retrieval | 5 results |

## Research Papers

This repository includes the following research papers for reference:
- **CLIP**: Learning Transferable Visual Models From Natural Language Supervision (papers/clip-paper.pdf)
- **MobileCLIP v1**: Fast Image-Text Models through Multi-Modal Reinforced Training (papers/mobileclip-v1.pdf)
- **MobileCLIP v2**: Efficient Vision-Language Models (papers/mobileclip-v2.pdf)

## References

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Apple MobileCLIP](https://github.com/apple/ml-mobileclip)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
