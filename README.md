# CLIP Experiments

A comprehensive exploration of OpenAI's CLIP (Contrastive Language-Image Pre-training) and Apple's MobileCLIP models, featuring various experiments and analyses on vision-language understanding.

## Overview

This repository contains a series of experiments designed to deeply understand the capabilities, performance, and behavior of CLIP models. The experiments cover zero-shot classification, embedding space analysis, prompt engineering strategies, and performance benchmarking.

## Features

The main exploration script (`clip_paper_exploration.py`) includes 9 different experiment modules:

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

## Project Structure

```
clip-experiments/
├── clip_paper_exploration.py  # Main experiments script
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

Run specific experiments by uncommenting the desired function calls in the `__main__` section of `clip_paper_exploration.py`:

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

Then run the script:
```bash
python clip_paper_exploration.py
```

### Example: Running Zero-Shot Classification

```python
if __name__ == "__main__":
    zero_shot_classification()
```

This will classify a sample image using natural language labels.

## Available CLIP Models

The repository supports multiple CLIP model variants:
- RN50
- RN101
- RN50x4
- RN50x16
- RN50x64
- ViT-B/32
- ViT-B/16
- ViT-L/14
- ViT-L/14@336px

## Datasets

The experiments use the following datasets:
- **CIFAR-10**: 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **MNIST**: Handwritten digits (0-9)
- **Custom Test Images**: Various images for demonstration purposes

Datasets are automatically downloaded when running the experiments for the first time.

## Key Insights

Some interesting findings from the experiments:

- **Prompt Engineering**: Detailed prompts like "a photo of a {class}" generally outperform simple class names
- **Temperature Scaling**: Lower temperatures make predictions more confident, higher temperatures create more uniform distributions
- **Embedding Space**: CLIP creates semantically meaningful embeddings where similar concepts cluster together
- **Zero-Shot Performance**: CLIP achieves competitive accuracy on CIFAR-10 without any fine-tuning

## Research Papers

This repository includes the following research papers for reference:
- **CLIP**: Learning Transferable Visual Models From Natural Language Supervision (papers/clip-paper.pdf)
- **MobileCLIP v1**: Fast Image-Text Models through Multi-Modal Reinforced Training (papers/mobileclip-v1.pdf)
- **MobileCLIP v2**: Efficient Vision-Language Models (papers/mobileclip-v2.pdf)

## References

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Apple MobileCLIP](https://github.com/apple/ml-mobileclip)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
