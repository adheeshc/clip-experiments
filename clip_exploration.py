"""CLIP Deep Dive"""

import clip
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CIFAR10, MNIST
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# ============================================
# PART 1: Model Exploration
# ============================================
def model_exploration():
    """Explore different CLIP models and their parameters"""
    print("\n === Model Exploration === \n")
    model_list = clip.available_models()
    for model_id in model_list:
        model, preprocess = clip.load(model_id, device=device)
        model = model.to(device)
        model = model.eval()

        input_resolution = model.visual.input_resolution
        context_length = model.context_length
        vocab_size = model.vocab_size

        print("-" * 40)
        print(f"Model ID : {model_id}")
        print(f"Model parameters: {np.sum([int(np.prod(p.shape)) for p in model.parameters()])}")
        print("Input resolution:", input_resolution)
        print("Context length:", context_length)
        print("Vocab size:", vocab_size)
        print("-" * 40)


# ============================================
# PART 2: Zero Shot Classification
# ============================================


def zero_shot_classification():
    """Zero Shot Classification Example"""
    print("\n === zero shot classification === \n")
    model, preprocess = clip.load("ViT-B/32", device=device)
    filename = "./data/clip_test_images/bird.jpg"
    image_tensor = preprocess(Image.open(filename))
    assert isinstance(image_tensor, torch.Tensor)
    image = image_tensor.unsqueeze(0).to(device)
    text_labels = ["a dog", "a cat", "a bird", "a car"]
    text = clip.tokenize(text_labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    values, indices = similarity[0].topk(4)
    print("\nPredictions:")
    for value, index in zip(values, indices):
        print(f"{text_labels[index.item()]:20s} : {100*value.item():.2f}%")


# ===================================================
# PART 3: CIFAR 10 Dataset Zero Shot Classification
# ===================================================


def cifar_zero_shot():
    """CIFAR-10 Zero Shot Classification"""
    print("\n === cifar zero shot classification === \n")

    dataset = CIFAR10(root="./data", download=True, train=False)
    model, preprocess = clip.load("ViT-B/32", device=device)

    # CIFAR-10 classes
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    text_prompts = [f"a photo of a {c}" for c in classes]
    text_tokens = clip.tokenize(text_prompts).to(device)

    correct = 0
    total = len(dataset)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        for i in tqdm(range(total)):
            image, label = dataset[i]
            image_input = preprocess(image).unsqueeze(0).to(device)

            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predicted_index = similarity[0].argmax().item()

            if predicted_index == label:
                correct += 1

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{total} images, Current Accuracy: {100 * correct/(i + 1):.2f}%")

    accuracy = 100 * correct / total
    print(f"CIFAR-10 Zero-Shot Classification Accuracy: {accuracy:.2f}%")


# ===================================================
# PART 4: MNIST Dataset Zero Shot Classification
# ===================================================


def mnist_zero_shot():
    """MNIST Zero Shot Classification with different prompt engineering strategies"""
    print("\n=== mnist zero shot classification === \n")
    model, preprocess = clip.load("ViT-B/32", device=device)

    dataset = MNIST(root="./data", download=True, train=False)

    prompt_strategies = [
        [f"{i}" for i in range(10)],
        [f"a photo of the digit {i}" for i in range(10)],
        [f"a handwritten digit {i}" for i in range(10)],
        [f"the number {i}" for i in range(10)],
    ]

    for strategy, prompts in enumerate(prompt_strategies, 1):
        print(f"Evaluating strategy {strategy}: {prompts}")

        text_tokens = clip.tokenize(prompts).to(device)

        correct = 0
        total = len(dataset)

        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            for i in tqdm(range(total)):
                image, label = dataset[i]
                image_input = preprocess(image.convert("RGB")).unsqueeze(0).to(device)

                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                predicted_index = similarity[0].argmax().item()

                if predicted_index == label:
                    correct += 1

        accuracy = 100 * correct / total
        print(f"MNIST Zero-Shot Classification Accuracy with strategy {strategy}: {accuracy:.2f}%")


# ===================================================
# PART 5: Text-to-Image Retrieval
# ===================================================


def text_to_image_retrieval():
    """Demonstrate retrieving images using text queries"""
    print("\n=== Text-to-Image Retrieval ===")
    model, preprocess = clip.load("ViT-B/32", device=device)
    image_filenames = [
        "./data/clip_test_images/dog.jpg",
        "./data/clip_test_images/cat.jpg",
        "./data/clip_test_images/bird.jpg",
        "./data/clip_test_images/car.jpg",
        "./data/clip_test_images/building.jpg",
        "./data/clip_test_images/food.jpg",
        "./data/clip_test_images/flower.jpg",
        "./data/clip_test_images/person.jpg",
    ]
    images = [preprocess(Image.open(fn)).unsqueeze(0).to(device) for fn in image_filenames]  # type: ignore
    text_queries = ["dogs sleeping", "salad", "vehicles", "flying", "skyscraper"]

    text_tokens = clip.tokenize(text_queries).to(device)

    with torch.no_grad():
        image_features = torch.cat([model.encode_image(img) for img in images], dim=0)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * text_features @ image_features.T).softmax(dim=-1)
    for i, query in enumerate(text_queries):
        values, indices = similarity[i].topk(2)
        print(f"\nTop images for query '{query}':")
        for value, index in zip(values, indices):
            print(f"Image: {image_filenames[index.item()]:30s} : {100*value.item():.2f}%")


# ==================================================
# PART 6: Understanding Embeddings
# ==================================================


def explore_embeddings():
    """Explore CLIPs embedding space"""
    print("\n=== Understanding Embeddings ===\n")
    model, preprocess = clip.load("ViT-B/32", device=device)

    texts = ["a dog", "a puppy", "a cat", "a kitten", "a car", "a vehicle"]
    text_tokens = clip.tokenize(texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarity = (text_features[i] @ text_features[j].T).item()
            print(f"Similarity between '{texts[i]}' and '{texts[j]}': {similarity:.4f}")


# ===================================================
# PART 7: Performance Benchmarking
# ===================================================


def performance_benchmark():
    """Benchmark CLIP model performance on image and text encoding"""
    print("\n=== Performance Benchmark ===\n")
    model, preprocess = clip.load("ViT-B/32", device=device)

    dummy_image = torch.randn(1, 3, model.visual.input_resolution, model.visual.input_resolution).to(device)
    dummy_text = clip.tokenize(["This is a sample text for benchmarking."]).to(device)

    import time

    # warm-up
    for _ in range(10):
        with torch.no_grad():
            _ = model.encode_image(dummy_image)
            _ = model.encode_text(dummy_text)

    num_runs = 100
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark image encoding
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.encode_image(dummy_image)
    image_time = time.time() - start_time
    print(f"Image encoding time for {num_runs} runs: {image_time:.4f} seconds")

    # Benchmark text encoding
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.encode_text(dummy_text)
    text_time = time.time() - start_time
    print(f"Text encoding time for {num_runs} runs: {text_time:.4f} seconds")

    # Overall benchmarking
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model.encode_image(dummy_image)
            _ = model.encode_text(dummy_text)
    total_time = time.time() - start_time

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    avg_time = total_time / num_runs * 1000  # ms
    print("\nInference Latency:")
    print(f"- Average: {avg_time:.2f} ms")
    print(f"- Throughput: {1000/avg_time:.2f} images/second")

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        print("\nGPU Memory:")
        print(f"- Allocated: {memory_allocated:.2f} GB")
        print(f"- Reserved: {memory_reserved:.2f} GB")


# ===================================================
# MAIN EXECUTION
# ===================================================


if __name__ == "__main__":
    print("=" * 50)
    print("CLIP EXPLORATION")
    print("=" * 50)

    model_exploration()

    zero_shot_classification()

    cifar_zero_shot()

    mnist_zero_shot()

    text_to_image_retrieval()

    explore_embeddings()

    performance_benchmark()
