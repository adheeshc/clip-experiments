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
# PART 6: Embedding Space Analysis
# ==================================================


def embedding_space_analysis():
    """Explore CLIPs embedding space"""
    print("\n=== Embedding Space Analysis ===\n")
    model, preprocess = clip.load("ViT-B/32", device=device)

    concepts = {
        "animals": ["dog", "cat", "bird", "fish", "horse"],
        "vehicles": ["car", "truck", "airplane", "boat", "bicycle"],
        "food": ["pizza", "burger", "salad", "pasta", "sushi"],
        "furniture": ["chair", "table", "sofa", "bed", "desk"],
    }

    texts = []
    category = []

    for cat, items in concepts.items():
        for item in items:
            texts.append(f"a photo of a {item}")
            category.append(cat)

    text_tokens = clip.tokenize(texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity_matrix = (text_features @ text_features.T).cpu().numpy()

    print("Intra-category similarities (should be high):")
    start_idx = 0
    for category, items in concepts.items():
        end_idx = start_idx + len(items)
        category_sims = similarity_matrix[start_idx:end_idx, start_idx:end_idx]

        # Get off-diagonal similarities (within category)
        mask = np.ones_like(category_sims, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_sim = category_sims[mask].mean()

        print(f"{category:10s}: {avg_sim:.4f}")
        start_idx = end_idx

    print("\nInter-category similarities (should be lower):")
    categories = list(concepts.keys())
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            cat1 = categories[i]
            cat2 = categories[j]

            items1 = concepts[cat1]
            items2 = concepts[cat2]

            idx1_start = sum(len(concepts[c]) for c in categories[:i])
            idx1_end = idx1_start + len(items1)
            idx2_start = sum(len(concepts[c]) for c in categories[:j])
            idx2_end = idx2_start + len(items2)

            inter_sims = similarity_matrix[idx1_start:idx1_end, idx2_start:idx2_end]
            avg_sim = inter_sims.mean()

            print(f"{cat1:10s} - {cat2:10s}: {avg_sim:.4f}")


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


# ============================================
# PART 8: Temperature Scaling Analysis
# ============================================


def temperature_scaling_analysis():
    """Analyze the effect of temperature scaling on CLIP similarity scores"""
    # Key insight: Lower temperatures make the model more confident (sharper distributions), while higher temperatures lead to more uniform distributions.

    print("\n=== Temperature Scaling Analysis ===\n")
    model, preprocess = clip.load("ViT-B/32", device=device)

    image_filenames = [
        "./data/clip_test_images/dog.jpg",
        "./data/clip_test_images/cat.jpg",
        "./data/clip_test_images/bird.jpg",
    ]
    images = [preprocess(Image.open(fn)).unsqueeze(0).to(device) for fn in image_filenames]  # type: ignore
    text_labels = ["a dog", "a cat", "a bird"]

    text_tokens = clip.tokenize(text_labels).to(device)

    with torch.no_grad():
        image_features = torch.cat([model.encode_image(img) for img in images], dim=0)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Raw similarity (before temperature/softmax)
        logits = 100.0 * image_features @ text_features.T
        print("Raw similarity scores (logits):")
        for i, label in enumerate(text_labels):
            print(f"  {label:20s}: {logits[0][i].item():.4f}")

        print("\nProbabilities with different temperatures:")
        temperatures = [0.01, 0.1, 1.0, 10.0, 100.0]

        for temp in temperatures:
            similarity = (logits / temp).softmax(dim=-1)
            print(f"\nTemperature: {temp}")
            for i, query in enumerate(text_labels):
                values, indices = similarity[i].topk(2)
                print(f"Top images for query '{query}':")
                for value, index in zip(values, indices):
                    print(f"Image: {image_filenames[index.item()]:30s} : {100*value.item():.2f}%")


# ============================================
# PART 9: Prompt Engineering Deep Dive
# ============================================


def prompt_engineering_deep_dive():
    """Systematic Analysis of Prompt Engineering Strategies"""
    print("\n=== Prompt Engineering Deep Dive ===\n")
    model, preprocess = clip.load("ViT-B/32", device=device)

    dataset = CIFAR10(root="./data", download=True, train=False)

    prompt_strategies = {
        "simple": [f"{c}" for c in dataset.classes],
        "detailed": [f"a photo of a {c}" for c in dataset.classes],
        "photo_of_the": [f"a photo of the {c}" for c in dataset.classes],
        "photo_of_a_small": [f"a photo of a small {c}" for c in dataset.classes],
        "photo_of_a_large": [f"a photo of a large {c}" for c in dataset.classes],
        "blurry_photo": [f"a blurry photo of a {c}" for c in dataset.classes],
        "good_photo": [f"a good photo of a {c}" for c in dataset.classes],
        "contextual": [f"a high-resolution image of a {c} in nature" for c in dataset.classes],
        "humorous": [f"a funny picture of a {c}" for c in dataset.classes],
    }

    results = {}
    num_samples = 1000  # Use subset for speed

    for strategy_name, prompts in prompt_strategies.items():
        print(f"\nEvaluating strategy '{strategy_name}': {prompts[0]}")

        text_tokens = clip.tokenize(prompts).to(device)

        correct = 0
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            for i in tqdm(range(num_samples)):
                image, label = dataset[i]
                image_input = preprocess(image).unsqueeze(0).to(device)

                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                predicted_index = similarity[0].argmax().item()

                if predicted_index == label:
                    correct += 1

        accuracy = 100 * correct / num_samples
        results[strategy_name] = accuracy
        print(f"Accuracy with strategy '{strategy_name}': {accuracy:.2f}%")

    print("=" * 50)
    print("Ranking by Accuracy:")
    print("=" * 50)
    for template_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{template_name:30s} {acc:6.2f}%")


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

    embedding_space_analysis()

    performance_benchmark()

    temperature_scaling_analysis()

    prompt_engineering_deep_dive()
