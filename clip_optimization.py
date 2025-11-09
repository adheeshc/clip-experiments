"""
CLIP Model Optimization Experiments

demo_text_caching()

benchmark_fp16_quantization()

benchmark_batch_processing()

benchmark_model_variants()

demo_optimized_clip()

production_image_search()

"""

import pickle as pkl
import time
from typing import Any, Dict, List

import clip
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# ============================================
# OPTIMIZATION 1: Text Embedding Cache
# ============================================


class TextEmbeddingCache:
    """Cache for text embeddings to avoid recomputing them"""

    def __init__(self, model: Any, device: str = "cuda") -> None:
        self.model = model
        self.device = device
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text with caching"""
        cache_key = tuple(texts)

        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        self.cache_misses += 1
        text_tokens = clip.tokenize(texts).to(self.device)

        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_tokens)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

        self.cache[cache_key] = text_embeddings
        return text_embeddings

    def save_cache(self, path: str) -> None:
        """Save the cache to disk"""
        cache_data = {
            "cache": {k: v.cpu() for k, v in self.cache.items()},
            "stats": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
            },
        }

        with open(path, "wb") as f:
            pkl.dump(cache_data, f)

        print(f"Cache saved to {path}")

    def load_cache(self, path: str) -> None:
        """Load cache from disk"""
        with open(path, "rb") as f:
            cache_data = pkl.load(f)

        self.cache = {k: torch.from_numpy(v).to(self.device) for k, v in cache_data["cache"].items()}
        self.cache_hits = cache_data["stats"]["cache_hits"]
        self.cache_misses = cache_data["stats"]["cache_misses"]

        print(f"Cache loaded from {path}")

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            "cache_size": len(self.cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": hit_rate,
            "speedup_estimate": f"{1 / (1 - hit_rate):.2f}x" if hit_rate > 0 else "1.00x",
        }


def demo_text_caching() -> TextEmbeddingCache:
    """Demonstrate text embedding caching"""
    print("\n=== Text Embedding Cache ===\n")

    model, preprocess = clip.load("ViT-B/32", device=device)
    cache = TextEmbeddingCache(model, device=device)

    categories = ["a dog", "a cat", "a bird", "a car", "a person"]

    # Benchmark: No caching
    print("\n --- Benchmarking without caching ---")
    start_time = time.time()
    for i in range(10):
        text_tokens = clip.tokenize(categories).to(device)
        with torch.no_grad():
            text_embeddings = model.encode_text(text_tokens)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    duration1 = time.time() - start_time
    print(f"Time for 100 encodings: {duration1:.3f}s\n")

    # Benchmark: With caching
    print("\n --- Benchmarking with caching ---")

    start_time = time.time()
    for i in range(100):
        text_embeddings = cache.encode_text(categories)
    duration2 = time.time() - start_time
    print(f"Time for 100 encodings: {duration2:.3f}s\n")

    speedup = duration1 / duration2
    print(f"\nSpeedup with caching: {speedup:.2f}x")
    print("\nCache stats:", cache.get_stats())
    stats = cache.get_stats()
    for k, v in stats.items():
        print(f"{k}: {v}")

    cache.save_cache("text_embedding_cache.pkl")
    return cache


# ===========================================
# OPTIMIZATION 2: FP16 Quantization Benchmark
# ===========================================


def fp16_quantization_benchmark() -> None:
    """Compare FP32 vs FP16 inference performance across different batch sizes"""
    print("\n=== FP16 Quantization Benchmark ===\n")

    if not torch.cuda.is_available():
        print("FP16 quantization benchmark requires a CUDA-capable GPU.")
        return

    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    batch_sizes = [1, 8, 32, 64]
    num_runs = 50
    results = []

    print("\n" + "=" * 70)
    print(f"{'Batch Size':<12} {'FP32 Time':<12} {'FP16 Time':<12} {'Speedup':<12} {'Memory (GB)':<12}")
    print("=" * 70)

    for batch_size in batch_sizes:
        dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
        dummy_text = clip.tokenize([f"photo {i}" for i in range(batch_size)]).to(device)

        # Warm-up FP32
        for i in range(5):
            with torch.no_grad():
                img_features_fp32 = model.encode_image(dummy_image)
                text_features_fp32 = model.encode_text(dummy_text)

        # Benchmark FP32
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(num_runs):
            with torch.no_grad():
                img_features_fp32 = model.encode_image(dummy_image)
                text_features_fp32 = model.encode_text(dummy_text)
        torch.cuda.synchronize()
        duration_fp32 = time.time() - start_time

        # Warm-up FP16
        for i in range(5):
            with torch.no_grad(), torch.amp.autocast("cuda"):  # type: ignore
                img_features_fp16 = model.encode_image(dummy_image)
                text_features_fp16 = model.encode_text(dummy_text)

        # Benchmark FP16
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(num_runs):
            with torch.no_grad(), torch.amp.autocast("cuda"):  # type: ignore
                img_features_fp16 = model.encode_image(dummy_image)
                text_features_fp16 = model.encode_text(dummy_text)
        torch.cuda.synchronize()
        duration_fp16 = time.time() - start_time

        # Calculate metrics
        speedup = duration_fp32 / duration_fp16
        memory_gb = torch.cuda.memory_allocated() / 1e9

        # Store results
        results.append(
            {
                "batch_size": batch_size,
                "fp32_time": duration_fp32,
                "fp16_time": duration_fp16,
                "speedup": speedup,
                "memory": memory_gb,
            }
        )

        # Print row
        print(f"{batch_size:<12} {duration_fp32:<12.3f} {duration_fp16:<12.3f} {speedup:<12.2f}x {memory_gb:<12.3f}")

        torch.cuda.empty_cache()

    print("=" * 70)

    # Accuracy check with batch size 1
    print("\n--- Accuracy Check ---")
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    dummy_text = clip.tokenize(["a photo of a dog"]).to(device)

    with torch.no_grad():
        img_features_fp32 = model.encode_image(dummy_image)
        text_features_fp32 = model.encode_text(dummy_text)

    with torch.no_grad(), torch.amp.autocast("cuda"):  # type: ignore
        img_features_fp16 = model.encode_image(dummy_image)
        text_features_fp16 = model.encode_text(dummy_text)

    img_features_fp32 /= img_features_fp32.norm(dim=-1, keepdim=True)
    text_features_fp32 /= text_features_fp32.norm(dim=-1, keepdim=True)
    img_features_fp16 = img_features_fp16.float()
    img_features_fp16 /= img_features_fp16.norm(dim=-1, keepdim=True)
    text_features_fp16 = text_features_fp16.float()
    text_features_fp16 /= text_features_fp16.norm(dim=-1, keepdim=True)

    similarity_fp32 = (img_features_fp32 @ text_features_fp32.T).item()
    similarity_fp16 = (img_features_fp16 @ text_features_fp16.T).item()
    diff = np.abs(similarity_fp32 - similarity_fp16)

    print(f"FP32 Similarity: {similarity_fp32:.6f}")
    print(f"FP16 Similarity: {similarity_fp16:.6f}")
    print(f"Difference: {diff:.6f}")


# ===========================================
# OPTIMIZATION 3: Batch Processing Benchmark
# ===========================================


def batch_processing_benchmark() -> None:
    """Benchmark batch processing performance"""
    print("\n=== Batch Processing Benchmark ===\n")

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Generate dummy images
    num_images = 100
    dummy_images = [torch.randn(3, 224, 224) for i in range(num_images)]

    # Benchmark single image processing
    print("\n--- Single Image Processing (batch=1) ---")
    start_time = time.time()
    results_single = []
    for img in dummy_images:
        img_batch = img.unsqueeze(0).to(device)
        with torch.no_grad():
            img_features = model.encode_image(img_batch)
            results_single.append(img_features)
    duration_single = time.time() - start_time
    throughput_single = num_images / duration_single
    print(f"Time for {num_images} images: {duration_single:.3f}s")
    print(f"Throughput: {throughput_single:.2f} images/sec\n")

    # Benchmark batch image processing
    print("\n--- Batch Image Processing ---")

    batch_sizes = [1, 4, 8, 16, 32, 64]

    print("\n" + "=" * 50)
    print(f"{'Batch Size':<12} {'Time (s)':<12} {'Images/sec':<12} {'Speedup (x)':<12}")
    print("=" * 50)

    for batch_size in batch_sizes:
        start_time = time.time()
        results_batch = []

        for i in range(0, num_images, batch_size):
            img_batch = torch.stack(dummy_images[i : i + batch_size]).to(device)
            with torch.no_grad():
                img_features = model.encode_image(img_batch)
                results_batch.append(img_features)
        duration_batch = time.time() - start_time
        throughput_batch = num_images / duration_batch
        speedup = throughput_batch / throughput_single

        print(f"{batch_size:<12} {duration_batch:<12.3f} {throughput_batch:<12.2f} {speedup:.2f}x")

    print("=" * 50)


# ===========================================
# OPTIMIZATION 4: Model Variants Benchmark
# ===========================================


def model_variants_benchmark() -> None:
    """Benchmark different CLIP model variants"""
    print("\n=== Model Variants Benchmark ===\n")

    model_names = [
        "RN50",  # ResNet-50
        "RN101",  # ResNet-101
        "ViT-B/16",  # Vision Transformer Base 16
        "ViT-B/32",  # Vision Transformer Base 32
        "ViT-L/14",  # Vision Transformer Large 14
    ]
    batch_size = 16
    num_runs = 30

    print("\n" + "=" * 70)
    print(f"{'Model':<15} {'Time (s)':<12} {'Images/sec':<12}")
    print("=" * 70)

    for model_name in model_names:
        model, preprocess = clip.load(model_name, device=device)
        model.eval()

        dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)

        # Warm-up
        for i in range(5):
            with torch.no_grad():
                img_features = model.encode_image(dummy_image)

        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(num_runs):
            with torch.no_grad():
                img_features = model.encode_image(dummy_image)
        torch.cuda.synchronize()
        duration = time.time() - start_time

        throughput = (batch_size * num_runs) / duration

        print(f"{model_name:<15} {duration:<12.3f} {throughput:<12.2f}")

    print("=" * 70)


# ===========================================
# OPTIMIZATION 5: Combined Optimizations
# ===========================================


class OptimizedClip:
    """CLIP model with all optimizations applied"""

    def __init__(self, model_name: str = "ViT-B/32", use_fp16: bool = True, device: str = "cuda") -> None:
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.device = device
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.model.eval()

        # Text embedding cache
        self.text_cache = TextEmbeddingCache(self.model, device=device)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encodes text with caching"""
        if self.use_fp16:
            with torch.amp.autocast("cuda"):  # type: ignore
                text_features = self.text_cache.encode_text(texts)
            return text_features.float()  # Convert to float32 to match image features
        else:
            return self.text_cache.encode_text(texts)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encodes image with optional FP16"""
        if self.use_fp16:
            with torch.no_grad(), torch.amp.autocast("cuda"):  # type: ignore
                img_features = self.model.encode_image(images.to(self.device))
                img_features /= img_features.norm(dim=-1, keepdim=True)
            return img_features.float()
        else:
            with torch.no_grad():
                img_features = self.model.encode_image(images.to(self.device))
                img_features /= img_features.norm(dim=-1, keepdim=True)
            return img_features

    def classify_images(self, images: List[Image.Image], categories: List[str]) -> Dict[str, float]:
        """Classify images into categories"""
        img_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)  # type: ignore

        # Encode images
        img_features = self.encode_image(img_tensors)

        # Encode text
        text_features = self.encode_text(categories)

        # Compute similarity
        similarities = (100 * img_features @ text_features.T).softmax(dim=-1)

        values, indices = similarities[0].topk(len(categories))

        results = {
            "predictions": [
                {"category": categories[idx], "probability": val.item()} for val, idx in zip(values, indices)
            ],
            "cache_stats": self.text_cache.get_stats(),
        }

        return results


def combined_optimizations():
    print("\n=== Combined CLIP Optimizations Demo ===\n")
    opt_clip = OptimizedClip(model_name="ViT-B/32", use_fp16=True, device=device)
    categories = ["a dog", "a cat", "a bird", "a car", "a person"]
    num_images = 5
    dummy_images = [Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8)) for _ in range(num_images)]

    start_time = time.time()
    results_list = []

    batch_size = 16

    for i in range(0, num_images, batch_size):
        batch_images = dummy_images[i : i + batch_size]
        results = opt_clip.classify_images(batch_images, categories)
        results_list.append(results["predictions"])

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.3f}s\n")
    print(f"Throughput: {num_images / total_time:.2f} images/sec\n")
    print(f"Average Latency: {total_time / num_images *1000:.3f}ms per image\n")


def image_search():
    """Image search system using optimized CLIP"""
    print("\n=== Production Image Search Demo ===\n")

    opt_clip = OptimizedClip(model_name="ViT-B/32", use_fp16=True, device=device)

    num_images = 1000
    image_embeddings = []

    batch_size = 32
    for i in range(0, num_images, batch_size):
        batch = torch.randn(batch_size, 3, 224, 224).to(device)

        embeddings = opt_clip.encode_image(batch)
        image_embeddings.append(embeddings)

    image_embeddings = torch.cat(image_embeddings, dim=0)
    print(f"Indexed {num_images} images.")

    # Search query
    queries = [
        "a cute dog playing in the park",
        "modern architecture building",
        "delicious food on a plate",
        "beautiful sunset over ocean",
        "person riding a bicycle",
    ]

    search_start = time.time()
    for query in queries:
        text_embedding = opt_clip.encode_text([query])

        similarities = (100 * image_embeddings @ text_embedding.T).squeeze()

        top_k = 5
        values, indices = similarities.topk(top_k)

        print(f"\nQuery: {query}")
        for score, idx in zip(values, indices):
            print(f"Image Index: {idx.item()}, Similarity Score: {score.item():.4f}")
    search_time = time.time() - search_start

    print(f"Average search time: {search_time / len(queries):.3f}s per query")


if __name__ == "__main__":
    print("=" * 50)
    print("CLIP OPTIMIZATIONS")
    print("=" * 50)

    demo_text_caching()

    fp16_quantization_benchmark()

    batch_processing_benchmark()

    model_variants_benchmark()

    combined_optimizations()

    image_search()
