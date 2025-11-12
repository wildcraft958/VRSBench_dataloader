# Configuration Guide

Complete guide to configuring VRSBench DataLoader for production use.

## Table of Contents
1. [Environment Variables](#environment-variables)
2. [VRSBenchConfig Parameters](#vrsbenchconfig-parameters)
3. [DataLoader Parameters](#dataloader-parameters)
4. [Logging Configuration](#logging-configuration)
5. [Performance Tuning](#performance-tuning)
6. [Production Best Practices](#production-best-practices)

---

## Environment Variables

Set these before running the dataloader:

### HuggingFace Authentication

```bash
# Primary method (recommended)
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"

# Alternative
export HF_TOKEN="hf_your_token_here"
```

**Get your token:** https://huggingface.co/settings/tokens

### Logging Level

```bash
export LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

---

## VRSBenchConfig Parameters

### Dataset URLs

```python
config = VRSBenchConfig()

# Custom dataset URLs
config.IMAGES_URL = "https://your-host.com/images.zip"
config.ANNOTATIONS_TRAIN_URL = "https://your-host.com/train.zip"
config.ANNOTATIONS_VAL_URL = "https://your-host.com/val.zip"
```

### Download Settings

```python
config.MAX_RETRIES = 5              # Number of download retry attempts
config.BACKOFF_FACTOR = 1.5         # Exponential backoff multiplier
config.CHUNK_SIZE = 16384           # Download chunk size (bytes)
config.TIMEOUT = 60                 # Request timeout (seconds)
```

### Logging Settings

```python
config.LOG_LEVEL = "INFO"           # Logging verbosity
config.LOG_DIR = "./logs"           # Log directory
config.LOG_FILE = "vrsbench.log"    # Log filename
config.LOG_MAX_BYTES = 10*1024*1024 # Max log file size (10MB)
config.LOG_BACKUP_COUNT = 5         # Number of backup log files
config.JSON_LOGS = True             # Use JSON format (vs plain text)
```

### Cache Settings

```python
config.CACHE_DIR = "./hf_cache"     # Cache directory
config.VERIFY_CACHE = True          # Verify cached files integrity
```

### Image Processing

```python
config.DEFAULT_IMAGE_SIZE = (256, 256)  # Default resize dimensions
config.REGION_PADDING = 10              # Padding for region extraction
```

### Performance

```python
config.NUM_WORKERS = 4              # DataLoader worker processes
config.BATCH_SIZE = 16              # Default batch size
config.PIN_MEMORY = True            # Pin memory for GPU training
config.PREFETCH_FACTOR = 2          # Samples to prefetch per worker
```

### Task Configuration

```python
config.SUPPORTED_TASKS = [
    "classification",
    "detection",
    "captioning",
    "vqa",
    "grounding"
]
```

---

## DataLoader Parameters

### Basic Parameters

```python
dataloader = create_vrsbench_dataloader(
    # Required
    images_dir="./data/images",

    # Task selection
    task="classification",  # classification, detection, captioning, vqa, grounding

    # Annotation source (provide ONE of these)
    annotations=dataset,              # Pre-loaded HF Dataset or list
    annotations_jsonl="path.jsonl",   # Local JSONL file
    annotations_url="https://...",    # URL to download

    # Dataset split
    split="validation",  # train, validation, test

    # Optional
    image_key=None,  # Auto-detect image key in annotations
)
```

### Image Processing

```python
from torchvision import transforms

custom_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

dataloader = create_vrsbench_dataloader(
    images_dir="./data/images",
    task="classification",
    transform=custom_transform
)
```

### Performance Parameters

```python
dataloader = create_vrsbench_dataloader(
    images_dir="./data/images",
    task="classification",
    batch_size=32,              # Batch size
    num_workers=8,              # Worker processes (CPU cores)
)
```

**Worker Guidelines:**
- CPU-only: `num_workers = 4-8`
- GPU training: `num_workers = 4 * num_gpus`
- High-res images: `num_workers = 8-16`
- Testing/debugging: `num_workers = 0` (single process)

### Multi-Task Features

```python
# VQA with multi-annotation expansion
dataloader = create_vrsbench_dataloader(
    images_dir="./data/images",
    task="vqa",
    expand_multi_annotations=True,  # One sample per QA pair
)

# Grounding with region extraction
dataloader = create_vrsbench_dataloader(
    images_dir="./data/images",
    task="grounding",
    region_based=True,              # Extract cropped regions
    expand_multi_annotations=True,  # One sample per object
)
```

### Sampling and Filtering

```python
dataloader = create_vrsbench_dataloader(
    images_dir="./data/images",
    task="classification",
    sample_size=1000,    # Limit to first 1000 samples
    split="validation",  # Filter by split
)
```

### Download Options

```python
# Auto-download everything
dataloader = create_vrsbench_dataloader(
    images_dir="./data/images",
    task="classification",
    download_images=True,
    images_url="https://huggingface.co/.../images.zip",
    annotations_url="https://huggingface.co/.../annotations.zip",
)
```

### Metrics Collection

```python
dataloader, metrics = create_vrsbench_dataloader(
    images_dir="./data/images",
    task="classification",
    return_metrics=True  # Return (dataloader, metrics) tuple
)

# After training loop
summary = metrics.get_summary()
print(json.dumps(summary, indent=2))
```

---

## Logging Configuration

### Production Logging Setup

```python
config = VRSBenchConfig()

# JSON logs for production (machine-readable)
config.JSON_LOGS = True
config.LOG_LEVEL = "INFO"
config.LOG_DIR = "/var/log/vrsbench"
config.LOG_MAX_BYTES = 50 * 1024 * 1024  # 50MB
config.LOG_BACKUP_COUNT = 10

# Plain text for development
config.JSON_LOGS = False
config.LOG_LEVEL = "DEBUG"
```

### Log Rotation

Logs automatically rotate when reaching `LOG_MAX_BYTES`:

```
vrsbench_dataloader.log       # Current log
vrsbench_dataloader.log.1     # Previous
vrsbench_dataloader.log.2
...
vrsbench_dataloader.log.5     # Oldest (then deleted)
```

### Viewing Logs

```bash
# Real-time JSON logs
tail -f logs/vrsbench_dataloader.log | jq .

# Filter by level
cat logs/vrsbench_dataloader.log | jq 'select(.level=="ERROR")'

# Filter by logger
cat logs/vrsbench_dataloader.log | jq 'select(.logger=="DownloadManager")'

# Plain text logs
tail -f logs/vrsbench_dataloader.log | grep ERROR
```

---

## Performance Tuning

### GPU Training Optimization

```python
config = VRSBenchConfig()
config.PIN_MEMORY = True      # Pin memory for faster GPU transfer
config.NUM_WORKERS = 8        # 4 * num_gpus
config.PREFETCH_FACTOR = 2    # Prefetch next batches

dataloader = create_vrsbench_dataloader(
    images_dir="./data/images",
    task="classification",
    batch_size=64,              # Larger batches for GPU
    num_workers=config.NUM_WORKERS,
    config=config
)
```

### Memory Optimization

```python
# Reduce image size
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Smaller images
    transforms.ToTensor(),
])

# Smaller batch size
dataloader = create_vrsbench_dataloader(
    images_dir="./data/images",
    task="classification",
    transform=transform,
    batch_size=8,
    num_workers=2
)
```

### I/O Optimization

```python
# Use SSD storage for images
images_dir = "/ssd/vrsbench/images"  # Fast SSD

# Reduce workers if I/O bound
num_workers = 2  # Less workers on HDD

# Cache verification (skip for faster loading)
config = VRSBenchConfig()
config.VERIFY_CACHE = False  # Skip cache verification
```

### Profiling

```python
import time

dataloader, metrics = create_vrsbench_dataloader(
    images_dir="./data/images",
    task="classification",
    return_metrics=True
)

start = time.time()
for batch_idx, (images, metas) in enumerate(dataloader):
    # Your training code
    pass
duration = time.time() - start

# Check metrics
summary = metrics.get_summary()
print(f"Total time: {duration:.2f}s")
print(f"Images/sec: {summary['metrics']['images_loaded'] / duration:.1f}")
print(f"Avg load time: {summary['timings']['image_load']['mean']*1000:.1f}ms")
```

---

## Production Best Practices

### 1. Authentication

**Always set HuggingFace token in production:**

```bash
# In your deployment script
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN_SECRET}"
```

### 2. Error Handling

```python
from vrsbench_dataloader_production import create_vrsbench_dataloader
import logging

try:
    dataloader = create_vrsbench_dataloader(
        images_dir="./data/images",
        task="classification",
        annotations_url="https://...",
    )
except RuntimeError as e:
    logging.error(f"Failed to create dataloader: {e}")
    # Fallback or alert
    raise
```

### 3. Caching Strategy

```python
# Production: verify cache
config = VRSBenchConfig()
config.VERIFY_CACHE = True
config.CACHE_DIR = "/shared/cache"  # Shared cache for multiple workers

# Development: skip verification for speed
config.VERIFY_CACHE = False
```

### 4. Logging in Production

```python
# Production logging
config = VRSBenchConfig()
config.LOG_LEVEL = "WARNING"    # Only warnings and errors
config.JSON_LOGS = True         # Machine-readable
config.LOG_DIR = "/var/log/app"

# Send to monitoring system
# Parse JSON logs and send to CloudWatch/ELK/Sentry
```

### 5. Resource Limits

```python
# Limit workers based on available CPU cores
import os

num_cores = os.cpu_count()
config = VRSBenchConfig()
config.NUM_WORKERS = min(num_cores - 2, 8)  # Leave 2 cores for OS

# Limit batch size based on memory
import torch

gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
recommended_batch = int(gpu_mem_gb / 0.5)  # ~0.5GB per batch
```

### 6. Monitoring

```python
# Enable metrics in production
dataloader, metrics = create_vrsbench_dataloader(
    images_dir="./data/images",
    task="classification",
    return_metrics=True
)

# After each epoch
summary = metrics.get_summary()

# Send to monitoring (e.g., Prometheus)
from prometheus_client import Gauge

images_loaded = Gauge('vrsbench_images_loaded', 'Images loaded')
images_loaded.set(summary['metrics']['images_loaded'])

load_time = Gauge('vrsbench_avg_load_time', 'Avg image load time')
load_time.set(summary['timings']['image_load']['mean'])

# Reset metrics for next epoch
metrics.reset()
```

### 7. Health Checks

```python
def healthcheck_dataloader():
    """Health check for dataloader"""
    try:
        dataloader = create_vrsbench_dataloader(
            images_dir="./data/images",
            task="classification",
            annotations_jsonl="./data/test.jsonl",
            sample_size=1,
            num_workers=0
        )

        # Try loading one sample
        for images, metas in dataloader:
            break

        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### 8. Graceful Degradation

```python
# Try HF download, fallback to local
try:
    dataloader = create_vrsbench_dataloader(
        images_dir="./data/images",
        task="classification",
        annotations_url="https://huggingface.co/.../annotations.zip",
    )
except RuntimeError as e:
    logging.warning(f"HF download failed, using local: {e}")
    dataloader = create_vrsbench_dataloader(
        images_dir="./data/images",
        task="classification",
        annotations_jsonl="./data/backup_annotations.jsonl",
    )
```

---

## Environment-Specific Configs

### Development

```python
config = VRSBenchConfig()
config.LOG_LEVEL = "DEBUG"
config.VERIFY_CACHE = False
config.NUM_WORKERS = 0  # Easier debugging
config.JSON_LOGS = False
```

### Staging

```python
config = VRSBenchConfig()
config.LOG_LEVEL = "INFO"
config.VERIFY_CACHE = True
config.NUM_WORKERS = 4
config.JSON_LOGS = True
```

### Production

```python
config = VRSBenchConfig()
config.LOG_LEVEL = "WARNING"
config.VERIFY_CACHE = True
config.NUM_WORKERS = 8
config.JSON_LOGS = True
config.LOG_DIR = "/var/log/vrsbench"
config.CACHE_DIR = "/shared/cache"
```

---

## Troubleshooting Config

### High Memory Usage

```python
# Reduce batch size and workers
config.BATCH_SIZE = 8
config.NUM_WORKERS = 2

# Smaller images
transform = transforms.Resize((128, 128))
```

### Slow Loading

```python
# Increase workers
config.NUM_WORKERS = 8

# Disable cache verification
config.VERIFY_CACHE = False

# Enable GPU pinning
config.PIN_MEMORY = True
```

### Network Errors

```python
# Increase retries
config.MAX_RETRIES = 10
config.TIMEOUT = 120
config.BACKOFF_FACTOR = 2.0

# Set HF token
export HUGGINGFACE_HUB_TOKEN="hf_..."
```

---

For more information, see [README.md](README.md)
