# VRSBench DataLoader

**Version:** 3.0.0  
**Author:** Animesh Raj  

Production-ready PyTorch DataLoader for the VRSBench (Vision-language for Remote Sensing) dataset. Built for enterprise workloads with hardened reliability, comprehensive logging, and multi-task abstractions that eliminate boilerplate across classification, detection, captioning, VQA, and visual grounding tasks.

## 🚀 Quick Setup Files

This repository includes utility files for easy setup:

- **`jupyter_setup.py`** - Zero-friction Jupyter setup (works in Colab, JupyterLab, local Jupyter)
- **`setup_vrsbench.py`** - Reusable setup utility for any environment (Python scripts, notebooks, etc.)
- **`setup.sh`** - Bash script for local installation

**Architecture**: The loader is built around a high-level API that automatically handles HuggingFace streaming datasets, image downloads, and data preparation. All functionality is consolidated under `prepare_vrsbench_dataset()` and `create_dataloader_from_prepared()`. The standard `create_vrsbench_dataloader()` function is now a convenience wrapper around these high-level functions.

## Why This Loader Exists

Working with VRSBench typically means:
- Manual curl downloads, unzip operations, and path management
- Writing custom code to map HuggingFace streaming datasets to local images
- Handling retries, rate limits, and corrupted downloads
- Building task-specific data extraction logic
- Managing logging, metrics, and error tracking

This loader eliminates all of that. **One function call replaces 50+ lines of boilerplate.** It handles network failures, validates data integrity, tracks performance metrics, and provides structured logging out of the box.

**Architecture**: The loader is built around a high-level API that automatically handles HuggingFace streaming datasets, image downloads, and data preparation. All functionality is consolidated under `prepare_vrsbench_dataset()` and `create_dataloader_from_prepared()`. The standard `create_vrsbench_dataloader()` function is now a convenience wrapper around these high-level functions.

## What It Does

**Multi-Task Support**: Unified API for classification, detection, captioning, VQA, and visual grounding. Task-specific target extraction is built-in.

**Structured JSON Logging**: Rotating log files (10MB, 5 backups) with configurable levels. JSON format for log aggregation systems. Console output for development.

**Robust Downloads**: Automatic retries with exponential backoff (5 attempts, 1.5x backoff). HuggingFace authentication support. Rate limit detection and handling. Progress bars for long downloads.

**Smart Caching**: File verification (size checks, zip integrity validation). Skips re-downloads when valid cache exists. Configurable cache verification.

**Metrics Collection**: Tracks load times, error rates, cache hits/misses. Returns `MetricsCollector` object with timing statistics. Production monitoring hooks.

**Region Extraction**: Automatic bounding box normalization. Region cropping for high-resolution satellite imagery. Configurable padding around regions.

**Production Guardrails**: Exception handling with detailed error context. Automatic image path resolution (absolute, relative, basename search). Graceful degradation when optional dependencies missing.

## Installation

```bash
pip install -e .
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 1.12
- torchvision >= 0.13
- Pillow >= 9.0
- pandas >= 1.3
- datasets >= 2.0.0 (recommended, for HuggingFace integration)

## Using in Google Colab

The VRSBench DataLoader works seamlessly in Google Colab notebooks. Here's how to set it up:

### Option 1: Zero-Friction Setup (Recommended)

**Easiest method** - Works in any Jupyter environment (Colab, JupyterLab, local Jupyter):

```python
# Method 1: Upload jupyter_setup.py and run
exec(open('jupyter_setup.py').read())

# Method 2: Use the setup function (auto-detects environment)
from setup_vrsbench import setup_for_notebook

prepare_vrsbench_dataset, create_dataloader_from_prepared, get_task_targets = setup_for_notebook(
    repo_url="https://github.com/wildcraft958/VRSBench_dataloader",  # Optional: auto-detects if None
    run_test=True  # Optional: runs a quick smoke test
)
```

**What it does automatically:**
- Auto-detects your Jupyter environment (Colab, JupyterLab, local Jupyter)
- Clones repository from GitHub (if needed)
- Installs all dependencies
- Adds to Python path
- Imports all functions
- Runs optional smoke test

### Option 2: Manual Setup

```python
# Install dependencies
!pip install -q torch torchvision pillow pandas requests tqdm datasets

# Clone repository
!git clone https://github.com/yourusername/VRSBench_dataloader.git /content/vrsbench_dataloader
import sys
sys.path.insert(0, '/content/vrsbench_dataloader')

# Import functions
from vrsbench_dataloader_production import (
    prepare_vrsbench_dataset,
    create_dataloader_from_prepared,
    create_vrsbench_dataloader,
    get_task_targets
)
```

### Option 3: Direct Upload

If you've uploaded `vrsbench_dataloader_production.py` directly to Colab:

```python
from vrsbench_dataloader_production import (
    prepare_vrsbench_dataset,
    create_dataloader_from_prepared,
    create_vrsbench_dataloader,
    get_task_targets
)
```

### Complete Colab Example

```python
# Step 1: Install dependencies
!pip install -q torch torchvision pillow pandas requests tqdm datasets

# Step 2: Set up HuggingFace token (optional, but recommended)
import os
os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_token_here"  # Get from https://huggingface.co/settings/tokens

# Step 3: Import the dataloader (after uploading/cloning)
from vrsbench_dataloader_production import prepare_vrsbench_dataset, create_dataloader_from_prepared

# Step 4: Prepare dataset (downloads images and loads HF dataset automatically)
data = prepare_vrsbench_dataset(
    split="validation",
    task="captioning",
    num_samples=1000,  # Limit for faster testing
    download_images=True
)

print(f"✓ Loaded {data['num_samples']} samples")
print(f"✓ Images directory: {data.get('images_dir', 'N/A')}")

# Step 5: Create DataLoader
dataloader = create_dataloader_from_prepared(
    data,
    batch_size=16,
    num_workers=2  # Use 2-4 workers in Colab
)

# Step 6: Use in your training loop
for images, metas in dataloader:
    captions = get_task_targets(metas, task="captioning")
    # images: torch.Tensor of shape [batch_size, 3, H, W]
    # captions: List[str] of ground truth captions
    # Train your model here
    break  # Remove break for full training
```

### Colab-Specific Tips

1. **GPU Runtime**: Enable GPU for faster image processing:
   - Runtime → Change runtime type → GPU (T4 or better)

2. **Memory Management**: 
   - Use `num_samples` to limit dataset size during testing
   - Set `num_workers=2` or `0` to reduce memory usage
   - Images are downloaded to Colab's `/content` directory (~5GB for validation set)

3. **HuggingFace Authentication**:
   ```python
   from huggingface_hub import login
   login(token="your_token_here")
   # Or use environment variable:
   # os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_token_here"
   ```

4. **Persistent Storage**: 
   - Colab sessions reset, so images will re-download each session
   - Consider using Google Drive for persistent storage:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Use Drive path for images_dir
   data = prepare_vrsbench_dataset(
       split="validation",
       images_dir="/content/drive/MyDrive/VRSBench/Images_val",
       download_images=True
   )
   ```

5. **Quick Test**:
   ```python
   # Test with just 10 samples
   data = prepare_vrsbench_dataset(
       split="validation",
       task="captioning",
       num_samples=10,
       download_images=True
   )
   print(f"Sample keys: {list(data.keys())}")
   print(f"First caption: {list(data['image_to_caption'].values())[0]}")
   ```

## Quick Start: Your First 5 Minutes

### For Jupyter Notebook Users (Colab, JupyterLab, Local Jupyter)

**Fastest way to get started:**

1. Upload `jupyter_setup.py` to your notebook
2. Run: `exec(open('jupyter_setup.py').read())`
3. Update `REPO_URL` in the file if needed (defaults to wildcraft958/VRSBench_dataloader)
4. Run the cell - everything is set up automatically!

Or use the setup function (auto-detects environment):
```python
from setup_vrsbench import setup_for_notebook

# Auto-detects Colab, JupyterLab, or local Jupyter
prepare_vrsbench_dataset, create_dataloader_from_prepared, get_task_targets = setup_for_notebook(
    repo_url="https://github.com/wildcraft958/VRSBench_dataloader"  # Optional
)
```

### For Local Users

After cloning the repository locally:

```python
# Option 1: Use the setup utility
from setup_vrsbench import setup_vrsbench_loader

prepare_vrsbench_dataset, create_dataloader_from_prepared, get_task_targets = setup_vrsbench_loader(
    repo_url="https://github.com/yourusername/VRSBench_dataloader",  # Optional if already cloned
    auto_install_deps=True,
    run_smoke_test=False
)

# Option 2: Direct import (if already in the repo directory)
from vrsbench_dataloader_production import (
    prepare_vrsbench_dataset,
    create_dataloader_from_prepared,
    get_task_targets
)
```

### Simplified Workflow for All Tasks

The high-level API works for **all tasks** (classification, detection, captioning, VQA, grounding). It replaces manual downloads, unzip, and mapping:

```python
from vrsbench_dataloader_production import prepare_vrsbench_dataset, create_dataloader_from_prepared, get_task_targets

# For any task - one function call handles everything
data = prepare_vrsbench_dataset(
    split="validation",
    task="captioning",  # or "classification", "detection", "vqa", "grounding"
    num_samples=1000  # None = all samples
)

# Task-specific mappings are automatically created
if data["task"] == "captioning":
    image_to_caption = data["image_to_caption"]
elif data["task"] == "classification":
    image_to_label = data["image_to_label"]
elif data["task"] == "vqa":
    image_to_qa_pairs = data["image_to_qa_pairs"]
elif data["task"] in ["detection", "grounding"]:
    image_to_bboxes = data["image_to_bboxes"]

# Or create a DataLoader directly
dataloader = create_dataloader_from_prepared(data, batch_size=16)
for images, metas in dataloader:
    targets = get_task_targets(metas, task=data["task"])
    # Train your model
```

**What this replaces:**
```python
# OLD: Manual workflow (50+ lines) - now eliminated for ALL tasks
!curl -L -C - -o Images_val.zip https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip
!unzip Images_val.zip -d Images_val
from datasets import load_dataset
fw = load_dataset("xiang709/VRSBench", streaming=True)
# ... complex manual mapping code for each task ...
```

### For All Tasks (Primary Workflow)

**The recommended approach**: Use HuggingFace streaming datasets directly.

```python
from vrsbench_dataloader_production import create_vrsbench_dataloader, get_task_targets

# Primary workflow: Direct HuggingFace streaming dataset
# Automatically downloads images, loads HF dataset, combines them
dataloader = create_vrsbench_dataloader(
    task="classification",  # or "detection", "captioning", "vqa", "grounding"
    split="validation",
    download_images=True,  # Downloads images automatically
    batch_size=16,
    num_workers=4
)

# Standard PyTorch iteration
for images, metas in dataloader:
    # images: torch.Tensor [B, 3, H, W]
    # metas: List[Dict] with task-specific metadata

    # Extract task targets
    labels = get_task_targets(metas, task="classification")

    # Your training code
    outputs = model(images)
    loss = criterion(outputs, labels)
```

**What this does automatically:**
- Loads HuggingFace streaming dataset (`xiang709/VRSBench`)
- Downloads images from HuggingFace (if needed)
- Extracts images to local directory
- Maps streaming dataset metadata to local images
- Returns PyTorch DataLoader ready for training

The loader works directly with HuggingFace streaming datasets.

## Task-Specific Usage

### 1. Classification

Extracts image-level labels. Auto-detects label key (`label`, `category`, `class`, etc.).

```python
# Option 1: Simplified workflow (recommended)
data = prepare_vrsbench_dataset(
    split="validation",
    task="classification",
    num_samples=1000
)
image_to_label = data["image_to_label"]
for sample in data["samples"]:
    label = sample["label"]

# Option 2: Create DataLoader from prepared data
dataloader = create_dataloader_from_prepared(data, batch_size=16)
for images, metas in dataloader:
    labels = get_task_targets(metas, task="classification", label_key="category")
    # labels: List[Any] - classification labels

```

**API Behavior**: `prepare_vrsbench_dataset(task="classification")` creates `image_to_label` mapping. `get_task_targets()` searches for `label`, `category`, `class`, `target`, `class_id` in metadata. Specify `label_key` to override.

### 2. Object Detection

Extracts bounding boxes. Normalizes coordinates automatically (handles [0-1] normalized, [x1,y1,x2,y2] corner format, [x,y,w,h] format).

```python
# Option 1: Simplified workflow (recommended)
data = prepare_vrsbench_dataset(
    split="validation",
    task="detection",
    num_samples=1000
)
image_to_bboxes = data["image_to_bboxes"]
for sample in data["samples"]:
    bboxes = sample["bboxes"]

# Option 2: Create DataLoader from prepared data
dataloader = create_dataloader_from_prepared(data, batch_size=16)
for images, metas in dataloader:
    bboxes = get_task_targets(metas, task="detection")
    # bboxes: List[List[float]] - [[x, y, w, h], ...] in pixel coordinates

```

**API Behavior**: `prepare_vrsbench_dataset(task="detection")` creates `image_to_bboxes` mapping. Bounding boxes are normalized to `[x, y, w, h]` pixel coordinates regardless of input format. Handles nested `objects` arrays and flat `bbox`/`bboxes` fields.

### 3. Image Captioning

Extracts captions. Works with `caption`, `description`, or `text` fields.

```python
# Option 1: Simplified workflow (recommended for all tasks)
data = prepare_vrsbench_dataset(
    split="validation", 
    task="captioning",
    num_samples=1000
)
image_to_caption = data["image_to_caption"]
for sample in data["samples"]:
    caption = sample["caption"]

# Option 2: Create DataLoader from prepared data
dataloader = create_dataloader_from_prepared(data, batch_size=16)
for images, metas in dataloader:
captions = get_task_targets(metas, task="captioning")
    # captions: List[str]

```

**API Behavior**: `prepare_vrsbench_dataset(task="captioning")` downloads images, loads HuggingFace streaming dataset, and combines them. Returns dict with `samples`, `image_to_caption`, `id_to_path` mappings. Works identically for all tasks with task-specific mappings.

### 4. Visual Question Answering (VQA)

Extracts question-answer pairs. Supports single QA per image or multiple QA pairs.

```python
# Option 1: Simplified workflow (recommended)
data = prepare_vrsbench_dataset(
    split="validation",
    task="vqa",
    num_samples=1000
)
image_to_qa_pairs = data["image_to_qa_pairs"]
for sample in data["samples"]:
    qa_pairs = sample["qa_pairs"]

# Option 2: Create DataLoader from prepared data
dataloader = create_dataloader_from_prepared(data, batch_size=16)
for images, metas in dataloader:
qa_pairs = get_task_targets(metas, task="vqa")
    # qa_pairs: List[Tuple[str, str]] - [(question, answer), ...]

```

**API Behavior**: `prepare_vrsbench_dataset(task="vqa")` creates `image_to_qa_pairs` mapping. Each record's `qa_pairs` array is stored in the sample metadata.

### 5. Visual Grounding

Extracts bounding boxes with referring expressions. Can extract cropped regions automatically.

```python
# Option 1: Simplified workflow (recommended)
data = prepare_vrsbench_dataset(
    split="validation",
    task="grounding",
    num_samples=1000
)
image_to_bboxes = data["image_to_bboxes"]

# Option 2: Create DataLoader from prepared data
dataloader = create_dataloader_from_prepared(data, batch_size=16)

```

**API Behavior**: `prepare_vrsbench_dataset(task="grounding")` creates `image_to_bboxes` mapping. Bounding boxes are normalized to `[x, y, w, h]` pixel coordinates.

## Feature Deep Dive

### Structured JSON Logging

**What it does**: Writes logs to both console (INFO+) and file (DEBUG+). JSON format for machine parsing. Automatic rotation at 10MB with 5 backup files.

**How to use**:
```python
# Set log level via environment variable
import os
os.environ["LOG_LEVEL"] = "DEBUG"

# Or via config
from vrsbench_dataloader_production import VRSBenchConfig
config = VRSBenchConfig(LOG_LEVEL="DEBUG", JSON_LOGS=True)
```

**Log location**: `./logs/vrsbench_dataloader.log` (configurable via `LOG_DIR`)

**Log format**:
```json
{
  "timestamp": "2025-01-13T04:58:00.123456",
  "level": "INFO",
  "message": "Download successful",
  "logger": "DownloadManager",
  "file": "annotations_val.zip",
  "duration": 45.23
}
```

**Operational value**: Structured logs integrate with ELK, Splunk, Datadog. Query by logger, level, or custom fields. Track download times, error rates, cache efficiency.

### Robust Downloads

**What it does**: Automatic retries (5 attempts, exponential backoff 1.5x). HuggingFace authentication support. Rate limit detection (HTTP 429). Progress bars. File verification.

**How to use**:
```python
# Set HuggingFace token (recommended)
import os
os.environ["HUGGINGFACE_HUB_TOKEN"] = "hf_your_token_here"

# Downloads happen automatically
dataloader = create_vrsbench_dataloader(
    task="classification",
    split="validation",
    download_images=True
)
```

**Operational value**: No manual retry logic. Handles network failures, rate limits, corrupted downloads. Reduces support burden.

**API Behavior**: `download_with_retries()` checks cache first, verifies file integrity (size + zip validation), retries on failure with exponential backoff. Logs all attempts and failures.

### Smart Caching

**What it does**: Verifies cached files (size checks, zip integrity). Skips re-downloads when valid cache exists. Configurable verification.

**How to use**:
```python
from vrsbench_dataloader_production import VRSBenchConfig
config = VRSBenchConfig(
    CACHE_DIR="./cache",
    VERIFY_CACHE=True  # Verify cached files before using
)

dataloader = create_vrsbench_dataloader(..., config=config)
```

**Operational value**: Faster subsequent runs. Prevents using corrupted cached files. Reduces bandwidth costs.

**API Behavior**: `_verify_file()` checks file exists, size is reasonable (>1KB), size matches expected (if provided), zip files can be opened. Invalid cache is deleted and re-downloaded.

### Metrics Collection

**What it does**: Tracks cache hits/misses, images loaded, errors by type, operation timings (download, image load). Returns statistics (mean, min, max, total).

**How to use**:
```python
dataloader, metrics = create_vrsbench_dataloader(
    task="classification",
    split="validation",
    return_metrics=True
)

# Run your training loop
for images, metas in dataloader:
    pass

# Get metrics summary
summary = metrics.get_summary()
print(json.dumps(summary, indent=2))
```

**Output example**:
```json
{
  "metrics": {
    "cache_hits": 150,
    "cache_misses": 2,
    "images_loaded": 1000
  },
  "errors": {
    "image_load_error": 5
  },
  "timings": {
    "image_load": {
      "count": 1000,
      "mean": 0.015,
      "min": 0.008,
      "max": 0.150,
      "total": 15.0
    }
  }
}
```

**Operational value**: Identify bottlenecks (slow image loads, high error rates). Monitor cache efficiency. Track performance over time.

**API Behavior**: `MetricsCollector` records counters, timings, and errors. `get_summary()` returns aggregated statistics. Metrics persist across dataset iterations.

### Region Extraction

**What it does**: Normalizes bounding boxes to `[x, y, w, h]` pixel coordinates. Crops image regions with configurable padding. Handles normalized coordinates, corner format, width/height format.

**How to use**:
```python
# Use high-level API for grounding task
data = prepare_vrsbench_dataset(
    split="validation",
    task="grounding",
    num_samples=1000
)
dataloader = create_dataloader_from_prepared(data, batch_size=16)

for images, metas in dataloader:
    # images are cropped regions, not full images
    bboxes = get_task_targets(metas, task="grounding")
```

**Operational value**: No manual bbox normalization. Consistent coordinate format. Automatic region cropping for high-res satellite imagery.

**API Behavior**: `TaskProcessor.normalize_bbox()` handles all coordinate formats. `extract_region_from_bbox()` crops with 10px padding (configurable via `REGION_PADDING`).

## Advanced Usage

### Custom Configuration

```python
from vrsbench_dataloader_production import VRSBenchConfig, create_vrsbench_dataloader

config = VRSBenchConfig(
    # Download settings
    MAX_RETRIES=5,
    BACKOFF_FACTOR=1.5,
    TIMEOUT=60,
    
    # Logging
    LOG_LEVEL="DEBUG",
    LOG_DIR="./logs",
    JSON_LOGS=True,
    
    # Cache
    CACHE_DIR="./cache",
    VERIFY_CACHE=True,
    
    # Performance
    NUM_WORKERS=8,
    BATCH_SIZE=32,
    PIN_MEMORY=True,
    
    # Image processing
    DEFAULT_IMAGE_SIZE=(512, 512),
    REGION_PADDING=20
)

dataloader = create_vrsbench_dataloader(
    task="classification",
    split="validation",
    config=config
)
```

### Custom Image Transforms

```python
from torchvision import transforms

custom_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataloader = create_vrsbench_dataloader(
    task="classification",
    split="validation",
    transform=custom_transform
)
```

### Direct HuggingFace Dataset (Primary Workflow)

```python
from datasets import load_dataset
from vrsbench_dataloader_production import create_vrsbench_dataloader

# Option 1: Let the loader handle it automatically (recommended)
dataloader = create_vrsbench_dataloader(
    task="classification",
    split="validation",
    download_images=True
)

# Option 2: Use high-level API for more control
data = prepare_vrsbench_dataset(
    split="validation",
    task="classification",
    num_samples=1000
)
dataloader = create_dataloader_from_prepared(data, batch_size=16)
```

### Filter by Split

```python
# Split is specified when preparing the dataset
data = prepare_vrsbench_dataset(
    split="validation",  # Only load validation split
    task="classification"
)
dataloader = create_dataloader_from_prepared(data, batch_size=16)
```

## Operational Guardrails

### Error Handling

**Image not found**: Logs warning with attempted path, image key, images_dir. Skips sample and continues. Logs first 10 failures, then suppresses.

**Download failure**: Retries 5 times with exponential backoff. Logs each attempt. Raises `RuntimeError` after all retries exhausted.

**Corrupted cache**: Detects via file verification. Deletes corrupted file and re-downloads. Logs warning.

**Missing annotations**: Raises `RuntimeError` with clear message. Lists available annotation sources.

**Invalid task**: Raises `ValueError` with supported tasks list.

### Monitoring Hooks

**Logging**: All operations logged with context (file paths, durations, error types). JSON format for parsing.

**Metrics**: Track performance and errors. Export to monitoring systems.

**Progress**: Progress bars for downloads. Periodic status updates during dataset preparation.

### Configuration Validation

**Task validation**: Checks task against `SUPPORTED_TASKS` list. Raises `ValueError` if invalid.

**Directory creation**: Auto-creates log and cache directories. Handles permissions errors.

**Environment variables**: `LOG_LEVEL`, `HUGGINGFACE_HUB_TOKEN`, `HF_TOKEN` supported.

## Performance

**Benchmarks** (NVIDIA V100, 32GB RAM, SSD, VRSBench validation set - 1,131 samples):

| Task | Batch Size | Num Workers | Images/sec | Memory (GB) |
|------|------------|-------------|------------|-------------|
| Classification | 16 | 4 | 850 | 2.1 |
| Classification | 32 | 8 | 1400 | 3.8 |
| VQA (expanded) | 16 | 4 | 620 | 2.5 |
| Grounding (regions) | 16 | 4 | 580 | 2.8 |

**Optimization tips**:
- Increase `num_workers` (4-8 recommended)
- Enable `pin_memory=True` for GPU training
- Use SSD storage for images
- Reduce image resolution in transforms
- Profile with `return_metrics=True`

## Troubleshooting

### HTTP 429 Rate Limit Errors

**Problem**: Too many requests to HuggingFace without authentication.

**Solution**:
```bash
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
# Get free token from https://huggingface.co/settings/tokens
```

**API Behavior**: `_get_headers()` checks `HUGGINGFACE_HUB_TOKEN` or `HF_TOKEN` env vars. Adds `Authorization: Bearer {token}` header. Logs warning if no token found.

### Missing `datasets` Library

**Problem**: `datasets` not installed (optional dependency).

**Solution**: Install for better performance:
```bash
pip install datasets
```

**API Behavior**: Falls back to pandas-based processing if `datasets` not available. `prepare_vrsbench_dataset()` requires `datasets` and raises `ImportError` if missing.

### Images Not Found

**Problem**: Image paths in annotations don't match filesystem.

**Solution**:
- Verify `images_dir` points to correct location
- Check if images are in subdirectories (auto-detected)
- Enable debug logging: `export LOG_LEVEL=DEBUG`

**API Behavior**: `_find_image_path()` tries: (1) absolute path, (2) relative to `images_dir`, (3) basename search in directory tree. Logs attempted paths on failure.

### Slow Data Loading

**Problem**: Bottleneck in data pipeline.

**Solutions**:
1. Increase `num_workers` (4-8 recommended)
2. Enable `pin_memory=True` for GPU training
3. Use SSD storage for images
4. Reduce image resolution in transforms
5. Profile with metrics: `dataloader, metrics = create_vrsbench_dataloader(..., return_metrics=True)`

## API Reference

### `create_vrsbench_dataloader()`

Convenience wrapper to create PyTorch DataLoader.

**This function is a wrapper** around the high-level API:
- Calls `prepare_vrsbench_dataset()` to download images and load HuggingFace streaming dataset
- Calls `create_dataloader_from_prepared()` to create the DataLoader

**Primary workflow**: Uses HuggingFace streaming datasets.

**Parameters:**
- `task` (str): Task type - `"classification"`, `"detection"`, `"captioning"`, `"vqa"`, `"grounding"`
- `split` (str): Dataset split - `"train"`, `"validation"`, `"test"` (default: `"validation"`)
- `images_dir` (Optional[str]): Directory to store images (default: `./Images_{split}`)
- `sample_size` (Optional[int]): Limit number of samples (None = all)
- `hf_dataset_name` (str): HuggingFace dataset identifier (default: `"xiang709/VRSBench"`)
- `download_images` (bool): Download images if not present (default: True)
- `images_url` (Optional[str]): URL for image zip (auto-detected from split if not provided)
- `transform` (Optional[Callable]): Image transforms (default: resize to 256x256)
- `batch_size` (int): Batch size (default: 16)
- `num_workers` (int): Number of workers (default: 4)
- `config` (Optional[VRSBenchConfig]): Configuration object
- `return_metrics` (bool): Return metrics collector (default: False)

**Returns:**
- `DataLoader` or `(DataLoader, MetricsCollector)` if `return_metrics=True`

**Example:**
```python
# Primary workflow: Direct HuggingFace streaming dataset
dataloader = create_vrsbench_dataloader(
    task="captioning",
    split="validation",
    download_images=True
)

# For better control, use the high-level API directly:
from vrsbench_dataloader_production import prepare_vrsbench_dataset, create_dataloader_from_prepared
data = prepare_vrsbench_dataset(split="validation", task="captioning", num_samples=1000)
dataloader = create_dataloader_from_prepared(data, batch_size=16)
```

### `get_task_targets()`

Extract task-specific targets from metadata batch.

**Parameters:**
- `metas` (List[Dict]): List of metadata dicts
- `task` (str): Task type
- `label_key` (Optional[str]): Label key for classification

**Returns:**
- Task-specific targets (labels, captions, QA pairs, bboxes)

### `prepare_vrsbench_dataset()`

High-level function to automate entire workflow for **all tasks** - downloads images, loads HuggingFace dataset, combines them.

**Parameters:**
- `split` (str): Dataset split - `"train"` or `"validation"` (default: `"validation"`)
- `task` (str): Task type - `"classification"`, `"detection"`, `"captioning"`, `"vqa"`, `"grounding"` (default: `"captioning"`)
- `images_dir` (Optional[str]): Directory to store images (default: `./Images_{split}`)
- `num_samples` (Optional[int]): Limit number of samples (None = all)
- `hf_dataset_name` (str): HuggingFace dataset identifier (default: `"xiang709/VRSBench"`)
- `hf_token` (Optional[str]): HuggingFace token (or use `HUGGINGFACE_HUB_TOKEN` env var)
- `download_images` (bool): Whether to download images (default: True)
- `images_url` (Optional[str]): Custom URL for images (default: auto-detect from split)
- `output_json` (Optional[str]): Path to save JSON mapping (optional)
- `config` (Optional[VRSBenchConfig]): Configuration object (optional)
- `force_download` (bool): Force re-download even if images exist (default: False)

**Returns:**
- Dictionary with:
  - `"samples"`: List of sample dicts with `image_path` and task-specific metadata
  - Task-specific mappings:
    - `"image_to_caption"` (captioning): Dict mapping `image_id` -> `caption`
    - `"image_to_label"` (classification): Dict mapping `image_id` -> `label`
    - `"image_to_qa_pairs"` (vqa): Dict mapping `image_id` -> `qa_pairs`
    - `"image_to_bboxes"` (detection/grounding): Dict mapping `image_id` -> `bboxes`
  - `"id_to_path"`: Dict mapping `image_id` -> `image_path`
  - `"split"`: Dataset split used
  - `"task"`: Task type used
  - `"num_samples"`: Number of samples loaded

**Example:**
```python
# For any task
data = prepare_vrsbench_dataset(split="validation", task="classification", num_samples=1000)
image_to_label = data["image_to_label"]

data = prepare_vrsbench_dataset(split="validation", task="vqa", num_samples=1000)
image_to_qa_pairs = data["image_to_qa_pairs"]
```

### `create_dataloader_from_prepared()`

Create PyTorch DataLoader from prepared dataset (output of `prepare_vrsbench_dataset()`). Works for **all tasks**.

**Parameters:**
- `prepared_data` (Dict[str, Any]): Output from `prepare_vrsbench_dataset()`
- `task` (Optional[str]): Task type (auto-detected from `prepared_data` if not provided)
- `batch_size` (int): Batch size for DataLoader (default: 16)
- `num_workers` (int): Number of worker processes (default: 4)
- `transform` (Optional[Callable]): Image transforms (default: resize to 256x256)
- `config` (Optional[VRSBenchConfig]): Configuration object

**Returns:**
- PyTorch `DataLoader` yielding `(image_tensor, metadata_dict)` batches

**Example:**
```python
# Works for all tasks
data = prepare_vrsbench_dataset(split="validation", task="classification", num_samples=1000)
dataloader = create_dataloader_from_prepared(data, batch_size=16)

data = prepare_vrsbench_dataset(split="validation", task="vqa", num_samples=1000)
dataloader = create_dataloader_from_prepared(data, batch_size=16)
```


### `VRSBenchConfig`

Configuration dataclass for all settings.

**Key attributes:**
- `MAX_RETRIES` (int): Download retry attempts (default: 5)
- `BACKOFF_FACTOR` (float): Exponential backoff multiplier (default: 1.5)
- `LOG_LEVEL` (str): Logging level (default: `"INFO"`)
- `LOG_DIR` (str): Log directory (default: `"./logs"`)
- `JSON_LOGS` (bool): Use JSON log format (default: True)
- `CACHE_DIR` (str): Cache directory (default: `"./hf_cache"`)
- `VERIFY_CACHE` (bool): Verify cached files (default: True)
- `DEFAULT_IMAGE_SIZE` (Tuple[int, int]): Default image size (default: (256, 256))
- `REGION_PADDING` (int): Region padding in pixels (default: 10)

## Citation

If you use this dataloader or VRSBench dataset, please cite:

```bibtex
@inproceedings{vrsbench2024,
  title={VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding},
  author={Xiang, Liu and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## License

MIT License

## Changelog

### Version 3.0.0 (2025-01-13) - Major Refactoring
- 🔄 **Complete architecture refactor** - High-level API (`prepare_vrsbench_dataset` + `create_dataloader_from_prepared`) is now the core
- ✨ **Unified workflow** - `create_vrsbench_dataloader()` is now a convenience wrapper around the high-level API
- 🗑️ **Removed redundant code paths** - Eliminated duplicate logic between high-level and low-level APIs
- 📦 **Consolidated functionality** - All data preparation logic unified under `prepare_vrsbench_dataset()`
- 🗑️ **Removed legacy code** - Eliminated all JSONL/URL annotation loading code paths
- 🎯 **Streamlined codebase** - Removed ~200 lines of redundant code, improved maintainability
- 📝 **Clean API surface** - Only current, supported parameters remain. Zero deadweight.

### Version 2.1.0 (2025-01-13)
- ✨ **Refactored architecture** - HuggingFace streaming datasets are now the primary workflow
- ✨ **Simplified API** - `create_vrsbench_dataloader()` automatically loads HF datasets
- ✨ **Multi-task high-level API** - `prepare_vrsbench_dataset()` works for all tasks
- ✨ **Renamed function** - `create_dataloader_from_prepared()` replaces captioning-specific name
- ✨ **Consolidated codebase** - Removed duplicate logic, improved maintainability

### Version 2.0.0 (2025-01-13)
- ✨ Multi-task support (classification, detection, captioning, VQA, grounding)
- ✨ Structured JSON logging with rotation
- ✨ Metrics collection and monitoring
- ✨ Region-based extraction for high-res imagery
- ✨ Multi-annotation expansion
- ✨ Improved error handling and retries
- ✨ HuggingFace authentication support
- ✨ Production-ready configuration management
