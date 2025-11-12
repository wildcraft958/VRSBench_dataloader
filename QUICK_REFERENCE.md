# Quick Reference Card

## Installation

```bash
pip install -r requirements.txt
export HUGGINGFACE_HUB_TOKEN="hf_your_token"  # Optional but recommended
```

## Basic Usage

### Classification
```python
from vrsbench_dataloader_production import create_vrsbench_dataloader, get_task_targets

dataloader = create_vrsbench_dataloader(
    images_dir="./data/images",
    task="classification",
    annotations_jsonl="./data/annotations.jsonl",
    batch_size=16
)

for images, metas in dataloader:
    labels = get_task_targets(metas, task="classification")
    # Train your model...
```

### VQA
```python
dataloader = create_vrsbench_dataloader(
    images_dir="./data/images",
    task="vqa",
    annotations_jsonl="./data/vqa.jsonl",
    expand_multi_annotations=True
)

for images, metas in dataloader:
    qa_pairs = get_task_targets(metas, task="vqa")
    # qa_pairs: [(question, answer), ...]
```

### Grounding with Regions
```python
dataloader = create_vrsbench_dataloader(
    images_dir="./data/images",
    task="grounding",
    annotations_jsonl="./data/grounding.jsonl",
    region_based=True,
    expand_multi_annotations=True
)

for images, metas in dataloader:
    bboxes = get_task_targets(metas, task="grounding")
    # images are cropped to regions
```

## Configuration

```python
from vrsbench_dataloader_production import VRSBenchConfig

config = VRSBenchConfig()
config.LOG_LEVEL = "INFO"
config.NUM_WORKERS = 8
config.BATCH_SIZE = 32

dataloader = create_vrsbench_dataloader(..., config=config)
```

## Metrics

```python
dataloader, metrics = create_vrsbench_dataloader(
    ..., 
    return_metrics=True
)

# After training
summary = metrics.get_summary()
```

## Supported Tasks

| Task | Description | Target Format |
|------|-------------|---------------|
| `classification` | Image classification | `List[int]` |
| `detection` | Object detection | `List[List[float]]` |
| `captioning` | Image captioning | `List[str]` |
| `vqa` | Visual QA | `List[Tuple[str, str]]` |
| `grounding` | Visual grounding | `List[List[float]]` |

## Common Parameters

```python
create_vrsbench_dataloader(
    images_dir="./data/images",           # Required
    task="classification",                # Required
    annotations_jsonl="path.jsonl",       # Annotation source
    batch_size=16,                        # Batch size
    num_workers=4,                        # Worker processes
    split="validation",                   # Data split
    transform=custom_transform,           # Image transforms
    sample_size=1000,                     # Limit samples
    expand_multi_annotations=False,       # Expand multi-annotations
    region_based=False,                   # Extract regions
    download_images=False,                # Auto-download
    return_metrics=False                  # Return metrics
)
```

## Environment Variables

```bash
export HUGGINGFACE_HUB_TOKEN="hf_..."  # HF authentication
export LOG_LEVEL="INFO"                 # Logging level
```

## Troubleshooting

### Rate Limit (HTTP 429)
```bash
export HUGGINGFACE_HUB_TOKEN="hf_your_token"
```

### Slow Loading
```python
config.NUM_WORKERS = 8
config.VERIFY_CACHE = False
```

### High Memory
```python
batch_size=8
num_workers=2
transform=transforms.Resize((128, 128))
```

## CLI Testing

```bash
python vrsbench_dataloader_production.py \
    --images-dir ./data/images \
    --annotations-jsonl ./data/annotations.jsonl \
    --task classification \
    --batch-size 4 \
    --sample-size 20
```

## Logging

```bash
# View logs
tail -f logs/vrsbench_dataloader.log | jq .

# Filter errors
cat logs/vrsbench_dataloader.log | jq 'select(.level=="ERROR")'
```

## File Structure

```
├── vrsbench_dataloader_production.py  # Main module
├── requirements.txt                   # Dependencies
├── README.md                          # Documentation
├── CONFIGURATION.md                   # Config guide
├── example_classification.py          # Example scripts
├── example_vqa.py
├── example_grounding.py
├── setup.sh                          # Setup script
└── logs/                             # Log files
    └── vrsbench_dataloader.log
```

## Performance Tips

1. **GPU Training:** `num_workers = 4 * num_gpus`, `pin_memory=True`
2. **CPU-Only:** `num_workers = 4-8`
3. **Debugging:** `num_workers = 0`, `LOG_LEVEL="DEBUG"`
4. **Production:** `LOG_LEVEL="WARNING"`, `JSON_LOGS=True`
5. **SSD Storage:** Store images on SSD for faster I/O
6. **Batch Size:** Larger for GPU (32-64), smaller for CPU (8-16)

## Links

- **Dataset:** https://huggingface.co/datasets/xiang709/VRSBench
- **Paper:** https://arxiv.org/abs/2406.12384
- **GitHub:** https://github.com/lx709/VRSBench
- **HF Token:** https://huggingface.co/settings/tokens
