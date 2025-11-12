# Example Log Outputs

This document shows example log outputs from the VRSBench DataLoader in different scenarios.

## Console Output (Human-Readable)

### Successful DataLoader Creation

```
2025-01-13 04:58:15 - DataLoaderFactory - INFO - Creating DataLoader for task=classification, split=validation, batch_size=16
2025-01-13 04:58:15 - DownloadManager - DEBUG - Using HuggingFace authentication token
2025-01-13 04:58:15 - DownloadManager - INFO - Using cached file: ./hf_cache/Annotations_val.zip
2025-01-13 04:58:16 - AnnotationProcessor - INFO - Parsing ./hf_cache/Annotations_val.jsonl with safe coercion...
2025-01-13 04:58:17 - AnnotationProcessor - INFO - Loaded 1131 annotation records
2025-01-13 04:58:17 - AnnotationProcessor - INFO - Creating HuggingFace Dataset
2025-01-13 04:58:18 - VRSBenchDataset - INFO - Initializing VRSBench dataset for task: classification
2025-01-13 04:58:18 - DataLoaderFactory - INFO - DataLoader created successfully
```

### Training Loop Output

```
Batch 1:
  Images: torch.Size([16, 3, 256, 256])
  Metadata count: 16
  Targets: [0, 1, 2, 1, 0, 3, 2, 1, 0, 0, 1, 2, 3, 1, 0, 2]

Batch 2:
  Images: torch.Size([16, 3, 256, 256])
  Metadata count: 16
  Targets: [1, 1, 2, 0, 3, 1, 2, 2, 0, 1, 0, 0, 3, 2, 1, 1]

Batch 3:
  Images: torch.Size([16, 3, 256, 256])
  Metadata count: 16
  Targets: [2, 3, 1, 0, 1, 2, 0, 1, 3, 2, 1, 0, 0, 1, 2, 3]
```

## JSON Log File (Machine-Readable)

### Download Success

```json
{
  "timestamp": "2025-01-13T04:58:15.234567",
  "level": "INFO",
  "message": "Download successful",
  "logger": "DownloadManager",
  "file": "Annotations_val.zip",
  "size_bytes": 15728640,
  "duration": 45.23,
  "cache_hit": false
}
```

### Annotation Loading

```json
{
  "timestamp": "2025-01-13T04:58:17.123456",
  "level": "INFO",
  "message": "Loaded 1131 annotation records",
  "logger": "AnnotationProcessor",
  "jsonl_path": "./hf_cache/Annotations_val.jsonl",
  "records_count": 1131,
  "skipped_lines": 0
}
```

### Dataset Iteration

```json
{
  "timestamp": "2025-01-13T04:58:45.678901",
  "level": "INFO",
  "message": "Dataset iteration complete",
  "logger": "VRSBenchDataset",
  "emitted": 1131,
  "skipped": 12,
  "duration": 27.5
}
```

### Image Load Performance

```json
{
  "timestamp": "2025-01-13T04:58:20.456789",
  "level": "DEBUG",
  "message": "Image loaded",
  "logger": "VRSBenchDataset",
  "image_path": "./data/images/scene_0045.jpg",
  "load_time_ms": 12.3,
  "size": [1024, 1024],
  "format": "JPEG"
}
```

## Warning Logs

### Missing Image

```json
{
  "timestamp": "2025-01-13T04:58:22.123456",
  "level": "WARNING",
  "message": "Image not found",
  "logger": "VRSBenchDataset",
  "image_ref": "missing_image.jpg",
  "annotation_id": 247
}
```

### No HF Token

```json
{
  "timestamp": "2025-01-13T04:58:15.000000",
  "level": "WARNING",
  "message": "No HF token found. Downloads may be rate-limited. Set HUGGINGFACE_HUB_TOKEN env var.",
  "logger": "DownloadManager"
}
```

### Malformed JSON

```json
{
  "timestamp": "2025-01-13T04:58:16.789012",
  "level": "WARNING",
  "message": "Skipped malformed JSON at line 342",
  "logger": "AnnotationProcessor",
  "jsonl_path": "./hf_cache/Annotations_val.jsonl",
  "line_number": 342,
  "error": "Expecting ',' delimiter"
}
```

## Error Logs

### Download Failure

```json
{
  "timestamp": "2025-01-13T04:58:30.123456",
  "level": "ERROR",
  "message": "HTTP 429 Too Many Requests. Set HUGGINGFACE_HUB_TOKEN env var. Retrying in 1.5s...",
  "logger": "DownloadManager",
  "url": "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/images.zip",
  "attempt": 2,
  "max_retries": 5,
  "backoff_seconds": 1.5
}
```

### Corrupted File

```json
{
  "timestamp": "2025-01-13T04:58:35.567890",
  "level": "ERROR",
  "message": "Corrupted zip file",
  "logger": "DownloadManager",
  "file_path": "./hf_cache/images.zip",
  "file_size": 1024,
  "expected_min_size": 1048576
}
```

### Image Load Failure

```json
{
  "timestamp": "2025-01-13T04:58:40.234567",
  "level": "ERROR",
  "message": "Failed to load image",
  "logger": "VRSBenchDataset",
  "image_path": "./data/images/corrupted.jpg",
  "error": "PIL.UnidentifiedImageError: cannot identify image file",
  "annotation_id": 523
}
```

## Metrics Summary Output

### Complete Metrics Report

```json
{
  "metrics": {
    "cache_hits": 2,
    "cache_misses": 1,
    "images_loaded": 1119,
    "annotations_loaded": 1131,
    "annotations_extracted": 1
  },
  "errors": {
    "image_load_error": 12,
    "jsonl_parse_errors": 0
  },
  "timings": {
    "download": {
      "count": 1,
      "mean": 45.23,
      "min": 45.23,
      "max": 45.23,
      "total": 45.23
    },
    "image_load": {
      "count": 1119,
      "mean": 0.0142,
      "min": 0.0085,
      "max": 0.156,
      "total": 15.89
    }
  }
}
```

## Debug Logs (LOG_LEVEL=DEBUG)

### Detailed Image Processing

```json
{
  "timestamp": "2025-01-13T04:58:20.123456",
  "level": "DEBUG",
  "message": "Processing annotation item",
  "logger": "VRSBenchDataset",
  "item_keys": ["id", "image", "label", "caption", "bbox"],
  "image_key": "image",
  "split": "validation"
}
```

### Cache Verification

```json
{
  "timestamp": "2025-01-13T04:58:15.456789",
  "level": "DEBUG",
  "message": "Verifying cached file",
  "logger": "DownloadManager",
  "file_path": "./hf_cache/Annotations_val.zip",
  "size_bytes": 15728640,
  "is_valid_zip": true,
  "verification_passed": true
}
```

### Transform Application

```json
{
  "timestamp": "2025-01-13T04:58:20.567890",
  "level": "DEBUG",
  "message": "Applying transforms",
  "logger": "VRSBenchDataset",
  "original_size": [1024, 768],
  "target_size": [256, 256],
  "transform_steps": ["Resize", "ToTensor", "Normalize"]
}
```

## Production Monitoring Logs

### Hourly Performance Summary

```json
{
  "timestamp": "2025-01-13T05:00:00.000000",
  "level": "INFO",
  "message": "Hourly performance summary",
  "logger": "MetricsCollector",
  "period": "2025-01-13 04:00-05:00",
  "total_batches": 2400,
  "total_images": 38400,
  "avg_throughput_images_per_sec": 850.5,
  "error_rate_percent": 0.3,
  "cache_hit_rate_percent": 98.5
}
```

### Resource Usage

```json
{
  "timestamp": "2025-01-13T04:58:30.000000",
  "level": "INFO",
  "message": "Resource usage snapshot",
  "logger": "VRSBenchDataset",
  "cpu_percent": 45.2,
  "memory_used_gb": 4.3,
  "memory_percent": 13.4,
  "active_workers": 8,
  "queue_size": 32
}
```

## Error Recovery Logs

### Successful Retry

```json
{
  "timestamp": "2025-01-13T04:58:18.123456",
  "level": "INFO",
  "message": "Download successful after retry",
  "logger": "DownloadManager",
  "url": "https://huggingface.co/.../images.zip",
  "attempts_needed": 3,
  "total_duration": 92.5,
  "final_attempt_duration": 48.7
}
```

### Fallback to Local Cache

```json
{
  "timestamp": "2025-01-13T04:58:25.789012",
  "level": "WARNING",
  "message": "Download failed, using local cache",
  "logger": "DownloadManager",
  "url": "https://huggingface.co/.../annotations.zip",
  "fallback_path": "./hf_cache/Annotations_val.zip",
  "cache_age_hours": 12.5
}
```

## VQA Task-Specific Logs

### Multi-Annotation Expansion

```json
{
  "timestamp": "2025-01-13T04:58:25.123456",
  "level": "DEBUG",
  "message": "Expanding multi-annotation item",
  "logger": "VRSBenchDataset",
  "image_id": 1234,
  "qa_pairs_count": 5,
  "expanded_samples": 5
}
```

### QA Pair Processing

```json
{
  "timestamp": "2025-01-13T04:58:26.234567",
  "level": "DEBUG",
  "message": "Processing VQA sample",
  "logger": "TaskProcessor",
  "question": "What type of building is shown?",
  "answer": "Commercial building",
  "question_length": 29,
  "answer_length": 18
}
```

## Grounding Task-Specific Logs

### Region Extraction

```json
{
  "timestamp": "2025-01-13T04:58:28.345678",
  "level": "DEBUG",
  "message": "Extracting region from bbox",
  "logger": "TaskProcessor",
  "original_image_size": [2048, 1536],
  "bbox": [512, 384, 256, 192],
  "padding": 10,
  "extracted_region_size": [276, 212]
}
```

### Object Expansion

```json
{
  "timestamp": "2025-01-13T04:58:27.456789",
  "level": "DEBUG",
  "message": "Expanding objects for grounding",
  "logger": "VRSBenchDataset",
  "image_id": 5678,
  "objects_count": 8,
  "expanded_samples": 8
}
```

## Log Analysis Examples

### Count Errors by Type

```bash
cat logs/vrsbench_dataloader.log | jq -r 'select(.level=="ERROR") | .error' | sort | uniq -c
```

Output:
```
  12 image_load_error
   2 download_error
   1 bad_zip
```

### Average Load Time

```bash
cat logs/vrsbench_dataloader.log | jq -r 'select(.load_time_ms) | .load_time_ms' | awk '{sum+=$1; count+=1} END {print "Avg:", sum/count, "ms"}'
```

Output:
```
Avg: 14.2 ms
```

### Timeline of Events

```bash
cat logs/vrsbench_dataloader.log | jq -r '[.timestamp, .level, .message] | @tsv'
```

Output:
```
2025-01-13T04:58:15.234567  INFO    Creating DataLoader
2025-01-13T04:58:16.345678  INFO    Loaded 1131 annotation records
2025-01-13T04:58:17.456789  INFO    DataLoader created successfully
```

## Real-Time Monitoring

### Watch for Errors

```bash
tail -f logs/vrsbench_dataloader.log | jq -r 'select(.level=="ERROR" or .level=="WARNING") | "[\(.timestamp)] \(.level): \(.message)"'
```

### Monitor Throughput

```bash
tail -f logs/vrsbench_dataloader.log | jq -r 'select(.message=="Dataset iteration complete") | "Processed \(.emitted) images in \(.duration)s = \((.emitted / .duration) | floor) img/s"'
```

---

These logs demonstrate the comprehensive observability provided by the production dataloader, enabling effective monitoring, debugging, and optimization.
