from __future__ import annotations

"""
vrsbench_dataloader_production.py

Production-ready VRSBench DataLoader with comprehensive logging, multi-task support,
and robust error handling for satellite imagery tasks (classification, detection, 
captioning, VQA, grounding).

Author: Animesh Raj
Date: 2025-01-13
Version: 3.0.0

Features:
- Structured JSON logging with rotation
- Multi-task support (classification, detection, captioning, VQA, grounding)
- Region-level extraction for high-res satellite imagery
- Automatic retry with exponential backoff
- Metrics collection (load times, error rates, cache hits)
- HuggingFace streaming dataset integration (primary workflow)
- JSONL file support (fallback for custom datasets)
- Production-ready error handling
"""

from typing import Optional, Callable, Iterator, Dict, Any, Tuple, List, Union
import os
import sys
import time
import json
import zipfile
import io
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import requests
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from multiprocessing import Manager
import multiprocessing as mp

# Optional dependencies - HuggingFace datasets library
# This library provides better performance for large datasets and seamless integration
# with HuggingFace Hub. Required for prepare_vrsbench_dataset() functions.
try:
    from datasets import load_dataset, Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    load_dataset = None
    Dataset = None

# Optional dependencies - Parquet support for faster I/O
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    from pyarrow.lib import ArrowInvalid
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    pq = None
    pa = None
    # Create a dummy ArrowInvalid class for error handling
    class ArrowInvalid(Exception):
        pass

# ========================= Configuration =========================

@dataclass
class VRSBenchConfig:
    """
    Central configuration for VRSBench dataloader.
    
    This dataclass holds all configuration parameters in one place, making it easy to
    customize behavior without modifying code. All settings have sensible defaults but
    can be overridden via environment variables or direct instantiation.
    """

    # Dataset URLs (direct HF access)
    # These are the official HuggingFace dataset URLs for VRSBench
    # Using /resolve/main/ ensures we get the actual files, not the git repository
    IMAGES_URL: str = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/images.zip"
    ANNOTATIONS_TRAIN_URL: str = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Annotations_train.zip"
    ANNOTATIONS_VAL_URL: str = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Annotations_val.zip"

    # Download settings
    MAX_RETRIES: int = 5  # Number of retry attempts for failed downloads
    BACKOFF_FACTOR: float = 1.5  # Exponential backoff multiplier (1.5x delay each retry)
    CHUNK_SIZE: int = 16384  # 16KB chunks - balance between memory and network efficiency
    TIMEOUT: int = 60  # Request timeout in seconds

    # Logging settings
    # LOG_LEVEL can be set via environment variable for runtime control
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = "./logs"  # Directory for log files
    LOG_FILE: str = "vrsbench_dataloader.log"  # Main log file name
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB - rotate logs when this size is reached
    LOG_BACKUP_COUNT: int = 5  # Keep 5 backup log files (total ~50MB of logs)
    JSON_LOGS: bool = True  # Use JSON format for machine-readable logs (better for parsing/monitoring)

    # Cache settings
    CACHE_DIR: str = "./hf_cache"  # Directory for caching downloaded files
    VERIFY_CACHE: bool = True  # Verify cached files are valid before using (prevents corruption issues)

    # Image processing
    DEFAULT_IMAGE_SIZE: Tuple[int, int] = (256, 256)  # Default resize size for images
    REGION_PADDING: int = 10  # Pixels to pad around bounding boxes when extracting regions

    # Performance settings
    NUM_WORKERS: int = 4  # Number of parallel workers for data loading (0 = main process only)
    BATCH_SIZE: int = 16  # Default batch size
    PIN_MEMORY: bool = True  # Pin memory for faster GPU transfer (only useful with GPU)
    PREFETCH_FACTOR: int = 2  # Number of batches to prefetch per worker (reduces idle time)

    # Checkpointing / fault-tolerance
    CHECKPOINT_DIR: str = "./checkpoints"
    CHECKPOINT_EVERY_SAMPLES: int = 1000  # Persist progress after every N processed samples

    # HuggingFace streaming fault tolerance
    HF_MAX_CONSECUTIVE_PARSE_ERRORS: int = 5

    # Data sourcing preferences
    PREFER_LOCAL_ANNOTATIONS: bool = os.getenv(
        "VRSBENCH_PREFER_LOCAL_ANNOTATIONS", "1"
    ).strip().lower() not in {"0", "false", "no"}

    # Task-specific settings
    SUPPORTED_TASKS: List[str] = None  # Will be set in __post_init__

    def __post_init__(self):
        """
        Initialize default values and create necessary directories.
        
        Called automatically after dataclass instantiation to set defaults
        that depend on other fields or require side effects (like creating directories).
        """
        if self.SUPPORTED_TASKS is None:
            # All supported vision-language tasks in VRSBench
            self.SUPPORTED_TASKS = ["classification", "detection", "captioning", "vqa", "grounding", "complete", "all"]

        # Create directories if they don't exist
        # exist_ok=True prevents errors if directories already exist
        # Skip log directory creation if it's /dev/null (Colab workaround)
        if self.LOG_DIR != "/dev/null":
            os.makedirs(self.LOG_DIR, exist_ok=True)
        if self.CACHE_DIR:
            os.makedirs(self.CACHE_DIR, exist_ok=True)
        if self.CHECKPOINT_DIR:
            os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

# ========================= Logging Setup =========================

class StructuredLogger:
    """
    Production-ready structured logger with JSON support.
    
    This logger provides two output formats:
    1. JSON format: Machine-readable logs perfect for log aggregation systems (ELK, Splunk, etc.)
    2. Plain text: Human-readable logs for development and debugging
    
    Features:
    - Dual output: Console (INFO+) and file (DEBUG+) for different verbosity levels
    - Log rotation: Prevents log files from growing unbounded
    - Structured data: JSON logs include timestamps, levels, and custom fields
    """

    def __init__(self, name: str, config: VRSBenchConfig):
        """
        Initialize structured logger with console and file handlers.
        
        Args:
            name: Logger name (typically class/module name)
            config: Configuration object with logging settings
        """
        self.config = config
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
        # Prevent propagation to root logger to avoid duplicate logs
        self.logger.propagate = False

        # Console handler - outputs to stdout for immediate visibility
        # Set to INFO level to avoid cluttering console with DEBUG messages
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if config.JSON_LOGS:
            # For JSON logs: output raw JSON without additional formatting
            # The JSON is already formatted by _format_json(), so we just pass it through
            console_format = logging.Formatter('%(message)s')
        else:
            # For plain text: human-readable with timestamp and level
            # Useful for development and debugging
            console_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler with rotation - prevents log files from growing too large
        # RotatingFileHandler automatically rotates when maxBytes is reached
        # Skip file handler if LOG_DIR is /dev/null (Colab workaround to prevent log bloat)
        if config.LOG_DIR != "/dev/null":
            log_path = Path(config.LOG_DIR) / config.LOG_FILE
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=config.LOG_MAX_BYTES,  # Rotate at 10MB
                backupCount=config.LOG_BACKUP_COUNT  # Keep 5 backup files
            )
            # File handler captures all levels (DEBUG+) for complete audit trail
            file_handler.setLevel(logging.DEBUG)

            if config.JSON_LOGS:
                # Raw JSON format - each line is a complete JSON object
                # This makes it easy to parse with tools like jq or log aggregation systems
                file_format = logging.Formatter('%(message)s')
            else:
                # Plain text with function name and line number for debugging
                file_format = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
                )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
        # If LOG_DIR is /dev/null, skip file handler entirely (prevents log bloat in Colab)

    def _format_json(self, level: str, message: str, **kwargs) -> str:
        """
        Format log entry as JSON string.
        
        Creates a structured JSON log entry with standard fields plus any additional
        context passed via kwargs. This makes logs machine-readable and queryable.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Main log message
            **kwargs: Additional fields to include in log (e.g., file_path, duration)
        
        Returns:
            JSON string representation of log entry
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),  # ISO 8601 format for easy parsing
            "level": level,
            "message": message,
            "logger": self.logger.name,  # Identifies which component logged this
            **kwargs  # Merge any additional context fields
        }
        return json.dumps(log_entry)

    def debug(self, message: str, **kwargs):
        if self.config.JSON_LOGS:
            self.logger.debug(self._format_json("DEBUG", message, **kwargs))
        else:
            self.logger.debug(f"{message} {kwargs if kwargs else ''}")

    def info(self, message: str, **kwargs):
        if self.config.JSON_LOGS:
            self.logger.info(self._format_json("INFO", message, **kwargs))
        else:
            self.logger.info(f"{message} {kwargs if kwargs else ''}")

    def warning(self, message: str, **kwargs):
        if self.config.JSON_LOGS:
            self.logger.warning(self._format_json("WARNING", message, **kwargs))
        else:
            self.logger.warning(f"{message} {kwargs if kwargs else ''}")

    def error(self, message: str, exc_info: bool = False, **kwargs):
        if self.config.JSON_LOGS:
            self.logger.error(self._format_json("ERROR", message, **kwargs), exc_info=exc_info)
        else:
            self.logger.error(f"{message} {kwargs if kwargs else ''}", exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = False, **kwargs):
        if self.config.JSON_LOGS:
            self.logger.critical(self._format_json("CRITICAL", message, **kwargs), exc_info=exc_info)
        else:
            self.logger.critical(f"{message} {kwargs if kwargs else ''}", exc_info=exc_info)

# ========================= Metrics Collection =========================

class MetricsCollector:
    """
    Collect and report dataloader performance metrics.
    
    This class tracks various metrics during data loading:
    - Counters: Cache hits/misses, images loaded, errors encountered
    - Timings: Download times, image load times, processing durations
    - Errors: Categorized error counts for monitoring and debugging
    
    Metrics are useful for:
    - Performance optimization (identifying bottlenecks)
    - Monitoring production systems (error rates, cache efficiency)
    - Debugging issues (which operations are slow/failing)
    """

    def __init__(self):
        # Counters for discrete events (cache hits, images loaded, etc.)
        self.metrics = defaultdict(int)
        # Timing data for operations (allows computing mean/min/max)
        self.timings = defaultdict(list)
        # Error counts by type (helps identify common failure modes)
        self.errors = defaultdict(int)

    def record_time(self, operation: str, duration: float):
        self.timings[operation].append(duration)

    def increment(self, metric: str, value: int = 1):
        self.metrics[metric] += value

    def record_error(self, error_type: str):
        self.errors[error_type] += 1

    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "metrics": dict(self.metrics),
            "errors": dict(self.errors),
            "timings": {}
        }

        for op, times in self.timings.items():
            if times:
                summary["timings"][op] = {
                    "count": len(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times)
                }

        return summary

    def reset(self):
        self.metrics.clear()
        self.timings.clear()
        self.errors.clear()


def _json_default(value):
    """Best-effort JSON serializer for objects that default json can't handle."""
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("latin-1", errors="ignore")
    return str(value)


class ProgressCheckpointWriter:
    """Incrementally persists processed dataset snapshots to avoid losing work."""

    def __init__(
        self,
        enabled: bool,
        checkpoint_dir: str,
        split: str,
        task: str,
        every: int,
        logger: StructuredLogger,
    ):
        self.enabled = enabled and every is not None and every > 0
        self.every = every if every else 0
        self.logger = logger
        self.prefix = f"{split}_{task.replace('/', '_')}"
        self.last_checkpoint_count = 0
        self.checkpoint_dir = checkpoint_dir
        if self.enabled:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def maybe_checkpoint(self, data: Dict[str, Any], processed_count: int) -> bool:
        if not self.enabled or processed_count - self.last_checkpoint_count < self.every:
            return False
        file_name = f"{self.prefix}_progress_{processed_count:06d}.json"
        self._write_snapshot(data, file_name)
        self.last_checkpoint_count = processed_count
        return True

    def finalize(self, data: Dict[str, Any]):
        if not self.enabled:
            return
        file_name = f"{self.prefix}_final.json"
        self._write_snapshot(data, file_name)

    def _write_snapshot(self, data: Dict[str, Any], file_name: str):
        path = os.path.join(self.checkpoint_dir, file_name)
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2, default=_json_default)
            os.replace(tmp_path, path)
            self.logger.debug(f"Checkpoint persisted: {path}")
        except Exception as exc:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            self.logger.warning(f"Failed to write checkpoint {path}: {exc}")


def _normalize_sample_for_consistency(sample: Any) -> Any:
    """Normalize known inconsistent fields (e.g., ques_id types) in-place."""
    if not isinstance(sample, dict):
        return sample

    if "qa_pairs" in sample and isinstance(sample["qa_pairs"], list):
        normalized_pairs = []
        for qa in sample["qa_pairs"]:
            if not isinstance(qa, dict):
                continue
            qa_copy = dict(qa)
            if "ques_id" in qa_copy and qa_copy["ques_id"] is not None:
                qa_copy["ques_id"] = str(qa_copy["ques_id"])
            normalized_pairs.append(qa_copy)
        sample["qa_pairs"] = normalized_pairs

    if "image_id" in sample and sample["image_id"] is not None:
        sample["image_id"] = str(sample["image_id"])

    return sample


def _extract_sample_key(sample: Dict[str, Any], idx: int, split: str) -> str:
    """Derive a deterministic key for deduplicating samples."""
    candidate_keys = [
        sample.get("image_id"),
        sample.get("image_name"),
        sample.get("file_name"),
        sample.get("id"),
    ]

    image_field = sample.get("image")
    if isinstance(image_field, dict):
        candidate_keys.append(image_field.get("path"))
        candidate_keys.append(image_field.get("filename"))
    elif isinstance(image_field, str):
        candidate_keys.append(image_field)

    for key in candidate_keys:
        if key:
            return str(key)
    return f"{split}_{idx:06d}"


def _ensure_annotations_zip(
    split: str,
    annotations_dir: Optional[str],
    annotations_url: Optional[str],
    config: VRSBenchConfig,
    download_mgr: DownloadManager,
    logger: StructuredLogger,
    force_download: bool = False,
) -> Optional[str]:
    """Download annotations zip if needed and return path."""
    if annotations_dir is None:
        annotations_dir = f"./Annotations_{split}"
    os.makedirs(annotations_dir, exist_ok=True)

    if annotations_url is None:
        if split == "validation":
            annotations_url = config.ANNOTATIONS_VAL_URL
        elif split == "train":
            annotations_url = config.ANNOTATIONS_TRAIN_URL
        else:
            logger.warning(f"Unknown split '{split}' for annotations fallback; cannot download")
            return None

    zip_name = os.path.basename(annotations_url)
    zip_path = os.path.join(annotations_dir, zip_name)

    should_download = force_download or not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1024
    if should_download:
        logger.info(f"Downloading annotations for {split} split...")
        download_mgr.download_with_retries(annotations_url, zip_path, force=force_download)

    if not os.path.exists(zip_path):
        logger.error(f"Annotations zip not found at {zip_path}")
        return None

    return zip_path


def _iterate_annotations_from_zip(
    zip_path: str,
    logger: StructuredLogger,
    stats: Optional[Dict[str, int]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield samples from a local annotations zip, normalizing inconsistencies."""
    if not zip_path or not os.path.exists(zip_path):
        logger.error(f"Cannot iterate annotations from missing zip: {zip_path}")
        return iter(())

    def _iter() -> Iterator[Dict[str, Any]]:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                json_files = [name for name in zf.namelist() if name.lower().endswith(".json")]
                if not json_files:
                    logger.warning(f"No JSON files found inside {zip_path}")
                for member in json_files:
                    try:
                        with zf.open(member, "r") as fh:
                            text = io.TextIOWrapper(fh, encoding="utf-8")
                            data = json.load(text)
                    except json.JSONDecodeError as exc:
                        logger.warning(f"Failed to parse {member} in {zip_path}: {exc}")
                        if stats is not None:
                            stats["annotation_parse_errors"] = stats.get("annotation_parse_errors", 0) + 1
                        continue
                    except Exception as exc:
                        logger.warning(f"Failed to read {member} in {zip_path}: {exc}")
                        if stats is not None:
                            stats["annotation_read_errors"] = stats.get("annotation_read_errors", 0) + 1
                        continue

                    if isinstance(data, list):
                        for entry in data:
                            yield _normalize_sample_for_consistency(entry)
                    elif isinstance(data, dict):
                        annotations = data.get("annotations")
                        if isinstance(annotations, list):
                            metadata = {k: v for k, v in data.items() if k != "annotations"}
                            for entry in annotations:
                                combined = dict(metadata)
                                combined.update(entry if isinstance(entry, dict) else {})
                                yield _normalize_sample_for_consistency(combined)
                        else:
                            yield _normalize_sample_for_consistency(data)
                    else:
                        logger.warning(f"Unsupported JSON structure in {member}: {type(data)}")
        except zipfile.BadZipFile as exc:
            logger.error(f"Annotations zip {zip_path} is corrupted: {exc}")
            if stats is not None:
                stats["annotation_bad_zip"] = stats.get("annotation_bad_zip", 0) + 1

    return _iter()


def _resilient_sample_generator(
    split: str,
    hf_dataset_name: str,
    hf_token: Optional[str],
    logger: StructuredLogger,
    config: VRSBenchConfig,
    download_mgr: DownloadManager,
    annotations_dir: Optional[str],
    annotations_url: Optional[str],
    force_download: bool = False,
    stats: Optional[Dict[str, int]] = None,
    prefer_local_annotations: Optional[bool] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield samples from HuggingFace streaming dataset with automatic fallback."""
    prefer_local = (
        config.PREFER_LOCAL_ANNOTATIONS if prefer_local_annotations is None else prefer_local_annotations
    )
    fallback_zip_path: Optional[str] = None

    def _local_iterator(reason: str) -> Optional[Iterator[Dict[str, Any]]]:
        nonlocal fallback_zip_path
        fallback_zip_path = fallback_zip_path or _ensure_annotations_zip(
            split=split,
            annotations_dir=annotations_dir,
            annotations_url=annotations_url,
            config=config,
            download_mgr=download_mgr,
            logger=logger,
            force_download=force_download,
        )
        if fallback_zip_path:
            logger.info(reason)
            return _iterate_annotations_from_zip(fallback_zip_path, logger, stats=stats)
        return None

    if prefer_local:
        local_iter = _local_iterator(
            "Preferring local annotations zip before falling back to HuggingFace streaming"
        )
        if local_iter is not None:
            yield from local_iter
            return
        else:
            logger.warning(
                "Prefer-local requested but annotations zip unavailable; attempting HuggingFace streaming"
            )

    hf_iterator: Optional[Iterator[Dict[str, Any]]] = None
    if HAS_DATASETS:
        if hf_token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        try:
            hf_dataset = load_dataset(hf_dataset_name, streaming=True)
            if split not in hf_dataset:
                logger.warning(f"Split '{split}' not present in dataset {hf_dataset_name}")
            else:
                hf_iterator = iter(hf_dataset[split])
        except Exception as exc:
            logger.warning(f"Failed to initialize HuggingFace streaming dataset: {exc}")
    else:
        logger.warning("datasets library unavailable; skipping HuggingFace streaming")

    fallback_triggered = False
    if hf_iterator is not None:
        consecutive_errors = 0
        while True:
            try:
                sample = next(hf_iterator)
                consecutive_errors = 0
                yield _normalize_sample_for_consistency(sample)
            except StopIteration:
                return
            except (ArrowInvalid, ValueError, TypeError, KeyError) as exc:
                consecutive_errors += 1
                if stats is not None:
                    stats["hf_parsing_errors"] = stats.get("hf_parsing_errors", 0) + 1
                logger.warning(
                    f"Streaming parse error ({type(exc).__name__}): {exc}. "
                    f"[{consecutive_errors}/{config.HF_MAX_CONSECUTIVE_PARSE_ERRORS}]"
                )
                if consecutive_errors >= config.HF_MAX_CONSECUTIVE_PARSE_ERRORS:
                    logger.error("Too many parsing errors from HuggingFace stream. Switching to fallback.")
                    fallback_triggered = True
                    break
                continue
            except Exception as exc:
                consecutive_errors += 1
                if stats is not None:
                    stats["hf_unexpected_errors"] = stats.get("hf_unexpected_errors", 0) + 1
                logger.warning(
                    f"Unexpected streaming error ({type(exc).__name__}): {exc}. "
                    f"[{consecutive_errors}/{config.HF_MAX_CONSECUTIVE_PARSE_ERRORS}]"
                )
                if consecutive_errors >= config.HF_MAX_CONSECUTIVE_PARSE_ERRORS:
                    logger.error("Too many unexpected errors from HuggingFace stream. Switching to fallback.")
                    fallback_triggered = True
                    break
                continue

    if hf_iterator is None or fallback_triggered:
        local_iter = _local_iterator("Using local annotations fallback loader")
        if local_iter is not None:
            for sample in local_iter:
                yield sample

# ========================= Download Utilities =========================

class DownloadManager:
    """
    Manages robust file downloads with caching and verification.
    
    This class handles downloading datasets from HuggingFace with:
    - Automatic retries with exponential backoff (handles network failures)
    - File caching (avoids re-downloading on subsequent runs)
    - Integrity verification (ensures files aren't corrupted)
    - Progress bars (visual feedback during long downloads)
    - Rate limit handling (detects and handles HTTP 429 errors)
    - HuggingFace authentication (uses tokens to avoid rate limits)
    """

    def __init__(self, config: VRSBenchConfig, logger: StructuredLogger, metrics: MetricsCollector):
        """
        Initialize download manager with configuration and dependencies.
        
        Args:
            config: Configuration with download settings (retries, timeouts, etc.)
            logger: Logger for tracking download progress and errors
            metrics: Metrics collector for tracking cache hits/misses and download times
        """
        self.config = config
        self.logger = logger
        self.metrics = metrics

    def _get_headers(self) -> Dict[str, str]:
        """
        Get HTTP headers with HuggingFace authentication if available.
        
        HuggingFace allows anonymous downloads but rate-limits them. Providing a token
        (free from https://huggingface.co/settings/tokens) increases rate limits significantly.
        
        Returns:
            Dictionary of HTTP headers including authentication if token is available
        """
        # Check for token in environment variables (supports both common variable names)
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        headers = {"User-Agent": "vrsbench-production-loader/2.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
            self.logger.debug("Using HuggingFace authentication token")
        else:
            # Warn but don't fail - downloads will still work, just may be slower
            self.logger.warning("No HF token found. Downloads may be rate-limited. Set HUGGINGFACE_HUB_TOKEN env var.")
        return headers

    def _verify_file(self, path: str, expected_size: Optional[int] = None) -> bool:
        """
        Verify downloaded file integrity before using cached files.
        
        This prevents using corrupted or incomplete downloads. Checks:
        1. File exists
        2. File size is reasonable (not suspiciously small)
        3. File size matches expected size (if provided)
        4. Zip files are valid (can be opened and read)
        
        Args:
            path: Path to file to verify
            expected_size: Expected file size in bytes (optional, for validation)
        
        Returns:
            True if file is valid, False otherwise
        """
        if not os.path.exists(path):
            return False

        file_size = os.path.getsize(path)

        # Check minimum size - files smaller than 1KB are likely error pages or empty
        if file_size < 1024:
            self.logger.warning(f"File {path} is suspiciously small: {file_size} bytes")
            return False

        # Check expected size if provided - allows detecting incomplete downloads
        # Tolerance of 1KB accounts for minor variations
        if expected_size and abs(file_size - expected_size) > 1024:
            self.logger.warning(f"File size mismatch: expected ~{expected_size}, got {file_size}")
            return False

        # Check if it's a zip file - try to open it to verify it's not corrupted
        if path.endswith('.zip'):
            try:
                with zipfile.ZipFile(path, 'r') as zf:
                    zf.namelist()  # Try to read file list - will fail if corrupted
                return True
            except zipfile.BadZipFile:
                self.logger.error(f"Corrupted zip file: {path}")
                return False

        return True

    def download_with_retries(self, url: str, output_path: str, force: bool = False) -> str:
        """
        Download file with automatic retries, progress bar, and caching.
        
        This is the main download method that orchestrates the entire download process:
        1. Checks cache first (avoids unnecessary downloads)
        2. Downloads with progress bar (visual feedback)
        3. Verifies file integrity (ensures download succeeded)
        4. Retries on failure with exponential backoff (handles transient errors)
        5. Handles rate limiting (detects HTTP 429 and waits)

        Args:
            url: Source URL to download from
            output_path: Local path where file should be saved
            force: If True, re-download even if cached file exists

        Returns:
            Path to downloaded file (same as output_path)

        Raises:
            RuntimeError: If download fails after all retry attempts
        """
        start_time = time.time()

        # Check cache first - avoids re-downloading if file already exists
        if not force and os.path.exists(output_path):
            if self.config.VERIFY_CACHE and self._verify_file(output_path):
                # Cache hit with verification - file exists and is valid
                self.logger.info(f"Using cached file: {output_path}")
                self.metrics.increment("cache_hits")
                return output_path
            elif not self.config.VERIFY_CACHE:
                # Cache hit without verification - faster but less safe
                self.logger.info(f"Using cached file (unverified): {output_path}")
                self.metrics.increment("cache_hits")
                return output_path
            else:
                # Cached file exists but failed verification - delete and re-download
                self.logger.warning(f"Cached file invalid, re-downloading: {output_path}")
                os.remove(output_path)

        # Cache miss - need to download
        self.metrics.increment("cache_misses")

        # Download with retries
        backoff = 1.0
        last_exception = None

        for attempt in range(1, self.config.MAX_RETRIES + 1):
            try:
                self.logger.info(f"Downloading {url} (attempt {attempt}/{self.config.MAX_RETRIES})")

                response = requests.get(
                    url,
                    stream=True,
                    headers=self._get_headers(),
                    timeout=self.config.TIMEOUT
                )
                response.raise_for_status()

                # Check for HTML response (rate limiting)
                content_type = response.headers.get("content-type", "")
                if "html" in content_type.lower():
                    raise RuntimeError(
                        f"Received HTML response (likely rate-limited). "
                        f"Set HUGGINGFACE_HUB_TOKEN to increase rate limits."
                    )

                # Download with progress bar
                total_size = int(response.headers.get("content-length", 0))
                temp_path = output_path + ".partial"

                with open(temp_path, "wb") as f:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=os.path.basename(output_path),
                        disable=self.config.LOG_LEVEL == "ERROR"
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=self.config.CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                # Verify and rename
                if self._verify_file(temp_path, expected_size=total_size):
                    os.replace(temp_path, output_path)
                    duration = time.time() - start_time
                    self.metrics.record_time("download", duration)
                    self.logger.info(f"Download successful: {output_path} ({duration:.2f}s)")
                    return output_path
                else:
                    raise RuntimeError("Downloaded file failed verification")

            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    self.metrics.record_error("rate_limit")
                    self.logger.error(
                        f"HTTP 429 Too Many Requests. Set HUGGINGFACE_HUB_TOKEN env var. "
                        f"Retrying in {backoff:.1f}s..."
                    )
                else:
                    self.metrics.record_error("http_error")
                    self.logger.error(f"HTTP error: {e}")
                last_exception = e

            except Exception as e:
                self.metrics.record_error("download_error")
                self.logger.error(f"Download error: {e}", exc_info=True)
                last_exception = e

            # Cleanup partial download
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Exponential backoff
            if attempt < self.config.MAX_RETRIES:
                time.sleep(backoff)
                backoff *= self.config.BACKOFF_FACTOR

        # All retries failed
        self.metrics.record_error("download_failed")
        raise RuntimeError(
            f"Failed to download {url} after {self.config.MAX_RETRIES} attempts"
        ) from last_exception

# ========================= Multi-Task Support =========================

class TaskProcessor:
    """Process annotations for different tasks"""

    @staticmethod
    def normalize_bbox(bbox: List[float], image_width: int, image_height: int) -> List[float]:
        """
        Normalize bounding box coordinates to pixel values if needed.
        
        Handles:
        - Normalized coords [0-1] -> pixel coords
        - Corner format [x1, y1, x2, y2] -> [x, y, w, h]
        
        Args:
            bbox: Bounding box [x, y, w, h] or [x1, y1, x2, y2]
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            [x, y, w, h] in pixel coordinates
        """
        if not bbox or len(bbox) < 4:
            return bbox
        
        x1, y1, x2_or_w, y2_or_h = bbox[:4]
        
        # Detect normalized coordinates (all values <= 1)
        if all(0 <= val <= 1 for val in [x1, y1, x2_or_w, y2_or_h]):
            # Check if it's corner format [x1, y1, x2, y2] or [x, y, w, h]
            # If x2 > x1 and values look like corners, treat as corners
            if x2_or_w > x1 and y2_or_h > y1 and x2_or_w <= 1 and y2_or_h <= 1:
                # Corner format normalized -> convert to pixel coords
                x = int(x1 * image_width)
                y = int(y1 * image_height)
                w = int((x2_or_w - x1) * image_width)
                h = int((y2_or_h - y1) * image_height)
            else:
                # [x, y, w, h] normalized -> convert to pixels
                x = int(x1 * image_width)
                y = int(y1 * image_height)
                w = int(x2_or_w * image_width)
                h = int(y2_or_h * image_height)
            return [x, y, w, h]
        
        # Already in pixel coordinates
        # Check if corner format [x1, y1, x2, y2] in pixels
        if x2_or_w > x1 and y2_or_h > y1 and x2_or_w <= image_width and y2_or_h <= image_height:
            # Likely corner format -> convert to [x, y, w, h]
            x = int(x1)
            y = int(y1)
            w = int(x2_or_w - x1)
            h = int(y2_or_h - y1)
            return [x, y, w, h]
        
        # Already [x, y, w, h] in pixels
        return [int(x1), int(y1), int(x2_or_w), int(y2_or_h)]

    @staticmethod
    def extract_classification_labels(metas: List[Dict[str, Any]], label_key: Optional[str] = None) -> List[Any]:
        """Extract classification labels"""
        if not metas:
            return []

        if label_key is None:
            for key in ['label', 'category', 'class', 'target', 'class_id']:
                if key in metas[0]:
                    label_key = key
                    break

        if label_key is None:
            raise ValueError("Could not find label key in metadata")

        return [m.get(label_key) for m in metas]

    @staticmethod
    def extract_captions(metas: List[Dict[str, Any]]) -> List[str]:
        """Extract captions for captioning tasks"""
        captions = []
        for m in metas:
            caption = m.get('caption') or m.get('description') or m.get('text') or ""
            captions.append(str(caption))
        return captions

    @staticmethod
    def extract_vqa_pairs(metas: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Extract question-answer pairs for VQA"""
        pairs = []
        for m in metas:
            # Single QA pair in record
            if 'question' in m and 'answer' in m:
                pairs.append((str(m['question']), str(m['answer'])))
            # Multiple QA pairs
            elif 'qa_pairs' in m:
                qa_list = m['qa_pairs']
                if isinstance(qa_list, list) and qa_list:
                    # Take first QA pair for batching
                    qa = qa_list[0]
                    pairs.append((str(qa.get('question', '')), str(qa.get('answer', ''))))
                else:
                    pairs.append(("", ""))
            else:
                pairs.append(("", ""))
        return pairs

    @staticmethod
    def extract_bboxes(metas: List[Dict[str, Any]]) -> List[Union[List, Dict]]:
        """Extract bounding boxes for detection/grounding"""
        boxes = []
        for m in metas:
            if 'bbox' in m:
                boxes.append(m['bbox'])
            elif 'bboxes' in m:
                boxes.append(m['bboxes'])
            elif 'objects' in m:
                obj_list = m['objects']
                if isinstance(obj_list, list):
                    boxes.append([obj.get('bbox', []) for obj in obj_list])
                else:
                    boxes.append([])
            else:
                boxes.append([])
        return boxes

    @staticmethod
    def extract_region_from_bbox(image: Image.Image, bbox: List[float], padding: int = 10) -> Image.Image:
        """Extract image region using bounding box for region-level tasks"""
        if not bbox or len(bbox) < 4:
            return image

        # Normalize bbox to [x, y, w, h] in pixels
        img_w, img_h = image.size
        normalized_bbox = TaskProcessor.normalize_bbox(bbox, img_w, img_h)
        x, y, w, h = normalized_bbox[:4]

        x1 = max(0, int(x - padding))
        y1 = max(0, int(y - padding))
        x2 = min(img_w, int(x + w + padding))
        y2 = min(img_h, int(y + h + padding))

        return image.crop((x1, y1, x2, y2))

# ========================= Helper Functions =========================

def _detect_image_directory(base_dir: str) -> str:
    """
    Automatically detect the actual directory containing images after extraction.
    
    When a zip file is extracted, images might be in:
    1. The base directory directly
    2. A single subdirectory (e.g., Images_val.zip -> Images_val/)
    3. Multiple subdirectories (use the one with most images)
    
    Args:
        base_dir: Base directory where zip was extracted
        
    Returns:
        Path to directory containing the most images
    """
    if not os.path.exists(base_dir):
        return base_dir
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.BMP')
    
    # Count images in base directory
    base_images = [f for f in os.listdir(base_dir) 
                   if f.endswith(image_extensions) and os.path.isfile(os.path.join(base_dir, f))]
    base_count = len(base_images)
    
    # Find subdirectories (excluding hidden and system dirs)
    subdirs = [d for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d)) 
               and not d.startswith('.') 
               and d != '__MACOSX']
    
    if not subdirs:
        # No subdirectories, images are in base
        return base_dir
    
    # Count images in each subdirectory
    best_dir = base_dir
    best_count = base_count
    
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        try:
            subdir_images = [f for f in os.listdir(subdir_path)
                           if f.endswith(image_extensions) and os.path.isfile(os.path.join(subdir_path, f))]
            subdir_count = len(subdir_images)
            
            if subdir_count > best_count:
                best_dir = subdir_path
                best_count = subdir_count
        except (OSError, PermissionError):
            # Skip directories we can't read
            continue
    
    return best_dir

# ========================= Parquet Support =========================

def save_to_parquet(data: Dict[str, Any], output_path: str, logger: Optional['StructuredLogger'] = None):
    """
    Save prepared dataset to parquet format (10-100x faster than JSON for large datasets).
    
    Args:
        data: Output from prepare_vrsbench_dataset() or prepare_vrsbench_dataset_parallel()
        output_path: Path to save parquet file (should end with .parquet)
        logger: Optional logger for messages
    """
    if not HAS_PARQUET:
        raise ImportError(
            "pyarrow required for parquet support. Install with: pip install pyarrow"
        )
    
    try:
        import pandas as pd
        import json
        
        # Convert samples to DataFrame, converting complex types to strings
        samples = data["samples"]
        samples_processed = []
        for sample in samples:
            sample_processed = {}
            for key, value in sample.items():
                # Handle None values
                if value is None:
                    sample_processed[key] = None
                # Convert complex types (list, dict, tuple) to JSON strings
                elif isinstance(value, (list, dict, tuple)):
                    sample_processed[key] = json.dumps(value)
                else:
                    # Convert all other types to strings
                    sample_processed[key] = str(value)
            samples_processed.append(sample_processed)
        
        # Convert to DataFrame
        df_samples = pd.DataFrame(samples_processed)
        
        # Convert all columns to string type to avoid type inference issues
        # Replace None with empty string to avoid "None" string issues
        for col in df_samples.columns:
            df_samples[col] = df_samples[col].astype(str)
            # Replace "None" string (from None values) with empty string
            df_samples[col] = df_samples[col].replace('None', '')
        
        # Save samples to parquet
        df_samples.to_parquet(output_path, compression='snappy', index=False)
        
        # Save task-specific mappings and metadata separately (as JSON in parquet metadata or separate files)
        metadata = {
            "id_to_path": data.get("id_to_path", {}),
            "split": data.get("split", ""),
            "task": data.get("task", ""),
            "num_samples": data.get("num_samples", 0),
            "image_to_caption": data.get("image_to_caption", {}),
            "image_to_label": data.get("image_to_label", {}),
            "image_to_qa_pairs": data.get("image_to_qa_pairs", {}),
            "image_to_bboxes": data.get("image_to_bboxes", {})
        }
        
        # Save metadata as JSON alongside parquet
        metadata_path = output_path.replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if logger:
            logger.info(f"✓ Saved to parquet: {output_path}")
            logger.info(f"✓ Saved metadata: {metadata_path}")
        else:
            print(f"✓ Saved to parquet: {output_path}")
            print(f"✓ Saved metadata: {metadata_path}")
            
    except Exception as e:
        if logger:
            logger.error(f"Failed to save parquet: {e}")
        raise


def load_from_parquet(parquet_path: str, logger: Optional['StructuredLogger'] = None) -> Dict[str, Any]:
    """
    Load prepared dataset from parquet format.
    
    Args:
        parquet_path: Path to parquet file
        logger: Optional logger for messages
    
    Returns:
        Dictionary with same structure as prepare_vrsbench_dataset() output
    """
    if not HAS_PARQUET:
        raise ImportError(
            "pyarrow required for parquet support. Install with: pip install pyarrow"
        )
    
    try:
        import pandas as pd
        import json
        
        # Load samples from parquet
        df_samples = pd.read_parquet(parquet_path)
        
        # Convert back from strings to original types
        samples = []
        for _, row in df_samples.iterrows():
            sample = {}
            for key, value in row.items():
                # Handle empty strings (were None originally)
                if pd.isna(value) or (isinstance(value, str) and value == ''):
                    sample[key] = None
                # Try to parse JSON strings back to complex types
                elif isinstance(value, str):
                    # Check if it's a JSON string (starts with [ or {)
                    if value.strip().startswith(('{', '[')):
                        try:
                            sample[key] = json.loads(value)
                        except (json.JSONDecodeError, ValueError):
                            # If parsing fails, keep as string
                            sample[key] = value
                    else:
                        sample[key] = value
                else:
                    sample[key] = value
            samples.append(sample)
        
        # Load metadata
        metadata_path = parquet_path.replace('.parquet', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Reconstruct data structure
        data = {
            "samples": samples,
            "id_to_path": metadata.get("id_to_path", {}),
            "split": metadata.get("split", ""),
            "task": metadata.get("task", ""),
            "num_samples": metadata.get("num_samples", len(samples)),
            "image_to_caption": metadata.get("image_to_caption", {}),
            "image_to_label": metadata.get("image_to_label", {}),
            "image_to_qa_pairs": metadata.get("image_to_qa_pairs", {}),
            "image_to_bboxes": metadata.get("image_to_bboxes", {})
        }
        
        if logger:
            logger.info(f"✓ Loaded from parquet: {parquet_path} ({len(samples)} samples)")
        else:
            print(f"✓ Loaded from parquet: {parquet_path} ({len(samples)} samples)")
        
        return data
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to load parquet: {e}")
        raise

# ========================= DataLoader Factory =========================

def create_vrsbench_dataloader(
    task: Union[str, None] = "classification",
    split: str = "validation",
    images_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    hf_dataset_name: str = "xiang709/VRSBench",
    hf_token: Optional[str] = None,
    download_images: bool = True,
    images_url: Optional[str] = None,
    transform: Optional[Callable] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    config: Optional[VRSBenchConfig] = None,
    return_metrics: bool = False
) -> Union[DataLoader, Tuple[DataLoader, MetricsCollector]]:
    """
    Create production-ready VRSBench DataLoader.
    
    **This is a convenience wrapper** around the high-level API:
    - Calls `prepare_vrsbench_dataset()` to download images and load HuggingFace streaming dataset
    - Calls `create_dataloader_from_prepared()` to create the DataLoader
    
    Uses HuggingFace streaming datasets as the primary and only workflow.

    Args:
        task: Task type (classification, detection, captioning, vqa, grounding, "all", or None)
              If "all" or None, loads data for all tasks simultaneously
        split: Dataset split (train/validation/test)
        images_dir: Directory to store images (default: ./Images_{split})
        num_samples: Limit number of samples (None = all)
        hf_dataset_name: HuggingFace dataset identifier (default: "xiang709/VRSBench")
        hf_token: HuggingFace token (or use HUGGINGFACE_HUB_TOKEN env var)
        download_images: Download images if not present (default: True)
        images_url: URL for image zip (auto-detected from split if not provided)
        transform: Image transforms (defaults to 256x256 resize)
        batch_size: Batch size
        num_workers: Number of worker processes
        config: Configuration object
        return_metrics: Return metrics collector along with dataloader

    Returns:
        DataLoader or (DataLoader, MetricsCollector)
    
    Example:
        >>> # Primary workflow: Direct HuggingFace streaming dataset
        >>> dataloader = create_vrsbench_dataloader(
        ...     task="captioning",
        ...     split="validation",
        ...     download_images=True
        ... )
        >>> 
        >>> # For better control, use the high-level API directly:
        >>> from vrsbench_dataloader_production import prepare_vrsbench_dataset, create_dataloader_from_prepared
        >>> data = prepare_vrsbench_dataset(split="validation", task="captioning", num_samples=1000)
        >>> dataloader = create_dataloader_from_prepared(data, batch_size=16)
    """

    config = config or VRSBenchConfig()
    logger = StructuredLogger("DataLoaderFactory", config)
    
    logger.info(f"Creating DataLoader for task={task}, split={split} using high-level API")
    
    # Prepare dataset using high-level API
    prepared_data = prepare_vrsbench_dataset(
        split=split,
        task=task,
        images_dir=images_dir,
        num_samples=num_samples,
        hf_dataset_name=hf_dataset_name,
        hf_token=hf_token,
        download_images=download_images,
        images_url=images_url,
        config=config,
        force_download=False
    )
    
    # Create DataLoader from prepared data
    dataloader = create_dataloader_from_prepared(
        prepared_data=prepared_data,
        task=task,
        batch_size=batch_size,
        num_workers=num_workers,
        transform=transform,
        config=config
    )
    
    logger.info("DataLoader created successfully using high-level API")
    
    if return_metrics:
        # Create a metrics collector for compatibility
        metrics = MetricsCollector()
        return dataloader, metrics
    return dataloader

# ========================= Task-Specific Helpers =========================

def get_task_targets(
    metas: List[Dict[str, Any]], 
    task: str,
    label_key: Optional[str] = None
) -> Union[List[Any], List[Tuple], List[str]]:
    """
    Extract task-specific targets from metadata batch

    Args:
        metas: List of metadata dicts
        task: Task type
        label_key: Label key for classification

    Returns:
        Task-specific targets
    """
    processor = TaskProcessor()

    if task == "classification":
        return processor.extract_classification_labels(metas, label_key)
    elif task == "captioning":
        return processor.extract_captions(metas)
    elif task == "vqa":
        return processor.extract_vqa_pairs(metas)
    elif task in ["detection", "grounding"]:
        return processor.extract_bboxes(metas)
    else:
        raise ValueError(f"Unsupported task: {task}")

# ========================= High-Level Workflow API =========================

def prepare_vrsbench_dataset(
    split: str = "validation",
    task: Union[str, None] = "captioning",
    images_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    hf_dataset_name: str = "xiang709/VRSBench",
    hf_token: Optional[str] = None,
    download_images: bool = True,
    images_url: Optional[str] = None,
    output_json: Union[Optional[str], bool] = None,
    config: Optional[VRSBenchConfig] = None,
    force_download: bool = False,
    prefer_local_annotations: Optional[bool] = None,
    annotations_dir: Optional[str] = None,
    annotations_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level function to prepare VRSBench dataset for any task.
    
    This function automates the entire workflow:
    1. Downloads images from HuggingFace (if needed)
    2. Extracts images to local directory
    3. Streams HuggingFace dataset with automatic fallback to local annotations when parsing fails
    4. Normalizes inconsistent records (e.g., mixed ques_id types) to avoid ArrowInvalid crashes
    5. Combines streaming dataset metadata with local images
    6. Saves periodic checkpoints (every `config.CHECKPOINT_EVERY_SAMPLES`) so work is never lost
    7. Returns easy-to-use data structure with task-specific mappings
    8. Optionally saves to JSON for easy access
    
    This replaces the manual workflow of:
    - curl downloads
    - unzip operations
    - HuggingFace dataset loading
    - Manual image-metadata mapping
    
    Args:
        split: Dataset split ("train" or "validation")
        task: Task type ("classification", "detection", "captioning", "vqa", "grounding", "all", or None)
              If "all" or None, loads data for all tasks simultaneously
        images_dir: Directory to store images (default: ./Images_{split})
        num_samples: Limit number of samples (None = all)
        hf_dataset_name: HuggingFace dataset identifier
        hf_token: HuggingFace token (or use HUGGINGFACE_HUB_TOKEN env var)
        download_images: Whether to download images (default: True)
    images_url: Custom URL for images (default: auto-detect from split)
        output_json: Path to save JSON mapping (optional)
        config: Configuration object (optional)
    force_download: Force re-download even if images exist
    prefer_local_annotations: Force local annotations to be the primary source (default: config flag)
    annotations_dir: Directory to cache annotations zip used by the fallback loader
    annotations_url: Custom URL for annotations zip (default: auto-detect from split)
    
        Returns:
                Dictionary with:
                - "samples": List of sample dicts with image_path and task-specific metadata
                - Task-specific mappings (e.g., "image_to_caption" for captioning, "image_to_label" for classification)
                    When task="all", all mappings are included
                - "id_to_path": Dict mapping image_id -> image_path
                - "split": Dataset split used
                - "task": Task type used ("all" if loading all tasks)
                - "num_samples": Number of samples loaded
                - On-disk checkpoints saved to `config.CHECKPOINT_DIR` during processing for fault tolerance
    
    Example:
        >>> # For captioning task
        >>> data = prepare_vrsbench_dataset(split="validation", task="captioning", num_samples=1000)
        >>> image_to_caption = data["image_to_caption"]
        >>> 
        >>> # For classification task
        >>> data = prepare_vrsbench_dataset(split="validation", task="classification", num_samples=1000)
        >>> image_to_label = data["image_to_label"]
        >>> 
        >>> # For VQA task
        >>> data = prepare_vrsbench_dataset(split="validation", task="vqa", num_samples=1000)
        >>> # Access qa_pairs from samples
        >>> 
        >>> # For all tasks (complete data loading)
        >>> data = prepare_vrsbench_dataset(split="validation", task="all", num_samples=1000)
        >>> # Access all mappings: image_to_caption, image_to_label, image_to_qa_pairs, image_to_bboxes
    """
    config = config or VRSBenchConfig()
    prefer_local_annotations = (
        config.PREFER_LOCAL_ANNOTATIONS if prefer_local_annotations is None else prefer_local_annotations
    )
    
    # Handle "all" tasks mode
    load_all_tasks = (task is None or task == "all")
    if load_all_tasks:
        task = "all"
    
    logger = StructuredLogger("VRSBenchWorkflow", config)
    if load_all_tasks:
        logger.info("Loading data for all tasks (complete data loading mode)")
    else:
        # Validate task
        if task not in config.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}. Choose from {config.SUPPORTED_TASKS} or 'all'")
    if not HAS_DATASETS:
        logger.warning(
            "datasets library not found; will stream via local annotations fallback instead of HuggingFace."
        )
    
    metrics = MetricsCollector()
    download_mgr = DownloadManager(config, logger, metrics)
    task_processor = TaskProcessor()
    
    # Set up images directory
    if images_dir is None:
        images_dir = f"./Images_{split}"
    
    os.makedirs(images_dir, exist_ok=True)
    
    # Download images if needed (improved logic to prevent multiple downloads)
    if download_images:
        # Auto-detect image URL based on split
        if images_url is None:
            if split == "validation":
                images_url = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip"
            elif split == "train":
                images_url = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_train.zip"
            else:
                raise ValueError(f"Unknown split: {split}. Use 'train' or 'validation'")
        
        # Check if images already exist (check both extracted images and zip file)
        zip_name = os.path.basename(images_url)
        zip_path = os.path.join(images_dir, zip_name)
        
        # First check if images are already extracted
        actual_images_dir = _detect_image_directory(images_dir)
        has_extracted_images = False
        if os.path.exists(actual_images_dir):
            try:
                image_files = [f for f in os.listdir(actual_images_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                has_extracted_images = len(image_files) > 0
            except (OSError, PermissionError):
                has_extracted_images = False
        
        # Check if zip file exists and is valid
        has_zip = os.path.exists(zip_path) and os.path.getsize(zip_path) > 1024 * 1024  # At least 1MB
        
        if not force_download and has_extracted_images:
            logger.info(f"Images already exist in {actual_images_dir}, skipping download")
            images_dir = actual_images_dir
        elif not force_download and has_zip:
            # Zip exists but images not extracted - just extract
            logger.info(f"Zip file exists, extracting images...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(images_dir)
            actual_images_dir = _detect_image_directory(images_dir)
            if actual_images_dir != images_dir:
                logger.info(f"Images found in subdirectory: {actual_images_dir}")
                images_dir = actual_images_dir
        else:
            # Need to download
            logger.info(f"Downloading images for {split} split...")
            download_mgr.download_with_retries(images_url, zip_path, force=force_download)
            
            # Extract images
            logger.info(f"Extracting images to {images_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(images_dir)
            
            # Auto-detect actual image directory (might be in subdirectory)
            actual_images_dir = _detect_image_directory(images_dir)
            if actual_images_dir != images_dir:
                logger.info(f"Images found in subdirectory: {actual_images_dir}")
                images_dir = actual_images_dir
    
    # Prepare resilient sample generator (HF streaming + fallback)
    logger.info(
        f"Combining annotations (prefer_local={prefer_local_annotations}) with local images..."
    )
    if num_samples:
        logger.info(f"Extracting {num_samples} samples from {split} set...")
    else:
        logger.info(f"Extracting all samples from {split} set...")

    sample_stats: Dict[str, int] = {}
    dataset_iterator = _resilient_sample_generator(
        split=split,
        hf_dataset_name=hf_dataset_name,
        hf_token=hf_token,
        logger=logger,
        config=config,
        download_mgr=download_mgr,
        annotations_dir=annotations_dir,
        annotations_url=annotations_url,
        force_download=force_download,
        stats=sample_stats,
        prefer_local_annotations=prefer_local_annotations,
    )
    
    # Initialize task-specific mappings
    test_data = {
        "samples": [],
        "id_to_path": {},
        "split": split,
        "task": task,
        "num_samples": 0
    }
    
    # Add task-specific mapping keys
    if load_all_tasks:
        # Initialize all task mappings for complete data loading
        test_data["image_to_caption"] = {}
        test_data["image_to_label"] = {}
        test_data["image_to_qa_pairs"] = {}
        test_data["image_to_bboxes"] = {}
    elif task == "captioning":
        test_data["image_to_caption"] = {}
    elif task == "classification":
        test_data["image_to_label"] = {}
    elif task == "vqa":
        test_data["image_to_qa_pairs"] = {}
    elif task in ["detection", "grounding"]:
        test_data["image_to_bboxes"] = {}
    
    count = 0
    skipped = 0
    duplicates_skipped = 0
    sample_keys_seen = set()
    checkpoint_writer = ProgressCheckpointWriter(
        enabled=bool(config.CHECKPOINT_EVERY_SAMPLES and config.CHECKPOINT_EVERY_SAMPLES > 0),
        checkpoint_dir=config.CHECKPOINT_DIR,
        split=split,
        task=task,
        every=config.CHECKPOINT_EVERY_SAMPLES or 0,
        logger=logger,
    )
    
    # Get list of local image files for matching
    logger.info(f"Building image file index...")
    local_image_files = {}
    if os.path.exists(images_dir):
        all_image_files = []
        for root, dirs, files in os.walk(images_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__MACOSX']
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_files.append((root, file))

        for root, file in tqdm(all_image_files, desc="Indexing image files", disable=config.LOG_LEVEL == "ERROR"):
            file_path = os.path.join(root, file)
            basename = os.path.basename(file)

            if basename not in local_image_files:
                local_image_files[basename] = file_path
            else:
                logger.debug(f"Duplicate image basename detected: {basename} -> {file_path}")

            basename_no_ext = os.path.splitext(basename)[0]
            if basename_no_ext not in local_image_files:
                local_image_files[basename_no_ext] = file_path
        logger.info(f"Indexed {len(local_image_files)} image files for split '{split}'")
    else:
        logger.warning(f"Images directory not found: {images_dir}. Samples without local images will be skipped.")

    progress_total = num_samples if num_samples else None
    disable_progress = config.LOG_LEVEL == "ERROR"

    try:
        with tqdm(total=progress_total, desc=f"Processing {split} samples", disable=disable_progress) as pbar:
            idx = 0
            for sample in dataset_iterator:
                if num_samples and count >= num_samples:
                    break

                if sample is None:
                    idx += 1
                    continue

                sample_key = _extract_sample_key(sample, idx, split)
                if sample_key in sample_keys_seen:
                    duplicates_skipped += 1
                    idx += 1
                    continue
                sample_keys_seen.add(sample_key)

                try:
                    image_id = None
                    if 'image' in sample:
                        image_obj = sample['image']
                        if hasattr(image_obj, 'filename'):
                            image_id = os.path.basename(image_obj.filename)
                        elif isinstance(image_obj, str):
                            image_id = os.path.basename(image_obj)
                        elif isinstance(image_obj, dict):
                            image_id = image_obj.get('path') or image_obj.get('filename') or image_obj.get('file_name')
                            if image_id:
                                image_id = os.path.basename(str(image_id))

                    if not image_id:
                        if 'image_id' in sample:
                            image_id = str(sample['image_id'])
                        elif 'id' in sample:
                            image_id = str(sample['id'])
                        elif 'image_name' in sample:
                            image_id = str(sample['image_name'])
                        elif 'file_name' in sample:
                            image_id = str(sample['file_name'])
                        else:
                            image_id = f"{split}_{idx:05d}"

                    image_id_base = os.path.splitext(image_id)[0]
                    image_path = None
                    if image_id in local_image_files:
                        image_path = local_image_files[image_id]
                    elif image_id_base in local_image_files:
                        image_path = local_image_files[image_id_base]
                    else:
                        possible_names = [
                            image_id,
                            image_id_base,
                            f"{image_id}.png",
                            f"{image_id}.jpg",
                            f"{image_id}.jpeg",
                            f"{image_id_base}.png",
                            f"{image_id_base}.jpg",
                            f"{image_id_base}.jpeg",
                        ]
                        for name in possible_names:
                            if name in local_image_files:
                                image_path = local_image_files[name]
                                break

                    if image_path and os.path.exists(image_path):
                        test_data["id_to_path"][image_id] = image_path
                        sample_info = {
                            "image_id": image_id,
                            "image_path": image_path,
                            "dataset_index": idx
                        }

                        if load_all_tasks:
                            caption = sample.get('caption', '') or sample.get('description', '') or sample.get('text', '')
                            if caption:
                                sample_info["caption"] = caption
                                test_data["image_to_caption"][image_id] = caption

                            label = None
                            for key in ['label', 'category', 'class', 'target', 'class_id']:
                                if key in sample:
                                    label = sample[key]
                                    break
                            if label is not None:
                                sample_info["label"] = label
                                test_data["image_to_label"][image_id] = label

                            qa_pairs = sample.get('qa_pairs', [])
                            if qa_pairs:
                                sample_info["qa_pairs"] = qa_pairs
                                test_data["image_to_qa_pairs"][image_id] = qa_pairs
                            elif 'question' in sample and 'answer' in sample:
                                qa_pair = [(sample['question'], sample['answer'])]
                                sample_info["qa_pairs"] = qa_pair
                                test_data["image_to_qa_pairs"][image_id] = qa_pair

                            bboxes = []
                            if 'bbox' in sample:
                                bboxes = [sample['bbox']]
                            elif 'bboxes' in sample:
                                bboxes = sample['bboxes']
                            elif 'objects' in sample:
                                bboxes = [obj.get('bbox', []) for obj in sample['objects'] if 'bbox' in obj]
                            if bboxes:
                                sample_info["bboxes"] = bboxes
                                test_data["image_to_bboxes"][image_id] = bboxes
                        elif task == "captioning":
                            caption = sample.get('caption', '') or sample.get('description', '') or sample.get('text', '')
                            sample_info["caption"] = caption
                            test_data["image_to_caption"][image_id] = caption
                        elif task == "classification":
                            label = None
                            for key in ['label', 'category', 'class', 'target', 'class_id']:
                                if key in sample:
                                    label = sample[key]
                                    break
                            if label is not None:
                                sample_info["label"] = label
                                test_data["image_to_label"][image_id] = label
                            else:
                                raise ValueError("No label found for classification task")
                        elif task == "vqa":
                            qa_pairs = sample.get('qa_pairs', [])
                            if qa_pairs:
                                sample_info["qa_pairs"] = qa_pairs
                                test_data["image_to_qa_pairs"][image_id] = qa_pairs
                            elif 'question' in sample and 'answer' in sample:
                                qa_pair = [(sample['question'], sample['answer'])]
                                sample_info["qa_pairs"] = qa_pair
                                test_data["image_to_qa_pairs"][image_id] = qa_pair
                            else:
                                raise ValueError("No QA pairs found for VQA task")
                        elif task in ["detection", "grounding"]:
                            bboxes = []
                            if 'bbox' in sample:
                                bboxes = [sample['bbox']]
                            elif 'bboxes' in sample:
                                bboxes = sample['bboxes']
                            elif 'objects' in sample:
                                bboxes = [obj.get('bbox', []) for obj in sample['objects'] if 'bbox' in obj]
                            if bboxes:
                                sample_info["bboxes"] = bboxes
                                test_data["image_to_bboxes"][image_id] = bboxes
                            else:
                                raise ValueError("No bounding boxes found for detection/grounding task")

                        test_data["samples"].append(sample_info)
                        count += 1
                        test_data["num_samples"] = count
                        pbar.update(1)
                        pbar.set_postfix({"processed": count, "skipped": skipped + duplicates_skipped})
                        checkpoint_writer.maybe_checkpoint(test_data, count)
                    else:
                        skipped += 1
                        pbar.set_postfix({"processed": count, "skipped": skipped + duplicates_skipped})
                        if skipped <= 10:
                            logger.warning(f"⚠ Skipped sample at index {idx} due to missing image path: {image_id}")
                except Exception as e:
                    skipped += 1
                    pbar.set_postfix({"processed": count, "skipped": skipped + duplicates_skipped})
                    if skipped <= 10:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        if len(error_msg) > 200:
                            error_msg = error_msg[:200] + "..."
                        logger.warning(f"⚠ Skipped sample at index {idx} due to processing error ({error_type}): {error_msg}")

                idx += 1
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        raise
    finally:
        checkpoint_writer.finalize(test_data)

    test_data["num_samples"] = count
    
    logger.info(f"✓ Created dataset with {count} samples (skipped {skipped})")
    
    # Save to JSON if requested
    if output_json:
        # Handle boolean True - generate default filename
        if output_json is True:
            output_json = f"vrsbench_{split}_{task}.json"
        
        logger.info(f"Saving to JSON: {output_json}")
        with open(output_json, 'w') as f:
            json.dump(test_data, f, indent=2)
        logger.info(f"✓ Saved to: {output_json}")
    
    return test_data


# ========================= Multiprocessing Helper =========================

def _process_single_sample_mp(args):
    """
    Process a single sample for multiprocessing (process-safe version).
    
    This is a module-level function to ensure it can be pickled for multiprocessing.
    Used with ProcessPoolExecutor for true parallelism.
    
    Args:
        args: Tuple of (idx, sample, image_files_dict, task, split)
    
    Returns:
        Dictionary with sample_info, task_data, and id_to_path, or None if processing failed
    """
    idx, sample, image_files_dict, task_mp, split_mp = args
    
    try:
        # Try to get image_id or construct one
        # PRIORITY 1: Check 'image' field first (contains actual image ID/filename)
        image_id = None
        if 'image' in sample:
            image_obj = sample['image']
            # Handle PIL Image object with filename attribute
            if hasattr(image_obj, 'filename'):
                image_id = os.path.basename(image_obj.filename)
            # Handle string path
            elif isinstance(image_obj, str):
                image_id = os.path.basename(image_obj)
            # Handle dict with path/filename
            elif isinstance(image_obj, dict):
                image_id = image_obj.get('path') or image_obj.get('filename') or image_obj.get('file_name')
                if image_id:
                    image_id = os.path.basename(str(image_id))
        
        # PRIORITY 2: Check other common fields
        if not image_id:
            if 'image_id' in sample:
                image_id = str(sample['image_id'])
            elif 'id' in sample:
                image_id = str(sample['id'])
            elif 'image_name' in sample:
                image_id = str(sample['image_name'])
            elif 'file_name' in sample:
                image_id = str(sample['file_name'])
            else:
                # Last resort: construct from index (but this should rarely happen)
                image_id = f"{split_mp}_{idx:05d}"
        
        image_id_base = os.path.splitext(image_id)[0]
        image_path = None
        
        # Strategy 1: Direct match
        if image_id in image_files_dict:
            image_path = image_files_dict[image_id]
        elif image_id_base in image_files_dict:
            image_path = image_files_dict[image_id_base]
        else:
            # Strategy 2: Try common patterns
            possible_names = [
                image_id, image_id_base,
                f"{image_id}.png", f"{image_id}.jpg", f"{image_id}.jpeg",
                f"{image_id_base}.png", f"{image_id_base}.jpg", f"{image_id_base}.jpeg",
            ]
            for name in possible_names:
                if name in image_files_dict:
                    image_path = image_files_dict[name]
                    break
            
            # REMOVED: Index-based fallback (this was causing the bug)
            # If image_path is still None, we should skip this sample rather than assign wrong image
            # This ensures data integrity
        
        # File existence check
        if not image_path or not os.path.exists(image_path):
            return None
        
        # Extract task-specific metadata
        sample_info = {
            "image_id": image_id,
            "image_path": image_path,
            "dataset_index": idx
        }
        
        task_data = {}
        
        # OPTIMIZED: Extract all task data in one pass for "complete" mode
        if task_mp == "complete" or task_mp == "all":
            # Extract all available task data efficiently
            has_data = False
            
            # Captioning
            caption = sample.get('caption', '') or sample.get('description', '') or sample.get('text', '')
            if caption:
                sample_info["caption"] = caption
                task_data["image_to_caption"] = {image_id: caption}
                has_data = True
            
            # Classification
            label = None
            for key in ['label', 'category', 'class', 'target', 'class_id']:
                if key in sample:
                    label = sample[key]
                    break
            if label is not None:
                sample_info["label"] = label
                task_data["image_to_label"] = {image_id: label}
                has_data = True
            
            # VQA
            qa_pairs = sample.get('qa_pairs', [])
            if not qa_pairs and 'question' in sample and 'answer' in sample:
                qa_pairs = [(sample['question'], sample['answer'])]
            if qa_pairs:
                sample_info["qa_pairs"] = qa_pairs
                task_data["image_to_qa_pairs"] = {image_id: qa_pairs}
                has_data = True
            
            # Detection/Grounding
            bboxes = []
            if 'bbox' in sample:
                bboxes = [sample['bbox']]
            elif 'bboxes' in sample:
                bboxes = sample['bboxes']
            elif 'objects' in sample:
                bboxes = [obj.get('bbox', []) for obj in sample['objects'] if 'bbox' in obj]
            if bboxes:
                sample_info["bboxes"] = bboxes
                task_data["image_to_bboxes"] = {image_id: bboxes}
                has_data = True
            
            # Always include sample if image exists (even if no task data)
            # Add all other fields
            for key in ['objects', 'attributes', 'relationships', 'question', 'answer', 
                       'bbox', 'bboxes', 'category', 'label', 'caption', 'description', 'text']:
                if key in sample and key not in sample_info:
                    sample_info[key] = sample[key]
        
        # Single task mode (original behavior - faster, filters samples)
        else:
            # Extract task-specific targets
            if task_mp == "captioning":
                caption = sample.get('caption', '') or sample.get('description', '') or sample.get('text', '')
                sample_info["caption"] = caption
                task_data["image_to_caption"] = {image_id: caption}
            elif task_mp == "classification":
                label = None
                for key in ['label', 'category', 'class', 'target', 'class_id']:
                    if key in sample:
                        label = sample[key]
                        break
                if label is not None:
                    sample_info["label"] = label
                    task_data["image_to_label"] = {image_id: label}
                else:
                    return None  # Skip if no label
            elif task_mp == "vqa":
                qa_pairs = sample.get('qa_pairs', [])
                if qa_pairs:
                    sample_info["qa_pairs"] = qa_pairs
                    task_data["image_to_qa_pairs"] = {image_id: qa_pairs}
                elif 'question' in sample and 'answer' in sample:
                    qa_pair = [(sample['question'], sample['answer'])]
                    sample_info["qa_pairs"] = qa_pair
                    task_data["image_to_qa_pairs"] = {image_id: qa_pair}
                else:
                    return None  # Skip if no QA pairs
            elif task_mp in ["detection", "grounding"]:
                bboxes = []
                if 'bbox' in sample:
                    bboxes = [sample['bbox']]
                elif 'bboxes' in sample:
                    bboxes = sample['bboxes']
                elif 'objects' in sample:
                    bboxes = [obj.get('bbox', []) for obj in sample['objects'] if 'bbox' in obj]
                if bboxes:
                    sample_info["bboxes"] = bboxes
                    task_data["image_to_bboxes"] = {image_id: bboxes}
                else:
                    return None  # Skip if no bboxes
            
            # Add all other fields
            for key in ['objects', 'attributes', 'relationships', 'question', 'answer', 
                       'bbox', 'bboxes', 'category', 'label', 'caption', 'description', 'text']:
                if key in sample and key not in sample_info:
                    sample_info[key] = sample[key]
        
        return {
            "sample_info": sample_info,
            "task_data": task_data,
            "id_to_path": {image_id: image_path}
        }
        
    except Exception as e:
        return None


def prepare_vrsbench_dataset_parallel(
    split: str = "validation",
    task: Union[str, None] = "captioning",
    images_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    hf_dataset_name: str = "xiang709/VRSBench",
    hf_token: Optional[str] = None,
    download_images: bool = True,
    images_url: Optional[str] = None,
    annotations_dir: Optional[str] = None,
    annotations_url: Optional[str] = None,
    output_json: Union[Optional[str], bool] = None,
    output_parquet: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: Optional[int] = None,
    config: Optional[VRSBenchConfig] = None,
    force_download: bool = False,
    num_workers: Optional[int] = None,
    use_multiprocessing: bool = True,
    prefer_local_annotations: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    High-level parallel function to prepare VRSBench dataset for any task.
    
    This is a parallelized version of prepare_vrsbench_dataset() that uses:
    - Non-streaming mode: Loads full dataset into memory for faster access
    - ThreadPoolExecutor: Processes samples in parallel (5-10x faster)
    
    This function automates the entire workflow:
    1. Downloads images from HuggingFace (if needed)
    2. Extracts images to local directory
    3. Loads HuggingFace non-streaming dataset (full dataset in memory)
    4. Combines dataset metadata with local images in parallel
    5. Returns easy-to-use data structure with task-specific mappings
    6. Optionally saves to JSON for easy access
    
    Performance:
    - 5-10x faster than streaming version for small/medium datasets
    - Best for datasets that fit in memory (< 10GB typically)
    - Uses ProcessPoolExecutor for true multi-core parallelism (bypasses Python GIL)
    - Optimal worker count: os.cpu_count() for CPU-bound, 4-16 for I/O-bound
    - Shares the resilient sample generator: automatically falls back to local annotations when HF parsing fails
    - Writes periodic checkpoints so a crash never forces you to start over
    
    Args:
        split: Dataset split ("train" or "validation")
        task: Task type ("classification", "detection", "captioning", "vqa", "grounding", "all", or None)
              If "all" or None, loads data for all tasks simultaneously
        images_dir: Directory to store images (default: ./Images_{split})
        num_samples: Limit number of samples (None = all)
        hf_dataset_name: HuggingFace dataset identifier
        hf_token: HuggingFace token (or use HUGGINGFACE_HUB_TOKEN env var)
        download_images: Whether to download images (default: True)
        images_url: Custom URL for images (default: auto-detect from split)
    annotations_dir: Directory to cache annotations zip used by the fallback loader
        annotations_url: Custom URL for annotations zip (default: auto-detect from split)
        output_json: Path to save JSON mapping (optional, or True for auto-filename)
        output_parquet: Path to save parquet file (optional, 10-100x faster than JSON)
        checkpoint_dir: Directory for periodic JSON checkpoints (default: config.CHECKPOINT_DIR)
        checkpoint_every: Persist checkpoints every N processed samples (default: config.CHECKPOINT_EVERY_SAMPLES)
        config: Configuration object (optional)
        force_download: Force re-download even if images exist
        num_workers: Number of parallel workers (default: min(32, os.cpu_count()) for multiprocessing, min(8, os.cpu_count() * 2) for threading)
        use_multiprocessing: Use ProcessPoolExecutor for true parallelism (default: True)
                             Set False to use ThreadPoolExecutor (I/O-bound only)
    prefer_local_annotations: Force local annotations to be the primary source (default: config flag)
    
    Returns:
        Dictionary with:
        - "samples": List of sample dicts with image_path and task-specific metadata
        - Task-specific mappings (e.g., "image_to_caption" for captioning, "image_to_label" for classification)
          When task="all", all mappings are included
        - "id_to_path": Dict mapping image_id -> image_path
        - "split": Dataset split used
        - "task": Task type used ("all" if loading all tasks)
        - "num_samples": Number of samples loaded
    
    Example:
        >>> # For captioning task (parallel version)
        >>> data = prepare_vrsbench_dataset_parallel(
        ...     split="validation",
        ...     task="captioning",
        ...     num_samples=1000,
        ...     num_workers=8
        ... )
        >>> image_to_caption = data["image_to_caption"]
        >>> 
        >>> # For classification task
        >>> data = prepare_vrsbench_dataset_parallel(
        ...     split="validation",
        ...     task="classification",
        ...     num_samples=1000,
        ...     num_workers=8
        ... )
        >>> image_to_label = data["image_to_label"]
    """
    if not HAS_DATASETS:
        raise ImportError(
            "HuggingFace datasets library is required. Install with: pip install datasets"
        )
    
    config = config or VRSBenchConfig()
    prefer_local_annotations = (
        config.PREFER_LOCAL_ANNOTATIONS if prefer_local_annotations is None else prefer_local_annotations
    )
    
    # Handle "all" tasks mode
    load_all_tasks = (task is None or task == "all")
    if load_all_tasks:
        task = "all"
    
    logger = StructuredLogger("VRSBenchWorkflowParallel", config)
    if load_all_tasks:
        logger.info("Loading data for all tasks (complete data loading mode)")
    else:
        # Validate task
        if task not in config.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}. Choose from {config.SUPPORTED_TASKS} or 'all'")
    
    # Normalize task name
    if task == "all":
        task = "complete"
    
    # Set optimal worker count if not provided
    if num_workers is None:
        if use_multiprocessing:
            # Limit to reasonable number (32 is optimal for most cases)
            # Too many workers cause overhead and memory issues
            num_workers = min(32, os.cpu_count() or 8)
        else:
            num_workers = min(8, (os.cpu_count() or 4) * 2)  # 4-16 for threading
    
    logger = StructuredLogger("VRSBenchWorkflowParallel", config)
    metrics = MetricsCollector()
    download_mgr = DownloadManager(config, logger, metrics)
    
    # Set up images directory
    if images_dir is None:
        images_dir = f"./Images_{split}"
    
    os.makedirs(images_dir, exist_ok=True)
    
    # Download images if needed (improved logic to prevent multiple downloads)
    if download_images:
        # Auto-detect image URL based on split
        if images_url is None:
            if split == "validation":
                images_url = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip"
            elif split == "train":
                images_url = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_train.zip"
            else:
                raise ValueError(f"Unknown split: {split}. Use 'train' or 'validation'")
        
        # Check if images already exist (check both extracted images and zip file)
        zip_name = os.path.basename(images_url)
        zip_path = os.path.join(images_dir, zip_name)
        
        # First check if images are already extracted
        actual_images_dir = _detect_image_directory(images_dir)
        has_extracted_images = False
        if os.path.exists(actual_images_dir):
            try:
                image_files = [f for f in os.listdir(actual_images_dir) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                has_extracted_images = len(image_files) > 0
            except (OSError, PermissionError):
                has_extracted_images = False

        # Check if zip file exists and is valid
        has_zip = os.path.exists(zip_path) and os.path.getsize(zip_path) > 1024 * 1024  # At least 1MB
        
        if not force_download and has_extracted_images:
            logger.info(f"Images already exist in {actual_images_dir}, skipping download")
            images_dir = actual_images_dir
        elif not force_download and has_zip:
            # Zip exists but images not extracted - just extract
            logger.info(f"Zip file exists, extracting images...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(images_dir)
            actual_images_dir = _detect_image_directory(images_dir)
            if actual_images_dir != images_dir:
                logger.info(f"Images found in subdirectory: {actual_images_dir}")
                images_dir = actual_images_dir
        else:
            # Need to download
            logger.info(f"Downloading images for {split} split...")
            download_mgr.download_with_retries(images_url, zip_path, force=force_download)
            
            # Extract images
            logger.info(f"Extracting images to {images_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(images_dir)
            
            # Auto-detect actual image directory (might be in subdirectory)
            actual_images_dir = _detect_image_directory(images_dir)
            if actual_images_dir != images_dir:
                logger.info(f"Images found in subdirectory: {actual_images_dir}")
                images_dir = actual_images_dir
    
    # Build local image files index (same as original)
    logger.info(f"Building image file index...")
    local_image_files = {}
    if os.path.exists(images_dir):
        # Collect all image files first for progress tracking
        all_image_files = []
        for root, dirs, files in os.walk(images_dir):
            # Skip hidden dirs and macOS metadata
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__MACOSX']
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_files.append((root, file))
        
        # Index files with progress bar
        for root, file in tqdm(all_image_files, desc="Indexing image files", disable=config.LOG_LEVEL == "ERROR"):
            file_path = os.path.join(root, file)
            # Store by basename for easy lookup
            basename = os.path.basename(file)
            # Remove extension for matching
            name_without_ext = os.path.splitext(basename)[0]
            local_image_files[name_without_ext] = file_path
            # Also store with full basename
            local_image_files[basename] = file_path
    
    logger.info(f"Found {len(local_image_files)} local image files")

    # Resilient sample generator with automatic fallback to local annotations
    logger.info(
        f"Collecting samples using resilient loader (primary: {hf_dataset_name}, fallback: local annotations, prefer_local={prefer_local_annotations})"
    )
    effective_checkpoint_every = (
        checkpoint_every if checkpoint_every is not None else config.CHECKPOINT_EVERY_SAMPLES
    )
    collection_checkpoint_writer = ProgressCheckpointWriter(
        enabled=bool(effective_checkpoint_every and effective_checkpoint_every > 0),
        checkpoint_dir=checkpoint_dir or config.CHECKPOINT_DIR,
        split=split,
        task=f"{task}_collection",
        every=effective_checkpoint_every or 0,
        logger=logger,
    )
    sample_stats: Dict[str, int] = {}
    sample_iterator = _resilient_sample_generator(
        split=split,
        hf_dataset_name=hf_dataset_name,
        hf_token=hf_token,
        logger=logger,
        config=config,
        download_mgr=download_mgr,
        annotations_dir=annotations_dir,
        annotations_url=annotations_url,
        force_download=force_download,
        stats=sample_stats,
        prefer_local_annotations=prefer_local_annotations,
    )

    all_samples: List[Tuple[int, Dict[str, Any]]] = []
    sample_keys_seen = set()
    duplicates_skipped = 0
    pbar_collect = tqdm(desc="Collecting samples", disable=config.LOG_LEVEL == "ERROR")
    idx = 0
    collection_buffer: List[Dict[str, Any]] = []
    collection_chunk_start = 0

    try:
        for sample in sample_iterator:
            if num_samples and len(all_samples) >= num_samples:
                break
            if sample is None:
                idx += 1
                continue

            sample_key = _extract_sample_key(sample, idx, split)
            if sample_key in sample_keys_seen:
                duplicates_skipped += 1
                idx += 1
                continue

            sample_keys_seen.add(sample_key)
            all_samples.append((idx, sample))
            collection_buffer.append({
                "dataset_index": idx,
                "sample": sample,
            })
            pbar_collect.update(1)
            pbar_collect.set_postfix({
                "collected": len(all_samples),
                "skipped": duplicates_skipped + sample_stats.get("hf_parsing_errors", 0)
            })
            if collection_checkpoint_writer.maybe_checkpoint(
                {
                    "stage": "collecting_samples",
                    "split": split,
                    "task": task,
                    "chunk_start_index": collection_chunk_start,
                    "chunk_size": len(collection_buffer),
                    "total_collected": len(all_samples),
                    "duplicates_skipped": duplicates_skipped,
                    "stats": sample_stats,
                    "samples": collection_buffer,
                },
                len(all_samples),
            ):
                collection_chunk_start = len(all_samples)
                collection_buffer = []
            idx += 1
    except KeyboardInterrupt:
        logger.warning("Sample collection interrupted by user")
        raise
    except Exception as e:
        logger.error(f"Fatal error during sample collection: {e}")
        raise
    finally:
        pbar_collect.close()

    collection_checkpoint_writer.finalize({
        "stage": "collecting_samples",
        "split": split,
        "task": task,
        "chunk_start_index": collection_chunk_start,
        "chunk_size": len(collection_buffer),
        "total_collected": len(all_samples),
        "duplicates_skipped": duplicates_skipped,
        "stats": sample_stats,
        "samples": collection_buffer,
    })

    skipped_parsing = (
        sample_stats.get("hf_parsing_errors", 0)
        + sample_stats.get("hf_unexpected_errors", 0)
        + sample_stats.get("annotation_parse_errors", 0)
        + sample_stats.get("annotation_read_errors", 0)
    )
    total_skipped = skipped_parsing + duplicates_skipped

    if total_skipped > 0:
        logger.warning(
            f"⚠ Skipped {total_skipped} samples (parse errors: {skipped_parsing}, duplicates: {duplicates_skipped})"
        )
        logger.info(
            f"✓ Successfully collected {len(all_samples)} valid samples out of {len(all_samples) + total_skipped} attempted"
        )
    else:
        logger.info(f"✓ Successfully collected {len(all_samples)} samples")
    
    if len(all_samples) == 0:
        raise ValueError(
            "No valid samples could be collected from the dataset. "
            "This may indicate a dataset format issue or all samples had parsing errors. "
            f"Total samples skipped: {total_skipped}"
        )
    
    logger.info(f"Collected {len(all_samples)} samples, processing in parallel with {num_workers} workers...")
    
    # Initialize task-specific mappings
    test_data = {
        "samples": [],
        "id_to_path": {},
        "split": split,
        "task": task,
        "num_samples": 0
    }
    
    # Add task-specific mapping keys
    if load_all_tasks:
        # Initialize all task mappings for complete data loading
        test_data["image_to_caption"] = {}
        test_data["image_to_label"] = {}
        test_data["image_to_qa_pairs"] = {}
        test_data["image_to_bboxes"] = {}
    elif task == "captioning":
        test_data["image_to_caption"] = {}
    elif task == "classification":
        test_data["image_to_label"] = {}
    elif task == "vqa":
        test_data["image_to_qa_pairs"] = {}
    elif task in ["detection", "grounding"]:
        test_data["image_to_bboxes"] = {}
    
    # Checkpoint writer for incremental persistence
    checkpoint_writer = ProgressCheckpointWriter(
        enabled=bool(effective_checkpoint_every and effective_checkpoint_every > 0),
        checkpoint_dir=checkpoint_dir or config.CHECKPOINT_DIR,
        split=split,
        task=task,
        every=effective_checkpoint_every or 0,
        logger=logger,
    )

    # Thread-safe counters (for threading mode)
    count_lock = threading.Lock()
    count = 0
    skipped = 0
    
    def process_single_sample(args):
        """
        Process a single sample - designed for parallel execution (threading mode).
        
        This function is thread-safe and processes one sample at a time.
        It extracts image_id, finds corresponding image file, and extracts
        task-specific metadata.
        
        Args:
            args: Tuple of (idx, sample) where idx is the dataset index and sample is the sample dict
        
        Returns:
            Dictionary with sample_info, task_data, and id_to_path, or None if processing failed
        """
        idx, sample = args
        # Use closure to access local_image_files (works for threading)
        image_files_dict = local_image_files
        
        try:
            # Try to get image_id or construct one
            # PRIORITY 1: Check 'image' field first (contains actual image ID/filename)
            image_id = None
            if 'image' in sample:
                image_obj = sample['image']
                # Handle PIL Image object with filename attribute
                if hasattr(image_obj, 'filename'):
                    image_id = os.path.basename(image_obj.filename)
                # Handle string path
                elif isinstance(image_obj, str):
                    image_id = os.path.basename(image_obj)
                # Handle dict with path/filename
                elif isinstance(image_obj, dict):
                    image_id = image_obj.get('path') or image_obj.get('filename') or image_obj.get('file_name')
                    if image_id:
                        image_id = os.path.basename(str(image_id))
            
            # PRIORITY 2: Check other common fields
            if not image_id:
                if 'image_id' in sample:
                    image_id = str(sample['image_id'])
                elif 'id' in sample:
                    image_id = str(sample['id'])
                elif 'image_name' in sample:
                    image_id = str(sample['image_name'])
                elif 'file_name' in sample:
                    image_id = str(sample['file_name'])
                else:
                    # Last resort: construct from index (but this should rarely happen)
                    image_id = f"{split}_{idx:05d}"
            
            # Remove extension from image_id for matching
            image_id_base = os.path.splitext(image_id)[0]
            
            # Find corresponding local image file
            image_path = None
            
            # Strategy 1: Direct match by image_id (with or without extension)
            if image_id in image_files_dict:
                image_path = image_files_dict[image_id]
            elif image_id_base in image_files_dict:
                image_path = image_files_dict[image_id_base]
            else:
                # Strategy 2: Try common patterns
                possible_names = [
                    image_id,
                    image_id_base,
                    f"{image_id}.png",
                    f"{image_id}.jpg",
                    f"{image_id}.jpeg",
                    f"{image_id_base}.png",
                    f"{image_id_base}.jpg",
                    f"{image_id_base}.jpeg",
                ]
                
                for name in possible_names:
                    if name in image_files_dict:
                        image_path = image_files_dict[name]
                        break
                
                # REMOVED: Index-based matching fallback (this was causing the bug)
                # If image_path is still None, we should skip this sample rather than assign wrong image
            
            # File existence check (I/O bound - benefits from threading)
            if not image_path or not os.path.exists(image_path):
                return None
            
            # Extract task-specific metadata
            sample_info = {
                "image_id": image_id,
                "image_path": image_path,
                "dataset_index": idx
            }
            
            task_data = {}
            
            # OPTIMIZED: Extract all task data in one pass for "complete" mode
            if task == "complete" or task == "all":
                # Extract all available task data efficiently
                has_data = False
                
                # Captioning
                caption = sample.get('caption', '') or sample.get('description', '') or sample.get('text', '')
                if caption:
                    sample_info["caption"] = caption
                    task_data["image_to_caption"] = {image_id: caption}
                    has_data = True
                
                # Classification
                label = None
                for key in ['label', 'category', 'class', 'target', 'class_id']:
                    if key in sample:
                        label = sample[key]
                        break
                if label is not None:
                    sample_info["label"] = label
                    task_data["image_to_label"] = {image_id: label}
                    has_data = True
                
                # VQA
                qa_pairs = sample.get('qa_pairs', [])
                if not qa_pairs and 'question' in sample and 'answer' in sample:
                    qa_pairs = [(sample['question'], sample['answer'])]
                if qa_pairs:
                    sample_info["qa_pairs"] = qa_pairs
                    task_data["image_to_qa_pairs"] = {image_id: qa_pairs}
                    has_data = True
                
                # Detection/Grounding
                bboxes = []
                if 'bbox' in sample:
                    bboxes = [sample['bbox']]
                elif 'bboxes' in sample:
                    bboxes = sample['bboxes']
                elif 'objects' in sample:
                    bboxes = [obj.get('bbox', []) for obj in sample['objects'] if 'bbox' in obj]
                if bboxes:
                    sample_info["bboxes"] = bboxes
                    task_data["image_to_bboxes"] = {image_id: bboxes}
                    has_data = True
                
                # Always include sample if image exists (even if no task data)
                # Add all other fields
                for key in ['objects', 'attributes', 'relationships', 'question', 'answer', 
                           'bbox', 'bboxes', 'category', 'label', 'caption', 'description', 'text']:
                    if key in sample and key not in sample_info:
                        sample_info[key] = sample[key]
            
            # Single task mode (original behavior - faster, filters samples)
            else:
                # Extract task-specific targets
                if task == "captioning":
                    caption = sample.get('caption', '') or sample.get('description', '') or sample.get('text', '')
                    sample_info["caption"] = caption
                    task_data["image_to_caption"] = {image_id: caption}
                elif task == "classification":
                    # Try to find label
                    label = None
                    for key in ['label', 'category', 'class', 'target', 'class_id']:
                        if key in sample:
                            label = sample[key]
                            break
                    if label is not None:
                        sample_info["label"] = label
                        task_data["image_to_label"] = {image_id: label}
                    else:
                        return None
                elif task == "vqa":
                    qa_pairs = sample.get('qa_pairs', [])
                    if qa_pairs:
                        sample_info["qa_pairs"] = qa_pairs
                        task_data["image_to_qa_pairs"] = {image_id: qa_pairs}
                    elif 'question' in sample and 'answer' in sample:
                        qa_pair = [(sample['question'], sample['answer'])]
                        sample_info["qa_pairs"] = qa_pair
                        task_data["image_to_qa_pairs"] = {image_id: qa_pair}
                    else:
                        return None
                elif task in ["detection", "grounding"]:
                    bboxes = []
                    if 'bbox' in sample:
                        bboxes = [sample['bbox']]
                    elif 'bboxes' in sample:
                        bboxes = sample['bboxes']
                    elif 'objects' in sample:
                        bboxes = [obj.get('bbox', []) for obj in sample['objects'] if 'bbox' in obj]
                    if bboxes:
                        sample_info["bboxes"] = bboxes
                        task_data["image_to_bboxes"] = {image_id: bboxes}
                    else:
                        return None
                
                # Add all other fields from dataset
                for key in ['objects', 'attributes', 'relationships', 'question', 'answer', 
                           'bbox', 'bboxes', 'category', 'label', 'caption', 'description', 'text']:
                    if key in sample and key not in sample_info:
                        sample_info[key] = sample[key]
            
            return {
                "sample_info": sample_info,
                "task_data": task_data,
                "id_to_path": {image_id: image_path}
            }
            
        except Exception as e:
            # Log error but don't fail entire batch
            return None
    
    # Prepare arguments for parallel processing
    args_list = all_samples  # Already in (idx, sample) format
    
    # Process in parallel using ProcessPoolExecutor or ThreadPoolExecutor
    # Note: On Windows, multiprocessing requires if __name__ == "__main__" guard
    # For better compatibility, we check if we're in main or if multiprocessing is safe
    if use_multiprocessing and num_workers > 1:
        # Detect Colab environment
        is_colab = 'google.colab' in sys.modules
        
        # Set multiprocessing start method (if not already set)
        try:
            if sys.platform == 'win32' or is_colab:
                # Windows and Colab require 'spawn' method
                mp.set_start_method('spawn', force=False)
                if is_colab:
                    logger.info("Colab detected: Using 'spawn' method for multiprocessing")
            else:
                # On Linux/Unix, 'fork' is default and faster, but try to set it explicitly
                try:
                    mp.set_start_method('fork', force=False)
                except RuntimeError:
                    # Already set, ignore
                    pass
        except RuntimeError:
            # Already set, ignore
            pass
        logger.info(f"Using ProcessPoolExecutor with {num_workers} processes for true multi-core parallelism")
        start_time = time.time()
        
        # Create progress bar
        pbar = tqdm(total=len(args_list), desc=f"Processing samples ({task})", disable=config.LOG_LEVEL == "ERROR")
        
        # For multiprocessing, we need to pass image_files as parameter
        # Note: Passing a copy of the dict to each process (memory trade-off for speed)
        # For very large image file dicts, consider using Manager().dict() but it's slower
        image_files_copy = dict(local_image_files)  # Create once, reuse for all processes
        args_list_mp = [(idx, sample, image_files_copy, task, split) 
                       for idx, sample in args_list]
        
        # Try ProcessPoolExecutor with error handling and fallback
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks using module-level function (pickleable)
                future_to_args = {executor.submit(_process_single_sample_mp, args): args for args in args_list_mp}
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_args):
                    completed += 1
                    result = future.result()
                    
                    with count_lock:
                        if result:
                            test_data["samples"].append(result["sample_info"])
                            test_data["id_to_path"].update(result["id_to_path"])
                            
                            # OPTIMIZED: Update all task mappings (supports complete mode)
                            # Use if-elif only for single task mode, use separate ifs for complete mode
                            if task == "complete" or task == "all":
                                # Update all available task mappings
                                if "image_to_caption" in result["task_data"]:
                                    test_data["image_to_caption"].update(result["task_data"]["image_to_caption"])
                                if "image_to_label" in result["task_data"]:
                                    test_data["image_to_label"].update(result["task_data"]["image_to_label"])
                                if "image_to_qa_pairs" in result["task_data"]:
                                    test_data["image_to_qa_pairs"].update(result["task_data"]["image_to_qa_pairs"])
                                if "image_to_bboxes" in result["task_data"]:
                                    test_data["image_to_bboxes"].update(result["task_data"]["image_to_bboxes"])
                            else:
                                # Single task mode - update only the relevant mapping
                                if "image_to_caption" in result["task_data"]:
                                    test_data["image_to_caption"].update(result["task_data"]["image_to_caption"])
                                elif "image_to_label" in result["task_data"]:
                                    test_data["image_to_label"].update(result["task_data"]["image_to_label"])
                                elif "image_to_qa_pairs" in result["task_data"]:
                                    test_data["image_to_qa_pairs"].update(result["task_data"]["image_to_qa_pairs"])
                                elif "image_to_bboxes" in result["task_data"]:
                                    test_data["image_to_bboxes"].update(result["task_data"]["image_to_bboxes"])
                            
                            count += 1
                            checkpoint_writer.maybe_checkpoint(test_data, count)
                        else:
                            skipped += 1
                        
                        # Update progress bar
                        elapsed = time.time() - start_time
                        rate = count / elapsed if elapsed > 0 else 0
                        pbar.update(1)
                        pbar.set_postfix({
                            "processed": count,
                            "skipped": skipped,
                            "rate": f"{rate:.1f}/s"
                        })
                        
                        # Progress logging (every 500 samples)
                        if count % 500 == 0:
                            logger.info(f"✓ [{count}/{len(all_samples)}] Processed samples... ({rate:.1f} samples/sec)")
            
            pbar.close()
            
        except (AttributeError, TypeError, Exception) as e:
            # If pickling fails (e.g., nested function issue), fall back to threading
            logger.warning(f"ProcessPoolExecutor failed ({type(e).__name__}: {e}), falling back to ThreadPoolExecutor")
            logger.warning("This may indicate the function cannot be pickled. Using threading mode instead.")
            use_multiprocessing = False
            num_workers = min(8, (os.cpu_count() or 4) * 2)
            # Continue with ThreadPoolExecutor below
        
    if not use_multiprocessing or num_workers <= 1:
        # Use ThreadPoolExecutor for I/O-bound operations or when multiprocessing disabled
        logger.info(f"Using ThreadPoolExecutor with {num_workers} threads (I/O-bound mode)")
        start_time = time.time()
        
        # Create progress bar
        pbar = tqdm(total=len(args_list), desc=f"Processing samples ({task})", disable=config.LOG_LEVEL == "ERROR")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks (args_list already contains (idx, sample) tuples)
            future_to_args = {executor.submit(process_single_sample, args): args for args in args_list}
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_args):
                completed += 1
                result = future.result()
                
                with count_lock:
                    if result:
                        test_data["samples"].append(result["sample_info"])
                        test_data["id_to_path"].update(result["id_to_path"])
                        
                        # OPTIMIZED: Update all task mappings (supports complete mode)
                        if task == "complete" or task == "all":
                            # Update all available task mappings
                            if "image_to_caption" in result["task_data"]:
                                test_data["image_to_caption"].update(result["task_data"]["image_to_caption"])
                            if "image_to_label" in result["task_data"]:
                                test_data["image_to_label"].update(result["task_data"]["image_to_label"])
                            if "image_to_qa_pairs" in result["task_data"]:
                                test_data["image_to_qa_pairs"].update(result["task_data"]["image_to_qa_pairs"])
                            if "image_to_bboxes" in result["task_data"]:
                                test_data["image_to_bboxes"].update(result["task_data"]["image_to_bboxes"])
                        else:
                            # Single task mode - update only the relevant mapping
                            if "image_to_caption" in result["task_data"]:
                                test_data["image_to_caption"].update(result["task_data"]["image_to_caption"])
                            elif "image_to_label" in result["task_data"]:
                                test_data["image_to_label"].update(result["task_data"]["image_to_label"])
                            elif "image_to_qa_pairs" in result["task_data"]:
                                test_data["image_to_qa_pairs"].update(result["task_data"]["image_to_qa_pairs"])
                            elif "image_to_bboxes" in result["task_data"]:
                                test_data["image_to_bboxes"].update(result["task_data"]["image_to_bboxes"])
                        
                        count += 1
                        checkpoint_writer.maybe_checkpoint(test_data, count)
                    else:
                        skipped += 1
                    
                    # Update progress bar
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    pbar.update(1)
                    pbar.set_postfix({
                        "processed": count,
                        "skipped": skipped,
                        "rate": f"{rate:.1f}/s"
                    })
                    
                    # Progress logging (every 500 samples)
                    if count % 500 == 0:
                        logger.info(f"✓ [{count}/{len(all_samples)}] Processed samples... ({rate:.1f} samples/sec)")
        
        pbar.close()
    
    # Calculate final elapsed time (start_time was set in both branches)
    elapsed = time.time() - start_time
    test_data["num_samples"] = count
    rate = count / elapsed if elapsed > 0 else 0
    logger.info(f"✓ Created dataset with {count} samples (skipped {skipped}) in {elapsed:.2f}s ({rate:.1f} samples/sec)")
    checkpoint_writer.finalize(test_data)
    
    # Record metrics
    metrics.record_time("parallel_preparation", elapsed)
    metrics.increment("samples_processed", count)
    metrics.increment("samples_skipped", skipped)
    
    # Save to parquet if requested (faster than JSON)
    if output_parquet:
        try:
            save_to_parquet(test_data, output_parquet, logger)
        except ImportError:
            logger.warning("pyarrow not available, falling back to JSON")
            if not output_json:
                output_json = output_parquet.replace('.parquet', '.json')
    
    # Save to JSON if requested
    if output_json:
        # Handle boolean True - generate default filename
        if output_json is True:
            output_json = f"vrsbench_{split}_{task}.json"
        
        logger.info(f"Saving to JSON: {output_json}")
        with open(output_json, 'w') as f:
            json.dump(test_data, f, indent=2)
        logger.info(f"✓ Saved to: {output_json}")
    
    return test_data


def create_dataloader_from_prepared(
    prepared_data: Dict[str, Any],
    task: Optional[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    config: Optional[VRSBenchConfig] = None
) -> DataLoader:
    """
    Create a PyTorch DataLoader from prepared dataset.
    
    This is a convenience function that converts the output of prepare_vrsbench_dataset()
    into a PyTorch DataLoader for training/inference. Works for all tasks.
    
    Args:
        prepared_data: Output from prepare_vrsbench_dataset()
        task: Task type (auto-detected from prepared_data if not provided)
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        transform: Image transforms (default: resize to 256x256)
        config: Configuration object
    
    Returns:
        PyTorch DataLoader yielding (image_tensor, metadata_dict) batches
    
    Example:
        >>> # For any task
        >>> data = prepare_vrsbench_dataset(split="validation", task="classification", num_samples=1000)
        >>> dataloader = create_dataloader_from_prepared(data, batch_size=16)
        >>> 
        >>> for images, metas in dataloader:
        ...     labels = get_task_targets(metas, task="classification")
        ...     # Train your model
    """
    config = config or VRSBenchConfig()
    
    # Auto-detect task from prepared_data if not provided
    if task is None:
        task = prepared_data.get("task", "captioning")
    
    # Validate task
    if task not in config.SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}. Choose from {config.SUPPORTED_TASKS}")
    
    logger = StructuredLogger("PreparedDataLoader", config)
    
    if not prepared_data.get("samples"):
        raise ValueError("prepared_data has no samples")
    
    # Default transform
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(config.DEFAULT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create a simple dataset from prepared data
    class PreparedDataset(IterableDataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
        
        def __iter__(self):
            for sample in self.samples:
                try:
                    image_path = sample["image_path"]
                    with Image.open(image_path) as img:
                        img = img.convert('RGB')
                        if self.transform:
                            img_tensor = self.transform(img)
                        else:
                            img_tensor = transforms.ToTensor()(img)
                    
                    # Return image and metadata
                    yield img_tensor, sample
                except Exception as e:
                    logger.warning(f"Failed to load image {sample.get('image_path', 'unknown')}: {e}")
                    continue
    
    dataset = PreparedDataset(prepared_data["samples"], transform)
    
    def collate_fn(batch):
        images = torch.stack([b[0] for b in batch])
        metas = [b[1] for b in batch]
        return images, metas
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR if num_workers > 0 else None
    )
    
    logger.info(f"Created DataLoader with {len(prepared_data['samples'])} samples for task: {task}")
    return dataloader




