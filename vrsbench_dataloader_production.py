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
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import requests
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Optional dependencies - HuggingFace datasets library
# This library provides better performance for large datasets and seamless integration
# with HuggingFace Hub. If not available, the code falls back to pandas-based processing.
try:
    from datasets import load_dataset, Dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    load_dataset = None
    Dataset = None

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
            self.SUPPORTED_TASKS = ["classification", "detection", "captioning", "vqa", "grounding"]

        # Create directories if they don't exist
        # exist_ok=True prevents errors if directories already exist
        # Skip log directory creation if it's /dev/null (Colab workaround)
        if self.LOG_DIR != "/dev/null":
            os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)

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

        # Remove existing handlers to avoid duplicates if logger is reused
        self.logger.handlers.clear()

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

# ========================= DataLoader Factory =========================

def create_vrsbench_dataloader(
    task: str = "classification",
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
        task: Task type (classification, detection, captioning, vqa, grounding)
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
    task: str = "captioning",
    images_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    hf_dataset_name: str = "xiang709/VRSBench",
    hf_token: Optional[str] = None,
    download_images: bool = True,
    images_url: Optional[str] = None,
    output_json: Optional[str] = None,
    config: Optional[VRSBenchConfig] = None,
    force_download: bool = False
) -> Dict[str, Any]:
    """
    High-level function to prepare VRSBench dataset for any task.
    
    This function automates the entire workflow:
    1. Downloads images from HuggingFace (if needed)
    2. Extracts images to local directory
    3. Loads HuggingFace streaming dataset
    4. Combines streaming dataset metadata with local images
    5. Returns easy-to-use data structure with task-specific mappings
    6. Optionally saves to JSON for easy access
    
    This replaces the manual workflow of:
    - curl downloads
    - unzip operations
    - HuggingFace dataset loading
    - Manual image-metadata mapping
    
    Args:
        split: Dataset split ("train" or "validation")
        task: Task type ("classification", "detection", "captioning", "vqa", "grounding")
        images_dir: Directory to store images (default: ./Images_{split})
        num_samples: Limit number of samples (None = all)
        hf_dataset_name: HuggingFace dataset identifier
        hf_token: HuggingFace token (or use HUGGINGFACE_HUB_TOKEN env var)
        download_images: Whether to download images (default: True)
        images_url: Custom URL for images (default: auto-detect from split)
        output_json: Path to save JSON mapping (optional)
        config: Configuration object (optional)
        force_download: Force re-download even if images exist
    
    Returns:
        Dictionary with:
        - "samples": List of sample dicts with image_path and task-specific metadata
        - Task-specific mappings (e.g., "image_to_caption" for captioning, "image_to_label" for classification)
        - "id_to_path": Dict mapping image_id -> image_path
        - "split": Dataset split used
        - "task": Task type used
        - "num_samples": Number of samples loaded
    
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
    """
    if not HAS_DATASETS:
        raise ImportError(
            "HuggingFace datasets library is required. Install with: pip install datasets"
        )
    
    config = config or VRSBenchConfig()
    
    # Validate task
    if task not in config.SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}. Choose from {config.SUPPORTED_TASKS}")
    
    logger = StructuredLogger("VRSBenchWorkflow", config)
    metrics = MetricsCollector()
    download_mgr = DownloadManager(config, logger, metrics)
    task_processor = TaskProcessor()
    
    # Set up images directory
    if images_dir is None:
        images_dir = f"./Images_{split}"
    
    os.makedirs(images_dir, exist_ok=True)
    
    # Download images if needed
    if download_images:
        # Auto-detect image URL based on split
        if images_url is None:
            if split == "validation":
                images_url = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip"
            elif split == "train":
                images_url = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_train.zip"
            else:
                raise ValueError(f"Unknown split: {split}. Use 'train' or 'validation'")
        
        logger.info(f"Downloading images for {split} split...")
        
        # Check if images already exist
        zip_name = os.path.basename(images_url)
        zip_path = os.path.join(images_dir, zip_name)
        
        # Check if images are already extracted
        actual_images_dir = _detect_image_directory(images_dir)
        has_images = False
        if os.path.exists(actual_images_dir):
            image_files = [f for f in os.listdir(actual_images_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            has_images = len(image_files) > 0
        
        if not force_download and has_images:
            logger.info(f"Images already exist in {actual_images_dir}, skipping download")
            images_dir = actual_images_dir
        else:
            # Download zip file
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
    
    # Load HuggingFace streaming dataset
    logger.info(f"Loading HuggingFace dataset: {hf_dataset_name}")
    
    # Set up authentication if token provided
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    
    try:
        hf_dataset = load_dataset(hf_dataset_name, streaming=True)
    except Exception as e:
        logger.error(f"Failed to load HuggingFace dataset: {e}")
        raise
    
    if split not in hf_dataset:
        raise ValueError(f"Split '{split}' not found in dataset. Available: {list(hf_dataset.keys())}")
    
    dataset_iterator = hf_dataset[split]
    
    # Combine streaming dataset with local images
    logger.info(f"Combining streaming dataset with local images...")
    if num_samples:
        logger.info(f"Extracting {num_samples} samples from {split} set...")
    else:
        logger.info(f"Extracting all samples from {split} set...")
    
    # Initialize task-specific mappings
    test_data = {
        "samples": [],
        "id_to_path": {},
        "split": split,
        "task": task,
        "num_samples": 0
    }
    
    # Add task-specific mapping keys
    if task == "captioning":
        test_data["image_to_caption"] = {}
    elif task == "classification":
        test_data["image_to_label"] = {}
    elif task == "vqa":
        test_data["image_to_qa_pairs"] = {}
    elif task in ["detection", "grounding"]:
        test_data["image_to_bboxes"] = {}
    
    count = 0
    skipped = 0
    
    # Get list of local image files for matching
    local_image_files = {}
    if os.path.exists(images_dir):
        for root, dirs, files in os.walk(images_dir):
            # Skip hidden dirs and macOS metadata
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__MACOSX']
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    # Store by basename for easy lookup
                    basename = os.path.basename(file)
                    # Remove extension for matching
                    name_without_ext = os.path.splitext(basename)[0]
                    local_image_files[name_without_ext] = file_path
                    # Also store with full basename
                    local_image_files[basename] = file_path
    
    logger.info(f"Found {len(local_image_files)} local image files")
    
    for idx, sample in enumerate(dataset_iterator):
        if num_samples and count >= num_samples:
            break
        
        try:
            # Try to get image_id or construct one
            image_id = None
            if 'image_id' in sample:
                image_id = str(sample['image_id'])
            elif 'id' in sample:
                image_id = str(sample['id'])
            elif 'image_name' in sample:
                image_id = str(sample['image_name'])
            elif 'file_name' in sample:
                image_id = str(sample['file_name'])
            else:
                # Construct from index
                image_id = f"{split}_{idx:05d}"
            
            # Remove extension from image_id for matching
            image_id_base = os.path.splitext(image_id)[0]
            
            # Find corresponding local image file
            image_path = None
            
            # Strategy 1: Direct match by image_id (with or without extension)
            if image_id in local_image_files:
                image_path = local_image_files[image_id]
            elif image_id_base in local_image_files:
                image_path = local_image_files[image_id_base]
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
                    if name in local_image_files:
                        image_path = local_image_files[name]
                        break
                
                # Strategy 3: Index-based matching (fallback)
                if image_path is None:
                    sorted_files = sorted(local_image_files.items())
                    if idx < len(sorted_files):
                        image_path = sorted_files[idx][1]
            
            if image_path and os.path.exists(image_path):
                # Store path mapping
                test_data["id_to_path"][image_id] = image_path
                
                # Extract task-specific metadata
                sample_info = {
                    "image_id": image_id,
                    "image_path": image_path,
                    "dataset_index": idx
                }
                
                # Extract task-specific targets
                if task == "captioning":
                    caption = sample.get('caption', '') or sample.get('description', '') or sample.get('text', '')
                    sample_info["caption"] = caption
                    test_data["image_to_caption"][image_id] = caption
                elif task == "classification":
                    # Try to find label
                    label = None
                    for key in ['label', 'category', 'class', 'target', 'class_id']:
                        if key in sample:
                            label = sample[key]
                            break
                    if label is not None:
                        sample_info["label"] = label
                        test_data["image_to_label"][image_id] = label
                elif task == "vqa":
                    qa_pairs = sample.get('qa_pairs', [])
                    if qa_pairs:
                        sample_info["qa_pairs"] = qa_pairs
                        test_data["image_to_qa_pairs"][image_id] = qa_pairs
                    elif 'question' in sample and 'answer' in sample:
                        qa_pair = [(sample['question'], sample['answer'])]
                        sample_info["qa_pairs"] = qa_pair
                        test_data["image_to_qa_pairs"][image_id] = qa_pair
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
                
                # Add all other fields from dataset
                for key in ['objects', 'attributes', 'relationships', 'question', 'answer', 
                           'bbox', 'bboxes', 'category', 'label', 'caption', 'description', 'text']:
                    if key in sample and key not in sample_info:
                        sample_info[key] = sample[key]
                
                test_data["samples"].append(sample_info)
                
                count += 1
                if count % 100 == 0:
                    logger.info(f"✓ [{count}/{num_samples if num_samples else 'all'}] Processed samples...")
            else:
                skipped += 1
                if skipped <= 10:
                    logger.warning(f"⚠ Skipped sample {idx}: Image not found for image_id={image_id}")
                
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                logger.warning(f"⚠ Error processing sample {idx}: {e}")
            continue
    
    test_data["num_samples"] = count
    
    logger.info(f"✓ Created dataset with {count} samples (skipped {skipped})")
    
    # Save to JSON if requested
    if output_json:
        logger.info(f"Saving to JSON: {output_json}")
        with open(output_json, 'w') as f:
            json.dump(test_data, f, indent=2)
        logger.info(f"✓ Saved to: {output_json}")
    
    return test_data


def prepare_vrsbench_dataset_parallel(
    split: str = "validation",
    task: str = "captioning",
    images_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    hf_dataset_name: str = "xiang709/VRSBench",
    hf_token: Optional[str] = None,
    download_images: bool = True,
    images_url: Optional[str] = None,
    output_json: Optional[str] = None,
    config: Optional[VRSBenchConfig] = None,
    force_download: bool = False,
    num_workers: int = 8
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
    - Optimal worker count: 4-16 threads (default: 8)
    
    Args:
        split: Dataset split ("train" or "validation")
        task: Task type ("classification", "detection", "captioning", "vqa", "grounding")
        images_dir: Directory to store images (default: ./Images_{split})
        num_samples: Limit number of samples (None = all)
        hf_dataset_name: HuggingFace dataset identifier
        hf_token: HuggingFace token (or use HUGGINGFACE_HUB_TOKEN env var)
        download_images: Whether to download images (default: True)
        images_url: Custom URL for images (default: auto-detect from split)
        output_json: Path to save JSON mapping (optional)
        config: Configuration object (optional)
        force_download: Force re-download even if images exist
        num_workers: Number of parallel threads (default: 8, optimal for I/O-bound operations)
    
    Returns:
        Dictionary with:
        - "samples": List of sample dicts with image_path and task-specific metadata
        - Task-specific mappings (e.g., "image_to_caption" for captioning, "image_to_label" for classification)
        - "id_to_path": Dict mapping image_id -> image_path
        - "split": Dataset split used
        - "task": Task type used
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
    
    # Validate task
    if task not in config.SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task}. Choose from {config.SUPPORTED_TASKS}")
    
    logger = StructuredLogger("VRSBenchWorkflowParallel", config)
    metrics = MetricsCollector()
    download_mgr = DownloadManager(config, logger, metrics)
    
    # Set up images directory
    if images_dir is None:
        images_dir = f"./Images_{split}"
    
    os.makedirs(images_dir, exist_ok=True)
    
    # Download images if needed (same logic as original)
    if download_images:
        # Auto-detect image URL based on split
        if images_url is None:
            if split == "validation":
                images_url = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip"
            elif split == "train":
                images_url = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_train.zip"
            else:
                raise ValueError(f"Unknown split: {split}. Use 'train' or 'validation'")
        
        logger.info(f"Downloading images for {split} split...")
        
        # Check if images already exist
        zip_name = os.path.basename(images_url)
        zip_path = os.path.join(images_dir, zip_name)
        
        # Check if images are already extracted
        actual_images_dir = _detect_image_directory(images_dir)
        has_images = False
        if os.path.exists(actual_images_dir):
            image_files = [f for f in os.listdir(actual_images_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            has_images = len(image_files) > 0
        
        if not force_download and has_images:
            logger.info(f"Images already exist in {actual_images_dir}, skipping download")
            images_dir = actual_images_dir
        else:
            # Download zip file
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
        for root, dirs, files in os.walk(images_dir):
            # Skip hidden dirs and macOS metadata
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__MACOSX']
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    # Store by basename for easy lookup
                    basename = os.path.basename(file)
                    # Remove extension for matching
                    name_without_ext = os.path.splitext(basename)[0]
                    local_image_files[name_without_ext] = file_path
                    # Also store with full basename
                    local_image_files[basename] = file_path
    
    logger.info(f"Found {len(local_image_files)} local image files")
    
    # KEY CHANGE: Load non-streaming dataset (loads full dataset into memory)
    logger.info(f"Loading HuggingFace dataset (non-streaming mode): {hf_dataset_name}")
    
    # Set up authentication if token provided
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    
    try:
        # Non-streaming mode - loads full dataset for parallel processing
        hf_dataset = load_dataset(hf_dataset_name, streaming=False)
    except Exception as e:
        logger.error(f"Failed to load HuggingFace dataset: {e}")
        raise
    
    if split not in hf_dataset:
        raise ValueError(f"Split '{split}' not found in dataset. Available: {list(hf_dataset.keys())}")
    
    # Convert to list for parallel processing
    dataset_split = hf_dataset[split]
    all_samples = list(dataset_split)
    
    # Limit samples if requested
    if num_samples:
        all_samples = all_samples[:num_samples]
    
    logger.info(f"Loaded {len(all_samples)} samples, processing in parallel with {num_workers} workers...")
    
    # Initialize task-specific mappings
    test_data = {
        "samples": [],
        "id_to_path": {},
        "split": split,
        "task": task,
        "num_samples": 0
    }
    
    # Add task-specific mapping keys
    if task == "captioning":
        test_data["image_to_caption"] = {}
    elif task == "classification":
        test_data["image_to_label"] = {}
    elif task == "vqa":
        test_data["image_to_qa_pairs"] = {}
    elif task in ["detection", "grounding"]:
        test_data["image_to_bboxes"] = {}
    
    # Thread-safe counters
    count_lock = threading.Lock()
    count = 0
    skipped = 0
    
    def process_single_sample(args):
        """
        Process a single sample - designed for parallel execution.
        
        This function is thread-safe and processes one sample at a time.
        It extracts image_id, finds corresponding image file, and extracts
        task-specific metadata.
        
        Args:
            args: Tuple of (idx, sample) where idx is the dataset index and sample is the sample dict
        
        Returns:
            Dictionary with sample_info, task_data, and id_to_path, or None if processing failed
        """
        idx, sample = args
        
        try:
            # Try to get image_id or construct one
            image_id = None
            if 'image_id' in sample:
                image_id = str(sample['image_id'])
            elif 'id' in sample:
                image_id = str(sample['id'])
            elif 'image_name' in sample:
                image_id = str(sample['image_name'])
            elif 'file_name' in sample:
                image_id = str(sample['file_name'])
            else:
                # Construct from index
                image_id = f"{split}_{idx:05d}"
            
            # Remove extension from image_id for matching
            image_id_base = os.path.splitext(image_id)[0]
            
            # Find corresponding local image file
            image_path = None
            
            # Strategy 1: Direct match by image_id (with or without extension)
            if image_id in local_image_files:
                image_path = local_image_files[image_id]
            elif image_id_base in local_image_files:
                image_path = local_image_files[image_id_base]
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
                    if name in local_image_files:
                        image_path = local_image_files[name]
                        break
                
                # Strategy 3: Index-based matching (fallback)
                if image_path is None:
                    sorted_files = sorted(local_image_files.items())
                    if idx < len(sorted_files):
                        image_path = sorted_files[idx][1]
            
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
    args_list = [(idx, sample) for idx, sample in enumerate(all_samples)]
    
    # Process in parallel using ThreadPoolExecutor
    logger.info(f"Starting parallel processing with {num_workers} workers...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(process_single_sample, args): idx for idx, args in enumerate(args_list)}
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            completed += 1
            result = future.result()
            
            with count_lock:
                if result:
                    test_data["samples"].append(result["sample_info"])
                    test_data["id_to_path"].update(result["id_to_path"])
                    
                    # Update task-specific mappings
                    if "image_to_caption" in result["task_data"]:
                        test_data["image_to_caption"].update(result["task_data"]["image_to_caption"])
                    elif "image_to_label" in result["task_data"]:
                        test_data["image_to_label"].update(result["task_data"]["image_to_label"])
                    elif "image_to_qa_pairs" in result["task_data"]:
                        test_data["image_to_qa_pairs"].update(result["task_data"]["image_to_qa_pairs"])
                    elif "image_to_bboxes" in result["task_data"]:
                        test_data["image_to_bboxes"].update(result["task_data"]["image_to_bboxes"])
                    
                    count += 1
                else:
                    skipped += 1
                
                # Progress logging (every 500 samples)
                if count % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    logger.info(f"✓ [{count}/{len(all_samples)}] Processed samples... ({rate:.1f} samples/sec)")
            
            # Batch progress (every 1000 tasks)
            if completed % 1000 == 0:
                logger.info(f"Completed {completed}/{len(args_list)} tasks...")
    
    test_data["num_samples"] = count
    
    elapsed = time.time() - start_time
    rate = count / elapsed if elapsed > 0 else 0
    logger.info(f"✓ Created dataset with {count} samples (skipped {skipped}) in {elapsed:.2f}s ({rate:.1f} samples/sec)")
    
    # Record metrics
    metrics.record_time("parallel_preparation", elapsed)
    metrics.increment("samples_processed", count)
    metrics.increment("samples_skipped", skipped)
    
    # Save to JSON if requested
    if output_json:
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




