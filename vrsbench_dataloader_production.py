"""
vrsbench_dataloader_production.py

Production-ready VRSBench DataLoader with comprehensive logging, multi-task support,
and robust error handling for satellite imagery tasks (classification, detection, 
captioning, VQA, grounding).

Author: Animesh Raj
Date: 2025-01-13
Version: 2.0.0

Features:
- Structured JSON logging with rotation
- Multi-task support (classification, detection, captioning, VQA, grounding)
- Region-level extraction for high-res satellite imagery
- Automatic retry with exponential backoff
- Metrics collection (load times, error rates, cache hits)
- HuggingFace fallback strategies
- Production-ready error handling
"""

from typing import Optional, Callable, Iterator, Dict, Any, Tuple, List, Union
import os
import sys
import time
import json
import zipfile
import hashlib
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
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

# Optional dependencies
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
    """Central configuration for VRSBench dataloader"""

    # Dataset URLs (direct HF access)
    IMAGES_URL: str = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/images.zip"
    ANNOTATIONS_TRAIN_URL: str = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Annotations_train.zip"
    ANNOTATIONS_VAL_URL: str = "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Annotations_val.zip"

    # Download settings
    MAX_RETRIES: int = 5
    BACKOFF_FACTOR: float = 1.5
    CHUNK_SIZE: int = 16384  # 16KB chunks
    TIMEOUT: int = 60

    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = "./logs"
    LOG_FILE: str = "vrsbench_dataloader.log"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    JSON_LOGS: bool = True

    # Cache settings
    CACHE_DIR: str = "./hf_cache"
    VERIFY_CACHE: bool = True

    # Image processing
    DEFAULT_IMAGE_SIZE: Tuple[int, int] = (256, 256)
    REGION_PADDING: int = 10

    # Performance
    NUM_WORKERS: int = 4
    BATCH_SIZE: int = 16
    PIN_MEMORY: bool = True
    PREFETCH_FACTOR: int = 2

    # Task-specific settings
    SUPPORTED_TASKS: List[str] = None

    def __post_init__(self):
        if self.SUPPORTED_TASKS is None:
            self.SUPPORTED_TASKS = ["classification", "detection", "captioning", "vqa", "grounding"]

        # Create directories
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)

# ========================= Logging Setup =========================

class StructuredLogger:
    """Production-ready structured logger with JSON support"""

    def __init__(self, name: str, config: VRSBenchConfig):
        self.config = config
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
        self.logger.propagate = False

        # Remove existing handlers
        self.logger.handlers.clear()

        # Console handler (human-readable or JSON)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if config.JSON_LOGS:
            # For JSON logs: output raw JSON without additional formatting
            console_format = logging.Formatter('%(message)s')
        else:
            # For plain text: human-readable with timestamp and level
            console_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler with rotation (JSON or plain)
        log_path = Path(config.LOG_DIR) / config.LOG_FILE
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT
        )
        file_handler.setLevel(logging.DEBUG)

        if config.JSON_LOGS:
            file_format = logging.Formatter('%(message)s')  # Raw JSON
        else:
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

    def _format_json(self, level: str, message: str, **kwargs) -> str:
        """Format log as JSON"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "logger": self.logger.name,
            **kwargs
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
    """Collect and report dataloader metrics"""

    def __init__(self):
        self.metrics = defaultdict(int)
        self.timings = defaultdict(list)
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
    """Manages robust file downloads with caching and verification"""

    def __init__(self, config: VRSBenchConfig, logger: StructuredLogger, metrics: MetricsCollector):
        self.config = config
        self.logger = logger
        self.metrics = metrics

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with HF authentication if available"""
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        headers = {"User-Agent": "vrsbench-production-loader/2.0"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
            self.logger.debug("Using HuggingFace authentication token")
        else:
            self.logger.warning("No HF token found. Downloads may be rate-limited. Set HUGGINGFACE_HUB_TOKEN env var.")
        return headers

    def _verify_file(self, path: str, expected_size: Optional[int] = None) -> bool:
        """Verify downloaded file integrity"""
        if not os.path.exists(path):
            return False

        file_size = os.path.getsize(path)

        # Check minimum size
        if file_size < 1024:  # Less than 1KB is suspicious
            self.logger.warning(f"File {path} is suspiciously small: {file_size} bytes")
            return False

        # Check expected size if provided
        if expected_size and abs(file_size - expected_size) > 1024:
            self.logger.warning(f"File size mismatch: expected ~{expected_size}, got {file_size}")
            return False

        # Check if it's a zip file
        if path.endswith('.zip'):
            try:
                with zipfile.ZipFile(path, 'r') as zf:
                    zf.namelist()
                return True
            except zipfile.BadZipFile:
                self.logger.error(f"Corrupted zip file: {path}")
                return False

        return True

    def download_with_retries(self, url: str, output_path: str, force: bool = False) -> str:
        """
        Download file with automatic retries, progress bar, and caching

        Args:
            url: Source URL
            output_path: Destination path
            force: Force re-download even if cached

        Returns:
            Path to downloaded file

        Raises:
            RuntimeError: If download fails after all retries
        """
        start_time = time.time()

        # Check cache
        if not force and os.path.exists(output_path):
            if self.config.VERIFY_CACHE and self._verify_file(output_path):
                self.logger.info(f"Using cached file: {output_path}")
                self.metrics.increment("cache_hits")
                return output_path
            elif not self.config.VERIFY_CACHE:
                self.logger.info(f"Using cached file (unverified): {output_path}")
                self.metrics.increment("cache_hits")
                return output_path
            else:
                self.logger.warning(f"Cached file invalid, re-downloading: {output_path}")
                os.remove(output_path)

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

# ========================= Annotation Processing =========================

class AnnotationProcessor:
    """Process and parse VRSBench annotations"""

    def __init__(self, config: VRSBenchConfig, logger: StructuredLogger, metrics: MetricsCollector):
        self.config = config
        self.logger = logger
        self.metrics = metrics

    def extract_from_zip(self, zip_path: str, extract_to: Optional[str] = None) -> List[str]:
        """Extract annotation files from zip"""
        extract_to = extract_to or os.path.dirname(zip_path)
        os.makedirs(extract_to, exist_ok=True)

        found_files = []

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                all_files = zf.namelist()
                self.logger.debug(f"Zip contains {len(all_files)} files")

                # Look for annotation files
                for name in all_files:
                    if name.lower().endswith(('.json', '.jsonl')):
                        out_path = os.path.join(extract_to, os.path.basename(name))
                        with open(out_path, 'wb') as out:
                            out.write(zf.read(name))
                        found_files.append(out_path)
                        self.logger.info(f"Extracted: {out_path}")

                if not found_files:
                    raise RuntimeError(f"No JSON/JSONL files found in {zip_path}")

                self.metrics.increment("annotations_extracted", len(found_files))
                return found_files

        except zipfile.BadZipFile as e:
            self.metrics.record_error("bad_zip")
            raise RuntimeError(f"Invalid zip file: {zip_path}") from e

    def load_jsonl(self, jsonl_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load JSONL file into DataFrame with error handling"""
        rows = []
        skipped = 0

        self.logger.info(f"Loading annotations from {jsonl_path}")

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    skipped += 1
                    if skipped <= 5:  # Log first few errors
                        self.logger.warning(f"Skipped malformed JSON at line {i+1}: {e}")

        if skipped > 0:
            self.metrics.increment("jsonl_parse_errors", skipped)
            self.logger.warning(f"Skipped {skipped} malformed JSON lines")

        df = pd.DataFrame(rows)
        self.logger.info(f"Loaded {len(df)} annotation records")
        self.metrics.increment("annotations_loaded", len(df))

        return df

    def coerce_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Safely coerce DataFrame columns to avoid Arrow casting errors"""
        df_out = df.copy()

        for col in df_out.columns:
            if pd.api.types.is_object_dtype(df_out[col]):
                non_null = df_out[col].dropna()
                if non_null.empty:
                    continue

                # Try numeric conversion
                try:
                    numeric = pd.to_numeric(df_out[col], errors='coerce')
                    if numeric.notna().sum() > len(df_out) * 0.9:  # >90% numeric
                        df_out[col] = numeric
                        continue
                except Exception:
                    pass

                # Default to string
                df_out[col] = df_out[col].astype(str)

        return df_out

    def to_dataset(self, jsonl_path: str, max_rows: Optional[int] = None) -> Union[Any, List[Dict]]:
        """Convert JSONL to HF Dataset or list of dicts"""
        df = self.load_jsonl(jsonl_path, max_rows=max_rows)

        if df.empty:
            raise RuntimeError(f"No valid records in {jsonl_path}")

        df_safe = self.coerce_columns(df)

        if HAS_DATASETS and Dataset is not None:
            self.logger.info("Creating HuggingFace Dataset")
            return Dataset.from_pandas(df_safe, preserve_index=False)
        else:
            self.logger.warning("datasets library not available, using list fallback")
            return df_safe.to_dict(orient='records')

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

# ========================= Main Dataset Class =========================

class VRSBenchDataset(IterableDataset):
    """
    Production VRSBench Dataset with multi-task support

    Supports: classification, detection, captioning, VQA, grounding
    """

    def __init__(
        self,
        images_dir: str,
        task: str = "classification",
        annotations: Optional[Any] = None,
        annotations_jsonl: Optional[str] = None,
        annotations_url: Optional[str] = None,
        split: str = "validation",
        image_key: Optional[str] = None,
        transform: Optional[Callable] = None,
        sample_size: Optional[int] = None,
        expand_multi_annotations: bool = False,
        region_based: bool = False,
        config: Optional[VRSBenchConfig] = None,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None
    ):
        """
        Args:
            images_dir: Directory containing images
            task: Task type (classification, detection, captioning, vqa, grounding)
            annotations: Pre-loaded annotations (HF Dataset or list of dicts)
            annotations_jsonl: Path to local JSONL file
            annotations_url: URL to download annotations zip
            split: Dataset split (train/validation/test)
            image_key: Key for image reference in annotations
            transform: Image transforms
            sample_size: Limit number of samples
            expand_multi_annotations: Expand records with multiple annotations
            region_based: Extract image regions using bboxes (for grounding/VQA)
            config: Configuration object
            logger: Logger instance
            metrics: Metrics collector
        """
        super().__init__()

        self.images_dir = images_dir
        self.task = task
        self.annotations = annotations
        self.annotations_jsonl = annotations_jsonl
        self.annotations_url = annotations_url
        self.split = split
        self.image_key = image_key
        self.transform = transform
        self.sample_size = sample_size
        self.expand_multi_annotations = expand_multi_annotations
        self.region_based = region_based

        self.config = config or VRSBenchConfig()
        self.logger = logger or StructuredLogger("VRSBenchDataset", self.config)
        self.metrics = metrics or MetricsCollector()

        # Validate task
        if task not in self.config.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task: {task}. Choose from {self.config.SUPPORTED_TASKS}")

        self.logger.info(f"Initializing VRSBench dataset for task: {task}")

        # Setup annotation processors
        self.download_mgr = DownloadManager(self.config, self.logger, self.metrics)
        self.ann_processor = AnnotationProcessor(self.config, self.logger, self.metrics)
        self.task_processor = TaskProcessor()

    def _find_image_path(self, image_ref: Any) -> Optional[str]:
        """Robustly find image file path from annotation reference"""
        if image_ref is None:
            return None

        # Handle various formats
        if isinstance(image_ref, (list, tuple)):
            image_ref = image_ref[0]

        if isinstance(image_ref, dict):
            for key in ['path', 'filename', 'file_name', 'id', 'image']:
                if key in image_ref:
                    image_ref = image_ref[key]
                    break
            else:
                image_ref = str(image_ref)

        image_ref = str(image_ref)

        # Try absolute path
        if os.path.isabs(image_ref) and os.path.exists(image_ref):
            return image_ref

        # Try relative to images_dir
        candidate = os.path.join(self.images_dir, image_ref)
        if os.path.exists(candidate):
            return candidate

        # Try basename search
        basename = os.path.basename(image_ref)
        if os.path.exists(self.images_dir):
            for root, _, files in os.walk(self.images_dir):
                if basename in files:
                    return os.path.join(root, basename)

        return None

    def _normalize_bbox(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        Normalize bbox to [x, y, w, h] in pixels.
        Handles:
        - Normalized coords (0..1) -> convert to pixels
        - Corner format [x1,y1,x2,y2] -> convert to [x,y,w,h]
        - Already pixel coords [x,y,w,h] -> return as-is
        """
        if not bbox or len(bbox) < 4:
            return bbox

        x1, y1, x2_or_w, y2_or_h = bbox[:4]

        # Detect if normalized (all values <= 1.0)
        if all(v <= 1.0 for v in [x1, y1, x2_or_w, y2_or_h]):
            # Normalized coords - could be [x1,y1,x2,y2] or [x,y,w,h]
            # Check if x2_or_w + x1 > 1.0 (indicating it's width, not x2)
            if x2_or_w < x1 or y2_or_h < y1:
                # It's [x,y,w,h] in normalized form
                x = int(x1 * img_width)
                y = int(y1 * img_height)
                w = int(x2_or_w * img_width)
                h = int(y2_or_h * img_height)
            else:
                # It's [x1,y1,x2,y2] in normalized form
                x = int(x1 * img_width)
                y = int(y1 * img_height)
                w = int((x2_or_w - x1) * img_width)
                h = int((y2_or_h - y1) * img_height)
            return [x, y, w, h]

        # Check if it's corner format [x1,y1,x2,y2] in pixels
        # (x2 > x1 and x2 > img_width/2 suggests corner format)
        if x2_or_w > x1 and y2_or_h > y1 and x2_or_w > x1 + 10:
            # Convert corner to xywh
            x = int(x1)
            y = int(y1)
            w = int(x2_or_w - x1)
            h = int(y2_or_h - y1)
            return [x, y, w, h]

        # Already in [x,y,w,h] pixel format
        return [int(x1), int(y1), int(x2_or_w), int(y2_or_h)]

    def _guess_image_key(self, item: Dict[str, Any]) -> Optional[str]:
        """Guess image key from annotation dict"""
        candidates = [
            'image', 'image_id', 'image_name', 'file_name', 
            'img', 'img_name', 'filename', 'img_id', 'image_path'
        ]

        for key in candidates:
            if key in item:
                return key

        # Look for keys with image extensions
        for key, value in item.items():
            if isinstance(value, str) and any(
                value.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
            ):
                return key

        return None

    def _load_annotations(self) -> Iterator[Dict[str, Any]]:
        """Load annotations from various sources"""
        # Priority: annotations object -> local jsonl -> download URL

        if self.annotations is not None:
            self.logger.info("Using pre-loaded annotations")
            if hasattr(self.annotations, '__iter__') and not isinstance(self.annotations, dict):
                for item in self.annotations:
                    yield item
                return

        if self.annotations_jsonl is not None:
            self.logger.info(f"Loading from local JSONL: {self.annotations_jsonl}")
            with open(self.annotations_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
            return

        if self.annotations_url is not None:
            self.logger.info(f"Downloading annotations from: {self.annotations_url}")

            # Download zip
            zip_name = os.path.basename(self.annotations_url)
            zip_path = os.path.join(self.config.CACHE_DIR, zip_name)
            self.download_mgr.download_with_retries(self.annotations_url, zip_path)

            # Extract and load
            extracted = self.ann_processor.extract_from_zip(zip_path)
            if extracted:
                dataset = self.ann_processor.to_dataset(extracted[0])

                if isinstance(dataset, list):
                    for item in dataset:
                        yield item
                else:
                    for item in dataset:
                        yield item
            return

        raise RuntimeError("No annotation source provided")

    def _expand_annotations(self, item: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Expand items with multiple annotations (for VQA/grounding)"""
        if not self.expand_multi_annotations:
            yield item
            return

        # Expand QA pairs
        if self.task == 'vqa' and 'qa_pairs' in item:
            qa_list = item['qa_pairs']
            if isinstance(qa_list, list):
                for qa in qa_list:
                    yield {
                        **item,
                        'question': qa.get('question', ''),
                        'answer': qa.get('answer', ''),
                        '_task': 'vqa'
                    }
                return

        # Expand object references
        if self.task in ['grounding', 'detection'] and 'objects' in item:
            obj_list = item['objects']
            if isinstance(obj_list, list):
                for obj in obj_list:
                    yield {
                        **item,
                        'bbox': obj.get('bbox', []),
                        'category': obj.get('category', ''),
                        '_task': self.task
                    }
                return

        yield item

    def __iter__(self):
        """Iterate over dataset yielding (image_tensor, metadata_dict)"""
        emitted = 0
        skipped = 0

        for raw_item in self._load_annotations():
            # Filter by split
            if 'split' in raw_item and raw_item['split'] != self.split:
                continue

            # Expand multi-annotations if needed
            for item in self._expand_annotations(raw_item):
                if self.sample_size and emitted >= self.sample_size:
                    self.logger.info(f"Reached sample limit: {self.sample_size}")
                    return

                # Find image
                image_key = self.image_key or self._guess_image_key(item)
                if image_key is None:
                    skipped += 1
                    if skipped <= 10:
                        self.logger.warning(f"Could not find image key in item: {list(item.keys())[:5]}")
                    continue

                image_ref = item.get(image_key)
                image_path = self._find_image_path(image_ref)

                if image_path is None:
                    skipped += 1
                    if skipped <= 10:
                        # Enhanced error message with debugging info
                        attempted_path = os.path.join(self.images_dir, str(image_ref)) if self.images_dir else str(image_ref)
                        self.logger.warning(
                            f"Image not found: {image_ref}",
                            image_key=image_key,
                            images_dir=self.images_dir,
                            attempted_path=attempted_path,
                            image_exists=os.path.exists(self.images_dir) if self.images_dir else False
                        )
                    continue

                # Load image
                try:
                    start_time = time.time()
                    with Image.open(image_path) as img:
                        img = img.convert('RGB')
                        img_width, img_height = img.size

                        # Normalize bboxes if needed (convert normalized coords to pixels)
                        if 'bbox' in item:
                            item['bbox'] = self._normalize_bbox(item['bbox'], img_width, img_height)
                        if 'bboxes' in item:
                            item['bboxes'] = [self._normalize_bbox(b, img_width, img_height) for b in item['bboxes']]
                        if 'objects' in item and isinstance(item['objects'], list):
                            for obj in item['objects']:
                                if 'bbox' in obj:
                                    obj['bbox'] = self._normalize_bbox(obj['bbox'], img_width, img_height)

                        # Extract region if needed
                        if self.region_based and 'bbox' in item:
                            bbox = item['bbox']
                            if bbox and len(bbox) >= 4:
                                img = self.task_processor.extract_region_from_bbox(
                                    img, bbox, padding=self.config.REGION_PADDING
                                )

                        # Apply transforms
                        if self.transform:
                            img_tensor = self.transform(img)
                        else:
                            img_tensor = transforms.ToTensor()(img)

                    # Add resolved image path to metadata for downstream use
                    item['_image_path'] = image_path
                    item['_image_size'] = (img_width, img_height)

                    load_time = time.time() - start_time
                    self.metrics.record_time("image_load", load_time)
                    self.metrics.increment("images_loaded")

                    # Inject resolved image path into metadata for downstream use
                    item['_image_path'] = image_path

                except Exception as e:
                    self.metrics.record_error("image_load_error")
                    if skipped <= 10:
                        self.logger.error(f"Failed to load image {image_path}: {e}")
                    skipped += 1
                    continue

                emitted += 1
                yield img_tensor, item

        self.logger.info(f"Dataset iteration complete: {emitted} yielded, {skipped} skipped")
        if skipped > 10:
            self.logger.warning(f"Total skipped items: {skipped}")

# ========================= DataLoader Factory =========================

def create_vrsbench_dataloader(
    images_dir: str,
    task: str = "classification",
    annotations: Optional[Any] = None,
    annotations_jsonl: Optional[str] = None,
    annotations_url: Optional[str] = None,
    split: str = "validation",
    image_key: Optional[str] = None,
    transform: Optional[Callable] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    sample_size: Optional[int] = None,
    expand_multi_annotations: bool = False,
    region_based: bool = False,
    download_images: bool = False,
    images_url: Optional[str] = None,
    config: Optional[VRSBenchConfig] = None,
    return_metrics: bool = False
) -> Union[DataLoader, Tuple[DataLoader, MetricsCollector]]:
    """
    Create production-ready VRSBench DataLoader

    Args:
        images_dir: Directory containing images
        task: Task type (classification, detection, captioning, vqa, grounding)
        annotations: Pre-loaded annotations
        annotations_jsonl: Path to local JSONL
        annotations_url: URL to download annotations
        split: Dataset split
        image_key: Key for image in annotations
        transform: Image transforms (defaults to 256x256 resize)
        batch_size: Batch size
        num_workers: Number of worker processes
        sample_size: Limit number of samples
        expand_multi_annotations: Expand multi-annotation items
        region_based: Extract regions for grounding tasks
        download_images: Download images if not present
        images_url: URL for image zip
        config: Configuration object
        return_metrics: Return metrics collector along with dataloader

    Returns:
        DataLoader or (DataLoader, MetricsCollector)
    """

    config = config or VRSBenchConfig()
    logger = StructuredLogger("DataLoaderFactory", config)
    metrics = MetricsCollector()

    logger.info(f"Creating DataLoader for task={task}, split={split}, batch_size={batch_size}")

    # Download images if requested
    if download_images:
        if not images_url:
            images_url = config.IMAGES_URL

        logger.info(f"Downloading images to {images_dir}")
        os.makedirs(images_dir, exist_ok=True)

        download_mgr = DownloadManager(config, logger, metrics)
        zip_path = os.path.join(images_dir, os.path.basename(images_url))
        download_mgr.download_with_retries(images_url, zip_path)

        # Extract
        logger.info("Extracting images...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(images_dir)
        logger.info(f"Images extracted to {images_dir}")

    # Default transform
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(config.DEFAULT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Create dataset
    dataset = VRSBenchDataset(
        images_dir=images_dir,
        task=task,
        annotations=annotations,
        annotations_jsonl=annotations_jsonl,
        annotations_url=annotations_url,
        split=split,
        image_key=image_key,
        transform=transform,
        sample_size=sample_size,
        expand_multi_annotations=expand_multi_annotations,
        region_based=region_based,
        config=config,
        logger=logger,
        metrics=metrics
    )

    # Collate function
    def collate_fn(batch):
        images = torch.stack([b[0] for b in batch])
        metas = [b[1] for b in batch]
        return images, metas

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=config.PIN_MEMORY,
        prefetch_factor=config.PREFETCH_FACTOR if num_workers > 0 else None
    )

    logger.info("DataLoader created successfully")

    if return_metrics:
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

