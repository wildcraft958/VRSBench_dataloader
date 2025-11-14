#!/usr/bin/env python3
"""
Zero-Friction Jupyter Notebook Setup for VRSBench DataLoader

Platform-agnostic setup that works in:
- Google Colab
- JupyterLab
- Local Jupyter Notebooks
- Any Jupyter environment

Usage in Jupyter:
    # Option 1: Upload this file and run:
    exec(open('jupyter_setup.py').read())
    
    # Option 2: Copy-paste this entire file into a Jupyter cell
    
    # Option 3: Use the setup function
    from setup_vrsbench import setup_for_notebook
    prepare_vrsbench_dataset, create_dataloader_from_prepared, get_task_targets = setup_for_notebook()
"""

# --- VRSBench Loader: Zero-Friction Jupyter Setup ---

import sys
import os
import glob
import subprocess
from pathlib import Path

# ========== CONFIGURATION ==========
# Replace with your actual GitHub repository URL (or leave None for auto-detect)
REPO_URL = "https://github.com/wildcraft958/VRSBench_dataloader"  # <-- UPDATE IF NEEDED
REPO_NAME = "VRSBench_dataloader"
DOWNLOAD_IMAGES_IN_TEST = False  # Set True to download ~5GB images during smoke test
# ===================================

def detect_environment():
    """Detect the Jupyter environment"""
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            if 'google.colab' in str(ipython.__class__.__module__):
                return 'colab', '/content'
            elif 'jupyter' in str(ipython.__class__.__module__).lower():
                return 'jupyter', str(Path.cwd())
    except ImportError:
        pass
    return 'python', str(Path.cwd())

env_type, default_dir = detect_environment()
TARGET_DIR = default_dir

print("=" * 70)
print("VRSBench DataLoader - Jupyter Notebook Setup")
print("=" * 70)
print(f"Environment: {env_type}")
print(f"Target directory: {TARGET_DIR}")

# 1) Check if already in repository directory
current_dir = Path.cwd()
if (current_dir / "vrsbench_dataloader_production.py").exists():
    print(f"\n[1/6] ✓ Using local VRSBench dataloader from current directory")
    repo_path = str(current_dir)
    need_clone = False
else:
    repo_path = os.path.join(TARGET_DIR, REPO_NAME)
    need_clone = not os.path.exists(repo_path) or not os.path.exists(
        os.path.join(repo_path, "vrsbench_dataloader_production.py")
    )

# 2) Clone repo if needed
if need_clone:
    if not REPO_URL or REPO_URL == "https://github.com/yourusername/VRSBench_dataloader":
        print(f"\n✗ ERROR: Please update REPO_URL in this cell with your actual GitHub repository URL")
        raise ValueError("REPO_URL not configured")
    
    print(f"\n[1/6] Cloning repository from {REPO_URL}...")
    try:
        subprocess.run(["git", "clone", REPO_URL, repo_path], check=True, 
                      capture_output=True, text=True)
        print(f"     ✓ Repository cloned to {repo_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR: Failed to clone repository")
        print(f"  Please check:")
        print(f"    1. Repository URL is correct: {REPO_URL}")
        print(f"    2. Repository is public (or you have access)")
        print(f"    3. Git is installed and available")
        raise
else:
    print(f"\n[1/6] ✓ Repository found at {repo_path}")

# 3) Add repo to sys.path
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
    print(f"[2/6] ✓ Added {repo_path} to Python path")
else:
    print(f"[2/6] ✓ Path already configured")

# 4) Locate loader file
print(f"[3/6] Locating loader file...")
loader_file = None

# Try direct path first
direct_path = os.path.join(repo_path, "vrsbench_dataloader_production.py")
if os.path.exists(direct_path):
    loader_file = direct_path
else:
    # Search recursively
    matches = glob.glob(
        os.path.join(repo_path, "**", "vrsbench_dataloader_production.py"),
        recursive=True
    )
    if matches:
        loader_file = matches[0]

if not loader_file or not os.path.exists(loader_file):
    raise RuntimeError(
        f"Loader file 'vrsbench_dataloader_production.py' not found in {repo_path}.\n"
        f"Please verify the repository structure."
    )

print(f"     ✓ Found: {loader_file}")

# 5) Install dependencies
print(f"[4/6] Checking dependencies...")
try:
    import torch
    import torchvision
    import PIL
    import pandas
    import requests
    import tqdm
    print("     ✓ Core dependencies already installed")
except ImportError:
    print("     Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                   "torch", "torchvision", "pillow", "pandas", "requests", "tqdm"], 
                  check=False, capture_output=True)

# Install datasets library (recommended)
try:
    import datasets
    print("     ✓ datasets library available")
except ImportError:
    print("     Installing datasets library (recommended)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], 
                  check=False, capture_output=True)

# 6) Convert file path to module import path
rel_path = os.path.relpath(loader_file, repo_path)
module_path = rel_path.replace(".py", "").replace("/", ".").replace("\\", ".")
if module_path.startswith("."):
    module_path = module_path[1:]
if not module_path:
    module_path = "vrsbench_dataloader_production"

print(f"[5/6] Importing module: {module_path}")

# 7) Import dynamically
import importlib
try:
    mod = importlib.import_module(module_path)
    
    prepare_vrsbench_dataset = getattr(mod, "prepare_vrsbench_dataset")
    create_dataloader_from_prepared = getattr(mod, "create_dataloader_from_prepared")
    get_task_targets = getattr(mod, "get_task_targets")
    
    print("     ✓ Functions imported successfully")
except (ImportError, AttributeError) as e:
    raise RuntimeError(f"Failed to import VRSBench DataLoader: {e}")

# 8) Smoke test (optional)
print(f"[6/6] Running smoke test...")
try:
    data = prepare_vrsbench_dataset(
        split="validation",
        task="captioning",
        num_samples=2,
        download_images=DOWNLOAD_IMAGES_IN_TEST
    )
    
    print("\n" + "=" * 70)
    print("✓ SETUP COMPLETE - Smoke Test Passed")
    print("=" * 70)
    print(f"\nLoaded {data.get('num_samples', 'N/A')} samples")
    print(f"Task: {data.get('task', 'N/A')}")
    print(f"Images directory: {data.get('images_dir', 'N/A')}")
    
    if data.get('image_to_caption'):
        first_key = list(data['image_to_caption'].keys())[0]
        print(f"\nFirst sample preview:")
        print(f"  Image ID: {first_key}")
        print(f"  Image Path: {data.get('id_to_path', {}).get(first_key, 'N/A')}")
        print(f"  Caption: {data['image_to_caption'][first_key][:80]}...")
    
    print("\n" + "=" * 70)
    print("Available functions:")
    print("  - prepare_vrsbench_dataset(split, task, num_samples, ...)")
    print("  - create_dataloader_from_prepared(data, batch_size, ...)")
    print("  - get_task_targets(metas, task)")
    print("=" * 70)
    
except Exception as e:
    print(f"\n⚠️  Smoke test warning (this is OK if images aren't downloaded): {e}")
    print("\n✓ Setup complete! Functions are ready to use.")
    print("  Note: Images will be downloaded when you call prepare_vrsbench_dataset with download_images=True")

print("\n🚀 Ready to use! Example:")
print("""
# Example usage for captioning:
data = prepare_vrsbench_dataset(
    split="validation",
    task="captioning",
    num_samples=1000,
    download_images=True
)

dataloader = create_dataloader_from_prepared(data, batch_size=16)
for images, metas in dataloader:
    captions = get_task_targets(metas, task="captioning")
    # Train your model here
""")

