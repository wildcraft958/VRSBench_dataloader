#!/usr/bin/env python3
"""
VRSBench DataLoader Setup Utility

Simplifies setup and import of VRSBench DataLoader in any environment,
especially useful for Google Colab notebooks.

Usage:
    # In Colab or any Python environment:
    exec(open('setup_vrsbench.py').read())
    
    # Or import and use:
    from setup_vrsbench import setup_vrsbench_loader
    prepare_vrsbench_dataset, create_dataloader_from_prepared, get_task_targets = setup_vrsbench_loader()
"""

import sys
import os
import glob
import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Optional


def detect_jupyter_environment():
    """
    Detect the Jupyter environment (Colab, JupyterLab, local Jupyter, etc.)
    
    Returns:
        dict with environment info: {'type': str, 'notebook_dir': str, 'is_colab': bool}
    """
    env_info = {
        'type': 'unknown',
        'notebook_dir': str(Path.cwd()),
        'is_colab': False,
        'is_jupyter': False
    }
    
    # Check if running in Jupyter
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            env_info['is_jupyter'] = True
            
            # Check for Colab
            if 'google.colab' in str(ipython.__class__.__module__):
                env_info['type'] = 'colab'
                env_info['is_colab'] = True
                env_info['notebook_dir'] = '/content'
            # Check for JupyterLab
            elif 'jupyter_client' in str(ipython.__class__.__module__):
                env_info['type'] = 'jupyterlab'
                env_info['notebook_dir'] = str(Path.cwd())
            # Check for classic Jupyter
            elif 'IPython' in str(ipython.__class__):
                env_info['type'] = 'jupyter'
                env_info['notebook_dir'] = str(Path.cwd())
    except ImportError:
        pass
    
    return env_info


def setup_vrsbench_loader(
    repo_url: Optional[str] = None,
    repo_name: str = "VRSBench_dataloader",
    target_dir: Optional[str] = None,
    auto_install_deps: bool = True,
    run_smoke_test: bool = False
) -> Tuple:
    """
    Set up and import VRSBench DataLoader from GitHub repository.
    
    Platform-agnostic: Works in Colab, JupyterLab, local Jupyter, or any Python environment.
    
    Args:
        repo_url: GitHub repository URL (default: auto-detect from current dir or use placeholder)
        repo_name: Name of the repository folder (default: "VRSBench_dataloader")
        target_dir: Directory to clone into (default: auto-detected based on environment)
        auto_install_deps: Automatically install dependencies (default: True)
        run_smoke_test: Run a quick smoke test after setup (default: False)
    
    Returns:
        Tuple of (prepare_vrsbench_dataset, create_dataloader_from_prepared, get_task_targets)
    
    Example:
        >>> prepare_vrsbench_dataset, create_dataloader_from_prepared, get_task_targets = setup_vrsbench_loader()
        >>> data = prepare_vrsbench_dataset(split="validation", task="captioning", num_samples=10)
    """
    # Detect environment
    env = detect_jupyter_environment()
    
    # Set defaults based on environment
    if target_dir is None:
        if env['is_colab']:
            target_dir = "/content"
        elif env['is_jupyter']:
            # For Jupyter notebooks, use current working directory
            target_dir = str(Path.cwd())
        else:
            # For regular Python scripts
            target_dir = str(Path.cwd())
    
    if repo_url is None:
        # Try to detect from current directory (if already cloned)
        current_dir = Path.cwd()
        if (current_dir / "vrsbench_dataloader_production.py").exists():
            print("[INFO] Using local VRSBench dataloader from current directory")
            repo_path = str(current_dir)
            repo_url = None  # Not needed for local
        else:
            # Check if repo is already cloned in target_dir
            potential_repo_path = os.path.join(target_dir, repo_name)
            if os.path.exists(potential_repo_path) and os.path.exists(
                os.path.join(potential_repo_path, "vrsbench_dataloader_production.py")
            ):
                print(f"[INFO] Found existing repository at {potential_repo_path}")
                repo_path = potential_repo_path
                repo_url = None  # Not needed, already exists
            else:
                # Default repository URL
                repo_url = "https://github.com/wildcraft958/VRSBench_dataloader"
                print(f"[INFO] Using repository: {repo_url}")
                repo_path = os.path.join(target_dir, repo_name)
    else:
        repo_path = os.path.join(target_dir, repo_name)
    
    # Print environment info
    print(f"[INFO] Environment: {env['type']} (Jupyter: {env['is_jupyter']})")
    print(f"[INFO] Working directory: {target_dir}")
    
    # Step 1: Clone repo if needed and not using local
    if repo_url is not None and not os.path.exists(repo_path):
        print(f"[INFO] Cloning repository from {repo_url} into {repo_path}...")
        try:
            subprocess.run(["git", "clone", repo_url, repo_path], check=True, 
                         capture_output=True, text=True)
            print(f"[INFO] ✓ Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to clone repository: {e}\n"
                f"Please ensure:\n"
                f"  1. Repository URL is correct: {repo_url}\n"
                f"  2. Repository is public or you have access\n"
                f"  3. Git is installed and available"
            )
    elif os.path.exists(repo_path):
        print(f"[INFO] Repository already exists at {repo_path}")
    
    # Step 2: Add to sys.path
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        print(f"[INFO] ✓ Added {repo_path} to sys.path")
    
    # Step 3: Locate loader file
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
    
    print(f"[INFO] ✓ Found loader file: {loader_file}")
    
    # Step 4: Install dependencies if requested
    if auto_install_deps:
        print("[INFO] Installing dependencies...")
        try:
            import torch
            import torchvision
            import PIL
            import pandas
            import requests
            import tqdm
            try:
                import datasets
            except ImportError:
                print("[INFO] Installing datasets library (recommended)...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"], 
                             check=False)
            print("[INFO] ✓ Core dependencies already installed")
        except ImportError as e:
            missing = str(e).split()[-1]
            print(f"[INFO] Installing missing dependency: {missing}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", 
                          "torch", "torchvision", "pillow", "pandas", "requests", "tqdm", "datasets"],
                         check=False)
            print("[INFO] ✓ Dependencies installed")
    
    # Step 5: Import module
    rel_path = os.path.relpath(loader_file, repo_path)
    module_path = rel_path.replace(".py", "").replace(os.sep, ".")
    
    # Handle case where module is at root
    if module_path.startswith("."):
        module_path = module_path[1:]
    if not module_path:
        module_path = "vrsbench_dataloader_production"
    
    print(f"[INFO] Importing module: {module_path}")
    
    try:
        import importlib
        mod = importlib.import_module(module_path)
        
        # Extract functions
        prepare_vrsbench_dataset = getattr(mod, "prepare_vrsbench_dataset")
        create_dataloader_from_prepared = getattr(mod, "create_dataloader_from_prepared")
        get_task_targets = getattr(mod, "get_task_targets")
        
        print("[INFO] ✓ VRSBench DataLoader imported successfully")
        
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"Failed to import VRSBench DataLoader: {e}\n"
            f"Module path: {module_path}\n"
            f"Loader file: {loader_file}"
        )
    
    # Step 6: Optional smoke test
    if run_smoke_test:
        print("\n[INFO] Running smoke test...")
        try:
            data = prepare_vrsbench_dataset(
                split="validation",
                task="captioning",
                num_samples=2,
                download_images=False  # Skip download for quick test
            )
            print("\n=== ✓ Smoke Test Passed ===")
            print(f"  Samples loaded: {data.get('num_samples', 'N/A')}")
            print(f"  Task: {data.get('task', 'N/A')}")
            if data.get('image_to_caption'):
                first_key = list(data['image_to_caption'].keys())[0]
                print(f"  First caption preview: {data['image_to_caption'][first_key][:60]}...")
        except Exception as e:
            print(f"[WARNING] Smoke test failed (this is OK if images aren't downloaded): {e}")
    
    return prepare_vrsbench_dataset, create_dataloader_from_prepared, get_task_targets


# Convenience function for Jupyter notebooks (Colab, JupyterLab, etc.)
def setup_for_notebook(repo_url: Optional[str] = None, run_test: bool = True):
    """
    Convenience function for Jupyter notebook environments (Colab, JupyterLab, local Jupyter).
    Auto-detects the environment and sets up accordingly.
    
    Args:
        repo_url: Your GitHub repository URL (default: auto-detected)
        run_test: Run smoke test after setup (default: True)
    
    Example:
        >>> setup_for_notebook("https://github.com/yourusername/VRSBench_dataloader")
        >>> # Or let it auto-detect:
        >>> setup_for_notebook()
    """
    env = detect_jupyter_environment()
    target_dir = env['notebook_dir']
    
    return setup_vrsbench_loader(
        repo_url=repo_url,
        target_dir=target_dir,
        auto_install_deps=True,
        run_smoke_test=run_test
    )


# Alias for backward compatibility
setup_for_colab = setup_for_notebook


# Auto-setup if run directly
if __name__ == "__main__":
    print("=" * 70)
    print("VRSBench DataLoader Setup Utility")
    print("=" * 70)
    print("\nThis utility will:")
    print("  1. Clone the repository (if needed)")
    print("  2. Add it to Python path")
    print("  3. Install dependencies")
    print("  4. Import the loader functions")
    print("\nFor Jupyter notebooks (Colab, JupyterLab, etc.):")
    print("  setup_for_notebook()  # Auto-detects environment")
    print("\nFor any environment:")
    print("  setup_vrsbench_loader(repo_url='YOUR_REPO_URL')")
    print("\n" + "=" * 70)
    
    # Example usage
    try:
        env = detect_jupyter_environment()
        if env['is_jupyter']:
            print(f"\nDetected {env['type']} environment - using notebook setup")
            prepare_vrsbench_dataset, create_dataloader_from_prepared, get_task_targets = setup_for_notebook(
                run_test=False
            )
        else:
            prepare_vrsbench_dataset, create_dataloader_from_prepared, get_task_targets = setup_vrsbench_loader(
                run_smoke_test=False
            )
        print("\n✓ Setup complete! Functions are ready to use.")
        print("\nAvailable functions:")
        print("  - prepare_vrsbench_dataset")
        print("  - create_dataloader_from_prepared")
        print("  - get_task_targets")
    except Exception as e:
        print(f"\n✗ Setup failed: {e}")
        print("\nPlease provide a valid repo_url parameter or ensure repository is accessible.")

