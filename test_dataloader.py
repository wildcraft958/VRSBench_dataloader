#!/usr/bin/env python3
"""
Test script for VRSBench Dataloader
Tests the dataloader with a small sample of data
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from vrsbench_dataloader_production import (
    create_vrsbench_dataloader,
    VRSBenchConfig,
    get_task_targets
)
from PIL import Image
import torch

def test_dataloader():
    """Test the VRSBench dataloader with a small sample"""
    
    print("="*60)
    print("VRSBench Dataloader Test")
    print("="*60)
    
    # Configure dataloader
    config = VRSBenchConfig()
    config.LOG_LEVEL = "INFO"
    config.LOG_DIR = "./test_logs"
    config.CACHE_DIR = "./test_cache"
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    
    # Test 1: Test with annotations_url (will download if needed)
    print("\n" + "="*60)
    print("Test 1: Loading from HuggingFace URL (captioning task)")
    print("="*60)
    
    try:
        # Use a small sample size for testing
        dataloader, metrics = create_vrsbench_dataloader(
            images_dir="./test_images",
            task="captioning",
            annotations_url="https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Annotations_val.zip",
            images_url="https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip",
            download_images=True,  # Will download and extract automatically
            split=None,  # Load all samples regardless of split
            batch_size=1,
            num_workers=0,
            sample_size=10,  # Only load 10 samples for testing
            config=config,
            return_metrics=True
        )
        
        print("\n✓ Dataloader created successfully")
        print("Loading samples...")
        
        # Load a few samples
        samples_loaded = 0
        for i, (image_tensor, metadata) in enumerate(dataloader):
            if i >= 5:  # Only test first 5
                break
            
            samples_loaded += 1
            print(f"\n  Sample {i+1}:")
            print(f"    Image tensor shape: {image_tensor.shape}")
            print(f"    Metadata keys: {list(metadata[0].keys())[:10]}")
            
            # Check for caption
            caption = None
            for key in ['caption', 'captions', 'description', 'text']:
                if key in metadata[0]:
                    caption = metadata[0][key]
                    break
            
            if caption:
                if isinstance(caption, list):
                    caption = caption[0] if caption else None
                print(f"    Caption: {str(caption)[:80]}..." if caption else "    Caption: None")
            
            # Check image path
            image_path = metadata[0].get('_image_path')
            if image_path:
                print(f"    Image path: {image_path}")
                print(f"    Image exists: {os.path.exists(image_path)}")
        
        print(f"\n✓ Successfully loaded {samples_loaded} samples")
        
        # Show metrics
        metrics_summary = metrics.get_summary()
        print(f"\nMetrics:")
        print(f"  Images loaded: {metrics_summary.get('metrics', {}).get('images_loaded', 0)}")
        print(f"  Annotations loaded: {metrics_summary.get('metrics', {}).get('annotations_loaded', 0)}")
        print(f"  Cache hits: {metrics_summary.get('metrics', {}).get('cache_hits', 0)}")
        print(f"  Cache misses: {metrics_summary.get('metrics', {}).get('cache_misses', 0)}")
        
        if metrics_summary.get('errors'):
            print(f"  Errors: {metrics_summary.get('errors')}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_task_targets():
    """Test the get_task_targets function"""
    print("\n" + "="*60)
    print("Test 2: Testing get_task_targets function")
    print("="*60)
    
    try:
        # Create sample metadata
        sample_metas = [
            {
                'caption': 'A satellite image showing urban areas',
                'image': 'test1.jpg',
                'split': 'validation'
            },
            {
                'caption': 'Rural landscape with fields',
                'image': 'test2.jpg',
                'split': 'validation'
            }
        ]
        
        # Test captioning task
        captions = get_task_targets(sample_metas, task='captioning')
        print(f"\n✓ Captioning targets extracted: {len(captions)} captions")
        for i, caption in enumerate(captions):
            print(f"  Caption {i+1}: {caption}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\nStarting VRSBench Dataloader Tests...")
    print("Note: First run will download ~4GB of images. Subsequent runs use cache.\n")
    
    # Run tests
    test1_passed = test_dataloader()
    test2_passed = test_task_targets()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Test 1 (Dataloader): {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Test 2 (Task Targets): {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)

