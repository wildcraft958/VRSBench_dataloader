#!/usr/bin/env python3
"""
Smoke test for dataloader enhancements:
- Test that _image_path is injected into metadata
- Test bbox normalization helper
"""

import os
import json
import tempfile
from PIL import Image
import torch

# Import the production dataloader
from vrsbench_dataloader_production import (
    create_vrsbench_dataloader,
    TaskProcessor,
    VRSBenchConfig
)

def create_test_data():
    """Create minimal test data: synthetic image + annotation JSONL"""
    # Create temp directory
    test_dir = tempfile.mkdtemp(prefix="vrsbench_test_")
    images_dir = os.path.join(test_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Create a synthetic image
    img = Image.new('RGB', (512, 512), color='red')
    img_path = os.path.join(images_dir, "test_image_001.jpg")
    img.save(img_path)
    
    # Create annotation JSONL with normalized bbox
    annotation = {
        "id": 1,
        "image": "test_image_001.jpg",
        "caption": "A test image for validation",
        "split": "validation",
        "bbox": [0.1, 0.2, 0.5, 0.6],  # Normalized [x1, y1, x2, y2]
        "category": "test-object"
    }
    
    jsonl_path = os.path.join(test_dir, "annotations.jsonl")
    with open(jsonl_path, 'w') as f:
        f.write(json.dumps(annotation) + '\n')
    
    return test_dir, images_dir, jsonl_path

def test_image_path_injection():
    """Test that _image_path is injected into metadata"""
    print("=" * 60)
    print("TEST 1: Image Path Injection")
    print("=" * 60)
    
    test_dir, images_dir, jsonl_path = create_test_data()
    
    try:
        # Create dataloader
        config = VRSBenchConfig()
        config.LOG_LEVEL = "WARNING"
        
        dataloader = create_vrsbench_dataloader(
            images_dir=images_dir,
            task="captioning",
            annotations_jsonl=jsonl_path,
            batch_size=1,
            num_workers=0,
            config=config
        )
        
        # Iterate and check metadata
        for images, metas in dataloader:
            print(f"✓ Loaded batch with {len(metas)} samples")
            
            # Check _image_path exists
            if '_image_path' in metas[0]:
                print(f"✓ '_image_path' found in metadata: {metas[0]['_image_path']}")
                
                # Verify it's a valid path
                if os.path.exists(metas[0]['_image_path']):
                    print(f"✓ Image path is valid and file exists")
                else:
                    print(f"✗ Image path does not exist: {metas[0]['_image_path']}")
                    return False
            else:
                print(f"✗ '_image_path' NOT found in metadata")
                print(f"  Available keys: {list(metas[0].keys())}")
                return False
            
            break  # Test first sample only
        
        print(f"\n✅ TEST 1 PASSED: Image path injection works!\n")
        return True
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)

def test_bbox_normalization():
    """Test bbox normalization helper"""
    print("=" * 60)
    print("TEST 2: BBox Normalization")
    print("=" * 60)
    
    processor = TaskProcessor()
    
    # Test case 1: Normalized corner format [x1, y1, x2, y2] in range [0, 1]
    bbox_normalized = [0.1, 0.2, 0.5, 0.6]
    result = processor.normalize_bbox(bbox_normalized, 512, 512)
    expected = [51, 102, 204, 204]  # [x, y, w, h] in pixels
    
    print(f"Test 1 - Normalized corners [0.1, 0.2, 0.5, 0.6] (512x512):")
    print(f"  Result: {result}")
    print(f"  Expected: {expected}")
    
    # Allow some tolerance due to rounding
    if all(abs(r - e) <= 1 for r, e in zip(result, expected)):
        print(f"  ✓ PASS")
    else:
        print(f"  ✗ FAIL")
        return False
    
    # Test case 2: Already in pixel format [x, y, w, h]
    bbox_pixels = [50, 100, 200, 200]
    result2 = processor.normalize_bbox(bbox_pixels, 512, 512)
    print(f"\nTest 2 - Already in pixels [50, 100, 200, 200]:")
    print(f"  Result: {result2}")
    print(f"  Expected: {bbox_pixels}")
    
    if result2 == bbox_pixels:
        print(f"  ✓ PASS")
    else:
        print(f"  ✗ FAIL")
        return False
    
    # Test case 3: Corner format in pixels [x1, y1, x2, y2]
    bbox_corner_pixels = [50, 100, 250, 300]
    result3 = processor.normalize_bbox(bbox_corner_pixels, 512, 512)
    expected3 = [50, 100, 200, 200]  # Converted to [x, y, w, h]
    print(f"\nTest 3 - Corner pixels [50, 100, 250, 300]:")
    print(f"  Result: {result3}")
    print(f"  Expected: {expected3}")
    
    if result3 == expected3:
        print(f"  ✓ PASS")
    else:
        print(f"  ✗ FAIL")
        return False
    
    print(f"\n✅ TEST 2 PASSED: BBox normalization works!\n")
    return True

def test_region_extraction():
    """Test region extraction with normalized bbox"""
    print("=" * 60)
    print("TEST 3: Region Extraction with Normalized BBox")
    print("=" * 60)
    
    # Create test image
    img = Image.new('RGB', (512, 512), color='blue')
    
    # Normalized bbox [x1, y1, x2, y2] = [0.2, 0.3, 0.6, 0.7]
    # Should extract region from pixel (102, 153) to (307, 358)
    # Size: 205x205 pixels (plus padding)
    bbox_normalized = [0.2, 0.3, 0.6, 0.7]
    
    processor = TaskProcessor()
    region = processor.extract_region_from_bbox(img, bbox_normalized, padding=0)
    
    print(f"Original image size: {img.size}")
    print(f"Normalized bbox: {bbox_normalized}")
    print(f"Extracted region size: {region.size}")
    
    # Expected size (approximately)
    expected_w = int((0.6 - 0.2) * 512)  # 204 pixels
    expected_h = int((0.7 - 0.3) * 512)  # 204 pixels
    
    # Allow some tolerance
    if abs(region.size[0] - expected_w) <= 2 and abs(region.size[1] - expected_h) <= 2:
        print(f"✓ Region size is correct (expected ~{expected_w}x{expected_h})")
        print(f"\n✅ TEST 3 PASSED: Region extraction works!\n")
        return True
    else:
        print(f"✗ Region size mismatch (expected ~{expected_w}x{expected_h})")
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("VRSBench DataLoader Enhancement Tests")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Image path injection
    results.append(("Image Path Injection", test_image_path_injection()))
    
    # Test 2: BBox normalization
    results.append(("BBox Normalization", test_bbox_normalization()))
    
    # Test 3: Region extraction
    results.append(("Region Extraction", test_region_extraction()))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Dataloader enhancements working correctly.")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED. Please review the output above.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
