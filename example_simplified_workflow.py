"""
Example: Simplified VRSBench Workflow

This script demonstrates the new simplified workflow that replaces the manual
process of downloading images, extracting, loading HuggingFace datasets, and
manually mapping them together.

OLD WAY (manual):
- curl downloads
- unzip operations
- HuggingFace dataset loading
- Complex manual mapping code

NEW WAY (one function call):
- prepare_vrsbench_dataset() does everything automatically
"""

from vrsbench_dataloader_production import (
    prepare_vrsbench_dataset,
    create_captioning_dataloader_from_prepared,
    get_task_targets
)
from PIL import Image

# ============================================
# Example 1: Basic Usage for Captioning Task
# ============================================

print("=" * 70)
print("Example 1: Basic Usage for Captioning Task")
print("=" * 70)

# One function call does everything:
# 1. Downloads images from HuggingFace (if needed)
# 2. Extracts images
# 3. Loads HuggingFace streaming dataset
# 4. Combines metadata with local images
# 5. Returns easy-to-use data structure

data = prepare_vrsbench_dataset(
    split="validation",
    num_samples=10,  # Limit to 10 samples for demo (use None for all)
    output_json="./test_utility.json"  # Optional: save to JSON
)

print(f"\n✓ Loaded {data['num_samples']} samples from {data['split']} split")
print(f"✓ Saved to: ./test_utility.json")

# Easy access to mappings
image_to_caption = data["image_to_caption"]
id_to_path = data["id_to_path"]

print("\n" + "=" * 70)
print("IMAGE-CAPTION MAPPING:")
print("=" * 70)

# Display the mapping
for image_id, caption in list(image_to_caption.items())[:5]:  # Show first 5
    print(f"\n{image_id}:")
    print(f"  Caption: {caption[:80]}...")
    print(f"  Path: {id_to_path[image_id]}")

# ============================================
# Example 2: Iterate Over Samples
# ============================================

print("\n" + "=" * 70)
print("Example 2: Iterate Over Samples")
print("=" * 70)

# Iterate over samples
for sample in data["samples"][:3]:  # Show first 3
    image_id = sample["image_id"]
    image_path = sample["image_path"]
    gt_caption = sample["caption"]
    
    # Load image for your model
    image = Image.open(image_path)
    
    print(f"\nImage: {image_id}")
    print(f"  Size: {image.size}")
    print(f"  Ground Truth: {gt_caption[:80]}...")

# ============================================
# Example 3: Create PyTorch DataLoader
# ============================================

print("\n" + "=" * 70)
print("Example 3: Create PyTorch DataLoader")
print("=" * 70)

# Create a PyTorch DataLoader directly from prepared data
dataloader = create_captioning_dataloader_from_prepared(
    data,
    batch_size=4,  # Small batch for demo
    num_workers=0   # Use 0 for demo (set to 4+ for production)
)

print(f"✓ Created DataLoader with {len(data['samples'])} samples")

# Iterate over batches
for batch_idx, (images, metas) in enumerate(dataloader):
    if batch_idx >= 2:  # Show first 2 batches only
        break
    
    captions = get_task_targets(metas, task="captioning")
    
    print(f"\nBatch {batch_idx + 1}:")
    print(f"  Images shape: {images.shape}")
    print(f"  Number of samples: {len(metas)}")
    print(f"  First caption: {captions[0][:60]}...")

# ============================================
# Example 4: Load from Saved JSON
# ============================================

print("\n" + "=" * 70)
print("Example 4: Load from Saved JSON")
print("=" * 70)

import json

# Load the mapping from JSON (useful for sharing with colleagues)
with open("./test_utility.json", 'r') as f:
    loaded_data = json.load(f)

print(f"✓ Loaded {loaded_data['num_samples']} samples from JSON")
print(f"✓ Split: {loaded_data['split']}")
print(f"✓ Available keys: {list(loaded_data.keys())}")

# Easy access
image_to_caption = loaded_data["image_to_caption"]
id_to_path = loaded_data["id_to_path"]

print(f"\n✓ image_to_caption has {len(image_to_caption)} entries")
print(f"✓ id_to_path has {len(id_to_path)} entries")

# ============================================
# Example 5: Use with Your Model
# ============================================

print("\n" + "=" * 70)
print("Example 5: Use with Your Model")
print("=" * 70)

print("""
# Example usage with your captioning model:

for sample in data["samples"]:
    image_path = sample["image_path"]
    gt_caption = sample["caption"]
    
    # Load image
    image = Image.open(image_path)
    
    # Preprocess for your model
    # image_tensor = your_preprocess(image)
    
    # Generate caption with your model
    # predicted_caption = your_model(image_tensor)
    
    # Compare with ground truth
    # score = evaluate(predicted_caption, gt_caption)
    
    print(f"Image: {sample['image_id']}")
    print(f"GT: {gt_caption}")
    # print(f"Predicted: {predicted_caption}")
    # print(f"Score: {score}")
""")

print("\n" + "=" * 70)
print("✓ All examples completed!")
print("=" * 70)

