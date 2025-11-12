#!/usr/bin/env python3
"""
Example: Visual Grounding with Region Extraction

This example shows region-based visual grounding for satellite imagery.
"""

import torch
import matplotlib.pyplot as plt
from vrsbench_dataloader_production import (
    create_vrsbench_dataloader,
    get_task_targets,
    VRSBenchConfig
)

def main():
    print("Creating visual grounding dataloader with region extraction...\n")

    config = VRSBenchConfig()
    config.LOG_LEVEL = "INFO"
    config.REGION_PADDING = 20  # Padding around bounding boxes

    # Create dataloader with region-based extraction
    dataloader = create_vrsbench_dataloader(
        images_dir="./data/images",
        task="grounding",
        annotations_jsonl="./data/grounding_annotations.jsonl",

        # Enable region extraction - crops image regions using bboxes
        region_based=True,

        # Expand multi-annotations - one sample per object
        expand_multi_annotations=True,

        batch_size=4,
        num_workers=4,
        config=config
    )

    print("✓ DataLoader created\n")

    # Process grounding samples
    print("Processing grounding samples...\n")

    for batch_idx, (images, metas) in enumerate(dataloader):
        # images: [B, 3, H, W] - cropped regions around objects
        # metas: metadata with bbox and category info

        bboxes = get_task_targets(metas, task="grounding")

        print(f"Batch {batch_idx + 1}:")
        print(f"  Regions: {images.shape}")
        print(f"  BBoxes: {len(bboxes)}")

        # Show first region info
        if metas:
            meta = metas[0]
            print(f"  Category: {meta.get('category', 'N/A')}")
            print(f"  BBox: {meta.get('bbox', 'N/A')}")

        # Your grounding model inference here
        # predictions = grounding_model(images, referring_expressions)

        if batch_idx >= 2:
            break

    print("\n✓ Grounding example completed!")

if __name__ == "__main__":
    main()
