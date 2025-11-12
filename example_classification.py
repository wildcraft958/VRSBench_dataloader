#!/usr/bin/env python3
"""
Example: Classification Task with VRSBench

This example shows how to use the VRSBench dataloader for image classification.
"""

import torch
import torch.nn as nn
from vrsbench_dataloader_production import (
    create_vrsbench_dataloader,
    get_task_targets,
    VRSBenchConfig
)

def main():
    # Configuration
    config = VRSBenchConfig()
    config.LOG_LEVEL = "INFO"
    config.BATCH_SIZE = 16
    config.NUM_WORKERS = 4

    print("Creating classification dataloader...")

    # Create dataloader
    dataloader, metrics = create_vrsbench_dataloader(
        images_dir="./data/images",
        task="classification",
        annotations_jsonl="./data/annotations_val.jsonl",
        # Or download directly:
        # annotations_url="https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Annotations_val.zip",
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        split="validation",
        config=config,
        return_metrics=True
    )

    print("✓ DataLoader created\n")

    # Dummy model for demonstration
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 256 * 256, 512),
        nn.ReLU(),
        nn.Linear(512, 10)  # 10 classes
    )

    criterion = nn.CrossEntropyLoss()

    # Training loop example
    print("Running training loop (demo)...\n")

    for batch_idx, (images, metas) in enumerate(dataloader):
        # images: [B, 3, 256, 256]
        # metas: List[Dict]

        # Extract labels
        labels = get_task_targets(metas, task="classification")
        labels = torch.tensor(labels)

        # Forward pass (demo only - not actual training)
        outputs = model(images)
        loss = criterion(outputs, labels)

        print(f"Batch {batch_idx + 1}:")
        print(f"  Images: {images.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Loss: {loss.item():.4f}")

        # Limit to first 5 batches for demo
        if batch_idx >= 4:
            break

    # Print metrics
    print("\n" + "="*60)
    print("Metrics Summary:")
    print("="*60)
    import json
    summary = metrics.get_summary()
    print(json.dumps(summary, indent=2))

    print("\n✓ Example completed successfully!")

if __name__ == "__main__":
    main()
