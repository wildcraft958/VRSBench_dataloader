#!/usr/bin/env python3
"""
Example: Visual Question Answering (VQA) with VRSBench

This example demonstrates VQA task with multi-annotation expansion.
"""

import torch
from vrsbench_dataloader_production import (
    create_vrsbench_dataloader,
    get_task_targets,
    VRSBenchConfig
)

def main():
    print("Creating VQA dataloader with multi-annotation expansion...\n")

    config = VRSBenchConfig()
    config.LOG_LEVEL = "INFO"

    # Create dataloader
    dataloader = create_vrsbench_dataloader(
        images_dir="./data/images",
        task="vqa",
        annotations_jsonl="./data/vqa_annotations.jsonl",
        # Enable multi-annotation expansion
        # This creates separate samples for each QA pair per image
        expand_multi_annotations=True,
        batch_size=8,
        num_workers=4,
        config=config
    )

    print("✓ DataLoader created\n")

    # Iterate through VQA samples
    print("Processing VQA samples...\n")

    for batch_idx, (images, metas) in enumerate(dataloader):
        # Extract question-answer pairs
        qa_pairs = get_task_targets(metas, task="vqa")

        print(f"Batch {batch_idx + 1}:")
        print(f"  Images: {images.shape}")
        print(f"  QA pairs: {len(qa_pairs)}")

        # Show first QA pair
        if qa_pairs:
            question, answer = qa_pairs[0]
            print(f"  Example Q: {question}")
            print(f"  Example A: {answer}")

        # Your VQA model inference here
        # outputs = vqa_model(images, questions)

        if batch_idx >= 2:
            break

    print("\n✓ VQA example completed!")

if __name__ == "__main__":
    main()
