## 🎯 VRSBench Captioning Evaluation - Complete Colab Notebook

### Cell 1: Install Dependencies

```python
# Cell 1: Install all required packages
!pip install -q vllm==0.6.3
!pip install -q torch torchvision
!pip install -q pillow pandas requests tqdm
!pip install -q nltk sacrebleu rouge-score bert-score
!pip install -q accelerate transformers

# Download NLTK data for metrics
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

print("✓ All dependencies installed!")
print("\n⚠️  RESOURCE REQUIREMENTS:")
print("  • GPU: Minimum 24GB VRAM (A100/RTX 4090 recommended)")
print("  • RAM: 32GB+ recommended")
print("  • Disk: ~20GB for dataset + models")
print("  • Each 7B model needs ~14-16GB VRAM")
print("  • Evaluate 1 model at a time if GPU memory is limited")
```


### Cell 2: Upload Production DataLoader

```python
# Cell 2: Upload vrsbench_dataloader_production.py
from google.colab import files
import os

print("📤 Upload vrsbench_dataloader_production.py file:")
uploaded = files.upload()

# Verify upload
if 'vrsbench_dataloader_production.py' in uploaded:
    print("✓ DataLoader uploaded successfully!")
else:
    print("❌ Please upload vrsbench_dataloader_production.py")
```


### Cell 3: Download VRSBench Dataset

```python
# Cell 3: Download VRSBench images and annotations
import os

# Create directories
!mkdir -p Images_val
!mkdir -p hf_cache

print("Downloading VRSBench validation images (~4.8GB)...")
!curl -L -o Images_val.zip "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Images_val.zip"
print("\nExtracting images...")
!unzip -q Images_val.zip -d Images_val
print("✓ Images ready!")

print("\nDownloading annotations...")
!curl -L -o Annotations_val.zip "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Annotations_val.zip"
!unzip -q Annotations_val.zip -d hf_cache
print("✓ Annotations ready!")

# List extracted files
print("\n📁 Files extracted:")
!ls -lh Images_val | head -5
!ls -lh hf_cache/*.jsonl
```


### Cell 4: Setup DataLoader

```python
# Cell 4: Initialize VRSBench DataLoader for captioning
from vrsbench_dataloader_production import (
    create_vrsbench_dataloader,
    get_task_targets,
    VRSBenchConfig
)
import json

# Configure
config = VRSBenchConfig()
config.LOG_LEVEL = "WARNING"  # Reduce noise
config.CACHE_DIR = "./hf_cache"

# Find annotation file
annotation_file = None
for file in os.listdir('./hf_cache'):
    if file.endswith('.jsonl') and 'val' in file.lower():
        annotation_file = f'./hf_cache/{file}'
        break

print(f"Using annotation file: {annotation_file}")

# Create dataloader for captioning task
print("\nCreating captioning dataloader...")
dataloader = create_vrsbench_dataloader(
    images_dir="./Images_val",
    task="captioning",
    annotations_jsonl=annotation_file,
    batch_size=1,  # Process one at a time for inference
    num_workers=0,
    sample_size=100,  # Test on first 100 samples (remove for full eval)
    config=config
)

print("✓ DataLoader ready!")
print("✓ DataLoader now injects '_image_path' into metadata for easy access")

# Preview first sample
for images, metas in dataloader:
    captions = get_task_targets(metas, task="captioning")
    print(f"\nSample preview:")
    print(f"  Image shape: {images.shape}")
    print(f"  Image path: {metas[0].get('_image_path', 'N/A')}")
    print(f"  Ground truth caption: {captions[0][:100]}...")
    break
```


### Cell 5: Load Models with vLLM

```python
# Cell 5: Load all models using vLLM
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
import torch

print("🚀 Loading models with vLLM...\n")

# Define models to test
models_to_test = {
    "Qwen2-VL-7B": "Qwen/Qwen2-VL-7B-Instruct",
    "MiMo-VL-SFT": "XiaomiMiMo/MiMo-VL-7B-SFT-2508",
    "MiMo-VL-RL": "XiaomiMiMo/MiMo-VL-7B-RL-2508",
    "InternVL2-8B": "OpenGVLab/InternVL2-8B",
    "OLMo-2-7B": "allenai/OLMo-2-1124-7B-Instruct"
}

# Load models
loaded_models = {}

for model_name, model_id in models_to_test.items():
    try:
        print(f"Loading {model_name} ({model_id})...")
        
        llm = LLM(
            model=model_id,
            trust_remote_code=True,
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            dtype="float16"
        )
        
        loaded_models[model_name] = {
            "llm": llm,
            "model_id": model_id
        }
        
        print(f"  ✓ {model_name} loaded successfully!\n")
        
    except Exception as e:
        print(f"  ✗ Failed to load {model_name}: {e}\n")
        continue

print(f"\n✓ Loaded {len(loaded_models)}/{len(models_to_test)} models successfully!")
print(f"Models ready: {list(loaded_models.keys())}")
```


### Cell 6: Define Caption Generation Functions

```python
# Cell 6: Caption generation functions for each model
from PIL import Image
import base64
from io import BytesIO

def prepare_image_for_vllm(image_path):
    """Load and prepare image"""
    image = Image.open(image_path).convert('RGB')
    return image

def generate_caption_qwen(llm, image_path):
    """Generate caption using Qwen2-VL with robust error handling"""
    try:
        prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image in detail.<|im_end|>\n<|im_start|>assistant\n"
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=128,
            stop=["<|im_end|>"]
        )
        
        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": Image.open(image_path).convert('RGB')}
            },
            sampling_params=sampling_params
        )
        
        # Robust output extraction
        if outputs and len(outputs) > 0 and hasattr(outputs[0], 'outputs'):
            if len(outputs[0].outputs) > 0:
                return outputs[0].outputs[0].text.strip()
        return ""
    except Exception as e:
        print(f"Error in Qwen generation: {e}")
        return ""

def generate_caption_mimo(llm, image_path):
    """Generate caption using MiMo-VL with robust error handling"""
    try:
        prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=128
        )
        
        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": Image.open(image_path).convert('RGB')}
            },
            sampling_params=sampling_params
        )
        
        if outputs and len(outputs) > 0 and hasattr(outputs[0], 'outputs'):
            if len(outputs[0].outputs) > 0:
                return outputs[0].outputs[0].text.strip()
        return ""
    except Exception as e:
        print(f"Error in MiMo generation: {e}")
        return ""

def generate_caption_internvl(llm, image_path):
    """Generate caption using InternVL2 with robust error handling"""
    try:
        prompt = "<image>\nPlease describe this image in detail."
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=128
        )
        
        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": Image.open(image_path).convert('RGB')}
            },
            sampling_params=sampling_params
        )
        
        if outputs and len(outputs) > 0 and hasattr(outputs[0], 'outputs'):
            if len(outputs[0].outputs) > 0:
                return outputs[0].outputs[0].text.strip()
        return ""
    except Exception as e:
        print(f"Error in InternVL generation: {e}")
        return ""

def generate_caption_olmo(llm, image_path):
    """Generate caption using OLMo-2 with robust error handling"""
    try:
        prompt = "<image>\nDescribe this remote sensing image."
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=128
        )
        
        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": Image.open(image_path).convert('RGB')}
            },
            sampling_params=sampling_params
        )
        
        if outputs and len(outputs) > 0 and hasattr(outputs[0], 'outputs'):
            if len(outputs[0].outputs) > 0:
                return outputs[0].outputs[0].text.strip()
        return ""
    except Exception as e:
        print(f"Error in OLMo generation: {e}")
        return ""
    )
    
    return outputs[0].outputs[0].text.strip()

# Map models to generation functions
generation_functions = {
    "Qwen2-VL-7B": generate_caption_qwen,
    "MiMo-VL-SFT": generate_caption_mimo,
    "MiMo-VL-RL": generate_caption_mimo,
    "InternVL2-8B": generate_caption_internvl,
    "OLMo-2-7B": generate_caption_olmo
}

print("✓ Caption generation functions defined!")
```


### Cell 7: Run Inference on All Models

```python
# Cell 7: Generate captions for all models
from tqdm import tqdm
import pandas as pd

print("🔮 Running inference on VRSBench validation set...\n")

# Store results
results = {
    "image_id": [],
    "ground_truth": [],
}

for model_name in loaded_models.keys():
    results[f"{model_name}_prediction"] = []

# Process each sample
sample_count = 0
for images, metas in tqdm(dataloader, desc="Processing samples"):
    # Get ground truth caption
    gt_captions = get_task_targets(metas, task="captioning")
    gt_caption = gt_captions[0]
    
    # Get image path (now injected by dataloader)
    image_path = metas[0].get('_image_path')
    if not image_path:
        # Fallback to constructing path (shouldn't happen with updated loader)
        image_filename = metas[0].get('image', metas[0].get('image_name', metas[0].get('file_name')))
        image_path = f"./Images_val/{image_filename}"
    
    image_id = metas[0].get('id', sample_count)
    
    # Store ground truth
    results["image_id"].append(image_id)
    results["ground_truth"].append(gt_caption)
    
    # Generate captions for each model
    for model_name, model_info in loaded_models.items():
        try:
            llm = model_info["llm"]
            gen_func = generation_functions[model_name]
            
            # Generate caption
            prediction = gen_func(llm, image_path)
            results[f"{model_name}_prediction"].append(prediction)
            
        except Exception as e:
            print(f"\n⚠️ Error with {model_name} on sample {sample_count}: {e}")
            results[f"{model_name}_prediction"].append("")
    
    sample_count += 1
    
    # Print progress every 10 samples
    if sample_count % 10 == 0:
        print(f"\n✓ Processed {sample_count} samples")
        print(f"  Last GT: {gt_caption[:80]}...")
        if loaded_models:
            first_model = list(loaded_models.keys())[0]
            last_pred = results[f'{first_model}_prediction'][-1]
            print(f"  {first_model}: {last_pred[:80] if last_pred else 'N/A'}...")

# Create DataFrame
results_df = pd.DataFrame(results)
print(f"\n✓ Inference complete! Generated {sample_count} captions per model")
print(f"\nResults shape: {results_df.shape}")
results_df.head()
```


### Cell 8: Calculate BERTScore (Semantic Similarity)

```python
# Cell 8: Calculate BERTScore - better for semantic similarity than BLEU
from bert_score import score as bert_score
import numpy as np

print("📊 Calculating BERTScore (semantic similarity)...\n")
print("ℹ️  BERTScore measures semantic similarity using contextual embeddings")
print("   More suitable for image captioning than BLEU (which measures n-gram overlap)\n")

# Calculate BERTScore for each model
bertscore_results = {}

for model_name in loaded_models.keys():
    pred_col = f"{model_name}_prediction"
    
    # Prepare references and hypotheses
    references = []
    hypotheses = []
    
    for idx, row in results_df.iterrows():
        if row[pred_col]:  # Skip empty predictions
            references.append(row["ground_truth"])
            hypotheses.append(row[pred_col])
    
    if not hypotheses:
        print(f"⚠️  No predictions for {model_name}, skipping...")
        continue
    
    print(f"Computing BERTScore for {model_name}...")
    
    # Calculate BERTScore
    # Using distilbert-base-uncased for speed (can use roberta-large for better accuracy)
    P, R, F1 = bert_score(
        hypotheses, 
        references,
        model_type="distilbert-base-uncased",  # Fast; use "roberta-large" for better quality
        lang="en",
        verbose=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    bertscore_results[model_name] = {
        "BERTScore-P": P.mean().item() * 100,
        "BERTScore-R": R.mean().item() * 100,
        "BERTScore-F1": F1.mean().item() * 100
    }
    
    print(f"{model_name}:")
    print(f"  Precision:  {P.mean().item()*100:.2f}")
    print(f"  Recall:     {R.mean().item()*100:.2f}")
    print(f"  F1:         {F1.mean().item()*100:.2f}\n")

# Create BERTScore DataFrame
bertscore_df = pd.DataFrame(bertscore_results).T
bertscore_df.index.name = "Model"
print("\n📊 BERTScore Summary (F1 is primary metric):")
print(bertscore_df.round(2))
```


### Cell 9: Calculate Additional Metrics (ROUGE, METEOR, CIDEr)

```python
# Cell 9: Calculate ROUGE, METEOR, and other complementary metrics
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from collections import defaultdict

print("📊 Calculating additional metrics...\n")

additional_metrics = defaultdict(dict)

# ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Tokenize helper
def tokenize_caption(caption):
    """Tokenize caption"""
    return word_tokenize(caption.lower())

for model_name in loaded_models.keys():
    pred_col = f"{model_name}_prediction"
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    meteor_scores = []
    caption_lengths = []
    
    for idx, row in results_df.iterrows():
        if row[pred_col]:
            gt = row["ground_truth"]
            pred = row[pred_col]
            
            # ROUGE scores
            rouge_results = scorer.score(gt, pred)
            rouge1_scores.append(rouge_results['rouge1'].fmeasure)
            rouge2_scores.append(rouge_results['rouge2'].fmeasure)
            rougeL_scores.append(rouge_results['rougeL'].fmeasure)
            
            # METEOR score
            try:
                meteor = meteor_score([tokenize_caption(gt)], tokenize_caption(pred))
                meteor_scores.append(meteor)
            except:
                pass
            
            # Caption length
            caption_lengths.append(len(pred.split()))
    
    additional_metrics[model_name] = {
        "ROUGE-1": np.mean(rouge1_scores) * 100 if rouge1_scores else 0,
        "ROUGE-2": np.mean(rouge2_scores) * 100 if rouge2_scores else 0,
        "ROUGE-L": np.mean(rougeL_scores) * 100 if rougeL_scores else 0,
        "METEOR": np.mean(meteor_scores) * 100 if meteor_scores else 0,
        "Avg Length": np.mean(caption_lengths) if caption_lengths else 0
    }
    
    print(f"{model_name}:")
    print(f"  ROUGE-1: {np.mean(rouge1_scores)*100:.2f}" if rouge1_scores else "  ROUGE-1: N/A")
    print(f"  ROUGE-2: {np.mean(rouge2_scores)*100:.2f}" if rouge2_scores else "  ROUGE-2: N/A")
    print(f"  ROUGE-L: {np.mean(rougeL_scores)*100:.2f}" if rougeL_scores else "  ROUGE-L: N/A")
    print(f"  METEOR:  {np.mean(meteor_scores)*100:.2f}" if meteor_scores else "  METEOR: N/A")
    print(f"  Avg Length: {np.mean(caption_lengths):.1f} words\n" if caption_lengths else "  Avg Length: N/A\n")

# Create metrics DataFrame
metrics_df = pd.DataFrame(additional_metrics).T
metrics_df.index.name = "Model"
print("\n📊 Additional Metrics Summary:")
print(metrics_df.round(2))
```


### Cell 10: Create Complete Results Table

```python
# Cell 10: Combine all metrics into final results table
import pandas as pd

print("📋 Creating final results table...\n")

# Combine BERTScore and additional metrics
final_results = pd.concat([bertscore_df, metrics_df], axis=1)

# Reorder columns - BERTScore-F1 is primary metric
column_order = ['BERTScore-F1', 'BERTScore-P', 'BERTScore-R',
                'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR', 'Avg Length']
final_results = final_results[column_order]

# Sort by BERTScore-F1 (descending)
final_results = final_results.sort_values('BERTScore-F1', ascending=False)

print("="*90)
print("VRSBench Captioning Evaluation Results")
print("="*90)
print(final_results.round(2))
print("="*90)

# Highlight best model
best_model = final_results['BERTScore-F1'].idxmax()
print(f"\n🏆 Best Model (BERTScore-F1): {best_model} ({final_results.loc[best_model, 'BERTScore-F1']:.2f})")

# Additional rankings
print(f"\n📊 Alternative Rankings:")
print(f"  Best ROUGE-L: {final_results['ROUGE-L'].idxmax()} ({final_results['ROUGE-L'].max():.2f})")
print(f"  Best METEOR:  {final_results['METEOR'].idxmax()} ({final_results['METEOR'].max():.2f})")

# Save to CSV
final_results.to_csv('vrsbench_captioning_results.csv')
print("\n✓ Results saved to vrsbench_captioning_results.csv")
```


### Cell 11: Visualize Results

```python
# Cell 11: Create visualization of results
import matplotlib.pyplot as plt
import seaborn as sns

print("📊 Creating visualizations...\n")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. BERTScore Components
ax1 = axes[0, 0]
bertscore_cols = ['BERTScore-P', 'BERTScore-R', 'BERTScore-F1']
final_results[bertscore_cols].plot(kind='bar', ax=ax1, colormap='viridis')
ax1.set_title('BERTScore Components (Semantic Similarity)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Score (%)', fontsize=12)
ax1.set_xlabel('Model', fontsize=12)
ax1.legend(title='Metric', loc='upper right')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)

# 2. ROUGE Scores Comparison
ax2 = axes[0, 1]
rouge_cols = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
final_results[rouge_cols].plot(kind='bar', ax=ax2, colormap='plasma')
ax2.set_title('ROUGE Scores Comparison', fontsize=14, fontweight='bold')
ax2.set_ylabel('Score (%)', fontsize=12)
ax2.set_xlabel('Model', fontsize=12)
ax2.legend(title='Metric', loc='upper right')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

# 3. Overall Performance (BERTScore-F1 vs METEOR)
ax3 = axes[1, 0]
ax3.scatter(final_results['BERTScore-F1'], final_results['METEOR'], 
           s=200, alpha=0.6, c=range(len(final_results)), cmap='coolwarm')
for idx, model in enumerate(final_results.index):
    ax3.annotate(model, 
                (final_results.loc[model, 'BERTScore-F1'], 
                 final_results.loc[model, 'METEOR']),
                fontsize=9, ha='center')
ax3.set_title('BERTScore-F1 vs METEOR', fontsize=14, fontweight='bold')
ax3.set_xlabel('BERTScore-F1 (%)', fontsize=12)
ax3.set_ylabel('METEOR Score (%)', fontsize=12)
ax3.grid(alpha=0.3)

# 4. Caption Length Analysis
ax4 = axes[1, 1]
final_results['Avg Length'].plot(kind='barh', ax=ax4, color='coral')
ax4.set_title('Average Caption Length', fontsize=14, fontweight='bold')
ax4.set_xlabel('Words', fontsize=12)
ax4.set_ylabel('Model', fontsize=12)
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('vrsbench_captioning_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualization saved to vrsbench_captioning_results.png")
```


### Cell 12: Show Example Predictions

```python
# Cell 12: Display example predictions
import random

print("🔍 Example Predictions:\n")
print("="*100)

# Show 5 random examples
sample_indices = random.sample(range(len(results_df)), min(5, len(results_df)))

for i, idx in enumerate(sample_indices, 1):
    row = results_df.iloc[idx]
    
    print(f"\nExample {i} (Image ID: {row['image_id']})")
    print("-"*100)
    print(f"Ground Truth:\n  {row['ground_truth']}\n")
    
    for model_name in loaded_models.keys():
        pred_col = f"{model_name}_prediction"
        print(f"{model_name}:\n  {row[pred_col]}\n")
    
    print("="*100)

print("\n✓ Example predictions displayed!")
```


### Cell 13: Export Results

```python
# Cell 13: Export all results
from google.colab import files

print("💾 Exporting results...\n")

# Save detailed predictions
results_df.to_csv('vrsbench_detailed_predictions.csv', index=False)
print("✓ Detailed predictions saved to: vrsbench_detailed_predictions.csv")

# Save metrics summary
final_results.to_csv('vrsbench_metrics_summary.csv')
print("✓ Metrics summary saved to: vrsbench_metrics_summary.csv")

# Download files
print("\n📥 Downloading results to your computer...")
files.download('vrsbench_detailed_predictions.csv')
files.download('vrsbench_metrics_summary.csv')
files.download('vrsbench_captioning_results.png')

print("\n✅ All results exported and downloaded!")
```


### Cell 14: Summary Report

```python
# Cell 14: Generate final summary report
print("="*80)
print("VRSBench CAPTIONING EVALUATION - FINAL REPORT")
print("="*80)

print(f"\n📊 Dataset Statistics:")
print(f"  Total samples evaluated: {len(results_df)}")
print(f"  Models tested: {len(loaded_models)}")
print(f"  Models: {', '.join(loaded_models.keys())}")

print(f"\n🏆 Rankings by BERTScore-F1 (Primary Metric):")
for rank, (model, score) in enumerate(final_results['BERTScore-F1'].items(), 1):
    print(f"  {rank}. {model}: {score:.2f}")

print(f"\n📈 Performance Summary:")
print(f"  Best BERTScore-F1: {final_results['BERTScore-F1'].max():.2f} ({final_results['BERTScore-F1'].idxmax()})")
print(f"  Best METEOR: {final_results['METEOR'].max():.2f} ({final_results['METEOR'].idxmax()})")
print(f"  Best ROUGE-L: {final_results['ROUGE-L'].max():.2f} ({final_results['ROUGE-L'].idxmax()})")

print(f"\nℹ️  Metric Explanation:")
print(f"  • BERTScore-F1: Semantic similarity using contextual embeddings (BEST for captioning)")
print(f"  • ROUGE-L: Longest common subsequence match")
print(f"  • METEOR: Considers synonyms and paraphrases")

print(f"\n💬 Caption Analysis:")
print(f"  Avg caption length (GT): {results_df['ground_truth'].str.split().str.len().mean():.1f} words")
for model_name in loaded_models.keys():
    pred_col = f"{model_name}_prediction"
    avg_len = results_df[pred_col].str.split().str.len().mean()
    print(f"  Avg caption length ({model_name}): {avg_len:.1f} words")

print("\n" + "="*80)
print("✅ EVALUATION COMPLETE!")
print("="*80)

print("""
📁 Generated Files:
  • vrsbench_detailed_predictions.csv - All predictions
  • vrsbench_metrics_summary.csv - Metrics table
  • vrsbench_captioning_results.png - Visualization

🎯 Next Steps:
  1. Analyze which model performs best for your use case
  2. Check example predictions in Cell 12
  3. Compare caption lengths and styles
  4. Consider fine-tuning based on these results
""")
```


***

## 🚀 How to Use This Notebook

1. **Run Cell 1**: Install all dependencies
2. **Run Cell 2**: Upload your `vrsbench_dataloader_production.py` file
3. **Run Cell 3**: Download VRSBench dataset (~10 mins)
4. **Run Cell 4**: Setup dataloader
5. **Run Cell 5**: Load models (may take 5-10 mins per model)
6. **Run Cell 6**: Define caption generation functions
7. **Run Cell 7**: Run inference (main evaluation, ~30-60 mins)
8. **Run Cells 8-14**: Calculate metrics, visualize, and export results

The notebook uses your production dataloader and evaluates all requested models with comprehensive metrics! 🎉

