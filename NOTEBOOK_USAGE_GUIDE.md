# VRSBench Captioning Evaluation Notebook - Usage Guide

## Overview

This guide explains the Colab notebook (`vrsbench_captioning_evaluation.ipynb`) that evaluates multiple vision-language models on the VRSBench dataset for image captioning tasks using BLEU scores.

## What Has Been Created

### 1. **Complete Colab Notebook** (`vrsbench_captioning_evaluation.ipynb`)

A comprehensive Jupyter notebook with 26 cells that:

- **Sets up the environment** with all required packages
- **Clones your VRSBench dataloader repository** from GitHub
- **Loads 1,000 samples** from the VRSBench dataset
- **Loads 5 vision-language models** for evaluation:
  1. Qwen2.5-VL-8B
  2. XiaomiMiMo/MiMo-VL-7B-RL-2508
  3. InternVL3 Instruct
  4. allenai/OLMo-2-1124-7B-Instruct
  5. allenai/OlmoEarth-v1-Base
- **Generates captions** for all samples using each model
- **Calculates BLEU-4 scores** comparing generated vs reference captions
- **Displays results** in tables and visualizations
- **Saves results** to JSON file

### 2. **Key Features**

- ✅ **Robust error handling**: Models that fail to load are skipped gracefully
- ✅ **Multiple model architectures**: Supports BLIP, Vision2Seq, and CausalLM models
- ✅ **Flexible caption generation**: Handles different model input formats
- ✅ **Comprehensive evaluation**: BLEU scores with statistics (mean, median, min, max, std dev)
- ✅ **Progress tracking**: Progress bars and status updates throughout
- ✅ **Results visualization**: Bar charts comparing model performance
- ✅ **Export functionality**: Results saved to JSON for further analysis

## How to Use the Notebook

### Step 1: Prepare Your GitHub Repository

1. **Push your VRSBench dataloader to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Production-ready VRSBench dataloader"
   git push origin main
   ```

2. **Note your GitHub username and repository name**:
   - Username: `yourusername` (replace with your actual username)
   - Repository: `VRSBench_dataloader` (or your actual repo name)

### Step 2: Open in Google Colab

1. **Upload the notebook to Google Colab**:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click "File" → "Upload notebook"
   - Upload `vrsbench_captioning_evaluation.ipynb`

   OR

   - If your repo is on GitHub, you can open it directly:
   - `https://colab.research.google.com/github/yourusername/VRSBench_dataloader/blob/main/vrsbench_captioning_evaluation.ipynb`

2. **Enable GPU runtime** (recommended for faster inference):
   - Click "Runtime" → "Change runtime type"
   - Select "GPU" (T4 or better)
   - Click "Save"

### Step 3: Update Repository URL

**IMPORTANT**: Before running, update the GitHub URL in **Cell 2**:

```python
# Replace 'yourusername' with your actual GitHub username
!git clone https://github.com/yourusername/VRSBench_dataloader.git /content/vrsbench_dataloader
```

For example, if your username is `animeshraj`:
```python
!git clone https://github.com/animeshraj/VRSBench_dataloader.git /content/vrsbench_dataloader
```

### Step 4: Run the Notebook

1. **Run cells sequentially**:
   - Click "Runtime" → "Run all" to run everything at once
   - OR run each cell individually by clicking the play button

2. **Monitor progress**:
   - The notebook will show progress bars and status messages
   - Model loading may take several minutes per model
   - Inference on 1,000 samples may take 1-3 hours depending on GPU

3. **Handle errors gracefully**:
   - If a model fails to load, it will be skipped
   - The notebook will continue with successfully loaded models
   - Check the troubleshooting section if issues occur

### Step 5: Review Results

After completion, you'll see:

1. **BLEU Score Results**: Per-model statistics printed in the notebook
2. **Results Table**: Pandas DataFrame sorted by performance
3. **Visualization**: Bar chart comparing all models
4. **JSON File**: `vrsbench_captioning_results.json` with detailed results

## Notebook Structure

### Cell Breakdown

| Cell | Type | Purpose |
|------|------|---------|
| 0 | Markdown | Introduction and overview |
| 1 | Markdown | Step 1: Environment setup |
| 2 | Code | Install packages + clone repo |
| 3 | Code | Import libraries |
| 4 | Markdown | Step 2: Load dataset |
| 5 | Code | Load VRSBench from HuggingFace |
| 6 | Code | Verify dataset structure |
| 7 | Markdown | Step 3: Model loading functions |
| 8 | Code | `load_model_safely()` function |
| 9 | Markdown | Step 4: Caption generation |
| 10 | Code | Caption generation functions |
| 11 | Markdown | Step 5: Load models |
| 12 | Code | Load all 5 models |
| 13 | Markdown | Step 6: Prepare references |
| 14 | Code | Extract reference captions |
| 15 | Markdown | Step 7: Run inference |
| 16 | Code | Generate captions for all models |
| 17 | Markdown | Step 8: Calculate BLEU |
| 18 | Code | BLEU calculation functions |
| 19 | Code | Calculate BLEU scores |
| 20 | Markdown | Step 9: Display results |
| 21 | Code | Create results table |
| 22 | Code | Visualize results |
| 23 | Markdown | Step 10: Save results |
| 24 | Code | Save to JSON |
| 25 | Markdown | Troubleshooting guide |

## Customization Options

### Change Number of Samples

In **Cell 5**, modify:
```python
num_samples = min(1000, len(hf_dataset))  # Change 1000 to desired number
```

### Add/Remove Models

In **Cell 12**, modify the `MODELS_TO_EVALUATE` list:
```python
MODELS_TO_EVALUATE = [
    {
        "name": "Your Model Name",
        "hf_id": "huggingface/model-id",
        "type": "vision2seq"  # or "blip" or "causal"
    },
    # ... add more models
]
```

### Change Evaluation Metric

To use metrics other than BLEU (e.g., ROUGE, METEOR, CIDEr):

1. Install additional packages:
   ```python
   !pip install rouge-score pycocoevalcap
   ```

2. Add metric calculation functions in **Cell 18**

3. Modify **Cell 19** to calculate additional metrics

### Use Custom Dataloader

If you want to use your custom dataloader instead of HuggingFace datasets:

1. Ensure the repo is cloned (Cell 2)
2. Modify **Cell 5** to use:
   ```python
   from vrsbench_dataloader_production import create_vrsbench_dataloader
   
   dataloader = create_vrsbench_dataloader(
       images_dir="./images",
       task="captioning",
       annotations_url="https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Annotations_val.zip",
       batch_size=8,
       sample_size=1000
   )
   ```

## Expected Output

### Console Output
```
✓ Libraries imported successfully
✓ Loaded 1000 samples from VRSBench
✓ Qwen2.5-VL-8B ready for inference
✓ MiMo-VL-7B ready for inference
...
✓ Successfully loaded 5 out of 5 models
✓ Inference completed for all models

BLEU Score Results
============================================================
Evaluating Qwen2.5-VL-8B...
  Average BLEU-4: 0.3245
  Median BLEU-4: 0.3100
  ...
```

### Results Table
```
FINAL RESULTS SUMMARY
================================================================================
Model                  Average BLEU-4  Median BLEU-4  ...  Valid Samples
Qwen2.5-VL-8B         0.3245          0.3100        ...  1000/1000
MiMo-VL-7B            0.2987          0.2850        ...  1000/1000
...
```

### JSON Output
```json
{
  "evaluation_date": "2025-01-13 15:30:00",
  "dataset": "VRSBench (xiang709/VRSBench)",
  "task": "Image Captioning",
  "metric": "BLEU-4",
  "num_samples": 1000,
  "models_evaluated": ["Qwen2.5-VL-8B", "MiMo-VL-7B", ...],
  "results": {
    "Qwen2.5-VL-8B": {
      "average_bleu4": 0.3245,
      "median_bleu4": 0.3100,
      ...
    }
  }
}
```

## Troubleshooting

### Issue: Repository Clone Fails

**Solution**: 
- Verify your GitHub username and repository name are correct
- Ensure the repository is public (or use authentication)
- Check the URL format: `https://github.com/USERNAME/REPO_NAME.git`

### Issue: Model Fails to Load

**Solution**:
- Check the HuggingFace model identifier is correct
- Some models require `trust_remote_code=True` (already included)
- Verify you have enough GPU memory (try CPU if GPU fails)
- Check HuggingFace model card for specific requirements

### Issue: CUDA Out of Memory

**Solution**:
- Use smaller batch sizes
- Process fewer samples
- Use CPU instead of GPU (slower but works)
- Free up GPU memory: `torch.cuda.empty_cache()`

### Issue: Import Errors

**Solution**:
- Re-run Cell 2 to reinstall packages
- Restart runtime: "Runtime" → "Restart runtime"
- Check package versions are compatible

### Issue: BLEU Score is 0.0

**Possible causes**:
- Generated captions are None (model inference failed)
- Caption format mismatch
- Tokenization issues

**Solution**:
- Check that models generated valid captions
- Verify reference captions are not empty
- Check the generated captions in `all_generated_captions`

## Performance Estimates

| Component | Time Estimate |
|-----------|---------------|
| Package installation | 2-5 minutes |
| Dataset download | 5-10 minutes |
| Model loading (per model) | 2-5 minutes |
| Inference (1000 samples, per model) | 30-90 minutes |
| BLEU calculation | < 1 minute |
| **Total (5 models)** | **~3-8 hours** |

*Times vary based on GPU type, network speed, and model complexity*

## Best Practices

1. **Start Small**: Test with 10-50 samples first to verify everything works
2. **Save Progress**: Results are saved to JSON, so you can stop and resume
3. **Monitor GPU**: Watch GPU memory usage in Colab's resource monitor
4. **Check Logs**: Review error messages if models fail to load
5. **Verify Models**: Check HuggingFace for correct model identifiers before running

## Next Steps

After running the evaluation:

1. **Analyze Results**: Compare BLEU scores across models
2. **Examine Failures**: Check which samples/models had issues
3. **Fine-tune**: Adjust model parameters or prompts for better performance
4. **Extend Evaluation**: Add more metrics (ROUGE, METEOR, CIDEr)
5. **Visualize**: Create additional visualizations or export to reports

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review error messages in the notebook output
3. Verify model names on HuggingFace
4. Check your repository is accessible
5. Ensure all dependencies are installed correctly

---

**Created**: January 2025  
**Version**: 1.0  
**Compatible with**: Google Colab, Python 3.8+, PyTorch 1.12+

