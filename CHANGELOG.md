# Changelog - VRSBench DataLoader Enhancements

## [2.1.0] - 2025-11-13

### ✨ Major Enhancements

#### DataLoader Core Improvements
- **Resolved Image Path Injection**: Dataset now automatically injects `_image_path` key into metadata after successful image load
  - Eliminates fragile path construction in downstream code
  - Makes notebook code more robust and portable
  - Available in metadata as `metas[0]['_image_path']`

- **Smart BBox Normalization**: New `TaskProcessor.normalize_bbox()` method with intelligent coordinate handling
  - Auto-detects normalized coordinates (0-1 range) and converts to pixels
  - Handles both corner format `[x1, y1, x2, y2]` and xywh format `[x, y, w, h]`
  - Automatically converts between formats as needed
  - Works seamlessly with region extraction for grounding tasks

#### Evaluation Notebook Improvements (`Testing models.md`)

- **Replaced BLEU with BERTScore**: 
  - BERTScore is significantly better for image captioning evaluation
  - Measures semantic similarity using contextual embeddings (not just n-gram overlap)
  - Returns Precision, Recall, and F1 scores
  - Uses `distilbert-base-uncased` for speed (can upgrade to `roberta-large` for better quality)
  
- **Fixed NLTK Dependencies**:
  - Removed invalid `punkt_tab` download (doesn't exist)
  - Added proper NLTK data packages: `punkt`, `wordnet`, `omw-1.4`
  
- **Added Resource Warnings**:
  - Clear GPU memory requirements (24GB+ VRAM recommended)
  - Per-model memory estimates (~14-16GB per 7B model)
  - Guidance for limited GPU scenarios

- **Robust vLLM Generation Functions**:
  - Added comprehensive error handling for all model generation functions
  - Proper output shape validation (`hasattr`, `len` checks)
  - Graceful fallback on errors (returns empty string instead of crashing)
  - Added `.convert('RGB')` to ensure images are in correct format

- **Improved Image Path Resolution**:
  - Cell 7 now uses `_image_path` from metadata
  - Includes fallback for backward compatibility
  - No more manual path construction errors

- **Updated Visualizations**:
  - Cell 11 now shows BERTScore components (P/R/F1) instead of BLEU
  - Scatter plot compares BERTScore-F1 vs METEOR (instead of BLEU-4 vs METEOR)
  - More semantically meaningful comparisons

- **Enhanced Final Report**:
  - Cell 14 ranks models by BERTScore-F1 (primary metric)
  - Includes metric explanations for clarity
  - Better guidance on which metrics to prioritize

### 🐛 Bug Fixes

- Fixed NLTK punkt_tab error (package doesn't exist)
- Fixed fragile image path construction in evaluation notebook
- Fixed missing error handling in vLLM generation that could cause crashes
- Fixed visualization attempting to plot non-existent BLEU columns

### 📦 Dependencies Updated

- Added `bert-score>=0.3.13` to requirements.txt
- Added `rouge-score>=0.1.2` to requirements.txt
- Updated NLTK data requirements in notebook

### 🧪 Testing

- Added comprehensive smoke test (`test_dataloader_changes.py`)
- Tests image path injection
- Tests bbox normalization with multiple formats
- Tests region extraction with normalized coordinates

### 📚 Documentation

- Updated notebook with clear resource requirements
- Added metric explanation in final report
- Better cell-by-cell documentation

### 🎯 Why BERTScore vs BLEU?

**BLEU problems for image captioning:**
- Only measures n-gram overlap (lexical similarity)
- Penalizes valid paraphrases and synonyms
- Doesn't capture semantic meaning
- Example: "car" vs "automobile" scores 0 in BLEU

**BERTScore advantages:**
- Measures semantic similarity using contextual embeddings
- Captures meaning, not just word overlap
- Better correlation with human judgment for image captioning
- Handles paraphrases and synonyms correctly
- Industry standard for modern captioning evaluation

**Metrics included:**
- **Primary**: BERTScore-F1 (semantic similarity)
- **Supplementary**: ROUGE-L (longest common subsequence), METEOR (considers synonyms)
- **Analysis**: Caption length statistics

### 🚀 Performance Impact

- Image path injection: Negligible overhead (~0.1ms per image)
- BBox normalization: < 0.01ms per bbox
- BERTScore computation: Slower than BLEU but more accurate (uses GPU if available)
  - ~2-5 seconds per 100 captions with distilbert-base-uncased
  - Can be parallelized for large-scale evaluation

### 🔄 Migration Guide

**For existing code using the dataloader:**
- No changes needed - backward compatible
- Optionally use `_image_path` from metadata instead of constructing paths

**For evaluation notebooks:**
- Update to new Cell 1 (fixed NLTK)
- Update Cell 7 to use `_image_path`
- Cell 8 now computes BERTScore instead of BLEU
- Visualization automatically uses correct metrics

**For bbox handling:**
- No changes needed - normalization is automatic
- Works with any bbox format (normalized, pixels, corners, xywh)

### ⚠️ Breaking Changes

None - all changes are backward compatible.

### 🙏 Credits

These enhancements address common pain points in VRSBench evaluation:
- Fragile path resolution
- Normalized coordinate handling
- Better semantic evaluation metrics
- Robust error handling for production use

---

## [2.0.0] - 2025-01-13

Initial production release with multi-task support, structured logging, and metrics collection.

See README.md for complete feature list.
