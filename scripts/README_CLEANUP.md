# Chain Linking Results Cleanup Scripts

## 🎯 Quick Start

### NEW: Automatic Cleanup (No Manual Steps Needed!)

Starting now, **new experiments automatically clean up training data** during execution:

```bash
python src/experiments/chain_linking/chain_linking_parallel.py \
  --data-source-mode helm_classic \
  --n-base 6
```

**What happens automatically**:
- ✅ Training completes → Parameters saved
- 🧹 **Immediately after**: Deletes `*.jsonlines` (training datasets)
- 💾 **Savings**: 17MB → 1.7MB per IRT directory (90%)
- 📦 **Keeps**: `item_params.parquet` + metadata (can still use models!)

**Result**: Experiments now use **~150MB instead of 11GB** from the start! 🎉

---

### For Existing Experiments (Manual Cleanup):

```bash
# Check what will be deleted (dry run):
./scripts/batch_cleanup.sh /path/to/data/v3

# Actually delete:
./scripts/batch_cleanup.sh /path/to/data/v3 --force
```

**Result**: Saves ~90% storage by removing temporary files while keeping all important results!

### 💡 NEW: Keep IRT Parameters, Delete Training Data

Save even more space while keeping the ability to use trained models:

```bash
# Remove only training datasets (*.jsonlines) - saves ~88% from IRT dirs:
./scripts/clean_training_datasets.sh /path/to/experiment --force

# Or via Python:
python scripts/cleanup_chain_results.py /path/to/experiment --models --keep-params

# Compress metadata for additional ~70% savings:
python scripts/compress_irt_metadata.py /path/to/experiment --force --recursive
```

**What's kept**:
- ✅ `item_params.parquet` - Model parameters (needed!)
- ✅ `item_params.meta.json.gz` - Compressed metadata
- 🗑️ `irt_dataset_final.jsonlines` - Training data (not needed after training!)
- 🗑️ `irt_val_dataset_dim5.jsonlines` - Validation data (not needed!)

**Savings**: 17MB → ~1.5MB per IRT directory (91%)

---

## 📚 Available Scripts

### 1. **`batch_cleanup.sh`** - Best for cleaning multiple experiments
```bash
# Dry run on all experiments in directory:
./scripts/batch_cleanup.sh /path/to/data/v3

# Actually clean:
./scripts/batch_cleanup.sh /path/to/data/v3 --force
```

**What it cleans**:
- ✅ `.temp/` - temporary worker files (7GB in your case)
- ✅ `chain_cache/` - resumption cache (2.7GB)
- ✅ `irt_base/` - base IRT models (119MB)
- ✅ `dist_*/irt_*/` - all IRT models (~500MB)

**What it keeps**:
- ✅ `all_results.csv/json` - main results
- ✅ `config.json` - experiment settings
- ✅ `dist_*/results.json` - per-distance metrics
- ✅ `dist_*/validation_*.csv` - per-model results

---

### 2. **`cleanup_chain_results.py`** - Python version with more options
```bash
# Single experiment:
python scripts/cleanup_chain_results.py /path/to/experiment --all

# All experiments in directory:
python scripts/cleanup_chain_results.py /path/to/data --all --recursive

# Only specific types:
python scripts/cleanup_chain_results.py /path/to/experiment --temp --cache
python scripts/cleanup_chain_results.py /path/to/experiment --models
```

**Options**:
- `--temp` - Clean `.temp/` directory
- `--cache` - Clean `chain_cache/`
- `--models` - Clean IRT models from `dist_*/irt_*/`
- `--irt-base` - Clean `irt_base/`
- `--all` - Clean everything
- `--dry-run` - Show what would be deleted
- `--recursive` - Process all subdirectories

---

### 3. **`quick_cleanup.sh`** - Simple bash script for single experiment
```bash
# Dry run:
./scripts/quick_cleanup.sh /path/to/experiment

# Actually delete:
./scripts/quick_cleanup.sh /path/to/experiment --force
```

---

### 4. **`clean_training_datasets.sh`** - Remove only training data (SAFE!)
```bash
# Remove *.jsonlines files, keep item_params:
./scripts/clean_training_datasets.sh /path/to/experiment --force
```

Saves ~88% space from IRT directories while keeping model parameters intact.

---

### 5. **`compress_irt_metadata.py`** - Compress metadata files
```bash
# Compress item_params.meta.json → item_params.meta.json.gz:
python scripts/compress_irt_metadata.py /path/to/experiment --force

# Recursive (all experiments):
python scripts/compress_irt_metadata.py /path/to/data --recursive --force
```

Saves ~70% space from metadata files.

---

## 🚀 Automatic Cleanup (New Experiments)

When running new experiments, use these flags:

```bash
python src/experiments/chain_linking/chain_linking_parallel.py \
  --data-source-mode helm_classic \
  --cleanup-cache \      # Auto-delete chain_cache when done (default: True)
  --cleanup-models       # Auto-delete IRT models when done (default: False)
```

**Defaults**:
- ✅ `.temp/` - **Always cleaned automatically**
- ✅ `chain_cache/` - **Cleaned by default** (use `--no-cleanup-cache` to keep)
- ❌ `dist_*/irt_*/` - **Not cleaned** (use `--cleanup-models` to clean)

---

## 💾 Storage Savings Example

Your case (`full_chain_classic_seed_26_anchors_100_target_TruthfulQA`):

### Option A: Full Cleanup (Delete all IRT models)

| Directory | Before | After | Savings |
|-----------|--------|-------|---------|
| `.temp/` | 7.1GB | 0 | 7.1GB |
| `chain_cache/` | 2.7GB | 0 | 2.7GB |
| `irt_base/` | 119MB | 0 | 119MB |
| `dist_*/irt_*/` | ~500MB | 0 | ~500MB |
| **Results** | ~100MB | ~100MB | 0 |
| **Total** | **11GB** | **~100MB** | **~10.9GB (99%)** |

### Option B: Keep IRT Parameters (RECOMMENDED!)

| Directory | Before | After | Savings |
|-----------|--------|-------|---------|
| `.temp/` | 7.1GB | 0 | 7.1GB |
| `chain_cache/` | 2.7GB | 0 | 2.7GB |
| `irt_base/` | 119MB | 0 | 119MB |
| `dist_*/irt_*/*.jsonlines` | ~440MB | 0 | ~440MB |
| `dist_*/irt_*/item_params.*` | ~60MB | ~20MB | ~40MB |
| **Results** | ~100MB | ~100MB | 0 |
| **Total** | **11GB** | **~140MB** | **~10.86GB (98.7%)** |

**Option B Benefits**:
- ✅ Can still use trained models
- ✅ Minimal storage penalty (~40MB extra vs Option A)
- ✅ Enables future analysis without retraining

---

## 📖 Full Documentation

See [`docs/chain_linking_storage.md`](../docs/chain_linking_storage.md) for detailed information about:
- What each file type contains
- When to delete vs keep
- Manual cleanup commands
- Troubleshooting

---

## ⚠️ Important Notes

1. **Always keep**:
   - `all_results.csv/json`
   - `config.json`
   - `dist_*/results.json`
   - `dist_*/validation_*.csv/json`

2. **Safe to delete after experiment succeeds**:
   - `.temp/` (should be auto-deleted)
   - `chain_cache/` (only needed for resuming failed runs)

3. **Optional to delete**:
   - `dist_*/irt_*/` - Only needed if you want to reload trained models
   - `irt_base/` - Only needed for debugging base IRT

4. **Before deleting**:
   - Always run with dry-run first
   - Make sure experiment completed successfully
   - Keep one backup copy until you verify results are correct

