# Chain Linking Experiment Runner

Unified system for running chain linking experiments with preset configurations and automatic seed management.

## Quick Start

```bash
# Using a preset (output-dir is REQUIRED)
sbatch sh_run/run_chain_linking_unified.sh \
    --output-dir data/v24_lb \
    --preset lb_standard \
    --seed 11

# With specific target dataset
sbatch sh_run/run_chain_linking_unified.sh \
    --output-dir data/v24_lb \
    --preset lb_standard \
    --seed 11 \
    --target "MMLU"

# Override preset parameters
sbatch sh_run/run_chain_linking_unified.sh \
    --output-dir data/v24_lb \
    --preset lb_standard \
    --n-anchors 50 \
    --n-models 100

# Custom configuration (no preset)
sbatch sh_run/run_chain_linking_unified.sh \
    --output-dir data/v24_custom \
    --data-source lb \
    --n-base 1 \
    --n-anchors 100
```

## Available Presets

All presets are defined in [`experiment_presets.yaml`](experiment_presets.yaml).

### Open LLM Leaderboard (LB) - 6 datasets, 395 models

| Preset | Description | Anchors | Models |
|--------|-------------|---------|--------|
| `lb_standard` | Standard configuration | 100 | all |
| `lb_models_100` | With 100 chain models | 100 | 100 |
| `lb_models_50` | With 50 chain models | 100 | 50 |
| `lb_anchors_25` | Low anchor count | 25 | all |
| `lb_anchors_50` | Medium anchor count | 50 | all |
| `lb_anchors_200` | High anchor count | 200 | all |
| `lb_epochs_20000` | Extended training | 100 | all |

### HELM Lite - 9 datasets, 91 models

| Preset | Description | Base | Anchors |
|--------|-------------|------|---------|
| `helm_lite_base1` | 1 base dataset | 1 | 50 |
| `helm_lite_base4` | 4 base datasets | 4 | 25 |

### MMLU Fields - 57 datasets, 428 models

| Preset | Description | Base | Anchors |
|--------|-------------|------|---------|
| `mmlu_fields` | MMLU subdomains | 8 | 10 |

### Disjoint Experiments

| Preset | Description | Bridge | Isolated |
|--------|-------------|--------|----------|
| `lb_disjoint` | Fixed bridge models | 20 | 50 |
| `lb_disjoint_random` | Random bridge models | 20 | 50 |

## Command Line Parameters

### Core Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--output-dir` | **REQUIRED** Base directory for experiments | `--output-dir data/v24_lb` |
| `--preset` | Use named preset | `--preset lb_standard` |
| `--seed` | Shuffle seed (determines dataset order & model splits) | `--seed 11` |
| `--random-seed` | Seed for random baseline scenarios | `--random-seed 1000` |
| `--target` | Specific target dataset | `--target "MMLU"` |
| `--type` | Experiment type: parallel (default) or disjoint | `--type disjoint` |
| `--skip-existing` | Skip if output dir exists (resume failures only) | `--skip-existing` |
| `--setup-only` | Only create config, don't run experiment | `--setup-only` |

**Understanding the two seeds:**
- `--seed` (shuffle_seed): Controls which datasets become base/chain/target and train/test model split. Keep this constant across experiments where you want the same data configuration.
- `--random-seed`: Controls the random question selection in random baseline validation methods. Change this to get independent random baseline comparisons.

### Data Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--data-source` | Data source mode | `--data-source lb` |
| `--n-base` | Number of base datasets | `--n-base 1` |
| `--max-chain` | Maximum chain length | `--max-chain 5` |
| `--n-anchors` | Anchors per dataset | `--n-anchors 100` |
| `--n-models` | Models for chain training | `--n-models 50` |

### Training Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--dims` | IRT dimensions | `--dims 5` |
| `--epochs` | Concurrent training epochs | `--epochs 2000` |
| `--epochs-fixed` | Fixed-anchor epochs | `--epochs-fixed 1000` |
| `--num-workers` | Parallel workers | `--num-workers 4` |

### SLURM Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--memory` | Memory allocation | `--memory 12g` |
| `--gpus` | Number of GPUs | `--gpus 4` |
| `--time` | Time limit | `--time "5:0:0"` |

## Seed Management

The script supports **two modes** for handling existing experiments:

### 1. Auto-Increment Mode (Default)

The unified runner automatically finds the next available seed to avoid conflicts with running experiments. It checks for both exact directory matches AND directories with target suffixes:

```bash
# Request seed 11
sbatch run_chain_linking_unified.sh --output-dir data/v24_lb --preset lb_standard --seed 11

# If seed_11_target_* directory exists, automatically uses seed_12, seed_13, etc.
# Output: "Using shuffle_seed: 17" (next available)
```

**Benefits:**
- Multiple workers can submit jobs in parallel without conflicts
- All experiments complete, just with different seeds

### 2. Skip-Existing Mode (Resume Failures)

Use `--skip-existing` to skip experiments where the output directory already exists. This is useful when you've run many experiments and only want to re-run the ones that failed:

```bash
# Will skip if output directory exists, won't auto-increment seed
sbatch run_chain_linking_unified.sh \
    --output-dir data/v24_lb \
    --preset lb_standard \
    --seed 11 \
    --skip-existing

# Output if exists: "⏭️  SKIP: Output directory already exists..."
# Exit code: 0 (success, skipped)
```

**Benefits:**
- Resume a batch of experiments without running successful ones again
- Faster re-runs after cluster failures
- No need to track which experiments succeeded/failed manually

**Example with run_all_experiments.sh:**

```bash
# First run (some fail due to cluster issues)
bash sh_run/run_all_experiments.sh

# Later: resume only failed experiments
bash sh_run/run_all_experiments.sh --skip-existing
```
- No need to manually track which seeds have been used
- Each experiment gets a unique output directory
- Correctly detects existing experiments even with `_target_<name>` suffix

To disable this behavior:
```bash
sbatch run_chain_linking_unified.sh --preset lb_standard --seed 11 --no-auto-increment
```

## Output Directory Structure

You **must** specify `--output-dir` for all experiments. The Python script will automatically append the target dataset name to the directory:

```
data/v24_lb/  (your specified output-dir)
  ├── full_chain_classic_seed_11_anchors_100_target_TruthfulQA/
  ├── full_chain_classic_seed_12_anchors_100_models_50_target_MMLU/
  ├── full_chain_classic_seed_13_anchors_50_target_HellaSwag/
  └── full_chain_classic_seed_21_anchors_100_target_ARC/
```

**Important:** The Python script creates the final directory with the `_target_<name>` suffix after determining which dataset will be the target. This ensures each experiment directory clearly shows its target dataset.

Each experiment directory contains:
- `config.json` - Experiment configuration
- `all_results.csv` - Aggregated results
- `all_results.json` - Detailed results
- `dist_0_direct/`, `dist_1_*/`, etc. - Per-distance results

## Common Use Cases

### 1. Run all 6 LB target datasets

```bash
# Submit 6 jobs, each with auto-incremented seed
for i in {1..6}; do
    sbatch sh_run/run_chain_linking_unified.sh \
        --output-dir data/v24_lb_sweep \
        --preset lb_standard \
        --seed 11
done
```

### 2. Anchor sweep experiment

```bash
# Same dataset/model configuration, different anchors
# Use same --seed to keep dataset order, but vary --random-seed for independent random baselines
for anchors in 25 50 100 200; do
    sbatch sh_run/run_chain_linking_unified.sh \
        --output-dir data/v24_anchor_sweep \
        --preset lb_standard \
        --n-anchors $anchors \
        --seed 21 \
        --random-seed $((1000 + anchors))
done
```

### 3. Model count experiment

```bash
for models in 50 100 200; do
    sbatch sh_run/run_chain_linking_unified.sh \
        --output-dir data/v24_model_sweep \
        --preset lb_standard \
        --n-models $models \
        --seed 31
done
```

### 4. Specific target with explicit path

```bash
sbatch sh_run/run_chain_linking_unified.sh \
    --output-dir data/v24_lb_paper \
    --preset lb_standard \
    --seed 11 \
    --target "MMLU"
```

### 5. Disjoint experiment

```bash
# Using run_chain_linking_unified.sh (if using unified approach)
sbatch sh_run/run_chain_linking_unified.sh \
    --output-dir data/v24_disjoint \
    --preset lb_disjoint \
    --seed 21

# Using run_chain_linking_disjoint_lb.sh (direct disjoint script)
sbatch sh_run/run_chain_linking_disjoint_lb.sh \
    data/v24_disjoint/full_chain_disjoint 42 100 20 50 "" fixed

# With skip-existing (8th positional argument)
sbatch sh_run/run_chain_linking_disjoint_lb.sh \
    data/v24_disjoint/full_chain_disjoint 42 100 20 50 "" fixed skip
```

## Creating Custom Presets

Edit [`experiment_presets.yaml`](experiment_presets.yaml) to add your own presets:

```yaml
presets:
  my_custom_preset:
    description: "My custom configuration"
    data_source_mode: lb
    n_base: 1
    max_chain: 5
    n_anchors: 75
    n_models_per_chain: 150
    slurm:
      memory: 10g
      time: "8:0:0"
```

Then use it:
```bash
sbatch sh_run/run_chain_linking_unified.sh --preset my_custom_preset --seed 11
```

## Disjoint Experiments Script

The `run_chain_linking_disjoint_lb.sh` script runs experiments with zero model overlap between datasets.

**Positional Arguments:**
1. `output_dir` - Base output directory
2. `shuffle_seed` - Seed for dataset/model split (default: 42)
3. `n_anchors` - Anchors per dataset (default: 100)
4. `n_bridge` - Bridge models count (default: 20)
5. `n_isolated` - Isolated models per chain (default: 50)
6. `target_dataset` - Target dataset name (default: auto)
7. `bridge_mode` - "fixed" or "random" (default: "fixed")
8. `skip_existing` - Pass "skip" to skip if exists (optional)

**Examples:**
```bash
# Basic usage
sbatch sh_run/run_chain_linking_disjoint_lb.sh \
    data/v24_disjoint/output 42 100 20 50 "" fixed

# With skip-existing
sbatch sh_run/run_chain_linking_disjoint_lb.sh \
    data/v24_disjoint/output 42 100 20 50 "" fixed skip
```

## Monitoring Experiments

```bash
# Check SLURM queue
squeue -u $USER

# View job output
tail -f slurm-<job_id>.out

# Check experiment progress
ls -la data/v23_lb/
```

## Migration from Old Scripts

| Old Script | New Command |
|------------|-------------|
| `run_chain_linking_parallel_b.sh /path 42 100` | `--preset lb_standard --seed 42 --n-anchors 100` |
| `run_chain_linking_parallel_lb.sh /path 42 50` | `--preset lb_standard --seed 42 --n-anchors 50` |
| `run_chain_linking_parallel_mmlu_fields.sh` | `--preset mmlu_fields` |

Old scripts are archived in [`old/`](old/) folder for reference.

## Troubleshooting

**Problem:** "Error: Preset file not found"
- **Solution:** Make sure you're running from project root or specify full path

**Problem:** Seed auto-increment not working
- **Solution:** Check that version directory exists and is writable

**Problem:** SLURM submission fails
- **Solution:** Check SLURM parameters match your cluster configuration

## Additional Resources

- Experiment presets: [`experiment_presets.yaml`](experiment_presets.yaml)
- Python scripts: [`../src/experiments/chain_linking/`](../src/experiments/chain_linking/)
- Archived scripts: [`old/`](old/)

