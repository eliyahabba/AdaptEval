# Old Chain Linking Python Scripts

This folder contains archived Python scripts that have been superseded by the newer implementations.

## Archived Scripts

### `chain_linking_experiment.py` 
- Original chain linking implementation
- Multi-target design (evaluated all remaining datasets as targets)
- **Superseded by**: `chain_linking_parallel.py` (formerly `chain_linking_parallel_b.py`)
- **Why replaced**: The parallel version is faster and more efficient with multiple GPUs

### `chain_linking_v2.py`
- Second iteration with simplified single-target design
- Compared Fixed-Anchor vs Concurrent calibration
- **Superseded by**: `chain_linking_parallel.py` 
- **Why replaced**: Parallel version includes same functionality but with better parallelization

## Current Production Scripts

Located in parent directory ([`src/experiments/chain_linking/`](../)):

- **`chain_linking_parallel.py`** - Main production script for standard experiments
  - Full parallelization across all (distance, method) scenarios
  - Multi-GPU support with worker pool
  - Used by: `run_chain_linking_unified.sh --type parallel`

- **`chain_linking_disjoint.py`** - Zero-overlap model allocation experiments
  - Tests IRT chain linking with disjoint model groups
  - Used by: `run_chain_linking_unified.sh --type disjoint`

- **`aggregate_chain_results.py`** - Results aggregation utility

## Migration Notes

If you have old experiments referencing these scripts, they should continue to work. However, for new experiments, use the current production scripts via the unified runner:

```bash
# Instead of old scripts
sbatch sh_run/old/run_chain_linking.sh

# Use unified runner
sbatch sh_run/run_chain_linking_unified.sh --preset lb_standard
```

