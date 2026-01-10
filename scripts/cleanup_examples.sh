#!/bin/bash
# Example: Cleanup all experiments in v3 directory
# This script shows how to use the cleanup tools

DATA_DIR="/cslab/gabis_lab/ehabba/AdaptEval/data/v3"

echo "Example 1: Dry run on single experiment"
echo "========================================"
python scripts/cleanup_chain_results.py \
  "$DATA_DIR/full_chain_classic_seed_26_anchors_100_target_TruthfulQA" \
  --all --dry-run

echo ""
echo "Example 2: Actually clean single experiment"
echo "=========================================="
python scripts/cleanup_chain_results.py \
  "$DATA_DIR/full_chain_classic_seed_26_anchors_100_target_TruthfulQA" \
  --temp --cache

echo ""
echo "Example 3: Clean all experiments recursively (dry run)"
echo "====================================================="
python scripts/cleanup_chain_results.py \
  "$DATA_DIR" \
  --all --recursive --dry-run

echo ""
echo "Example 4: Quick cleanup with bash script"
echo "=========================================="
./scripts/quick_cleanup.sh \
  "$DATA_DIR/full_chain_classic_seed_26_anchors_100_target_TruthfulQA"

echo ""
echo "Example 5: Run new experiment with auto-cleanup"
echo "==============================================="
python src/experiments/chain_linking/chain_linking_parallel.py \
  --data-source-mode helm_classic \
  --n-base 6 \
  --max-chain 5 \
  --cleanup-cache \
  --cleanup-models \
  --output-dir "$DATA_DIR/new_experiment"

