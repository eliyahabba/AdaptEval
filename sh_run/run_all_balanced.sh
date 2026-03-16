#!/bin/bash
# run_all_balanced.sh
# THE SINGLE DEFINITIVE SCRIPT FOR ALL EXPERIMENTS
#
# Covers:
# - LB: 60 experiments (6 datasets × configs) - balanced
#   - Baseline Fixed: 24 experiments (4 seeds per dataset) → lb_baseline_fixed/
#   - Anchor Sweep: 24 experiments (4 anchor counts × 6 datasets)
#   - Model Sweep: 12 experiments (2 model counts × 6 datasets)
# - HELM Lite: 18 experiments (9 base1 + 9 base4)
# - MMLU Fields: 20 experiments (seeds 11-30)
# - Disjoint: 12 experiments (6 fixed + 6 random bridge)
# - LB Extended Model Sweep: 9 model counts × 6 datasets × N seeds → lb_model_sweep_extended_fixed/
# - MMLU Extended Model Sweep: 9 model counts × N seeds → mmlu_model_sweep_extended_fixed/
# - HELM Lite Extended Model Sweep: 6 model counts × 9 datasets × N seeds → helm_lite_model_sweep_extended_fixed/
# - MMLU Anchor Sweep: 4 seeds (targets) × 4 anchor counts (10, 25, 50, 100) → mmlu_anchor_sweep_controlled/
# - MMLU Baseline 50 anchors: same as cat 3 but 50 anchors → mmlu_baseline_50/
#
# Total: depends on --seeds N (default N=3)
#
# Usage:
#   bash run_all_balanced.sh                    # Run all missing
#   bash run_all_balanced.sh --category 1       # Only LB
#   bash run_all_balanced.sh --category 2       # Only HELM Lite
#   bash run_all_balanced.sh --category 3       # Only MMLU (10 anchors)
#   bash run_all_balanced.sh --category 4       # Only Disjoint
#   bash run_all_balanced.sh --category 5       # Only LB Model Sweep Extended
#   bash run_all_balanced.sh --category 6       # Only MMLU Model Sweep Extended
#   bash run_all_balanced.sh --category 7       # Only HELM Lite Model Sweep Extended
#   bash run_all_balanced.sh --category 8       # Only MMLU Anchor Sweep (10, 25, 50, 100)
#   bash run_all_balanced.sh --category 9       # Only MMLU Baseline 50 anchors
#   bash run_all_balanced.sh --seeds 5          # Use 5 seeds per config (categories 5,6,7)
#   bash run_all_balanced.sh --setup-only       # Test only
#   bash run_all_balanced.sh --force            # Re-run all (ignore existing)
#   bash run_all_balanced.sh --resume           # Resume incomplete experiments

set -e

BASE_DIR="data/v29_after_changes"
SETUP_ONLY=""
SKIP_EXISTING=""
FORCE_RESUME=""
RUN_CATEGORY=""
N_SEEDS=3  # Number of seeds per dataset per model count (for categories 5, 6, 7)

# LB datasets
LB_DATASETS=("MMLU" "ARC Challenge" "HellaSwag" "TruthfulQA" "Winogrande" "GSM8K")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --setup-only) SETUP_ONLY="--setup-only"; shift ;;
        --force) SKIP_EXISTING=""; FORCE_RESUME=""; shift ;;  # Force re-run
        --resume) FORCE_RESUME="--force-resume"; shift ;;
        --skip-existing) SKIP_EXISTING="--skip-existing"; shift ;;
        --category) RUN_CATEGORY="$2"; shift 2 ;;
        --base-dir) BASE_DIR="$2"; shift 2 ;;
        --seeds) N_SEEDS="$2"; shift 2 ;;  # Number of seeds for categories 5,6,7
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Default: skip existing if not forcing or resuming
if [ -z "$FORCE_RESUME" ] && [ -z "$SKIP_EXISTING" ]; then
    SKIP_EXISTING="--skip-existing"
fi

# =============================================================================
# Helper: Check if experiment is already complete
# Returns 0 (true) if complete, 1 (false) if incomplete/missing
# =============================================================================
is_experiment_complete() {
    local output_dir="$1"
    local seed="$2"
    local anchors="$3"
    local target="$4"
    local models="$5"
    
    # Build expected directory name base (same logic as run_chain_linking_unified.sh)
    local dir_base="full_chain_classic_seed_${seed}_anchors_${anchors}"
    if [ -n "$models" ]; then
        dir_base="${dir_base}_models_${models}"
    fi

    # Handle both target naming variants:
    # - raw target name (current run_chain_linking_unified.sh behavior)
    # - underscored target name (historical/other scripts)
    if [ -n "$target" ]; then
        local target_clean="${target// /_}"
        local full_path_raw="${output_dir}/${dir_base}_target_${target}"
        local full_path_clean="${output_dir}/${dir_base}_target_${target_clean}"

        if [ -f "${full_path_raw}/all_results.csv" ] || [ -f "${full_path_clean}/all_results.csv" ]; then
            return 0  # Complete
        fi
        return 1  # Incomplete or missing
    fi

    local full_path="${output_dir}/${dir_base}"
    if [ -f "${full_path}/all_results.csv" ]; then
        return 0  # Complete
    fi
    return 1  # Incomplete or missing
}

# Helper: Check if DISJOINT experiment is complete
is_disjoint_complete() {
    local output_base="$1"
    local seed="$2"
    local anchors="$3"
    local bridge="$4"
    local bridge_mode="$5"
    local isolated="$6"
    
    local dir_name="${output_base}_seed_${seed}_anchors_${anchors}_bridge_${bridge}_${bridge_mode}_isolated_${isolated}"
    
    if [ -f "${dir_name}/all_results.csv" ]; then
        return 0  # Complete
    fi
    return 1  # Incomplete or missing
}

# Counter for tracking
SUBMITTED=0
SKIPPED=0

# Helper: Submit job with full command echo
submit_job() {
    echo "   → $@"
    "$@"
}

echo "=================================================================="
echo "Complete Experiment Suite (Balanced)"
echo "=================================================================="
echo "Base: $BASE_DIR"
[ -n "$SETUP_ONLY" ] && echo "Mode: SETUP-ONLY (test mode)"
[ -n "$FORCE_RESUME" ] && echo "Mode: RESUME (continue incomplete experiments)"
[ -n "$SKIP_EXISTING" ] && echo "Mode: SKIP-EXISTING (skip completed)"
[ -n "$RUN_CATEGORY" ] && echo "Running: Category $RUN_CATEGORY only"
echo "Seeds per config (cat 5,6,7): $N_SEEDS"
echo ""

mkdir -p "${BASE_DIR}/lb_baseline_fixed" "${BASE_DIR}/lb_anchor_sweep" "${BASE_DIR}/lb_anchor_sweep_controlled" "${BASE_DIR}/lb_model_sweep" "${BASE_DIR}/lb_model_sweep_controlled"
mkdir -p "${BASE_DIR}/helm_lite_baseline" "${BASE_DIR}/helm_lite_base4"
mkdir -p "${BASE_DIR}/mmlu_baseline" "${BASE_DIR}/mmlu_baseline_50" "${BASE_DIR}/mmlu_anchor_sweep_controlled"
mkdir -p "${BASE_DIR}/lb_disjoint_fixed" "${BASE_DIR}/lb_disjoint_random"

# ============================================================
# CATEGORY 1: LB (42 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "1" ]; then
    echo "=================================================================="
    echo "CATEGORY 1: LB (60 experiments)"
    echo "=================================================================="
    echo ""
    
    # 1.1: Baseline (4 seeds per dataset = 24 experiments)
    # Output to lb_baseline_fixed to distinguish from old experiments with anchor bug
    echo "-- LB Baseline Fixed (24 experiments) --"
    SEED=21
    for target in "${LB_DATASETS[@]}"; do
        for offset in 0 1 2 3; do
            RUN_SEED=$((SEED + offset))
            # Check if already complete when resuming
            if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/lb_baseline_fixed" $RUN_SEED 100 "$target" ""; then
                echo "   ⏭️  SKIP (complete): $target (seed=$RUN_SEED)"
                SKIPPED=$((SKIPPED + 1))
            else
                echo "   Submitting $target (baseline, seed=$RUN_SEED)..."
                submit_job sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                    --output-dir ${BASE_DIR}/lb_baseline_fixed \
                    --preset lb_standard \
                    --shuffle-seed $RUN_SEED \
                    --seed $RUN_SEED \
                    --target "$target" \
                    --random-seed 1000 \
                    $SKIP_EXISTING $FORCE_RESUME
                SUBMITTED=$((SUBMITTED + 1))
            fi
        done
        SEED=$((SEED + 4))
    done
    
    # 1.2: Anchor Sweep - CONTROLLED (same seed per target, only anchors vary)
    # This allows fair comparison: same train/test split, same chain order
    # Output to NEW directory to avoid mixing with old uncontrolled experiments
    echo ""
    echo "-- LB Anchor Sweep CONTROLLED (24 experiments) --"
    echo "   Same seed per target, only anchors vary (25, 50, 100, 200)"
    echo "   Output: ${BASE_DIR}/lb_anchor_sweep_controlled"
    SEED=31
    for target in "${LB_DATASETS[@]}"; do
        # SAME seed for all anchor counts of this target!
        TARGET_SEED=$SEED
        for anchors in 25 50 100 200; do
            if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/lb_anchor_sweep_controlled" $TARGET_SEED $anchors "$target" ""; then
                echo "   ⏭️  SKIP (complete): $target anchors=$anchors (seed=$TARGET_SEED)"
                SKIPPED=$((SKIPPED + 1))
            else
                echo "   Submitting $target (anchors=$anchors, seed=$TARGET_SEED)..."
                submit_job sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                    --output-dir ${BASE_DIR}/lb_anchor_sweep_controlled \
                    --preset lb_standard \
                    --n-anchors $anchors \
                    --shuffle-seed $TARGET_SEED \
                    --seed $TARGET_SEED \
                    --target "$target" \
                    --random-seed $((1000 + anchors)) \
                    $SKIP_EXISTING $FORCE_RESUME
                SUBMITTED=$((SUBMITTED + 1))
            fi
        done
        SEED=$((SEED + 1))  # Next target gets next seed
    done
    
    # 1.3: Model Sweep - CONTROLLED (same seed per target, only model count varies)
    # Output to NEW directory to avoid mixing with old uncontrolled experiments
    echo ""
    echo "-- LB Model Sweep CONTROLLED (12 experiments) --"
    echo "   Same seed per target, only model count varies (50, 100)"
    echo "   Output: ${BASE_DIR}/lb_model_sweep_controlled"
    SEED=61
    for target in "${LB_DATASETS[@]}"; do
        # SAME seed for all model counts of this target!
        TARGET_SEED=$SEED
        for models in 50 100; do
            if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/lb_model_sweep_controlled" $TARGET_SEED 100 "$target" "$models"; then
                echo "   ⏭️  SKIP (complete): $target models=$models (seed=$TARGET_SEED)"
                SKIPPED=$((SKIPPED + 1))
            else
                echo "   Submitting $target (models=$models, seed=$TARGET_SEED)..."
                submit_job sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                    --output-dir ${BASE_DIR}/lb_model_sweep_controlled \
                    --preset lb_standard \
                    --n-models $models \
                    --shuffle-seed $TARGET_SEED \
                    --seed $TARGET_SEED \
                    --target "$target" \
                    --random-seed $((2000 + models)) \
                    $SKIP_EXISTING $FORCE_RESUME
                SUBMITTED=$((SUBMITTED + 1))
            fi
        done
        SEED=$((SEED + 1))  # Next target gets next seed
    done
    
    echo ""
    echo "✓ Category 1 complete: 60 experiments (24 baseline + 24 anchor + 12 model)"
fi

# ============================================================
# CATEGORY 2: HELM Lite (18 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "2" ]; then
    echo ""
    echo "=================================================================="
    echo "CATEGORY 2: HELM Lite (18 experiments)"
    echo "=================================================================="
    echo ""
    
    # 2.1: Base=1 (9 experiments, seeds 11-19)
    echo "-- HELM Lite Baseline (base=1, 9 experiments) --"
    for seed in 11 12 13 14 15 16 17 18 19; do
        if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/helm_lite_baseline" $seed 50 "" ""; then
            echo "   ⏭️  SKIP (complete): HELM Lite base=1 (seed=$seed)"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "   Submitting HELM Lite base=1 (seed=$seed)..."
            submit_job sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/helm_lite_baseline \
                --preset helm_lite_base1 \
                --shuffle-seed $seed \
                --seed $seed \
                --random-seed 1000 \
                $SKIP_EXISTING $FORCE_RESUME
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    
    # 2.2: Base=4 (9 experiments, seeds 11-19)
    echo ""
    echo "-- HELM Lite Base4 (base=4, 9 experiments) --"
    for seed in 11 12 13 14 15 16 17 18 19; do
        if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/helm_lite_base4" $seed 25 "" ""; then
            echo "   ⏭️  SKIP (complete): HELM Lite base=4 (seed=$seed)"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "   Submitting HELM Lite base=4 (seed=$seed)..."
            submit_job sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/helm_lite_base4 \
                --preset helm_lite_base4 \
                --shuffle-seed $seed \
                --seed $seed \
                --random-seed 3000 \
                $SKIP_EXISTING $FORCE_RESUME
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    
    echo ""
    echo "✓ Category 2 complete: 18 experiments submitted"
fi

# ============================================================
# CATEGORY 3: MMLU Fields (20 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "3" ]; then
    echo ""
    echo "=================================================================="
    echo "CATEGORY 3: MMLU Fields (20 experiments)"
    echo "=================================================================="
    echo ""
    
    echo "-- MMLU Fields Baseline (seeds 11-30, 10 anchors) --"
    for seed in $(seq 11 30); do
        if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/mmlu_baseline" $seed 10 "" ""; then
            echo "   ⏭️  SKIP (complete): MMLU Fields (seed=$seed)"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "   Submitting MMLU Fields (seed=$seed)..."
            submit_job sbatch --mem=8g --time=10:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/mmlu_baseline \
                --preset mmlu_fields \
                --shuffle-seed $seed \
                --seed $seed \
                --random-seed 1000 \
                $SKIP_EXISTING $FORCE_RESUME
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    
    echo ""
    echo "✓ Category 3 complete: 20 experiments submitted"
fi

# ============================================================
# CATEGORY 4: Disjoint Experiments - LB (12 experiments)
# ============================================================
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "4" ]; then
    echo ""
    echo "=================================================================="
    echo "CATEGORY 4: Disjoint Experiments - LB (12 experiments)"
    echo "=================================================================="
    echo ""
    
    # Disjoint skip mode
    DISJOINT_SKIP_ARG=""
    [ -n "$SKIP_EXISTING" ] && DISJOINT_SKIP_ARG="skip"
    
    # 4.1: Fixed Bridge (6 experiments, seeds 41-46)
    echo "-- Disjoint Fixed Bridge (6 experiments) --"
    echo "   bridge=20, isolated=50"
    for seed in 41 42 43 44 45 46; do
        if [ -n "$FORCE_RESUME" ] && is_disjoint_complete "${BASE_DIR}/lb_disjoint_fixed/full_chain_disjoint" $seed 100 20 "fixed" 50; then
            echo "   ⏭️  SKIP (complete): Disjoint fixed (seed=$seed)"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "   Submitting Disjoint fixed (seed=$seed)..."
            submit_job sbatch $SETUP_ONLY sh_run/run_chain_linking_disjoint_lb.sh \
                ${BASE_DIR}/lb_disjoint_fixed/full_chain_disjoint $seed 100 20 50 "" fixed $DISJOINT_SKIP_ARG
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    
    # 4.2: Random Bridge (6 experiments, seeds 41-46)
    echo ""
    echo "-- Disjoint Random Bridge (6 experiments) --"
    echo "   bridge=20, isolated=50"
    for seed in 41 42 43 44 45 46; do
        if [ -n "$FORCE_RESUME" ] && is_disjoint_complete "${BASE_DIR}/lb_disjoint_random/full_chain_disjoint" $seed 100 20 "random" 50; then
            echo "   ⏭️  SKIP (complete): Disjoint random (seed=$seed)"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "   Submitting Disjoint random (seed=$seed)..."
            submit_job sbatch $SETUP_ONLY sh_run/run_chain_linking_disjoint_lb.sh \
                ${BASE_DIR}/lb_disjoint_random/full_chain_disjoint $seed 100 20 50 "" random $DISJOINT_SKIP_ARG
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    
    echo ""
    echo "✓ Category 4 complete: 12 experiments submitted"
fi

# ============================================================
# CATEGORY 5: Extended Model Sweep (9 × 6 × N_SEEDS experiments)
# ============================================================
# Test how number of models in chain affects performance
# 9 model counts × 6 datasets × N_SEEDS seeds
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "5" ]; then
    CAT5_TOTAL=$((9 * 6 * N_SEEDS))
    echo "=================================================================="
    echo "CATEGORY 5: Extended Model Sweep ($CAT5_TOTAL experiments)"
    echo "=================================================================="
    echo ""
    
    mkdir -p "${BASE_DIR}/lb_model_sweep_extended_fixed"
    
    echo "-- LB Model Sweep Extended ($CAT5_TOTAL experiments) --"
    echo "   Model counts: 5, 10, 25, 50, 100, 150, 200, 250, 300"
    echo "   $N_SEEDS seeds per dataset"
    echo "   Order: seed → dataset → model_count (get full picture faster)"
    echo "   Output: ${BASE_DIR}/lb_model_sweep_extended_fixed"
    
    BASE_SEED=101
    for seed_offset in $(seq 0 $((N_SEEDS - 1))); do
        echo ""
        echo "   --- Seed round $((seed_offset + 1))/3 ---"
        TARGET_IDX=0
        for target in "${LB_DATASETS[@]}"; do
            RUN_SEED=$((BASE_SEED + TARGET_IDX * N_SEEDS + seed_offset))
            for models in 5 10 25 50 100 150 200 250 300; do
                if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/lb_model_sweep_extended_fixed" $RUN_SEED 100 "$target" "$models"; then
                    echo "   ⏭️  SKIP (complete): $target models=$models (seed=$RUN_SEED)"
                    SKIPPED=$((SKIPPED + 1))
                else
                    echo "   Submitting $target (models=$models, seed=$RUN_SEED)..."
                    submit_job sbatch --time=12:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                        --output-dir ${BASE_DIR}/lb_model_sweep_extended_fixed \
                        --preset lb_standard \
                        --n-models $models \
                        --shuffle-seed $RUN_SEED \
                        --seed $RUN_SEED \
                        --target "$target" \
                        --random-seed $((3000 + models)) \
                        $SKIP_EXISTING $FORCE_RESUME
                    SUBMITTED=$((SUBMITTED + 1))
                fi
            done
            TARGET_IDX=$((TARGET_IDX + 1))
        done
    done
    
    echo ""
    echo "✓ Category 5 complete: $CAT5_TOTAL experiments (9 model counts × 6 datasets × $N_SEEDS seeds)"
fi

# ============================================================
# CATEGORY 6: MMLU Extended Model Sweep (9 × N_SEEDS experiments)
# ============================================================
# Test how number of models in chain affects performance for MMLU Fields
# 9 model counts × N_SEEDS seeds
# NOTE: Each seed produces a DIFFERENT target dataset (unlike LB where target is explicit)
# So N_SEEDS seeds = N_SEEDS different targets, each with full model sweep coverage
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "6" ]; then
    CAT6_TOTAL=$((9 * N_SEEDS))
    echo "=================================================================="
    echo "CATEGORY 6: MMLU Extended Model Sweep ($CAT6_TOTAL experiments)"
    echo "=================================================================="
    echo ""
    
    mkdir -p "${BASE_DIR}/mmlu_model_sweep_extended_fixed"
    
    echo "-- MMLU Model Sweep Extended ($CAT6_TOTAL experiments) --"
    echo "   Model counts: 5, 10, 25, 50, 100, 150, 200, 250, 300 | 50 anchors"
    echo "   $N_SEEDS seeds (each seed = different target dataset)"
    echo "   Output: ${BASE_DIR}/mmlu_model_sweep_extended_fixed"
    
    SEED=201
    for seed_offset in $(seq 0 $((N_SEEDS - 1))); do
        RUN_SEED=$((SEED + seed_offset))
        for models in 5 10 25 50 100 150 200 250 300; do
            if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/mmlu_model_sweep_extended_fixed" $RUN_SEED 50 "" "$models"; then
                echo "   ⏭️  SKIP (complete): MMLU models=$models (seed=$RUN_SEED)"
                SKIPPED=$((SKIPPED + 1))
            else
                echo "   Submitting MMLU Fields (models=$models, seed=$RUN_SEED, 50 anchors)..."
                submit_job sbatch --mem=8g --time=24:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                    --output-dir ${BASE_DIR}/mmlu_model_sweep_extended_fixed \
                    --preset mmlu_fields \
                    --n-anchors 50 \
                    --n-models $models \
                    --shuffle-seed $RUN_SEED \
                    --seed $RUN_SEED \
                    --random-seed $((4000 + models)) \
                    $SKIP_EXISTING $FORCE_RESUME
                SUBMITTED=$((SUBMITTED + 1))
            fi
        done
    done
    
    echo ""
    echo "✓ Category 6 complete: $CAT6_TOTAL experiments (9 model counts × $N_SEEDS seeds/targets)"
fi

# ============================================================
# CATEGORY 7: HELM Lite Extended Model Sweep (6 × 9 × N_SEEDS experiments)
# ============================================================
# Test how number of models in chain affects performance for HELM Lite
# 6 model counts × 9 datasets × N_SEEDS seeds
# HELM Lite has 91 models, so model counts: 5, 10, 25, 50, 75, 91
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "7" ]; then
    CAT7_TOTAL=$((6 * 9 * N_SEEDS))
    echo "=================================================================="
    echo "CATEGORY 7: HELM Lite Extended Model Sweep ($CAT7_TOTAL experiments)"
    echo "=================================================================="
    echo ""
    
    mkdir -p "${BASE_DIR}/helm_lite_model_sweep_extended_fixed"
    
    # HELM Lite datasets (9 total)
    HELM_LITE_DATASETS=("GSM8K-Lite" "LegalBench" "MATH Competition" "MedQA" "MMLU-Lite" "NarrativeQA" "NaturalQA" "OpenBookQA" "WMT-14 Translation")
    
    echo "-- HELM Lite Model Sweep Extended ($CAT7_TOTAL experiments) --"
    echo "   Model counts: 5, 10, 25, 50, 75, 91 (HELM has 91 models total)"
    echo "   9 datasets × $N_SEEDS seeds per dataset"
    echo "   Output: ${BASE_DIR}/helm_lite_model_sweep_extended_fixed"
    
    BASE_SEED=301
    for seed_offset in $(seq 0 $((N_SEEDS - 1))); do
        echo ""
        echo "   --- Seed round $((seed_offset + 1))/$N_SEEDS ---"
        TARGET_IDX=0
        for target in "${HELM_LITE_DATASETS[@]}"; do
            RUN_SEED=$((BASE_SEED + TARGET_IDX * N_SEEDS + seed_offset))
            for models in 5 10 25 50 75 91; do
                if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/helm_lite_model_sweep_extended_fixed" $RUN_SEED 50 "$target" "$models"; then
                    echo "   ⏭️  SKIP (complete): $target models=$models (seed=$RUN_SEED)"
                    SKIPPED=$((SKIPPED + 1))
                else
                    echo "   Submitting $target (models=$models, seed=$RUN_SEED)..."
                    submit_job sbatch $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                        --output-dir ${BASE_DIR}/helm_lite_model_sweep_extended_fixed \
                        --preset helm_lite_base1 \
                        --n-models $models \
                        --shuffle-seed $RUN_SEED \
                        --seed $RUN_SEED \
                        --target "$target" \
                        --random-seed $((5000 + models)) \
                        $SKIP_EXISTING $FORCE_RESUME
                    SUBMITTED=$((SUBMITTED + 1))
                fi
            done
            TARGET_IDX=$((TARGET_IDX + 1))
        done
    done
    
    echo ""
    echo "✓ Category 7 complete: $CAT7_TOTAL experiments (6 model counts × 9 datasets × $N_SEEDS seeds)"
fi

# ============================================================
# CATEGORY 8: MMLU Anchor Sweep (4 seeds × 4 anchor counts = 16 experiments)
# ============================================================
# Same seed per "target" (each seed = different MMLU subject), only anchors vary: 10, 25, 50, 100
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "8" ]; then
    CAT8_TOTAL=16
    echo "=================================================================="
    echo "CATEGORY 8: MMLU Anchor Sweep ($CAT8_TOTAL experiments)"
    echo "=================================================================="
    echo ""
    
    echo "-- MMLU Anchor Sweep Controlled (16 experiments) --"
    echo "   4 seeds (4 different targets), anchors: 10, 25, 50, 100"
    echo "   Output: ${BASE_DIR}/mmlu_anchor_sweep_controlled"
    
    for RUN_SEED in 11 12 13 14; do
        for anchors in 10 25 50 100; do
            if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/mmlu_anchor_sweep_controlled" $RUN_SEED $anchors "" ""; then
                echo "   ⏭️  SKIP (complete): MMLU anchors=$anchors (seed=$RUN_SEED)"
                SKIPPED=$((SKIPPED + 1))
            else
                echo "   Submitting MMLU Fields (anchors=$anchors, seed=$RUN_SEED)..."
                submit_job sbatch --mem=8g --time=10:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                    --output-dir ${BASE_DIR}/mmlu_anchor_sweep_controlled \
                    --preset mmlu_fields \
                    --n-anchors $anchors \
                    --shuffle-seed $RUN_SEED \
                    --seed $RUN_SEED \
                    --random-seed $((6000 + anchors)) \
                    $SKIP_EXISTING $FORCE_RESUME
                SUBMITTED=$((SUBMITTED + 1))
            fi
        done
    done
    
    echo ""
    echo "✓ Category 8 complete: $CAT8_TOTAL experiments (4 seeds × 4 anchor counts)"
fi

# ============================================================
# CATEGORY 9: MMLU Baseline 50 anchors (20 experiments)
# ============================================================
# Same as Category 3 but with 50 anchors per dataset → mmlu_baseline_50/
if [ -z "$RUN_CATEGORY" ] || [ "$RUN_CATEGORY" = "9" ]; then
    CAT9_TOTAL=20
    echo "=================================================================="
    echo "CATEGORY 9: MMLU Baseline 50 Anchors ($CAT9_TOTAL experiments)"
    echo "=================================================================="
    echo ""
    
    echo "-- MMLU Fields Baseline 50 anchors (seeds 11-30) --"
    for seed in $(seq 11 30); do
        if [ -n "$FORCE_RESUME" ] && is_experiment_complete "${BASE_DIR}/mmlu_baseline_50" $seed 50 "" ""; then
            echo "   ⏭️  SKIP (complete): MMLU 50 anchors (seed=$seed)"
            SKIPPED=$((SKIPPED + 1))
        else
            echo "   Submitting MMLU Fields 50 anchors (seed=$seed)..."
            submit_job sbatch --mem=8g --time=10:0:0 $SETUP_ONLY sh_run/run_chain_linking_unified.sh \
                --output-dir ${BASE_DIR}/mmlu_baseline_50 \
                --preset mmlu_fields \
                --n-anchors 50 \
                --shuffle-seed $seed \
                --seed $seed \
                --random-seed 1000 \
                $SKIP_EXISTING $FORCE_RESUME
            SUBMITTED=$((SUBMITTED + 1))
        fi
    done
    
    echo ""
    echo "✓ Category 9 complete: $CAT9_TOTAL experiments (MMLU baseline, 50 anchors)"
fi

# Summary
echo ""
echo "=================================================================="
echo "SUMMARY"
echo "=================================================================="
if [ -n "$FORCE_RESUME" ]; then
    echo "Submitted: $SUBMITTED jobs"
    echo "Skipped (complete): $SKIPPED experiments"
elif [ -n "$RUN_CATEGORY" ]; then
    echo "Category $RUN_CATEGORY: $SUBMITTED jobs submitted"
else
    echo "All categories:"
    CAT5_EXP=$((9 * 6 * N_SEEDS))
    CAT6_EXP=$((9 * N_SEEDS))
    CAT7_EXP=$((6 * 9 * N_SEEDS))
    TOTAL_EXP=$((60 + 18 + 20 + 12 + 16 + 20 + CAT5_EXP + CAT6_EXP + CAT7_EXP))
    echo "  Category 1 (LB):              60 experiments (24 baseline + 24 anchor + 12 model)"
    echo "  Category 2 (HELM Lite):       18 experiments"
    echo "  Category 3 (MMLU):            20 experiments (10 anchors)"
    echo "  Category 4 (Disjoint):        12 experiments"
    echo "  Category 5 (LB Model Ext):    $CAT5_EXP experiments (9 counts × 6 datasets × $N_SEEDS seeds)"
    echo "  Category 6 (MMLU Model Ext):  $CAT6_EXP experiments (9 counts × $N_SEEDS targets)"
    echo "  Category 7 (HELM Model Ext):  $CAT7_EXP experiments (6 counts × 9 datasets × $N_SEEDS seeds)"
    echo "  Category 8 (MMLU Anchor):    16 experiments (4 seeds × 4 anchor counts 10,25,50,100)"
    echo "  Category 9 (MMLU 50 anchors): 20 experiments (same as cat 3, 50 anchors)"
    echo "  ───────────────────────────────────────"
    echo "  TOTAL:                        $TOTAL_EXP experiments (with --seeds $N_SEEDS)"
    echo ""
    echo "Actually submitted: $SUBMITTED"
    [ $SKIPPED -gt 0 ] && echo "Skipped (complete): $SKIPPED"
fi
echo ""
echo "Monitor: squeue -u \$USER"
echo "Check:   python scripts/check_experiment_plan.py $BASE_DIR"
echo "=================================================================="
