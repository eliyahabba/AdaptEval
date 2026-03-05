#!/bin/bash

#SBATCH --job-name=chain-unified
#SBATCH --mem=12g
#SBATCH --time=12:0:0
#SBATCH --mail-user=eliya.habba@mail.huji.ac.il
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --gres=gg:g0:2
#SBATCH --cpus-per-task=2
#SBATCH --killable
#SBATCH --requeue

# Unified Chain Linking Experiment Runner
#
# IMPORTANT: To override SLURM resources at submission time:
#   sbatch --mem=8g --time=24:0:0 run_chain_linking_unified.sh [options]
#
# Default resources (10g RAM, 6 hours, 4 GPUs, 2 CPUs):
#   - Sufficient for LB and HELM Lite
#   - Override with --mem=8g for MMLU fields (less memory needed)
#   - Override with --time=24:0:0 for MMLU fields (longer runtime)
#
# This script provides a single entry point for all chain linking experiments.
# It supports preset configurations, parameter overrides, and auto-incrementing seeds
# to avoid conflicts with parallel workers.
#
# Usage Examples:
#   # Basic usage
#   sbatch run_chain_linking_unified.sh --output-dir data/v24_lb --preset lb_standard --shuffle-seed 11 --seed 11
#
#   # Override SLURM resources for MMLU
#   sbatch --mem=8g --time=24:0:0 run_chain_linking_unified.sh \
#       --output-dir data/v24_mmlu --preset mmlu_fields --seed 11
#
#   # Override preset values
#   sbatch run_chain_linking_unified.sh --output-dir data/v24_lb --preset lb_standard --n-anchors 50
#
#   # Skip experiments where directory already exists (any experiment with that seed)
#   sbatch run_chain_linking_unified.sh --output-dir data/v24_lb --preset lb_standard --shuffle-seed 11 --seed 11 --skip-existing
#
#   # Resume incomplete experiments (skip COMPLETED ones with all_results.csv, resume incomplete)
#   sbatch run_chain_linking_unified.sh --output-dir data/v24_lb --preset lb_standard --shuffle-seed 11 --seed 11 --force-resume
#
#   # Fully custom (no preset)
#   sbatch run_chain_linking_unified.sh --output-dir data/v24_custom --data-source lb --n-anchors 100

# =============================================================================
# Cleanup Handler (runs on exit, error, or interrupt)
# =============================================================================

cleanup_on_exit() {
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "🚨 Script terminated with error (exit code: $exit_code)"
        echo "   Python emergency cleanup should have run automatically"
        echo "   If you see leftover .temp directories, run:"
        echo "   ./scripts/batch_cleanup.sh ${OUTPUT_BASE_DIR} --force"
    fi
}

# Register cleanup handler
trap cleanup_on_exit EXIT

# =============================================================================
# Configuration
# =============================================================================

# Set project directory
if [ -z "$PROJECT_DIR" ]; then
    PROJECT_DIR="/cs/labs/gabis/eliyahabba/AdaptEval"
fi

# If we're not on the cluster, try to detect local path
if [ ! -d "$PROJECT_DIR" ]; then
    # Try to detect from script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
fi

cd $PROJECT_DIR

# Set Hugging Face cache directory (skip if local)
if [ -d "/cs/snapless/gabis/gabis/shared/huggingface/" ]; then
    export HF_HOME=/cs/snapless/gabis/gabis/shared/huggingface/
fi

export PYTHONPATH=$PROJECT_DIR/src:$PROJECT_DIR:$PYTHONPATH
export UNITXT_ALLOW_UNVERIFIED_CODE="True"
export CUDA_LAUNCH_BLOCKING=1

# Activate virtual environment (skip if not on cluster)
VENV_PATH="/cs/snapless/gabis/eliyahabba/venvs/AdaptEval/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
fi

# Load YAML parser (simple python-based parser)
parse_yaml() {
    python3 - "$@" <<'EOF'
import sys
import yaml

yaml_file = sys.argv[1]
preset_name = sys.argv[2] if len(sys.argv) > 2 else None

with open(yaml_file) as f:
    config = yaml.safe_load(f)

if preset_name:
    preset = config['presets'].get(preset_name, {})
    defaults = config.get('defaults', {})
    slurm_defaults = config.get('slurm_defaults', {})
    
    # Merge defaults with preset
    result = {**defaults, **preset}
    
    # Handle slurm overrides
    if 'slurm' in preset:
        result['slurm'] = {**slurm_defaults, **preset['slurm']}
    else:
        result['slurm'] = slurm_defaults
    
    # Print as shell variables
    for key, value in result.items():
        if key == 'slurm':
            for skey, svalue in value.items():
                print(f"SLURM_{skey.upper()}='{svalue}'")
        elif value is not None:
            if isinstance(value, list):
                print(f"{key.upper()}=({' '.join(map(str, value))})")
            else:
                print(f"{key.upper()}='{value}'")
else:
    # Just print defaults
    defaults = config.get('defaults', {})
    slurm_defaults = config.get('slurm_defaults', {})
    for key, value in {**defaults, **slurm_defaults}.items():
        if value is not None:
            print(f"{key.upper()}='{value}'")
EOF
}

# =============================================================================
# Parse Arguments
# =============================================================================

PRESET=""
OUTPUT_BASE_DIR=""  # User must specify base output directory
EXPERIMENT_TYPE="parallel"  # parallel or disjoint
AUTO_INCREMENT_SEED=true
FORCE_RESUME=false  # If true, force resume existing experiment (pass --force-resume to Python)
SETUP_ONLY=false  # If true, only create directory and config, don't run experiment
SKIP_EXISTING=false  # If true, skip if output dir already exists (no auto-increment)

# Experiment parameters (will be filled from preset or CLI)
SHUFFLE_SEED=""
DATA_SOURCE_MODE=""
N_BASE=""
MAX_CHAIN=""
N_ANCHORS=""
N_MODELS_PER_CHAIN=""
TARGET_DATASET=""
SEED="42"
RANDOM_SEED="1000"
DIMS="5"
EPOCHS=""
EPOCHS_FIXED=""
TEST_RATIO=""
NUM_WORKERS=""

# Disjoint-specific parameters
N_BRIDGE_MODELS=""
N_ISOLATED_PER_CHAIN=""
BRIDGE_MODE=""

# SLURM parameters
SLURM_MEMORY=""
SLURM_GPUS=""
SLURM_CPUS=""
SLURM_TIME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --type)
            EXPERIMENT_TYPE="$2"
            shift 2
            ;;
        --seed)
            # Backward-compatible behavior:
            # - seed controls model split/random sampling in Python
            # - if shuffle seed is not explicitly provided, keep historical behavior
            #   where --seed also controlled dataset shuffle/target selection
            SEED="$2"
            if [ -z "$SHUFFLE_SEED" ]; then
                SHUFFLE_SEED="$2"
            fi
            shift 2
            ;;
        --shuffle-seed)
            SHUFFLE_SEED="$2"
            shift 2
            ;;
        --random-seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        --data-source|--data-source-mode)
            DATA_SOURCE_MODE="$2"
            shift 2
            ;;
        --n-base)
            N_BASE="$2"
            shift 2
            ;;
        --max-chain)
            MAX_CHAIN="$2"
            shift 2
            ;;
        --n-anchors)
            N_ANCHORS="$2"
            shift 2
            ;;
        --n-models|--n-models-per-chain)
            N_MODELS_PER_CHAIN="$2"
            shift 2
            ;;
        --target|--target-dataset)
            TARGET_DATASET="$2"
            shift 2
            ;;
        --dims)
            DIMS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --epochs-fixed)
            EPOCHS_FIXED="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --n-bridge-models)
            N_BRIDGE_MODELS="$2"
            shift 2
            ;;
        --n-isolated-per-chain)
            N_ISOLATED_PER_CHAIN="$2"
            shift 2
            ;;
        --bridge-mode)
            BRIDGE_MODE="$2"
            shift 2
            ;;
        --memory)
            SLURM_MEMORY="$2"
            shift 2
            ;;
        --gpus)
            SLURM_GPUS="$2"
            shift 2
            ;;
        --time)
            SLURM_TIME="$2"
            shift 2
            ;;
        --no-auto-increment)
            AUTO_INCREMENT_SEED=false
            shift
            ;;
        --force-resume)
            FORCE_RESUME=true
            AUTO_INCREMENT_SEED=false  # Also disable bash-level auto-increment
            shift
            ;;
        --setup-only|--dry-run)
            SETUP_ONLY=true
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            AUTO_INCREMENT_SEED=false  # Don't auto-increment when skipping existing
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Load Preset Configuration
# =============================================================================

if [ -n "$PRESET" ]; then
    echo "Loading preset: $PRESET"
    PRESET_FILE="$PROJECT_DIR/sh_run/experiment_presets.yaml"
    
    if [ ! -f "$PRESET_FILE" ]; then
        echo "Error: Preset file not found: $PRESET_FILE"
        exit 1
    fi
    
    # Store CLI arguments before loading preset
    CLI_SHUFFLE_SEED="$SHUFFLE_SEED"
    CLI_SEED="$SEED"
    CLI_DATA_SOURCE_MODE="$DATA_SOURCE_MODE"
    CLI_N_BASE="$N_BASE"
    CLI_MAX_CHAIN="$MAX_CHAIN"
    CLI_N_ANCHORS="$N_ANCHORS"
    CLI_N_MODELS="$N_MODELS_PER_CHAIN"
    CLI_EPOCHS="$EPOCHS"
    CLI_NUM_WORKERS="$NUM_WORKERS"
    CLI_RANDOM_SEED="$RANDOM_SEED"
    
    # Load preset configuration
    eval $(parse_yaml "$PRESET_FILE" "$PRESET")
    
    # Override preset with CLI arguments if provided
    [ -n "$CLI_SHUFFLE_SEED" ] && SHUFFLE_SEED="$CLI_SHUFFLE_SEED"
    [ -n "$CLI_SEED" ] && SEED="$CLI_SEED"
    [ -n "$CLI_DATA_SOURCE_MODE" ] && DATA_SOURCE_MODE="$CLI_DATA_SOURCE_MODE"
    [ -n "$CLI_N_BASE" ] && N_BASE="$CLI_N_BASE"
    [ -n "$CLI_MAX_CHAIN" ] && MAX_CHAIN="$CLI_MAX_CHAIN"
    [ -n "$CLI_N_ANCHORS" ] && N_ANCHORS="$CLI_N_ANCHORS"
    [ -n "$CLI_N_MODELS" ] && N_MODELS_PER_CHAIN="$CLI_N_MODELS"
    [ -n "$CLI_EPOCHS" ] && EPOCHS="$CLI_EPOCHS"
    [ -n "$CLI_NUM_WORKERS" ] && NUM_WORKERS="$CLI_NUM_WORKERS"
    [ -n "$CLI_RANDOM_SEED" ] && RANDOM_SEED="$CLI_RANDOM_SEED"
fi

# Set defaults if still empty
SHUFFLE_SEED="${SHUFFLE_SEED:-11}"
DATA_SOURCE_MODE="${DATA_SOURCE_MODE:-lb}"
N_BASE="${N_BASE:-1}"
MAX_CHAIN="${MAX_CHAIN:-10}"
N_ANCHORS="${N_ANCHORS:-100}"
EPOCHS="${EPOCHS:-2000}"
EPOCHS_FIXED="${EPOCHS_FIXED:-1000}"
TEST_RATIO="${TEST_RATIO:-0.25}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
RANDOM_SEED="${RANDOM_SEED:-1000}"

# Require output directory
if [ -z "$OUTPUT_BASE_DIR" ]; then
    echo "Error: --output-dir is required"
    echo ""
    echo "Usage: $0 --output-dir /path/to/experiments --preset <preset_name> [options]"
    echo ""
    echo "Example:"
    echo "  sbatch $0 --output-dir data/v24_lb --preset lb_standard --seed 11"
    echo ""
    echo "For MMLU (needs less memory but more time):"
    echo "  sbatch --mem=8g --time=24:0:0 $0 --output-dir data/v24_mmlu --preset mmlu_fields --seed 11"
    exit 1
fi

# =============================================================================
# Build Output Directory Name
# =============================================================================

build_output_dir_name() {
    local seed=$1
    local base_name="full_chain_classic_seed_${seed}_anchors_${N_ANCHORS}"
    
    if [ -n "$N_MODELS_PER_CHAIN" ] && [ "$N_MODELS_PER_CHAIN" != "null" ]; then
        base_name="${base_name}_models_${N_MODELS_PER_CHAIN}"
    fi
    
    if [ -n "$TARGET_DATASET" ]; then
        base_name="${base_name}_target_${TARGET_DATASET}"
    fi
    
    echo "$base_name"
}

# =============================================================================
# Auto-Increment Seed Logic
# =============================================================================

# Ensure output base directory is absolute or relative to PROJECT_DIR
if [[ "$OUTPUT_BASE_DIR" != /* ]]; then
    # Relative path - make it relative to PROJECT_DIR
    OUTPUT_BASE_DIR="${PROJECT_DIR}/${OUTPUT_BASE_DIR}"
fi

# Create base directory if it doesn't exist
mkdir -p "$OUTPUT_BASE_DIR"

find_next_available_seed() {
    local base_dir="$1"
    local initial_seed="$2"
    local current_seed=$initial_seed
    
    while true; do
        local output_dir_name=$(build_output_dir_name $current_seed)
        local full_path="${base_dir}/${output_dir_name}"
        
        # Check if directory exists (either exact match OR with any target suffix)
        # This handles the case where Python script appends _target_<name> to the directory
        if [ ! -d "$full_path" ] && ! ls -d "${full_path}_target_"* 2>/dev/null | grep -q .; then
            echo $current_seed
            return
        fi
        
        # Determine which directory caused the conflict
        if [ -d "$full_path" ]; then
            echo "  Seed $current_seed exists (${output_dir_name}), trying next..." >&2
        else
            local existing=$(ls -d "${full_path}_target_"* 2>/dev/null | head -1 | xargs -n1 basename)
            echo "  Seed $current_seed exists (${existing}), trying next..." >&2
        fi
        
        current_seed=$((current_seed + 1))
        
        # Safety limit
        if [ $current_seed -gt $((initial_seed + 100)) ]; then
            echo "Error: Could not find available seed after 100 attempts" >&2
            exit 1
        fi
    done
}

# Auto-increment seed if enabled
if [ "$AUTO_INCREMENT_SEED" = true ]; then
    SHUFFLE_SEED=$(find_next_available_seed "$OUTPUT_BASE_DIR" "$SHUFFLE_SEED")
    echo "Using shuffle_seed: $SHUFFLE_SEED"
fi

# Build final output directory
OUTPUT_DIR_NAME=$(build_output_dir_name $SHUFFLE_SEED)
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${OUTPUT_DIR_NAME}"

# Check if we should skip existing experiments
if [ "$SKIP_EXISTING" = true ]; then
    # Check if directory exists (either exact match OR with any target suffix)
    if [ -d "$OUTPUT_DIR" ] || ls -d "${OUTPUT_DIR}_target_"* 2>/dev/null | grep -q .; then
        if [ -d "$OUTPUT_DIR" ]; then
            echo "⏭️  SKIP: Output directory already exists: ${OUTPUT_DIR}"
        else
            existing=$(ls -d "${OUTPUT_DIR}_target_"* 2>/dev/null | head -1 | xargs -n1 basename)
            echo "⏭️  SKIP: Output directory already exists: ${OUTPUT_BASE_DIR}/${existing}"
        fi
        echo "   (Use without --skip-existing to auto-increment seed and create a new experiment)"
        exit 0
    fi
fi

# Check if experiment is already COMPLETE when resuming
# (Skip completed experiments, only resume incomplete ones)
if [ "$FORCE_RESUME" = true ]; then
    # Find the actual output directory (may have target suffix)
    ACTUAL_OUTPUT_DIR="$OUTPUT_DIR"
    if [ ! -d "$OUTPUT_DIR" ]; then
        ACTUAL_OUTPUT_DIR=$(ls -d "${OUTPUT_DIR}_target_"* 2>/dev/null | head -1)
    fi
    
    if [ -n "$ACTUAL_OUTPUT_DIR" ] && [ -f "${ACTUAL_OUTPUT_DIR}/all_results.csv" ]; then
        echo "⏭️  SKIP (COMPLETE): Experiment already finished: ${ACTUAL_OUTPUT_DIR}"
        echo "   Found: all_results.csv"
        echo "   (Use --force without --resume to re-run from scratch)"
        exit 0
    fi
fi

# =============================================================================
# Print Configuration
# =============================================================================

echo "========================================"
echo "Unified Chain Linking Experiment"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Preset: ${PRESET:-none}"
echo "Experiment Type: $EXPERIMENT_TYPE"
if [ "$SKIP_EXISTING" = true ]; then
    echo "Mode: SKIP-EXISTING (will not auto-increment seed)"
elif [ "$AUTO_INCREMENT_SEED" = true ]; then
    echo "Mode: AUTO-INCREMENT (finds next available seed)"
fi
echo ""
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Parameters:"
echo "  DATA_SOURCE_MODE: $DATA_SOURCE_MODE"
echo "  SHUFFLE_SEED: $SHUFFLE_SEED (controls dataset order / target selection)"
echo "  SEED: $SEED (controls model splits and task-level random sampling)"
echo "  RANDOM_SEED: $RANDOM_SEED (for random baseline scenarios)"
echo "  N_BASE: $N_BASE"
echo "  MAX_CHAIN: $MAX_CHAIN"
echo "  N_ANCHORS: $N_ANCHORS"
echo "  N_MODELS_PER_CHAIN: ${N_MODELS_PER_CHAIN:-all}"
echo "  TARGET_DATASET: ${TARGET_DATASET:-auto}"
echo "  DIMS: $DIMS"
echo "  EPOCHS: $EPOCHS"
echo "  EPOCHS_FIXED: $EPOCHS_FIXED"
echo "  NUM_WORKERS: $NUM_WORKERS"
echo ""
echo "SLURM Resources (override at submission with sbatch --mem=Xg --time=H:M:S):"
echo "  Defaults: 10g RAM, 6 hours, 4 GPUs, 2 CPUs"
echo "  Current job: $SLURM_JOB_ID"
echo "========================================"

# =============================================================================
# Load CUDA (skip if not on cluster)
# =============================================================================

if command -v module &> /dev/null; then
    module load cuda
    echo "Available GPUs:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
fi

# =============================================================================
# Create Output Directory and Save Config
# =============================================================================

# NOTE: We do NOT create the output directory here because the Python script
# will append the target name and create the final directory.
# Creating it here would leave behind empty directories without target suffixes.

# However, in setup-only mode, we create it for demonstration purposes
if [ "$SETUP_ONLY" = true ]; then
    mkdir -p "$OUTPUT_DIR"
    
    # Save configuration to JSON
    cat > "${OUTPUT_DIR}/config.json" << EOF
{
  "experiment_type": "$EXPERIMENT_TYPE",
  "preset": "${PRESET:-none}",
  "data_source_mode": "$DATA_SOURCE_MODE",
  "shuffle_seed": $SHUFFLE_SEED,
  "seed": $SEED,
  "random_seed": $RANDOM_SEED,
  "n_base": $N_BASE,
  "max_chain": $MAX_CHAIN,
  "n_anchors": $N_ANCHORS,
  "n_models_per_chain": ${N_MODELS_PER_CHAIN:-null},
  "target_dataset": "${TARGET_DATASET:-auto}",
  "dims": [$DIMS],
  "epochs": $EPOCHS,
  "epochs_fixed": $EPOCHS_FIXED,
  "test_ratio": $TEST_RATIO,
  "num_workers": $NUM_WORKERS,
  "output_dir": "$OUTPUT_DIR",
  "slurm_job_id": "${SLURM_JOB_ID:-local}",
  "node": "${SLURMD_NODENAME:-local}"
}
EOF

    echo ""
    echo "Configuration saved to: ${OUTPUT_DIR}/config.json"
    echo ""
    echo "======================================"
    echo "SETUP-ONLY MODE: Stopping here"
    echo "======================================"
    echo "Directory created: $OUTPUT_DIR"
    echo "Config saved, experiment NOT run"
    echo ""
    echo "NOTE: When running for real (without --setup-only), the Python script"
    echo "      will append '_target_<name>' to the directory after determining"
    echo "      the target dataset."
    exit 0
fi

# If not setup-only mode, Python script will create the directory
echo ""
if [ -n "$TARGET_DATASET" ]; then
    echo "NOTE: Output directory already includes target name (--target was specified)"
    echo "      Final directory will be: ${OUTPUT_DIR}"
else
    echo "NOTE: Python script will append target suffix after auto-selecting target"
    echo "      Final directory will be: ${OUTPUT_DIR}_target_<auto_selected_name>"
fi
echo ""

# =============================================================================
# Run Experiment
# =============================================================================

if [ "$EXPERIMENT_TYPE" = "disjoint" ]; then
    # Disjoint experiment
    echo "Running disjoint experiment..."
    
    # Build arguments
    ARGS="--output-dir \"${OUTPUT_DIR}\""
    ARGS="$ARGS --n-base ${N_BASE}"
    ARGS="$ARGS --max-chain 4"  # Disjoint uses fixed max-chain=4
    ARGS="$ARGS --n-anchors-per-dataset ${N_ANCHORS}"
    ARGS="$ARGS --seed ${SEED}"
    ARGS="$ARGS --shuffle-seed ${SHUFFLE_SEED}"
    ARGS="$ARGS --random-seed ${RANDOM_SEED}"
    ARGS="$ARGS --dims ${DIMS}"
    ARGS="$ARGS --epochs ${EPOCHS}"
    ARGS="$ARGS --data-source-mode ${DATA_SOURCE_MODE}"
    ARGS="$ARGS --num-workers ${NUM_WORKERS}"
    
    if [ -n "$N_BRIDGE_MODELS" ]; then
        ARGS="$ARGS --n-bridge-models ${N_BRIDGE_MODELS}"
    fi
    if [ -n "$N_ISOLATED_PER_CHAIN" ]; then
        ARGS="$ARGS --n-isolated-per-chain ${N_ISOLATED_PER_CHAIN}"
    fi
    if [ -n "$TARGET_DATASET" ]; then
        ARGS="$ARGS --target-dataset \"${TARGET_DATASET}\""
    fi
    if [ "$BRIDGE_MODE" = "random" ]; then
        ARGS="$ARGS --random-bridge"
    else
        ARGS="$ARGS --fixed-bridge"
    fi
    
    eval python src/experiments/chain_linking/chain_linking_disjoint.py $ARGS
    
else
    # Standard parallel experiment
    echo "Running parallel experiment..."
    
    # Build arguments
    ARGS="--output-dir \"${OUTPUT_DIR}\""
    ARGS="$ARGS --n-base ${N_BASE}"
    ARGS="$ARGS --max-chain ${MAX_CHAIN}"
    ARGS="$ARGS --n-anchors-per-dataset ${N_ANCHORS}"
    ARGS="$ARGS --test-ratio ${TEST_RATIO}"
    ARGS="$ARGS --seed ${SEED}"
    ARGS="$ARGS --shuffle-seed ${SHUFFLE_SEED}"
    ARGS="$ARGS --random-seed ${RANDOM_SEED}"
    ARGS="$ARGS --dims ${DIMS}"
    ARGS="$ARGS --epochs ${EPOCHS}"
    ARGS="$ARGS --epochs-fixed ${EPOCHS_FIXED}"
    ARGS="$ARGS --data-source-mode ${DATA_SOURCE_MODE}"
    ARGS="$ARGS --num-workers ${NUM_WORKERS}"
    
    if [ -n "$N_MODELS_PER_CHAIN" ] && [ "$N_MODELS_PER_CHAIN" != "null" ]; then
        ARGS="$ARGS --n-models-per-chain ${N_MODELS_PER_CHAIN}"
    fi
    if [ -n "$TARGET_DATASET" ]; then
        ARGS="$ARGS --target-dataset \"${TARGET_DATASET}\""
    fi
    if [ "$FORCE_RESUME" = true ]; then
        ARGS="$ARGS --force-resume"
    fi
    
    eval python src/experiments/chain_linking/chain_linking_parallel.py $ARGS
fi

# =============================================================================
# Print Resource Usage
# =============================================================================

echo ""
echo "Job resource usage:"

# NOTE:
# - For RUNNING jobs, sacct fields like MaxRSS may be empty due to accounting latency.
# - While RUNNING, prefer: sstat -j ${SLURM_JOB_ID}.batch --format=AveRSS,MaxRSS,MaxVMSize
# - After COMPLETION, prefer querying the batch step explicitly: ${SLURM_JOB_ID}.batch

if [ -n "$SLURM_JOB_ID" ]; then
    echo "sacct (job + steps):"
    sacct -j "${SLURM_JOB_ID}" --units=G --format=User,JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,AveRSS,ReqMem,AllocTRES%30
    echo ""
    echo "sacct (batch step):"
    sacct -j "${SLURM_JOB_ID}.batch" --units=G --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,AveRSS,ReqMem,AllocTRES%30
    echo ""
    if command -v seff &> /dev/null; then
        echo "seff summary:"
        seff "${SLURM_JOB_ID}" || true
    fi
else
    echo "SLURM_JOB_ID not set (local run?)"
fi

