#!/usr/bin/env python3
"""
check_experiment_plan.py

Compare actual experiments against the planned experiments in run_all_experiments.sh
Shows what exists, what's missing, and what's the status.
"""

import sys
from pathlib import Path
from collections import defaultdict
import json


# Define the expected experiment plan based on run_all_experiments.sh
EXPERIMENT_PLAN = {
    'lb_baseline': {
        'description': 'LB Baseline (Category 1)',
        'seeds': list(range(11, 17)),  # 11-16
        'anchors': [100],
        'models': [None],  # None = all models
        'n_base': [1],
        'expected_count': 6,
    },
    'helm_lite_baseline': {
        'description': 'HELM Lite Baseline (Category 1)',
        'seeds': list(range(11, 20)),  # 11-19
        'anchors': [50],
        'models': [None],
        'n_base': [1],
        'expected_count': 9,
    },
    'mmlu_baseline': {
        'description': 'MMLU Fields Baseline (Category 1)',
        'seeds': list(range(11, 31)),  # 11-30
        'anchors': [10],
        'models': [None],
        'n_base': [8],
        'expected_count': 20,
    },
    'lb_anchor_sweep': {
        'description': 'LB Anchor Sweep (Category 2)',
        'seeds': list(range(21, 27)),  # 21-26
        'anchors': [25, 50, 100, 200],
        'models': [None],
        'n_base': [1],
        'expected_count': 24,  # 6 seeds × 4 anchors
    },
    'lb_model_sweep': {
        'description': 'LB Model Count Sweep (Category 3)',
        'seeds': list(range(31, 37)),  # 31-36
        'anchors': [100],
        'models': [50, 100],
        'n_base': [1],
        'expected_count': 12,  # 6 seeds × 2 models
    },
    'helm_lite_base4': {
        'description': 'HELM Lite Base Count (Category 4)',
        'seeds': list(range(11, 20)),  # 11-19
        'anchors': [25],
        'models': [None],
        'n_base': [4],
        'expected_count': 9,
    },
    'lb_disjoint_fixed': {
        'description': 'LB Disjoint Fixed Bridge (Category 5)',
        'seeds': list(range(41, 47)),  # 41-46
        'anchors': [100],
        'models': [None],
        'n_base': [1],
        'expected_count': 6,
        'type': 'disjoint',
    },
    'lb_disjoint_random': {
        'description': 'LB Disjoint Random Bridge (Category 5)',
        'seeds': list(range(41, 47)),  # 41-46
        'anchors': [100],
        'models': [None],
        'n_base': [1],
        'expected_count': 6,
        'type': 'disjoint',
    },
}


def parse_experiment_name(exp_name: str) -> dict:
    """Extract parameters from experiment directory name."""
    parts = {}
    
    # Extract seed
    if '_seed_' in exp_name:
        seed_part = exp_name.split('_seed_')[1].split('_')[0]
        parts['seed'] = int(seed_part)
    
    # Extract anchors
    if '_anchors_' in exp_name:
        anchors_part = exp_name.split('_anchors_')[1].split('_')[0]
        parts['anchors'] = int(anchors_part)
    
    # Extract target
    if '_target_' in exp_name:
        target_part = exp_name.split('_target_')[1]
        parts['target'] = target_part
    
    # Extract models if present
    if '_models_' in exp_name:
        models_part = exp_name.split('_models_')[1].split('_')[0]
        parts['models'] = int(models_part)
    else:
        parts['models'] = None
    
    return parts


def get_experiment_status(exp_dir: Path) -> str:
    """Get simple status: complete, partial, or incomplete."""
    if not exp_dir.exists():
        return 'missing'
    
    results_file = exp_dir / 'all_results.json'
    if results_file.exists() and results_file.stat().st_size > 10:
        try:
            with open(results_file) as f:
                results = json.load(f)
            # Check if has reasonable number of distances
            if len(results) >= 3:
                return 'complete'
            else:
                return 'partial'
        except:
            return 'partial'
    
    # Has config but no results
    if (exp_dir / 'config.json').exists():
        return 'incomplete'
    
    return 'missing'


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_experiment_plan.py <experiments_dir>")
        print("\nExample:")
        print("  python check_experiment_plan.py data/v25_comprehensive")
        sys.exit(1)
    
    base_dir = Path(sys.argv[1])
    
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)
    
    print("="*90)
    print("EXPERIMENT PLAN vs ACTUAL STATUS")
    print("="*90)
    print(f"Base directory: {base_dir}")
    print(f"Plan source: sh_run/run_all_experiments.sh\n")
    
    total_planned = sum(plan['expected_count'] for plan in EXPERIMENT_PLAN.values())
    total_found = 0
    total_complete = 0
    total_missing = 0
    
    # Check each category
    for category, plan in EXPERIMENT_PLAN.items():
        print("="*90)
        print(f"{plan['description']}")
        print(f"Category: {category}")
        print("="*90)
        
        category_dir = base_dir / category
        
        print(f"\nPlanned: {plan['expected_count']} experiments")
        print(f"  Seeds: {min(plan['seeds'])}-{max(plan['seeds'])}")
        print(f"  Anchors: {plan['anchors']}")
        if plan['models'] != [None]:
            print(f"  Models: {plan['models']}")
        
        # Collect actual experiments
        actual_experiments = {}
        if category_dir.exists():
            for exp_dir in category_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                
                params = parse_experiment_name(exp_dir.name)
                status = get_experiment_status(exp_dir)
                
                # Create key for this experiment
                key = (params.get('seed'), params.get('anchors'), params.get('models'))
                actual_experiments[key] = {
                    'name': exp_dir.name,
                    'status': status,
                    'target': params.get('target'),
                    **params
                }
        
        # Check coverage
        found = 0
        complete = 0
        missing_list = []
        incomplete_list = []
        
        for seed in plan['seeds']:
            for anchor in plan['anchors']:
                for model in plan['models']:
                    key = (seed, anchor, model)
                    
                    if key in actual_experiments:
                        found += 1
                        exp = actual_experiments[key]
                        if exp['status'] == 'complete':
                            complete += 1
                        else:
                            incomplete_list.append((seed, anchor, model, exp['status'], exp.get('target')))
                    else:
                        missing_list.append((seed, anchor, model))
        
        total_found += found
        total_complete += complete
        total_missing += len(missing_list)
        
        # Print summary
        print(f"\nActual:")
        print(f"  Found: {found}/{plan['expected_count']}")
        print(f"  ✓ Complete: {complete}")
        print(f"  ⚠ Incomplete: {found - complete}")
        print(f"  ✗ Missing: {len(missing_list)}")
        
        # Target coverage (for LB categories)
        if 'lb' in category and actual_experiments:
            targets = set(exp.get('target') for exp in actual_experiments.values() if exp.get('target'))
            print(f"\nTargets covered: {len(targets)}")
            target_counts = defaultdict(int)
            for exp in actual_experiments.values():
                if exp.get('target'):
                    target_counts[exp['target']] += 1
            for target, count in sorted(target_counts.items()):
                complete_count = sum(1 for exp in actual_experiments.values() 
                                   if exp.get('target') == target and exp['status'] == 'complete')
                print(f"  {target:20s}: {complete_count}/{count} complete")
        
        # Show matrix for sweep experiments
        if len(plan['anchors']) > 1 or len(plan['models']) > 1:
            param_name = 'anchors' if len(plan['anchors']) > 1 else 'models'
            param_values = plan['anchors'] if len(plan['anchors']) > 1 else plan['models']
            
            print(f"\n--- Coverage Matrix (seed × {param_name}) ---")
            
            # Header
            print(f"{'Seed':<6}", end='')
            for val in param_values:
                val_str = str(val) if val is not None else 'all'
                print(f"  {val_str:>4s}", end='')
            print()
            print("-" * (6 + 6 * len(param_values)))
            
            # Rows
            for seed in plan['seeds']:
                print(f"{seed:<6d}", end='')
                for val in param_values:
                    key = (seed, val, None) if param_name == 'anchors' else (seed, plan['anchors'][0], val)
                    
                    if key in actual_experiments:
                        status = actual_experiments[key]['status']
                        symbol = {'complete': '✓', 'partial': '⚠', 'incomplete': '○', 'missing': '✗'}[status]
                        print(f"    {symbol} ", end='')
                    else:
                        print(f"    - ", end='')
                print()
            
            print("\nLegend: ✓=complete  ⚠=partial  ○=incomplete  -=missing")
        
        # Show missing combinations (limit to 10)
        if missing_list:
            print(f"\n--- Missing Experiments ({len(missing_list)}) ---")
            for i, (seed, anchor, model) in enumerate(missing_list[:10]):
                model_str = f"_models_{model}" if model else ""
                print(f"  ✗ seed_{seed}_anchors_{anchor}{model_str}")
            if len(missing_list) > 10:
                print(f"  ... and {len(missing_list) - 10} more")
        
        # Show incomplete experiments (limit to 10)
        if incomplete_list:
            print(f"\n--- Incomplete Experiments ({len(incomplete_list)}) ---")
            for i, (seed, anchor, model, status, target) in enumerate(incomplete_list[:10]):
                model_str = f"_models_{model}" if model else ""
                target_str = f"_target_{target}" if target else ""
                print(f"  ⚠ seed_{seed}_anchors_{anchor}{model_str}{target_str} [{status}]")
            if len(incomplete_list) > 10:
                print(f"  ... and {len(incomplete_list) - 10} more")
        
        print()
    
    # Overall summary
    print("="*90)
    print("OVERALL SUMMARY")
    print("="*90)
    
    print(f"\nPlan vs Actual:")
    print(f"  Planned total:       {total_planned:3d} experiments")
    print(f"  Found:               {total_found:3d} ({100*total_found/total_planned:.1f}%)")
    print(f"  ✓ Complete:          {total_complete:3d} ({100*total_complete/total_planned:.1f}%)")
    print(f"  ⚠ Incomplete/Partial: {total_found - total_complete:3d}")
    print(f"  ✗ Missing:           {total_missing:3d}")
    
    # Category summary
    print(f"\nBy Category:")
    for category, plan in EXPERIMENT_PLAN.items():
        category_dir = base_dir / category
        if not category_dir.exists():
            status = "✗ missing"
        else:
            count = sum(1 for d in category_dir.iterdir() if d.is_dir())
            complete = sum(1 for d in category_dir.iterdir() 
                          if d.is_dir() and get_experiment_status(d) == 'complete')
            status = f"{complete}/{plan['expected_count']} complete"
        
        print(f"  {category:25s}: {status}")
    
    print("\n" + "="*90)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if total_complete >= total_planned * 0.8:
        print("  ✓ Good coverage! >80% experiments complete")
        print("  → Proceed to analysis: python scripts/summarize_experiments.py")
    elif total_found >= total_planned * 0.9:
        print("  ⚠ Most experiments started but many incomplete")
        print("  → Check for failures (OOM, crashes)")
        print("  → Re-run incomplete with: bash sh_run/run_all_experiments.sh --skip-existing")
    else:
        print("  ✗ Significant gaps in coverage")
        print("  → Re-run missing experiments: bash sh_run/run_all_experiments.sh")
    
    print("="*90)


if __name__ == "__main__":
    main()

