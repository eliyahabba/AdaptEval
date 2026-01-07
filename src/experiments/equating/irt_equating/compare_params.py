
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json

def compare_item_parameters(skill_name: str, skills_root: str):
    root = Path(skills_root)
    skill_dir = root / skill_name
    
    if not skill_dir.exists():
        print(f"❌ Skill directory not found: {skill_dir}")
        return

    # Define paths
    baseline_path = skill_dir / "irt" / "item_params.parquet"
    fixed_path = skill_dir / "equating" / "fixed_anchor" / "item_params_fixed_anchor.parquet"
    baseline_meta_path = skill_dir / "irt" / "item_params.meta.json"
    fixed_meta_path = skill_dir / "equating" / "fixed_anchor" / "item_params_fixed_anchor.meta.json"

    if not baseline_path.exists():
        print(f"❌ Baseline params missing: {baseline_path}")
        return
    if not fixed_path.exists():
        print(f"❌ Fixed anchor params missing: {fixed_path}")
        return

    print(f"📊 Comparing parameters for skill: {skill_name}")
    print(f"   Baseline: {baseline_path.name}")
    print(f"   Fixed:    {fixed_path.name}")
    
    # Load data
    df_base = pd.read_parquet(baseline_path)
    df_fixed = pd.read_parquet(fixed_path)

    # Ensure question_id index
    if "question_id" in df_base.columns:
        df_base = df_base.set_index("question_id")
    if "question_id" in df_fixed.columns:
        df_fixed = df_fixed.set_index("question_id")

    # Find common items (Anchors candidates)
    common_ids = df_base.index.intersection(df_fixed.index)
    print(f"\n   Found {len(common_ids)} common items (potential anchors).")
    
    if len(common_ids) == 0:
        print("   ⚠️ No common items found.")
        return

    base_subset = df_base.loc[common_ids]
    fixed_subset = df_fixed.loc[common_ids]

    # Compare discrimination (a) and difficulty (b) scalars
    for param in ['a', 'b']:
        if param not in base_subset.columns:
            continue
            
        diff = np.abs(base_subset[param] - fixed_subset[param])
        
        print(f"\n   🔹 Scalar Parameter '{param}':")
        print(f"     Mean Diff: {diff.mean():.6f}")
        print(f"     Max Diff:  {diff.max():.6f}")
        print(f"     Std Dev:   {diff.std():.6f}")
        
        # Show top 5 changes
        top_diffs = diff.nlargest(5)
        print("     Top 5 biggest changes:")
        for q_id, d_val in top_diffs.items():
            orig = base_subset.loc[q_id, param]
            new = fixed_subset.loc[q_id, param]
            print(f"       - {q_id}: {orig:.4f} -> {new:.4f} (Δ {d_val:.4f})")
    
    # Compare full MIRT vectors if metadata exists
    if baseline_meta_path.exists() and fixed_meta_path.exists():
        print("\n   🔹 Full MIRT Vectors:")
        try:
            with open(baseline_meta_path) as f:
                base_meta = json.load(f)
            with open(fixed_meta_path) as f:
                fixed_meta = json.load(f)
            
            if "A_matrix" in base_meta and "A_matrix" in fixed_meta:
                A_base = np.array(base_meta["A_matrix"])  # (1, D, n_items)
                A_fixed = np.array(fixed_meta["A_matrix"])
                B_base = np.array(base_meta["B_matrix"])
                B_fixed = np.array(fixed_meta["B_matrix"])
                
                base_qids = list(df_base.index)
                fixed_qids = list(df_fixed.index)
                
                # Compare vectors for common items
                a_max_diff = 0.0
                b_max_diff = 0.0
                a_diffs = []
                b_diffs = []
                
                for qid in common_ids:
                    if qid in base_qids and qid in fixed_qids:
                        base_idx = base_qids.index(qid)
                        fixed_idx = fixed_qids.index(qid)
                        
                        a_diff = np.abs(A_base[0, :, base_idx] - A_fixed[0, :, fixed_idx]).max()
                        b_diff = np.abs(B_base[0, :, base_idx] - B_fixed[0, :, fixed_idx]).max()
                        
                        a_diffs.append(a_diff)
                        b_diffs.append(b_diff)
                        a_max_diff = max(a_max_diff, a_diff)
                        b_max_diff = max(b_max_diff, b_diff)
                
                print(f"     A vectors (discrimination):")
                print(f"       Mean Max Diff: {np.mean(a_diffs):.10f}")
                print(f"       Max Max Diff:  {a_max_diff:.10f}")
                print(f"     B vectors (difficulty):")
                print(f"       Mean Max Diff: {np.mean(b_diffs):.10f}")
                print(f"       Max Max Diff:  {b_max_diff:.10f}")
                
                if a_max_diff < 1e-6 and b_max_diff < 1e-6:
                    print("\n   ✅ FULL VECTORS ARE IDENTICAL!")
                else:
                    print("\n   ⚠️ VECTORS DIFFER - Check implementation")
        except Exception as e:
            print(f"     Error loading metadata: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", help="Name of the skill to analyze", default="Entailment & Bias")
    parser.add_argument("--skills-root", default=str(Path(__file__).resolve().parents[3] / "data" / "processed" / "skills"))
    args = parser.parse_args()
    
    compare_item_parameters(args.skill, args.skills_root)

