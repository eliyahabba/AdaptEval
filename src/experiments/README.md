# Experiments Folder Organization

This folder contains various experimental scripts organized by purpose.

## Folder Structure

### 📊 `visualization/`
Scripts for generating plots and visualizations:
- `visualize_paper_v3.py` - Main visualization script for paper (v3)
- `visualize_paper_focused.py` - Focused visualization for paper
- `plot_cost_vs_performance.py` - Cost vs performance analysis plots
- `visualize_per_model.py` - Per-model visualizations
- `visualize_chain_linking.py` - Chain linking visualizations
- `visualize_topographic.py` - Topographic map visualizations

### 🔗 `chain_linking/`
Chain linking experimental scripts:
- `chain_linking_disjoint.py` - Disjoint chain linking experiments
- `chain_linking_parallel_b.py` - Parallel chain linking experiments (version b)
- `chain_linking_v2.py` - Chain linking version 2
- `chain_linking_experiment.py` - Main chain linking experiments
- `aggregate_chain_results.py` - Aggregate results from chain linking

### ✅ `validation/`
Data validation and quality checking scripts:
- `simple_mmlu_validation.py` - MMLU validation script
- `simple_mmlu_validation_pickle.py` - MMLU validation from pickle files
- `helm_parquet_validation.py` - HELM parquet file validation

### 🔧 `data_processing/`
Data preparation and processing scripts:
- `prepare_reeval_dataset.py` - Prepare re-evaluation datasets
- `fix_pooled_results.py` - Fix pooled results data

### ⚙️ `config/`
Configuration files:
- `data_source_config.json` - Data source configuration
- `helm_classic_items.json` - HELM classic items configuration
- `dataset_items_config.json` - Dataset items configuration

### 🛠️ `utils/`
Utility scripts:
- `model_matcher.py` - Model name matching utilities

### 📈 `equating/`
IRT and cross-dataset equating:
- `cross_dataset_equating.py` - Cross-dataset equating script
- `irt_equating/` - IRT equating subfolder

### 🗄️ `old_visualization/`
Archived visualization scripts (old versions):
- Various old visualization scripts that are no longer actively used

### 🗂️ `old/`
Other archived/deprecated scripts

## Usage Notes

- Config files in `config/` are used by multiple scripts across folders
- Visualization scripts may depend on data processing outputs
- Chain linking scripts are experimental and may have dependencies on each other

