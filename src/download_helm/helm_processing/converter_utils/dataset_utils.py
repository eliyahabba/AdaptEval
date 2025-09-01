"""
Dataset utilities for HELM conversion.

Handles dataset name extraction, mapping, and processing.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .advanced_mapping import get_question_index_from_mapping


def validate_mapping_config(enabled: bool, mapping_dir: Optional[Path] = None) -> Tuple[bool, Optional[Path]]:
    """
    Validate mapping configuration arguments.
    
    Args:
        enabled: Whether to enable advanced mapping
        mapping_dir: Directory with mapping files (required if enabled=True)
    
    Returns:
        Tuple of (validated_enabled, validated_mapping_dir)
        
    Raises:
        ValueError: If configuration is invalid
    """
    if enabled:
        if mapping_dir is None:
            raise ValueError("mapping_dir is required when enabling mapping mode")
        if not mapping_dir.exists():
            raise ValueError(f"Mapping directory does not exist: {mapping_dir}")
        return True, mapping_dir
    else:
        return False, None


def load_dataset_mappings(mappings_file: Optional[Path] = None) -> Dict[str, str]:
    """
    Load dataset name mappings from JSON file.
    
    Args:
        mappings_file: Path to JSON file with mappings (optional)
        
    Returns:
        Dictionary of dataset name mappings
    """
    if mappings_file is None:
        # Try to find default mappings file
        current_dir = Path(__file__).parent.parent
        mappings_file = current_dir / "dataset_mappings.json"

    try:
        if mappings_file.exists():
            with open(mappings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("dataset_name_mappings", {})
    except Exception as e:
        print(f"Warning: Could not load mappings from {mappings_file}: {e}")

    return {}


def normalize_dataset_name(dataset_name: str,
                           custom_mappings: Optional[Dict[str, str]] = None,
                           mappings_file: Optional[Path] = None) -> str:
    """
    Normalize dataset name using configurable mappings from JSON file.
    
    Args:
        dataset_name: Raw dataset name
        custom_mappings: Optional custom mapping dictionary {old_name: new_name}
        mappings_file: Optional path to JSON mappings file
        
    Returns:
        Normalized dataset name
    """
    if not dataset_name:
        return dataset_name

    # Load mappings from JSON file
    mappings = load_dataset_mappings(mappings_file)

    # Override/extend with custom mappings if provided
    if custom_mappings:
        mappings.update(custom_mappings)

    # Handle compound names (e.g., mmlu.anatomy)
    if '.' in dataset_name:
        base_name, subject = dataset_name.split('.', 1)
        # Apply mapping to base name if exists
        normalized_base = mappings.get(base_name, base_name)
        return f"{normalized_base}.{subject}"

    # Apply direct mapping if exists, otherwise keep original
    return mappings.get(dataset_name, dataset_name)


def extract_dataset_name_from_run_spec(run_spec: Dict, scenario: Dict) -> Optional[str]:
    """
    Extract dataset name from HELM run specification and scenario.
    
    This function knows HELM's structure but doesn't hardcode specific dataset names.

    Args:
        run_spec: Run specification dictionary
        scenario: Scenario dictionary

    Returns:
        Extracted dataset name or None
    """
    dataset_base = None
    subject = None

    # Try to extract from scenario_spec class_name
    if run_spec and "scenario_spec" in run_spec:
        spec = run_spec.get("scenario_spec", {})
        class_name = spec.get("class_name", "")

        if class_name:
            # Extract dataset name from class name (remove common suffixes)
            dataset_from_class = class_name.lower()

            # Remove common HELM patterns
            if "_scenario" in dataset_from_class:
                dataset_from_class = dataset_from_class.replace("_scenario", "")

            # Handle nested class names (e.g., "commonsense_scenario.OpenBookQA")
            if "." in dataset_from_class:
                parts = dataset_from_class.split(".")
                # Use the last part (usually the actual dataset name)
                dataset_base = parts[-1].lower()
            else:
                dataset_base = dataset_from_class

        # Extract subject/subset from arguments
        if "args" in spec:
            args = spec["args"]
            # Try common argument names for subjects
            subject = (args.get("subject") or
                       args.get("subset") or
                       args.get("category"))

            # Handle translation datasets
            if not subject and "source_language" in args and "target_language" in args:
                subject = f"{args['source_language']}-{args['target_language']}"

    # Try to extract from run name parameters (e.g., "dataset=mmlu,subject=anatomy")
    if run_spec and "name" in run_spec:
        run_name = run_spec["name"]

        # Extract dataset parameter
        if not dataset_base and "dataset=" in run_name:
            dataset_base = _extract_parameter_from_string(run_name, "dataset")

        # Extract subject parameter
        if not subject:
            subject = (_extract_parameter_from_string(run_name, "subject") or
                       _extract_parameter_from_string(run_name, "subset"))

        # Fallback: use prefix before colon
        if not dataset_base and ":" in run_name:
            dataset_base = run_name.split(":")[0]

    # Last resort: use scenario name
    if not dataset_base and scenario and "name" in scenario:
        dataset_base = scenario["name"].lower()

    # Build final dataset name
    if dataset_base:
        if subject:
            return f"{dataset_base}.{subject}"
        else:
            return dataset_base

    return None


def _extract_parameter_from_string(text: str, param: str) -> Optional[str]:
    """Extract parameter value from comma-separated key=value string."""
    try:
        parts = [p.strip() for p in text.split(",") if f"{param}=" in p]
        if parts:
            return parts[0].split("=")[1]
        return None
    except:
        return None


def get_question_index_fallback(instance: Dict) -> Tuple[int, str]:
    """
    Fallback method to get question index from instance ID.
    
    This is the generic method that doesn't require mapping files.
    
    Args:
        instance: Instance dictionary with 'id' and 'split' fields
        
    Returns:
        Tuple of (instance_num, split)
    """
    try:
        # Extract instance number from ID (e.g., "id123" -> 123)
        instance_num = int(instance['id'].split('id')[1])
        split = instance.get('split', 'test')  # Default to 'test' if no split
        return instance_num, split
    except (ValueError, KeyError, IndexError) as e:
        print(f"Warning: Could not parse instance ID '{instance.get('id', 'N/A')}': {e}")
        return 0, 'test'


def get_question_index(dataset_name: str, question: str, choices: List[str], instance: Dict,
                       use_mapping: bool = False, mapping_dir: Optional[Path] = None) -> Tuple[int, str]:
    """
    Get question index and split, using mapping if available, otherwise fallback to ID parsing.

    Args:
        dataset_name: Name of the dataset
        question: Question text
        choices: List of choice texts
        instance: Instance dictionary (for fallback)
        use_mapping: Whether to use advanced mapping
        mapping_dir: Directory with mapping files (if use_mapping=True)

    Returns:
        Tuple of (hf_index, hf_split)
    """
    # Check if advanced mapping should be used
    if use_mapping and mapping_dir is not None:
        try:
            # Validate configuration
            validate_mapping_config(True, mapping_dir)

            # Try advanced mapping
            hf_index, hf_split = get_question_index_from_mapping(
                dataset_name, question, choices, mapping_dir
            )

            if hf_index is not None:
                return hf_index, hf_split
            else:
                print(f"Advanced mapping failed for dataset {dataset_name}, falling back to ID parsing")
        except Exception as e:
            print(f"Error in advanced mapping: {e}")

    # Fallback to ID-based approach
    return get_question_index_fallback(instance)


def create_instance_section(instance: Dict, display_request: Dict, dataset_name: str,
                            map_dir: Path = None, use_mapping: bool = False) -> Dict:
    """
    Create instance section for evaluation schema.

    Args:
        instance: Instance dictionary
        display_request: Display request dictionary
        dataset_name: Dataset name
        map_dir: Mapping directory (optional, for backward compatibility)
        use_mapping: Whether to use advanced mapping (if None, decides based on map_dir presence)

    Returns:
        Instance section dictionary
    """
    question_text = instance.get("input", {}).get("text")

    # Create mapping between letters and choice texts
    references = instance.get("references", [])
    choice_texts = []

    for ref in references:
        choice_text = ref.get("output", {}).get("text")
        choice_texts.append(choice_text)

    # Determine if mapping should be used
    if use_mapping is None:
        # Auto-detect: use mapping if map_dir is provided
        use_mapping = map_dir is not None

    # Get question index using the specified configuration
    instance_num, split = get_question_index(
        dataset_name=dataset_name,
        question=question_text,
        choices=choice_texts,
        instance=instance,
        use_mapping=use_mapping,
        mapping_dir=map_dir
    )

    return {
        "raw_input": question_text,
        "dataset_name": dataset_name,
        "hf_split": split,
        "hf_index": instance_num
    }
