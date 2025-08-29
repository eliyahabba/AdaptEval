"""
Dataset utilities for HELM conversion.

Handles dataset name extraction, mapping, and processing.
"""

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


def normalize_dataset_name(dataset_name: str) -> str:
    """
    Normalize dataset name to standard format.

    Args:
        dataset_name: Raw dataset name

    Returns:
        Normalized dataset name
    """
    if not dataset_name:
        return dataset_name

    # Specific normalizations
    if dataset_name == "openbookqa":
        return "openbook_qa"
    if dataset_name == "gsm":
        return "gsm8k"
    if dataset_name == "narrative_qa":
        return "narrativeqa"
    if dataset_name == "wmt-14":
        return "wmt14"

    # Handle compound names (e.g., mmlu.anatomy)
    if '.' in dataset_name:
        base_name, subject = dataset_name.split('.', 1)
        if base_name == "openbookqa":
            return f"openbook_qa.{subject}"
        if base_name == "gsm":
            return f"gsm8k.{subject}"
        if base_name == "narrative_qa":
            return f"narrativeqa.{subject}"
        if base_name == "wmt-14":
            return f"wmt14.{subject}"

    return dataset_name


def extract_dataset_name_from_run_spec(run_spec: Dict, scenario: Dict) -> Optional[str]:
    """
    Extract dataset name from run specification and scenario.

    Args:
        run_spec: Run specification dictionary
        scenario: Scenario dictionary

    Returns:
        Extracted dataset name or None
    """
    dataset_base = None
    subject = None

    # Extract from scenario_spec class_name
    if run_spec and "scenario_spec" in run_spec:
        spec = run_spec.get("scenario_spec", {})
        class_name = spec.get("class_name", "")

        if class_name:
            # Handle OpenBookQA in commonsense_scenario
            if "commonsense_scenario.OpenBookQA" in class_name:
                dataset_base = "openbook_qa"
            else:
                # Extract dataset name from class name
                for part in class_name.split('.'):
                    part_lower = part.lower()
                    if "_scenario" in part_lower:
                        dataset_base = part_lower.replace("_scenario", "")
                        break

        # Extract subject from arguments
        if "args" in spec:
            args = spec["args"]
            if "subject" in args and args["subject"]:
                subject = args["subject"]
            elif "subset" in args and args["subset"]:
                subject = args["subset"]
            elif 'source_language' in args and 'target_language' in args:
                # Translation case: extract languages
                source_lang = args['source_language']
                target_lang = args['target_language']
                subject = f"{source_lang}-{target_lang}"

    # Extract from run name parameters
    if not dataset_base and run_spec and "name" in run_spec:
        run_name = run_spec["name"]

        # Check for dataset= parameter
        if "dataset=" in run_name:
            dataset_parts = [p for p in run_name.split(",") if "dataset=" in p]
            if dataset_parts:
                dataset_value = dataset_parts[0].split("=")[1]
                if dataset_value.lower() == "openbookqa":
                    dataset_base = "openbook_qa"
                else:
                    dataset_base = dataset_value

        # Check for subset= parameter
        if "subset=" in run_name:
            subset_parts = [p for p in run_name.split(",") if "subset=" in p]
            if subset_parts:
                subject = subset_parts[0].split("=")[1]

        # Fallback: extract from run name prefix
        elif ":" in run_name and not dataset_base:
            dataset_base = run_name.split(":")[0]

        # Extract subject from run name
        if not subject and "subject=" in run_name:
            subject_part = [p for p in run_name.split(",") if "subject=" in p]
            if subject_part:
                subject = subject_part[0].split("=")[1]

    # Extract from scenario name as last resort
    if not dataset_base and scenario and "name" in scenario:
        scenario_name = scenario["name"]
        if scenario_name.lower() == "openbookqa":
            dataset_base = "openbook_qa"
        else:
            dataset_base = scenario_name

    # Build full dataset name
    if dataset_base:
        if subject:
            dataset_name = f"{dataset_base}.{subject}"
        else:
            dataset_name = dataset_base
        return dataset_name

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
