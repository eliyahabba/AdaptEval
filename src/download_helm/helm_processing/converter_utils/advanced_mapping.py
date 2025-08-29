"""
Advanced dataset mapping utilities.

This module contains the sophisticated dataset mapping logic that requires
JSON mapping files. It's only used when explicitly enabled via configuration.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .data_loading import load_dataset_mapping


# Global cache for dataset mappings
INDEX_MAP_CACHE = {}


def get_dataset_file_mapping_name(dataset_name: str) -> str:
    """
    Get the mapping file name for a dataset.

    Args:
        dataset_name: Dataset name

    Returns:
        Mapping file base name
    """
    map_file_name = dataset_name.split('.')[0] if '.' in dataset_name else dataset_name

    # Handle special cases
    if map_file_name == "ai2_arc":
        map_file_name = "ai2_arc"
    elif map_file_name in ("gsm", "gsm8k"):
        map_file_name = "gsm8k"
    elif map_file_name == "legalbench" and '.' in dataset_name:
        # legalbench.<subset> → use subset-specific map file name
        subset = dataset_name.split('.')[1]
        map_file_name = f"legalbench.{subset}"
    elif map_file_name == "narrativeqa":
        map_file_name = "narrativeqa"
    elif map_file_name == "wmt14" and '.' in dataset_name:
        # wmt14.cs-en → wmt14.cs-en_samples.json
        pair = dataset_name.split('.')[1]
        map_file_name = f"wmt14.{pair}"
    elif map_file_name.startswith("global_mmlu"):
        # Add language to file name
        map_file_name = f"{map_file_name}.{dataset_name.split('.')[1]}"

    return map_file_name


def find_question_in_mapping(question: str, choices: List[str], index_map: Dict) -> Tuple[Optional[int], Optional[str]]:
    """
    Find question in dataset mapping.

    Args:
        question: Question text
        choices: List of choice texts
        index_map: Dataset mapping dictionary

    Returns:
        Tuple of (index, source) or (None, None) if not found
    """
    # Clean and normalize choices
    clean_choices = [choice.strip() for choice in choices]

    # Create search key in mapping format
    key = f"{question}|||{'|||'.join(sorted(clean_choices))}"

    # Direct lookup
    if key in index_map:
        metadata = index_map[key]
        print(f"Match found - index: {metadata['index']}, source: {metadata['source']}")
        return metadata['index'], metadata['source']

    # Try question-only lookup
    key_question_only = f"{question}|||"
    if key_question_only in index_map:
        metadata = index_map[key_question_only]
        print(f"Match found - index: {metadata['index']}, source: {metadata['source']}")
        return metadata['index'], metadata['source']

    # Try removing "Question: " prefix
    if "Question: " in key:
        key_no_prefix = key.split('Question: ')[1]
        if key_no_prefix in index_map:
            metadata = index_map[key_no_prefix]
            print(f"Match found - index: {metadata['index']}, source: {metadata['source']}")
            return metadata['index'], metadata['source']

    # No match found
    print(f"No match found for key in mapping")
    print(f"Question: {question}")
    print(f"Choices: {clean_choices}")

    return None, None


def get_question_index_from_mapping(dataset_name: str, question: str, choices: List[str], map_dir: Path) -> Tuple[Optional[int], Optional[str]]:
    """
    Get question index and split from dataset mapping using JSON files.

    Args:
        dataset_name: Name of the dataset
        question: Question text
        choices: List of choice texts
        map_dir: Directory containing mapping files

    Returns:
        Tuple of (hf_index, hf_split) or (None, None) if not found
    """
    try:
        # Get mapping file name
        map_file_name = get_dataset_file_mapping_name(dataset_name)

        # Load mapping if not cached
        if map_file_name not in INDEX_MAP_CACHE:
            INDEX_MAP_CACHE[map_file_name] = load_dataset_mapping(map_dir, map_file_name)

        index_map = INDEX_MAP_CACHE[map_file_name]

        if index_map is None:
            return None, None

        # Search for question
        hf_index, hf_split = find_question_in_mapping(question, choices, index_map)

        return hf_index, hf_split

    except Exception as e:
        print(f"Error in get_question_index_from_mapping: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None, None


def clear_mapping_cache():
    """Clear the mapping cache. Useful for testing or memory management."""
    global INDEX_MAP_CACHE
    INDEX_MAP_CACHE.clear()
