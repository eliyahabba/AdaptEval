"""
Example usage of HELM processing with argument-based configuration.

This script demonstrates how to use the HELM processing system in both
advanced mapping mode (with JSON files) and generic fallback mode.
"""

from pathlib import Path
from converter_utils.dataset_utils import create_instance_section, normalize_dataset_name


def example_with_mapping_enabled():
    """Example: Using advanced mapping mode with JSON files."""
    print("=== Example: Advanced Mapping Mode ===")

    # Mapping directory - replace with your actual mapping directory
    mapping_dir = Path("/path/to/your/mapping/files")

    # Process data with advanced mapping (if directory exists)
    if mapping_dir.exists():
        print(f"✓ Using advanced mapping with directory: {mapping_dir}")
        result = process_helm_data_with_mapping(mapping_dir)
    else:
        print("✗ Mapping directory not found, using fallback mode")
        result = process_helm_data_fallback()

    return result


def example_with_mapping_disabled():
    """Example: Using fallback mode without JSON files."""
    print("\n=== Example: Generic Fallback Mode ===")
    print("✓ Using generic fallback mode (ID-based)")

    # Process data using only ID parsing
    result = process_helm_data_fallback()
    return result


def example_explicit_configuration():
    """Example: Explicit configuration with arguments."""
    print("\n=== Example: Explicit Configuration ===")

    # Process with explicit parameters
    result = process_helm_data_explicit()
    return result


def process_helm_data_with_mapping(mapping_dir: Path):
    """Process data using advanced mapping."""
    sample_instance = get_sample_instance()
    
    # Use any dataset name - completely generic
    dataset_name = "your_dataset.your_subject"  # Replace with actual dataset name
    
    result = create_instance_section(
        instance=sample_instance,
        display_request={},
        dataset_name=dataset_name,
        map_dir=mapping_dir,
        use_mapping=True
    )
    
    print(f"Result with mapping: {result}")
    return result


def process_helm_data_fallback():
    """Process data using fallback mode."""
    sample_instance = get_sample_instance()
    
    # Any dataset name works - no restrictions
    dataset_name = "any_dataset_name.any_subject"  # Replace with actual dataset name
    
    result = create_instance_section(
        instance=sample_instance,
        display_request={},
        dataset_name=dataset_name,
        use_mapping=False
    )
    
    print(f"Result with fallback: {result}")
    return result


def process_helm_data_explicit():
    """Process data with explicit parameters."""
    sample_instance = get_sample_instance()
    
    # Option 1: Use default JSON mappings
    dataset_name = normalize_dataset_name("gsm.mathematics")  # Will use dataset_mappings.json
    
    # Option 2: Use custom mappings (overrides JSON)
    custom_mappings = {"legacy_eval": "modern_eval"}
    dataset_name_custom = normalize_dataset_name("legacy_eval.category", custom_mappings)
    
    # Option 3: Use completely new dataset (no mapping needed)
    new_dataset_name = "my_company_dataset.finance"  # Works as-is
    
    result = create_instance_section(
        instance=sample_instance,
        display_request={},
        dataset_name=new_dataset_name
        # No map_dir or use_mapping specified = auto fallback
    )
    
    print(f"Result with auto-detect: {result}")
    return result


def get_sample_instance():
    """Get sample instance data."""
    return {
        "id": "id42",
        "split": "test",
        "input": {
            "text": "Sample question text"
        },
        "references": [
            {"output": {"text": "Option A"}},
            {"output": {"text": "Option B"}},
            {"output": {"text": "Option C"}},
            {"output": {"text": "Option D"}}
        ]
    }


if __name__ == "__main__":
    print("HELM Processing Configuration Examples")
    print("=" * 50)

    # Example 1: Try advanced mapping (will fallback if no files)
    example_with_mapping_enabled()

    # Example 2: Explicitly use fallback mode
    example_with_mapping_disabled()

    # Example 3: Explicit configuration
    example_explicit_configuration()

    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use advanced mapping in your own code:")
    print("1. Pass: map_dir=Path('/path/to/json/files'), use_mapping=True")
    print("2. Your code will use advanced mapping")
    print("\nTo use generic mode (for sharing with others):")
    print("1. Pass: use_mapping=False  # or don't pass map_dir at all")
    print("2. Your code will use ID-based fallback")
    print("3. No JSON files required!")
