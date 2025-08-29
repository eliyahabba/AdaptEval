"""
HELM data converter.

Converts HELM evaluation data to standardized evaluation schema format.
"""

import argparse
import json
import os
from typing import List

# Import centralized settings
from settings import HF_MAP_DATA_DIR, DATASET_REGISTRY, MODEL_METADATA_CSV
# Import utility modules
from .converter_utils.data_loading import (
    load_json_file,
    get_model_metadata,
    save_csv_file,
)
from .converter_utils.dataset_utils import (
    extract_dataset_name_from_run_spec,
    create_instance_section,
)
from .converter_utils.evaluation_utils import (
    create_evaluation_section,
    create_output_section,
    convert_nan_to_null,
)
from .converter_utils.model_utils import create_evaluation_id, create_model_section

# Global variables for caching
MODEL_INFO = None
DATASET_TO_HF_REPO = DATASET_REGISTRY


def initialize_model_info():
    """Initialize model metadata cache."""
    global MODEL_INFO
    if MODEL_INFO is None:
        MODEL_INFO = get_model_metadata(str(MODEL_METADATA_CSV))


# Initialize model info cache
initialize_model_info()


def process_helm_data(data_dir: str) -> List[dict]:
    """
    Process HELM data and create examples in the new schema format.

    Args:
        data_dir: Directory containing HELM data files

    Returns:
        List of processed examples
    """
    # Load JSON files
    run_spec = load_json_file(os.path.join(data_dir, "run_spec.json"))
    scenario = load_json_file(os.path.join(data_dir, "scenario.json"))
    instances = load_json_file(os.path.join(data_dir, "instances.json"))
    display_requests = load_json_file(os.path.join(data_dir, "display_requests.json"))
    display_predictions = load_json_file(os.path.join(data_dir, "display_predictions.json"))

    # Create mappings for efficient lookup
    request_map = {req["instance_id"]: req for req in display_requests}
    prediction_map = {pred["instance_id"]: pred for pred in display_predictions}

    examples = []

    # Extract dataset name once
    dataset_name = extract_dataset_name_from_run_spec(run_spec, scenario)
    print(f"Dataset name extracted: {dataset_name}")

    # Process each instance
    for instance in instances:
        instance_id = instance.get("id")

        # Check if we have both request and prediction data for this instance
        if instance_id in request_map and instance_id in prediction_map:
            request = request_map[instance_id]
            prediction = prediction_map[instance_id]

            # Create example using the modular structure
            example = {
                "evaluation_id": create_evaluation_id(run_spec, instance_id),
                "model": create_model_section(run_spec, MODEL_INFO),
                "instance": create_instance_section(instance, request, dataset_name, HF_MAP_DATA_DIR),
                "output": create_output_section(prediction),
                "evaluation": create_evaluation_section(prediction, instance)
            }

            # Flatten the example structure
            flattened = {**example, **example["instance"], **example["model"], **example["evaluation"]}
            flattened.pop("instance", None)
            flattened.pop("model", None)
            flattened.pop("evaluation", None)

            # Rename fields for consistency
            flattened["model_name"] = flattened.pop("name", None)
            flattened["model_family"] = flattened.pop("family", None)
            flattened["evaluation_method_name"] = flattened.pop("method_name", None)
            flattened["evaluation_score"] = flattened.pop("score", None)

            # Sort fields in desired order
            ordered_fields = [
                'evaluation_id', 'dataset_name', 'hf_split', 'hf_index',
                'raw_input', 'ground_truth', 'model_name', 'model_family',
                'output', 'evaluation_method_name', 'evaluation_score'
            ]

            ordered_example = {k: flattened[k] for k in ordered_fields if k in flattened}
            examples.append(ordered_example)

    return examples


def main(data_dir: str, output_file: str) -> None:
    """
    Main function to process and convert HELM data.

    Args:
        data_dir: Directory containing HELM data files
        output_file: Path where CSV should be saved
    """
    examples = process_helm_data(data_dir)
    examples_processed = convert_nan_to_null(examples)

    # Save results as CSV
    save_csv_file(examples_processed, output_file)

    print(f"Converted {len(examples)} examples and saved to {output_file}")

    # Also print the first example in JSON format for quick inspection
    # if examples:
    #     print("\nFirst example:")
    #     print(json.dumps(examples_processed[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert HELM data to the new schema format")
    parser.add_argument("--data-dir", required=True, help="Directory containing HELM data files")
    parser.add_argument("--output-file", required=True, help="Output CSV file path")
    args = parser.parse_args()

    main(args.data_dir, args.output_file)
