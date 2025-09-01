"""
Model utilities for HELM conversion.

Handles model information extraction, configuration, and metadata processing.
"""

import hashlib
from typing import Dict, Optional


def create_evaluation_id(run_spec: Dict, instance_id: str) -> str:
    """Create unique evaluation identifier."""
    run_name = run_spec.get("name")
    if not run_name:
        run_name = "unknown"
    evaluation_id = f"{run_name}-{instance_id}"
    hashed_id = hashlib.sha256(evaluation_id.encode()).hexdigest()
    return hashed_id


def get_model_info(model_name: str, model_metadata: Dict[str, dict]) -> Dict:
    """
    Get model information from metadata.

    Args:
        model_name: Name of the model
        model_metadata: Dictionary of model metadata

    Returns:
        Model information dictionary
    """
    name, family = model_name.split("/")
    # get model data if available
    # model_data = model_metadata[model_name]

    return {
    "name": model_name,
    "family": family
    }


def create_model_section(run_spec: Dict, model_metadata: Optional[Dict[str, dict]] = None) -> Dict:
    """
    Create model section for evaluation schema.

    Args:
        run_spec: Run specification dictionary
        model_metadata: Model metadata dictionary

    Returns:
        Model section dictionary
    """
    model_name = run_spec.get("adapter_spec", {}).get("model")
    return get_model_info(model_name, model_metadata)
