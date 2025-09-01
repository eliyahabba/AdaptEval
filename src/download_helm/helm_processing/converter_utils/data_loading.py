"""
Data loading utilities for HELM conversion.

Handles JSON file loading, CSV processing, and file system operations.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def load_json_file(filepath: str) -> dict:
    """Load JSON data from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_model_metadata(csv_path: str) -> Dict[str, dict]:
    """
    Load model metadata from CSV file.

    Args:
        csv_path: Path to model metadata CSV file

    Returns:
        Dictionary mapping model names to their metadata
    """
    try:

        df = pd.read_csv(csv_path)
        model_info = {}

        for _, row in df.iterrows():
            model_name = row['name']
            model_info[model_name] = {
                "family": row['family'],
                "architecture": row['architecture'],
                "parameters": row['parameters'],
                "context_window": row['context_window'],
                "is_instruct": row['is_instruct'] == 'true',
                "hf_path": row['hf_path'] if pd.notna(row['hf_path']) else None,
                "revision": row['revision'] if pd.notna(row['revision']) else None,
                "quantization": {
                    "bit_precision": row['quantization_bit_precision'],
                    "method": row['quantization_method']
                }
            }

        print(f"Loaded metadata for {len(model_info)} models from {csv_path}")
        return model_info

    except Exception as e:
        print(f"Error loading model metadata from CSV: {e}")
        return {}


def load_dataset_mapping(map_dir: Path, map_file_name: str) -> Optional[dict]:
    """
    Load dataset-to-HF mapping from JSON file.

    Args:
        map_dir: Directory containing mapping files
        map_file_name: Name of mapping file (without _samples.json suffix)

    Returns:
        Mapping dictionary or None if file not found
    """
    json_path = map_dir / f"{map_file_name}_samples.json"

    if not json_path.exists():
        print(f"Warning: Map file {json_path} not found. Using default values.")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_csv_file(data: List[dict], output_path: str) -> None:
    """
    Save data to CSV file.

    Args:
        data: List of dictionaries to save
        output_path: Path where CSV should be saved
    """
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
