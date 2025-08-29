"""
Evaluation utilities for HELM conversion.

Handles evaluation metrics, prompt format detection, and scoring.
"""

import re
from typing import Dict, List, Optional, Tuple
import math
import numpy as np


def get_evaluation_metrics() -> List[Tuple[str, Dict]]:
    """
    Get evaluation metrics in priority order.

    Returns:
        List of (metric_key, method_info) tuples in priority order
    """
    return [
        ("exact_match", {
            "method_name": "label_only_match",
            "description": "Compares only the choice identifier/label to evaluate the response."
        }),
        ("quasi_exact_match", {
            "method_name": "quasi_label_only_match",
            "description": "Compares only the choice identifier/label to evaluate the response with a tolerance for minor differences."
        }),
        ("final_number_exact_match", {
            "method_name": "final_number_match",
            "description": "Compares the final number of the response with the expected answer."
        }),
        ('math_equiv_chain_of_thought', {
            "method_name": "math_equiv_chain_of_thought",
            "description": "Evaluates the response based on mathematical equivalence in chain of thought reasoning."
        }),
        ('f1_score', {
            "method_name": "f1_score",
            "description": "Calculates the F1 score between the predicted and correct answers."
        }),
        ('exact_match_with_references', {
            "method_name": "exact_match_with_references",
            "description": "Compares the predicted answer with the correct answer using references."
        }),
        ('quasi_exact_match_with_references', {
            "method_name": "quasi_exact_match_with_references",
            "description": "Compares the predicted answer with the correct answer using references with a tolerance for minor differences."
        }),
        ('bleu_4', {
            "method_name": "bleu_4",
            "description": "Calculates the BLEU-4 score between the predicted and correct answers."
        }),
        ('rouge_l', {
            "method_name": "rouge_l",
            "description": "Calculates the ROUGE-L score between the predicted and correct answers."
        }),
        ('meteor', {
            "method_name": "meteor_match",
            "description": "Calculates the METEOR score between the predicted and correct answers."
        }),
    ]


def select_evaluation_score(stats: Dict) -> Tuple[str, float]:
    """
    Select evaluation score using supported metrics in priority order.

    Args:
        stats: Statistics dictionary from prediction

    Returns:
        Tuple of (method_name, score)

    Raises:
        ValueError: If no supported evaluation metric is found
    """
    metric_priority = get_evaluation_metrics()

    for key, method in metric_priority:
        value = stats.get(key)
        if value is not None:
            return method["method_name"], value

    raise ValueError(
        "No supported evaluation metric found in prediction stats. "
        "Expected one of: exact_match, quasi_exact_match, final_number_exact_match, math_equiv_chain_of_thought."
    )


def create_evaluation_section(prediction: Dict, instance: Dict) -> Dict:
    """
    Create evaluation section for evaluation schema.

    Args:
        prediction: Prediction dictionary
        instance: Instance dictionary

    Returns:
        Evaluation section dictionary
    """
    # Find correct answer
    references = instance.get("references", [])
    correct_id = None

    # Identify correct choice letter (A, B, C, D)
    for i, ref in enumerate(references):
        if "correct" in ref.get("tags", []):
            correct_id = chr(65 + i)  # A, B, C, D...
            break

    # Select score using supported metrics
    stats = prediction.get("stats", {}) or {}
    method_name, score = select_evaluation_score(stats)

    return {
        "ground_truth": correct_id,
        "method_name": method_name,
        "score": score,
    }


def create_output_section(prediction: Dict) -> str:
    """
    Create output section for evaluation schema.

    Args:
        prediction: Prediction dictionary

    Returns:
        Predicted text
    """
    predicted_text = prediction.get("predicted_text")
    if predicted_text:
        predicted_text = predicted_text.strip()
    return predicted_text


def convert_nan_to_null(obj):
    """
    Recursively convert NaN values to None.

    Args:
        obj: Object to process

    Returns:
        Object with NaN values converted to None
    """

    if isinstance(obj, dict):
        return {key: convert_nan_to_null(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_null(item) for item in obj]
    elif isinstance(obj, (float, np.float64, np.float32)) and (math.isnan(obj) or np.isnan(obj)):
        return None
    else:
        return obj
