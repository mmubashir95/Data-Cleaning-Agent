"""Helpers for generating a short viva-ready project summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _humanize_action_name(action_name: str) -> str:
    mapping = {
        "duplicates_removed": "duplicate removal",
        "missing_values_handled": "missing value handling",
        "wrong_data_types_fixed": "data type fixing",
        "outliers_detected_and_handled": "outlier handling",
        "categorical_columns_encoded": "categorical encoding",
        "numeric_columns_scaled": "numeric scaling",
        "nlp_text_cleaning_applied": "NLP text cleaning",
    }
    return mapping.get(action_name, action_name.replace("_", " "))


def build_viva_summary(
    *,
    original_file_name: str,
    cleaning_summary: dict[str, Any],
    ml_recommendation: dict[str, Any],
    cleaning_actions: dict[str, Any],
) -> dict[str, Any]:
    """Create a short presentation-friendly summary for viva use."""
    dataset_name = Path(original_file_name).stem or "dataset"
    problem_type = ml_recommendation.get("recommended_problem_type", "Unknown")
    target_column = (
        ml_recommendation.get("selected_target_column")
        or ml_recommendation.get("suggested_target_column")
        or "No target column"
    )
    algorithm_names = [
        algorithm.get("name", str(algorithm))
        for algorithm in ml_recommendation.get("algorithms", [])
    ]
    recommended_algorithm = algorithm_names[0] if algorithm_names else "No algorithm recommendation"
    performed_actions = [
        _humanize_action_name(action_name)
        for action_name, details in cleaning_actions.items()
        if details.get("performed")
    ]
    skipped_actions = cleaning_summary.get("skipped_steps", [])

    if performed_actions:
        performed_actions_text = ", ".join(performed_actions)
    else:
        performed_actions_text = (
            "The dataset was profiled and reviewed, but no cleaning actions were applied."
        )

    skipped_actions_text = (
        " ".join(skipped_actions[:2])
        if skipped_actions
        else "No major cleaning steps were skipped after execution."
    )
    algorithm_list_text = ", ".join(algorithm_names) if algorithm_names else "No algorithm recommendations"
    smartphone_preprocessing_applied = cleaning_summary.get("smartphone_preprocessing_applied", False)

    architecture_text = (
        "Streamlit handles dataset upload and user controls, Python runs the real cleaning engine, "
        "Pandas and NumPy handle profiling, validation, and preprocessing, Flowise provides a "
        "human-readable AI explanation layer, and the final outputs are a cleaned CSV plus a JSON report."
    )
    workflow_text = (
        "The workflow is Upload -> Profile -> Detect Issues -> Clean -> Generate Report -> "
        "AI Explanation -> Download."
    )
    rationale_text = (
        "This design is better than sending the full file directly to an LLM because it avoids token "
        "limit errors, reduces hallucination, keeps Python responsible for accurate calculations, sends "
        "only a compact dataset profile to Flowise, and scales better for large datasets."
    )
    if smartphone_preprocessing_applied:
        dataset_text = (
            "The sir-provided dataset is related to smartphones on an e-commerce website. "
            "The goal is not simple classification because the Segment column is empty and no user rating/history target is available. "
            "Therefore, I treated it as a content-based recommendation problem. "
            "The dataset is complex and tricky because many important values are hidden inside text columns. "
            "For example, processor contains chipset and GHz speed, display contains screen size and refresh rate, "
            "camera contains MP values, battery contains mAh and charging watts, and card/OS columns may contain noisy or shifted values. "
            "Because of this, generic cleaning is not enough. "
            "I used Pandas and NumPy to clean the dataset, remove noise, handle missing values, extract smartphone-specific features, "
            "encode categorical columns, scale numeric columns, and generate both a cleaned readable dataset and an ML-ready recommendation dataset. "
            "The final ML-ready dataset can be used with cosine similarity to recommend smartphones based on similar specifications such as price, RAM, storage, battery, display, camera, processor, OS, and connectivity features."
        )
    else:
        dataset_text = (
            f"For the current dataset '{dataset_name}', the app identified a {problem_type} problem. "
            f"The target column is '{target_column}'. Cleaning actions performed: {performed_actions_text} "
            f"Skipped or limited actions: {skipped_actions_text} The recommended starter algorithm is "
            f"{recommended_algorithm}, with other suggested options including {algorithm_list_text}."
        )

    plain_text = " ".join(
        [
            architecture_text,
            workflow_text,
            dataset_text,
            rationale_text,
        ]
    )

    markdown = "\n".join(
        [
            "## Project Summary for Viva",
            "",
            architecture_text,
            "",
            workflow_text,
            "",
            dataset_text,
            "",
            rationale_text,
        ]
    )

    return {
        "dataset_name": dataset_name,
        "problem_type": problem_type,
        "target_column": target_column,
        "recommended_algorithm": recommended_algorithm,
        "recommended_algorithms": algorithm_names,
        "cleaning_actions_performed": performed_actions,
        "skipped_cleaning_actions": skipped_actions,
        "architecture_summary": architecture_text,
        "workflow_summary": workflow_text,
        "dataset_summary": dataset_text,
        "why_this_approach_is_better": rationale_text,
        "plain_text": plain_text,
        "markdown": markdown,
    }
